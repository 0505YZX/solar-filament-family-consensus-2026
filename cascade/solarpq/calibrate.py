"""Decision-theoretic component selection.

For a candidate component c of an image with annotator records r=1..R let
        u_c = (1/R) * sum_r  IoU(c, g_r) * 1[IoU(c, g_r) > 0.5]      ("expected matched IoU")
Adding c changes the pooled PQ denominator by exactly +0.5 per record (hit: +1 TP -1/2 FN; miss: +1/2 FP) and the
numerator by R*u_c, so PQ increases  <=>  u_c > PQ/2   (metrics.include_threshold).

We therefore (i) regress u_c from shape / probability / physical-contrast descriptors with gradient boosting, cross-fitted
over *observations* so no image scores its own candidates, and (ii) keep candidates with u_hat >= tau, tau tuned on
out-of-fold predictions.  The fitted optimum can be compared with the theoretical PQ/2 (a built-in sanity check).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from . import geometry as G
from . import postprocess as P
from . import metrics as Mx
from .config import Config


@dataclass
class ObsCandidates:
    stem: str
    table: pd.DataFrame                      # one row per candidate component (features)
    labels: np.ndarray | None = None         # label image at work_res (kept only when requested)
    mats: dict = field(default_factory=dict) # record_id -> (n_gt, n_cand) IoU matrix at work_res
    n_gt: dict = field(default_factory=dict)

    @property
    def n_cand(self) -> int:
        return len(self.table)

    def u_target(self) -> np.ndarray:
        """Expected matched IoU per candidate (mean over records)."""
        if not self.mats or self.n_cand == 0:
            return np.zeros(self.n_cand)
        acc = np.zeros(self.n_cand)
        for iou in self.mats.values():
            if iou.shape[0] == 0:
                continue
            acc += (iou * (iou > Mx.IOU_THR)).sum(0)
        return acc / len(self.mats)


def flat_from_channel(c0: np.ndarray) -> np.ndarray:
    return 1.0 + 0.25 * c0.astype(np.float32)


def build_candidates(stem: str, prob: np.ndarray, chans: np.ndarray, disk: np.ndarray, cfg: Config,
                     gt_rles_work: dict | None = None, keep_labels: bool = False, light: bool = False,
                     t_hi=None, t_lo=None, close_radius=None, min_area=None) -> ObsCandidates:
    """prob: (H,W) float32 in [0,1] at work_res; chans (3,H,W); disk (cx,cy,R)."""
    H = prob.shape[0]
    rmap = G.radius_map((H, H), *disk)
    valid = rmap < cfg.disk_frac
    lab = P.extract_labels(prob, valid,
                           cfg.t_hi if t_hi is None else t_hi, cfg.t_lo if t_lo is None else t_lo,
                           cfg.close_radius if close_radius is None else close_radius,
                           cfg.min_area if min_area is None else min_area)
    if light:      # only what the rule-based extraction search needs: label + mean probability
        from scipy import ndimage as ndi
        n = int(lab.max()); ids = np.arange(1, n + 1)
        tab = pd.DataFrame({"label": ids, "p_mean": ndi.mean(prob, lab, ids) if n else []})
    else:
        flat = flat_from_channel(chans[0])
        tab = P.component_table(lab, prob, chans, flat, rmap, valid, stem[-2:])
    if keep_labels:
        lab = lab.astype(np.uint8 if lab.max() < 255 else np.int32)        # 1 MB instead of 4 MB per image
    oc = ObsCandidates(stem=stem, table=tab, labels=lab if keep_labels else None)
    if gt_rles_work is not None:
        crles = [Mx.encode(lab == k) for k in tab["label"].tolist()]   # (lab may be uint8 here; comparison is dtype-safe)
        for rid, gts in gt_rles_work.items():
            oc.mats[rid] = Mx.iou_matrix(gts, crles) if crles else np.zeros((len(gts), 0))
            oc.n_gt[rid] = len(gts)
    return oc


def pq_of_selection(cands: list, selected: list) -> Mx.PQ:
    """Pooled PQ (work-res approximation of the official metric) when only the selected candidates are submitted."""
    parts = []
    for oc, sel in zip(cands, selected):
        for rid, iou in oc.mats.items():
            sub = iou[:, sel] if iou.shape[1] else iou
            parts.append(Mx.score_matrix(sub, oc.n_gt[rid], int(np.sum(sel))))
    return Mx.pool(parts)


# ------------------------------------------------------------------ scorer
FEATURES = P.FEATURE_COLUMNS


def stack_features(cands: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.concatenate([c.table[FEATURES].to_numpy(np.float64) for c in cands if c.n_cand], 0)
    y = np.concatenate([c.u_target() for c in cands if c.n_cand], 0)
    g = np.concatenate([np.full(c.n_cand, i) for i, c in enumerate(cands) if c.n_cand], 0)
    return X, y, g


def _gbm(seed=0):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=12, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=seed)


def crossfit_scores(cands: list, n_splits: int = 5, seed: int = 0):
    """Out-of-fold u_hat for every candidate (GroupKFold over observations) + a final model fitted on everything."""
    X, y, g = stack_features(cands)
    oof = np.zeros(len(y))
    n_splits = min(n_splits, len(np.unique(g)))
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, g):
        oof[va] = _gbm(seed).fit(X[tr], y[tr]).predict(X[va])
    final = _gbm(seed).fit(X, y)
    return np.clip(oof, 0, 1), final, (X, y, g)


def split_by_obs(scores: np.ndarray, cands: list) -> list:
    out, k = [], 0
    for c in cands:
        out.append(scores[k:k + c.n_cand]); k += c.n_cand
    return out


def tune_tau(cands: list, u_hat_by_obs: list, grid=None):
    """PQ as a function of tau on out-of-fold scores. Returns (best_tau, best_PQ, curve DataFrame)."""
    grid = np.round(np.arange(0.0, 0.62, 0.02), 3) if grid is None else grid
    rows = []
    for t in grid:
        sel = [u >= t for u in u_hat_by_obs]
        r = pq_of_selection(cands, sel)
        rows.append(dict(tau=t, pq=r.pq, dq=r.dq, sq=r.sq, tp=r.tp, fp=r.fp, fn=r.fn, n_pred=int(sum(s.sum() for s in sel))))
    curve = pd.DataFrame(rows)
    best = curve.loc[curve.pq.idxmax()]
    return float(best.tau), float(best.pq), curve


def oracle_pq(cands: list, tau_grid=None):
    """PQ when candidates are selected by their TRUE u (upper bound for any scorer; also checks tau* ~ PQ*/2)."""
    return tune_tau(cands, [c.u_target() for c in cands], tau_grid)


# ------------------------------------------------------------------ batch helpers and extraction-parameter search
def build_all_candidates(idx, stems, prob_of, cache, cfg: Config, gtw: dict | None, n_jobs: int = 1, **kw) -> list:
    """Build ObsCandidates for cache indices `idx`. prob_of(i) -> float32 probability map. Threads (cv2/scipy release the GIL)."""
    from joblib import Parallel, delayed
    def one(i):
        return build_candidates(stems[i], prob_of(i), np.asarray(cache["chans"][i]), cache["disk"][i], cfg,
                                None if gtw is None else gtw[stems[i]], **kw)
    if n_jobs == 1:
        return [one(i) for i in idx]
    return Parallel(n_jobs=n_jobs, backend="threading")(delayed(one)(i) for i in idx)


def tune_extraction(idx, stems, prob_of, cache, cfg: Config, gtw: dict, grid: dict, p_mean_grid=(0.0, 0.2, 0.3, 0.4, 0.5),
                    n_jobs: int = 1, log=print) -> pd.DataFrame:
    """Coarse grid over hysteresis / closing parameters. Each setting is scored with the simple rule 'keep if mean prob >= q'
    (best q per setting) so that this step does not depend on the learned scorer."""
    import itertools
    rows = []
    for t_lo, t_hi, cr in itertools.product(grid["t_lo"], grid["t_hi"], grid["close_radius"]):
        if t_hi < t_lo:
            continue
        cands = build_all_candidates(idx, stems, prob_of, cache, cfg, gtw, n_jobs, t_hi=t_hi, t_lo=t_lo, close_radius=cr, light=True)
        best = max(((pq_of_selection(cands, [c.table["p_mean"].to_numpy() >= q for c in cands]), q) for q in p_mean_grid),
                   key=lambda t: t[0].pq)
        rows.append(dict(t_lo=t_lo, t_hi=t_hi, close_radius=cr, p_mean_thr=best[1], pq=best[0].pq, dq=best[0].dq, sq=best[0].sq,
                         n_cand=int(sum(c.n_cand for c in cands))))
        log(f"  t_lo={t_lo} t_hi={t_hi} close={cr}: PQ={best[0].pq:.4f} (q={best[1]})")
    return pd.DataFrame(rows).sort_values("pq", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ root-causing false negatives
def fn_breakdown(cands: list, u_by: list, tau: float, station_of=None) -> pd.DataFrame:
    """For every ground-truth instance, classify why it is or isn't a TP, using the FULL (pre-tau) candidate set:

        true_positive     a candidate with IoU > 0.5 exists AND its u_hat >= tau (correctly kept)
        rejected_by_tau    a candidate with IoU > 0.5 exists but ALL such candidates have u_hat < tau
                           -> a calibration problem: the extraction found the right shape, the scorer/threshold
                              discarded it. Fix: lower tau, or improve the scorer's features.
        shape_mismatch     some candidate overlaps (IoU > 0) but none exceeds 0.5
                           -> an extraction-geometry problem: fragmented, merged, wrong extent. Fix: closing radius,
                              hysteresis thresholds, or the network's boundary precision.
        no_candidate       no candidate overlaps this GT at all
                           -> the pixel-level foreground map never lit up here. Fix: recall of the network itself
                              (more training, more capacity, higher resolution, or a genuinely blind spot).

    This partitions FN by which pipeline stage is responsible, so effort goes to the stage that is actually broken."""
    rows = []
    for oc, u in zip(cands, u_by):
        for rid, iou in oc.mats.items():
            n_gt = oc.n_gt[rid]
            for j in range(n_gt):
                row = iou[j] if iou.shape[1] else np.zeros(0)
                max_iou = float(row.max()) if row.size else 0.0
                if max_iou > Mx.IOU_THR:
                    hit = row > Mx.IOU_THR
                    cat = "true_positive" if (u[hit] >= tau).any() else "rejected_by_tau"
                elif max_iou > 0:
                    cat = "shape_mismatch"
                else:
                    cat = "no_candidate"
                rows.append({"stem": oc.stem, "record": rid, "gt_index": j, "category": cat, "max_iou": max_iou,
                            "station": (station_of(oc.stem) if station_of else oc.stem[-2:])})
    return pd.DataFrame(rows)


def summarize_fn_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    overall = df["category"].value_counts().rename("count").to_frame()
    overall["fraction"] = overall["count"] / max(n, 1)
    by_station = df.groupby(["station", "category"]).size().unstack(fill_value=0)
    by_station = by_station.div(by_station.sum(1), axis=0).round(3)
    return overall, by_station
