"""Glue: probability map -> selected disjoint instances at 2048 (used identically for OOF tuning and for the test set)."""
from __future__ import annotations
import numpy as np
from .config import Config
from . import calibrate as C, postprocess as P, geometry as G


def select_ids(table, u: np.ndarray, tau: float, min_per_image: int = 1) -> list:
    """Component labels to submit: u_hat >= tau, but never fewer than `min_per_image` (best-scoring ones)."""
    keep = np.flatnonzero(u >= tau)
    if len(keep) < min_per_image:
        keep = np.argsort(-u)[:min_per_image]
    return table["label"].to_numpy()[keep].tolist()


def predict_instances(stem: str, prob: np.ndarray, chans: np.ndarray, disk, cfg: Config, scorer, tau: float,
                      min_per_image: int = 1, out_size: int = 2048):
    """Returns (list of RLE at out_size, info dict).

    min_per_image: every one of the 1,154 training records contains >= 1 filament (measured), so an image with no selected
    component is implausible; the best-scoring candidate is kept (a real prediction, never a dummy mask)."""
    oc = C.build_candidates(stem, prob, chans, np.asarray(disk), cfg, None, keep_labels=True)
    info = {"n_candidates": oc.n_cand, "relaxed": False}
    if oc.n_cand == 0 and min_per_image > 0 and prob.max() > 0:              # nothing above t_lo: relax to the map's own scale
        pm = float(prob.max())
        oc = C.build_candidates(stem, prob, chans, np.asarray(disk), cfg, None, keep_labels=True,
                                t_hi=0.8 * pm, t_lo=0.5 * pm)
        info["relaxed"] = True
    if oc.n_cand == 0:
        return [], info
    u = np.clip(scorer.predict(oc.table[C.FEATURES].to_numpy(np.float64)), 0, 1) if scorer is not None else oc.table["p_mean"].to_numpy()
    ids = select_ids(oc.table, u, tau, min_per_image)
    info["n_selected"] = len(ids)
    return P.labels_to_rles(oc.labels, ids, out_size), info


def fallback_instances(chans: np.ndarray, disk, cfg: Config, out_size: int = 2048):
    """Last-resort prediction when a probability map is unavailable: the largest connected blob among the strongest dark-ridge
    responses (physical cue, no learned model). Used only to guarantee that every test image has a row."""
    import cv2
    H = chans.shape[-1]
    valid = G.radius_map((H, H), *disk) < cfg.disk_frac
    s = cv2.GaussianBlur(np.asarray(chans[2], np.float32), (0, 0), 2.0)
    thr = np.percentile(s[valid], 99.7)
    m = ((s >= thr) & valid).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return []
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return P.labels_to_rles((lab == k).astype(np.int32), [1], out_size)
