"""Panoptic Quality exactly as implemented in the host's self-evaluation notebook, plus fast RLE equivalents.

Facts about the official scorer that shape the whole design (all verified in tests/test_metrics.py):
  * The prediction set of an image is scored *separately against every annotator record* of that image, and all
    records are pooled.  A prediction that matches annotator A but not B is a TP for A and an FP for B.
  * hit = IoU > 0.5 for *every* (gt, pred) cell -- there is no one-to-one assignment.  With disjoint predictions at most
    one prediction can hit a given GT (two disjoint masks cannot both exceed IoU 0.5 with the same GT), so the
    metric coincides with PQ of Kirillov et al.  Overlapping / duplicated predictions would inflate TP; we forbid them
    (see export.validate_submission).
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np
from pycocotools import mask as M

IOU_THR = 0.5


@dataclass
class PQ:
    pq: float
    dq: float
    sq: float
    tp: int
    fp: int
    fn: int
    sum_iou: float

    def __str__(self):
        return (f"PQ={self.pq:.4f} (DQ={self.dq:.4f} x SQ={self.sq:.4f})  TP={self.tp} FP={self.fp} FN={self.fn}")


# ------------------------------------------------------------------ RLE helpers
def encode(mask: np.ndarray) -> dict:
    return M.encode(np.asfortranarray(mask.astype(np.uint8)))


def to_counts_str(rle: dict) -> str:
    c = rle["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def from_counts_str(counts: str, h: int = 2048, w: int = 2048) -> dict:
    return {"size": [h, w], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def iou_matrix(gt_rles: list, pred_rles: list) -> np.ndarray:
    """(n_gt, n_pred) IoU matrix; identical to the official get_overlap_matrices (union==0 -> 0)."""
    if len(gt_rles) == 0 or len(pred_rles) == 0:
        return np.zeros((len(gt_rles), len(pred_rles)), np.float64)
    return np.asarray(M.iou(pred_rles, gt_rles, [0] * len(gt_rles))).T.astype(np.float64)


# ------------------------------------------------------------------ scoring one annotator record
def score_matrix(iou: np.ndarray, n_gt: int, n_pred: int) -> tuple[float, int, int, int]:
    """Mirror of the per-row logic in the official get_pq_score. Returns (sum_iou_tp, tp, fp, fn)."""
    if n_gt == 0:
        return 0.0, 0, n_pred, 0
    if n_pred == 0:
        return 0.0, 0, 0, n_gt
    hit = iou > IOU_THR
    return float(iou[hit].sum()), int(hit.sum()), int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum())


def pool(parts) -> PQ:
    s = sum(p[0] for p in parts)
    tp = sum(p[1] for p in parts); fp = sum(p[2] for p in parts); fn = sum(p[3] for p in parts)
    den = tp + 0.5 * fp + 0.5 * fn
    pq = s / den if den > 0 else 0.0
    dq = tp / den if den > 0 else 0.0
    sq = s / tp if tp > 0 else 0.0
    return PQ(pq, dq, sq, tp, fp, fn, s)


def evaluate(gt_records: dict, preds: dict) -> tuple[PQ, dict]:
    """gt_records : {record_id ("010101-2011...Lh") : [RLE, ...]}   preds : {image_stem : [RLE, ...]}.
    Returns pooled PQ and {record_id: iou_matrix} (for the one-to-many / many-to-one plots)."""
    parts, mats = [], {}
    for rid, gts in gt_records.items():
        stem = rid.split("-", 1)[1]
        pr = preds.get(stem, [])
        iou = iou_matrix(gts, pr)
        mats[rid] = iou
        parts.append(score_matrix(iou, len(gts), len(pr)))
    return pool(parts), mats


def relation_counts(mats: dict) -> tuple[Counter, Counter]:
    """Degree distributions used in the host's plot_m2n_counts: #predictions overlapping each GT (one-to-many) and
    #GTs overlapping each prediction (many-to-one), overlap meaning IoU > 0."""
    gt_deg, pred_deg = Counter(), Counter()
    for iou in mats.values():
        if iou.size == 0:
            n_gt, n_pr = iou.shape
            gt_deg.update([0] * n_gt); pred_deg.update([0] * n_pr)
            continue
        hit = iou > 0
        gt_deg.update(hit.sum(1).tolist()); pred_deg.update(hit.sum(0).tolist())
    return gt_deg, pred_deg


# ------------------------------------------------------------------ dense reference (verbatim maths of the host code)
def reference_pq_dense(gt_layers_by_record: dict, pred_layers_by_stem: dict) -> PQ:
    """Numpy transcription of the official torch code (matmul IoU on dense 0/1 layers). Used only in tests."""
    parts = []
    for rid, gt in gt_layers_by_record.items():
        stem = rid.split("-", 1)[1]
        pr = pred_layers_by_stem.get(stem, np.zeros((0,) + gt.shape[1:], np.float32))
        n_gt, n_pr = gt.shape[0], pr.shape[0]
        npx = gt.shape[1] * gt.shape[2]
        g = gt.reshape(n_gt, npx).astype(np.float32); p = pr.reshape(n_pr, npx).astype(np.float32)
        inter = g @ p.T
        union = g.sum(1, keepdims=True) + p.sum(1, keepdims=True).T - inter
        iou = np.where(union == 0, 0.0, inter / np.where(union == 0, 1, union))
        parts.append(score_matrix(iou, n_gt, n_pr))
    return pool(parts)


# ------------------------------------------------------------------ the algebra behind the decision rule
def include_threshold(current_pq: float) -> float:
    """Adding one prediction to an image with R annotator records changes the PQ denominator by exactly +0.5 per record
    (a hit turns an FN into a TP: +1-0.5; a miss adds an FP: +0.5) and the numerator by sum_r IoU_r*1[hit_r].
    Hence PQ increases iff  mean_r E[IoU_r * 1(hit_r)]  >  PQ / 2."""
    return 0.5 * current_pq
