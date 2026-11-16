"""Probability map -> disjoint filament instances.

Because instances are connected components of the annotators' foreground (notebook section 2), instance extraction is
deterministic and *disjoint by construction*: all output masks come from one label image.  No detector, no crops, no
affine warps, no overlap painting.

Steps:  hysteresis threshold -> morphological closing (bridges fragments) -> hole filling -> area filter -> components.
"""
from __future__ import annotations
import numpy as np
import cv2
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure
from .config import Config


def extract_labels(prob: np.ndarray, valid: np.ndarray, t_hi: float, t_lo: float,
                   close_radius: int, min_area: int) -> np.ndarray:
    """prob float32 in [0,1] (H,W); valid bool (H,W). Returns int32 label image (0 = background)."""
    extent = ((prob >= t_lo) & valid).astype(np.uint8)
    n, lab = cv2.connectedComponents(extent, connectivity=8)
    if n <= 1:
        return np.zeros(prob.shape, np.int32)
    seed_ids = np.unique(lab[(prob >= t_hi) & valid])
    seed_ids = seed_ids[seed_ids > 0]
    m = np.isin(lab, seed_ids)
    if close_radius > 0 and m.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool) & valid
    m = ndi.binary_fill_holes(m)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(prob.shape, np.int32)
    keep = np.where(st[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
    out = np.zeros(prob.shape, np.int32)
    for new_id, old in enumerate(keep, 1):
        out[lab == old] = new_id
    return out


FEATURE_COLUMNS = [
    "area", "bbox_h", "bbox_w", "extent", "solidity", "elongation", "perimeter_ratio", "max_dt", "mean_dt",
    "p_mean", "p_max", "p_sum", "p_p90", "p_min", "flat_mean", "flat_min", "contrast", "z_mean", "ridge_mean",
    "r_centroid", "r_min", "n_comp", "rank_psum", "nn_dist", "img_p_mean", "station",
]
STATIONS = {"Bh": 0, "Ch": 1, "Lh": 2, "Mh": 3, "Th": 4, "Uh": 5}


def component_table(labels: np.ndarray, prob: np.ndarray, chans: np.ndarray, flat: np.ndarray, rmap: np.ndarray,
                    valid: np.ndarray, station: str) -> pd.DataFrame:
    """One row per component with shape / probability / physical-contrast descriptors (26 features)."""
    n = int(labels.max())
    if n == 0:
        return pd.DataFrame(columns=["label"] + FEATURE_COLUMNS)
    idx = np.arange(1, n + 1)
    zc = chans[1].astype(np.float32); rc = chans[2].astype(np.float32)
    area = ndi.sum(np.ones_like(prob), labels, idx)
    p_mean = ndi.mean(prob, labels, idx); p_max = ndi.maximum(prob, labels, idx); p_min = ndi.minimum(prob, labels, idx)
    p_sum = ndi.sum(prob, labels, idx)
    flat_mean = ndi.mean(flat, labels, idx); flat_min = ndi.minimum(flat, labels, idx)
    z_mean = ndi.mean(zc, labels, idx); ridge_mean = ndi.mean(rc, labels, idx)
    r_min = ndi.minimum(rmap, labels, idx)
    # contrast to a surrounding ring (dilated 6px minus component) -- physical darkness relative to local background
    ring_mean = np.zeros(n, np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    props = measure.regionprops(labels)
    rows = []
    cents = np.array([p.centroid for p in props])
    for i, p in enumerate(props):
        y0, x0, y1, x1 = p.bbox
        pad = 8
        ya, xa, yb, xb = max(y0 - pad, 0), max(x0 - pad, 0), min(y1 + pad, labels.shape[0]), min(x1 + pad, labels.shape[1])
        sub = (labels[ya:yb, xa:xb] == p.label).astype(np.uint8)
        ring = cv2.dilate(sub, k).astype(bool) & ~sub.astype(bool) & (labels[ya:yb, xa:xb] == 0) & valid[ya:yb, xa:xb]
        ring_mean[i] = flat[ya:yb, xa:xb][ring].mean() if ring.any() else flat_mean[i]
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       # fast convex hull (skimage's was ~0.6 s)
        hull_area = cv2.contourArea(cv2.convexHull(np.concatenate(cnts))) if cnts else 0.0
        solidity = float(min(1.0, p.area / max(hull_area, p.area * 0.5, 1.0)))
        dt = cv2.distanceTransform(sub, cv2.DIST_L2, 3)
        maj, minr = max(p.major_axis_length, 1e-3), max(p.minor_axis_length, 1e-3)
        cy, cx = p.centroid
        rows.append(dict(label=p.label, area=p.area, bbox_h=y1 - y0, bbox_w=x1 - x0, extent=p.extent,
                         solidity=solidity, elongation=maj / minr,
                         perimeter_ratio=(p.perimeter ** 2) / (4 * np.pi * max(p.area, 1)),
                         max_dt=float(dt.max()), mean_dt=float(dt[sub > 0].mean()),
                         p_p90=float(np.percentile(prob[ya:yb, xa:xb][sub > 0], 90)),
                         r_centroid=float(rmap[int(round(cy)), int(round(cx))])))
    df = pd.DataFrame(rows)
    df["p_mean"], df["p_max"], df["p_sum"], df["p_min"] = p_mean, p_max, p_sum, p_min
    df["flat_mean"], df["flat_min"] = flat_mean, flat_min
    df["contrast"] = ring_mean - flat_mean            # positive => darker than surroundings
    df["z_mean"], df["ridge_mean"], df["r_min"] = z_mean, ridge_mean, r_min
    df["n_comp"] = n
    df["rank_psum"] = pd.Series(p_sum).rank(ascending=False).values / n
    if n > 1:
        dm = np.hypot(cents[:, None, 0] - cents[None, :, 0], cents[:, None, 1] - cents[None, :, 1])
        np.fill_diagonal(dm, np.inf)
        df["nn_dist"] = dm.min(1)
    else:
        df["nn_dist"] = 1e3
    df["img_p_mean"] = float(prob[valid].mean())
    df["station"] = STATIONS.get(station, -1)
    return df[["label"] + FEATURE_COLUMNS]


def labels_to_rles(labels: np.ndarray, keep_ids, out_size: int):
    """Nearest-neighbour upsample of the label image (guarantees disjointness) and COCO-RLE encode selected components."""
    from .metrics import encode
    if labels.shape[0] != out_size:
        labels = cv2.resize(labels.astype(np.int32), (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return [encode(labels == k) for k in keep_ids]
