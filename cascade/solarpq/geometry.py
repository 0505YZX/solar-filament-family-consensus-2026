"""Solar-disk geometry and radial flat-fielding.

A full-disk H-alpha frame is a bright circle on a black background. Two first-principles facts are exploited:
  * filaments only exist on the disk, so everything outside is masked;
  * the mean chromospheric intensity depends (to first order) only on r/R (centre-to-limb variation), so
    dividing by the radial *median* profile removes it without touching localised dark structures.
"""
from __future__ import annotations
import numpy as np
import cv2


def fit_disk(img: np.ndarray, iters: int = 3) -> tuple[float, float, float]:
    """Fit the solar disk circle (cx, cy, R) in pixel units of `img` (uint8, 2-D).

    Otsu threshold on a blurred copy -> largest component -> hole fill -> algebraic (Kasa) least-squares circle fit
    on the boundary, iteratively trimming boundary points that deviate (prominences, off-limb artefacts).
    """
    g = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    b = cv2.GaussianBlur(g, (0, 0), 3)
    _, m = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if n < 2:
        h, w = g.shape
        return w / 2.0, h / 2.0, min(h, w) * 0.45
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    # fill holes (filaments / sunspots darker than Otsu threshold must not perturb the boundary)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    keep = np.ones(len(c), bool)
    cx = cy = R = 0.0
    for _ in range(iters):
        p = c[keep]
        A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
        y = (p ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        cx, cy = sol[0], sol[1]
        R = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1.0)))
        resid = np.abs(np.hypot(c[:, 0] - cx, c[:, 1] - cy) - R)
        keep = resid < max(2.5 * np.median(resid[keep]) + 1.0, 3.0)
        if keep.sum() < 50:
            break
    return float(cx), float(cy), float(R)


def radius_map(shape: tuple[int, int], cx: float, cy: float, R: float) -> np.ndarray:
    """Normalised radius r/R for every pixel (float32)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return np.hypot(xx - cx, yy - cy).astype(np.float32) / np.float32(R)


def disk_mask(shape, cx, cy, R, frac: float = 0.985) -> np.ndarray:
    return radius_map(shape, cx, cy, R) < frac


def radial_median_profile(img: np.ndarray, rmap: np.ndarray, mask: np.ndarray,
                          nbins: int = 100, smooth: float = 1.5) -> np.ndarray:
    """Per-pixel radial median intensity (centre-to-limb profile), interpolated and lightly smoothed."""
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(rmap[mask], edges) - 1, 0, nbins - 1)
    vals = img[mask].astype(np.float32)
    order = np.argsort(idx, kind="stable")
    idx_s, val_s = idx[order], vals[order]
    bounds = np.searchsorted(idx_s, np.arange(nbins + 1))
    prof = np.full(nbins, np.nan, np.float32)
    for i in range(nbins):
        seg = val_s[bounds[i]:bounds[i + 1]]
        if seg.size >= 20:
            prof[i] = np.median(seg)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ok = ~np.isnan(prof)
    prof = np.interp(centres, centres[ok], prof[ok]).astype(np.float32)
    if smooth > 0:
        k = int(4 * smooth) * 2 + 1
        prof = cv2.GaussianBlur(prof.reshape(1, -1), (k, 1), smooth, borderType=cv2.BORDER_REPLICATE).ravel()
    return np.interp(rmap, centres, prof).astype(np.float32)


def normalized_blur(x: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: Gaussian blur of x restricted to the support w (no bleeding from outside the disk)."""
    num = cv2.GaussianBlur(x * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)
