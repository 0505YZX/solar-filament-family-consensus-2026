"""Physics-inspired input channels, all computed at `work_res` on the disk-normalised image.

  c0  flat-field intensity      (I / radial-median-profile - 1), i.e. centre-to-limb variation removed
  c1  local contrast z-score    (flat - local mean) / local std   (normalised convolution, disk-restricted)
  c2  dark-ridge scale space    max_sigma  sigma^2 * max(lambda_1(Hessian), 0)  -- Lindeberg scale-normalised
                                second-derivative response to *dark* structures of width ~ sigma

These replace the "replicate grayscale to RGB" convention with three *different, physically meaningful* views.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
from .config import Config
from . import geometry as G


def _gauss_kernels(sigma: float):
    r = int(np.ceil(4 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    g1 = -x / sigma ** 2 * g
    g2 = (x ** 2 / sigma ** 4 - 1.0 / sigma ** 2) * g
    return g.astype(np.float32), g1.astype(np.float32), g2.astype(np.float32)


def hessian_eigs(img: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale-normalised (sigma^2) eigenvalues (lam1 >= lam2) of the Gaussian-derivative Hessian."""
    g, g1, g2 = _gauss_kernels(sigma)
    ixx = cv2.sepFilter2D(img, cv2.CV_32F, g2, g, borderType=cv2.BORDER_REFLECT)
    iyy = cv2.sepFilter2D(img, cv2.CV_32F, g, g2, borderType=cv2.BORDER_REFLECT)
    ixy = cv2.sepFilter2D(img, cv2.CV_32F, g1, g1, borderType=cv2.BORDER_REFLECT)
    tr = 0.5 * (ixx + iyy)
    disc = np.sqrt(np.maximum(0.25 * (ixx - iyy) ** 2 + ixy ** 2, 0.0))
    s2 = sigma ** 2
    return s2 * (tr + disc), s2 * (tr - disc)


@dataclass
class Channels:
    chans: np.ndarray      # (3, H, W) float16
    mask: np.ndarray       # (H, W) bool   -- valid disk
    disk: tuple            # (cx, cy, R) in work_res pixels
    flat: np.ndarray       # (H, W) float32 flat-field image (kept for component features)
    rmap: np.ndarray       # (H, W) float32 r/R
    disk_ok: bool = True   # False => the fitted circle was implausible and a default circle was used


def compute_channels(img_full: np.ndarray, cfg: Config) -> Channels:
    """img_full: (2048, 2048) uint8 -> Channels at cfg.work_res."""
    W = cfg.work_res
    img = cv2.resize(img_full, (W, W), interpolation=cv2.INTER_AREA) if img_full.shape[0] != W else img_full
    cx, cy, R = G.fit_disk(img)
    disk_ok = bool(0.25 * W < R < 0.62 * W and 0 < cx < W and 0 < cy < W)
    if not disk_ok:                                     # never let one odd frame crash a multi-hour run
        cx, cy, R = W / 2.0, W / 2.0, 0.46 * W
    rmap = G.radius_map(img.shape, cx, cy, R)
    mask = rmap < cfg.disk_frac
    prof = G.radial_median_profile(img.astype(np.float32), rmap, mask)
    flat = np.where(mask, img.astype(np.float32) / np.maximum(prof, 1.0), 1.0).astype(np.float32)
    w = mask.astype(np.float32)

    mu = G.normalized_blur(flat, w, cfg.local_mu_sigma)
    d = (flat - mu) * w
    sd = np.sqrt(np.maximum(G.normalized_blur(d ** 2, w, cfg.local_sd_sigma), 1e-8))
    z = d / np.maximum(sd, 0.02)

    ridge = np.zeros_like(flat)
    for s in cfg.ridge_sigmas:
        l1, _ = hessian_eigs(flat, s)
        ridge = np.maximum(ridge, np.maximum(l1, 0.0))
    ridge = ridge * w
    # scale by robust dispersion of the response on the disk so units are comparable across stations
    rs = np.median(ridge[mask]) + 1e-6
    ridge = np.log1p(ridge / (4.0 * rs))

    c0 = np.clip((flat - 1.0) / 0.25, -3, 3) * w
    c1 = np.clip(z, -4, 4) * w
    c2 = np.clip(ridge, 0, 4) * w
    chans = np.stack([c0, c1, c2]).astype(np.float16)
    return Channels(chans=chans, mask=mask, disk=(cx, cy, R), flat=flat, rmap=rmap, disk_ok=disk_ok)
