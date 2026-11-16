"""MAGFiLO / COCO parsing, multi-annotator soft targets, fold assignment and on-disk caching.

Design decisions (all derived from measurements on the training JSON, see notebook section 2):
  * The unit of splitting is the *observation* (image file), never the annotator record: 41% of observations are
    annotated 2-3 times and their records must never straddle train / validation.
  * The training target of an observation is the pixel-wise MEAN of its annotators' union masks (a calibrated
    "fraction of annotators that marked this pixel"), not a single annotator's mask: minimising BCE against a soft
    target estimates exactly the annotator-marginal probability that governs the expected IoU of any prediction.
  * Instances are NOT learned: inside one annotator record only 0.7% of filaments touch another one, so the connected
    components of a foreground mask *are* the instances (oracle PQ 0.997, notebook section 2).
"""
from __future__ import annotations
import json, os, glob
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import cv2
from pycocotools import mask as M
from .config import Config


@dataclass
class Observation:
    stem: str
    instrument: str
    dt: str
    records: list = field(default_factory=list)


@dataclass
class Registry:
    obs: dict                 # stem -> Observation
    anns: dict                # record_id -> [annotation dict]

    def stems(self):
        return sorted(self.obs)


def load_registry(json_path: str) -> Registry:
    d = json.load(open(json_path))
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)
    obs = {}
    for im in d["images"]:
        stem = im["file_name"].rsplit(".", 1)[0]
        o = obs.setdefault(stem, Observation(stem=stem, instrument=stem[-2:], dt=stem[:-2]))
        o.records.append(im["id"])
    for o in obs.values():
        o.records.sort()
    return Registry(obs=obs, anns=dict(anns))


# ------------------------------------------------------------------ rasterisation (pycocotools => same convention as the scorer)
def _scaled(poly, s):
    return (np.asarray(poly, np.float64) * s).tolist()


def instance_rles(anns: list, size: int, full: int = 2048) -> list:
    s = size / full
    return [M.merge(M.frPyObjects([_scaled(p, s) for p in a["segmentation"]], size, size)) for a in anns]


def union_mask(anns: list, size: int, full: int = 2048) -> np.ndarray:
    if not anns:
        return np.zeros((size, size), np.uint8)
    return M.decode(M.merge(instance_rles(anns, size, full))).astype(np.uint8)


def spine_mask(anns: list, size: int, thickness: int = 3, full: int = 2048) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    s = size / full
    for a in anns:
        sp = a.get("spine")
        if not sp or len(sp) < 4:
            continue
        pts = np.round(np.asarray(sp, np.float64).reshape(-1, 2) * s).astype(np.int32)
        cv2.polylines(m, [pts], False, 1, thickness)
    return m


def soft_target(reg: Registry, stem: str, size: int) -> np.ndarray:
    """Mean of the annotators' union masks (float32 in [0,1])."""
    recs = reg.obs[stem].records
    acc = np.zeros((size, size), np.float32)
    for r in recs:
        acc += union_mask(reg.anns[r], size)
    return acc / max(len(recs), 1)


def spine_target(reg: Registry, stem: str, size: int) -> np.ndarray:
    acc = np.zeros((size, size), np.float32)
    recs = reg.obs[stem].records
    for r in recs:
        acc += spine_mask(reg.anns[r], size)
    return acc / max(len(recs), 1)


def gt_records_rles(reg: Registry, stems, size: int) -> dict:
    """{record_id: [RLE]} at resolution `size` (2048 for the official-equivalent evaluation)."""
    return {r: instance_rles(reg.anns[r], size) for s in stems for r in reg.obs[s].records}


# ------------------------------------------------------------------ folds
def make_folds(reg: Registry, k: int, seed: int = 0, stems: list | None = None) -> dict:
    """Deterministic, station- and time-balanced assignment of *observations* to folds.

    Observations are sorted by (instrument, timestamp) and dealt round-robin => every fold sees every station and the
    whole 2011-2022 range, and temporally adjacent frames of the same filament land in different folds, mirroring the
    competition's random observation-level train/test split."""
    order = sorted(reg.obs if stems is None else stems, key=lambda s: (reg.obs[s].instrument, reg.obs[s].dt))
    off = seed % k
    return {s: (i + off) % k for i, s in enumerate(order)}


# ------------------------------------------------------------------ caching (memory-mapped, resumable)
def find_image(stem: str, dirs: list[str]) -> str | None:
    for d in dirs:
        for ext in (".jpeg", ".jpg", ".png"):
            p = os.path.join(d, stem + ext)
            if os.path.exists(p):
                return p
    return None


def build_cache(reg: Registry, stems: list, img_dirs: list[str], cfg: Config, out_dir: str,
                with_targets: bool = True, verbose: bool = True) -> dict:
    """Compute channels (+targets) for `stems` and store them as .npy memmaps in out_dir. Returns a manifest dict."""
    from . import features as F
    os.makedirs(out_dir, exist_ok=True)
    N, W = len(stems), cfg.work_res
    ch = np.lib.format.open_memmap(os.path.join(out_dir, "chans.npy"), "w+", np.float16, (N, 3, W, W))
    disk = np.zeros((N, 3), np.float32)
    fallbacks = []
    if with_targets:
        soft = np.lib.format.open_memmap(os.path.join(out_dir, "soft.npy"), "w+", np.uint8, (N, W, W))
        spn = np.lib.format.open_memmap(os.path.join(out_dir, "spine.npy"), "w+", np.uint8, (N, W, W))
    for i, s in enumerate(stems):
        p = find_image(s, img_dirs)
        if p is None:
            raise FileNotFoundError(f"image for {s} not found in {img_dirs}")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (cfg.full_res, cfg.full_res):
            raise ValueError(f"unexpected image {p}: {None if img is None else img.shape}")
        r = F.compute_channels(img, cfg)
        ch[i] = r.chans
        disk[i] = r.disk
        if not r.disk_ok:
            fallbacks.append(s)
        if with_targets:
            soft[i] = np.round(soft_target(reg, s, W) * 255).astype(np.uint8)
            spn[i] = np.round(spine_target(reg, s, W) * 255).astype(np.uint8)
        if verbose and (i % 50 == 0 or i == N - 1):
            print(f"  cached {i + 1}/{N}", flush=True)
    ch.flush()
    if with_targets:
        soft.flush(); spn.flush()
    np.save(os.path.join(out_dir, "disk.npy"), disk)
    if fallbacks and verbose:
        print(f"  WARNING: default disk circle used for {len(fallbacks)} images: {fallbacks[:5]}")
    man = {"stems": list(stems), "work_res": W, "with_targets": with_targets, "disk_fallbacks": fallbacks}
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"))
    return man


def open_cache(out_dir: str) -> dict:
    man = json.load(open(os.path.join(out_dir, "manifest.json")))
    c = {"manifest": man, "stems": man["stems"],
         "chans": np.load(os.path.join(out_dir, "chans.npy"), mmap_mode="r"),
         "disk": np.load(os.path.join(out_dir, "disk.npy"))}
    if man["with_targets"]:
        c["soft"] = np.load(os.path.join(out_dir, "soft.npy"), mmap_mode="r")
        c["spine"] = np.load(os.path.join(out_dir, "spine.npy"), mmap_mode="r")
    return c
