"""Submission writing and strict validation (format, uniqueness, RLE round-trip, DISJOINTNESS)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pycocotools import mask as M
from .metrics import to_counts_str


def write_submission(preds: dict, path: str) -> pd.DataFrame:
    """preds: {image_stem: [RLE dict, ...]} (2048x2048). Writes filament_id, segmentation_rle (counts only, no quotes)."""
    rows = []
    for stem in sorted(preds):
        for k, rle in enumerate(preds[stem], 1):
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": to_counts_str(rle)})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(path, index=False)
    return df


def validate_submission(df: pd.DataFrame, test_stems: list, size: int = 2048, require_all_images: bool = True) -> dict:
    """Raises AssertionError on any violation; returns a small report dict."""
    assert list(df.columns) == ["filament_id", "segmentation_rle"], "columns must be filament_id, segmentation_rle"
    assert df["filament_id"].is_unique, "filament_id must be unique"
    assert not df["segmentation_rle"].isna().any(), "empty RLE string"
    assert not df["segmentation_rle"].str.contains("['\"]").any(), "RLE counts must not contain quote characters"
    stems = df["filament_id"].str.rsplit("_", n=1).str[0]
    unknown = set(stems) - set(test_stems)
    assert not unknown, f"ids for unknown images: {sorted(unknown)[:5]}"
    if require_all_images:
        missing = set(test_stems) - set(stems)
        assert not missing, f"{len(missing)} test images have no prediction, e.g. {sorted(missing)[:3]}"
    n_overlap = 0
    for stem, g in df.groupby(stems):
        rles = [{"size": [size, size], "counts": c.encode("ascii")} for c in g["segmentation_rle"]]
        layers = M.decode(rles)                                   # (H, W, n) -- also proves the counts decode
        assert layers.shape[:2] == (size, size)
        areas = layers.reshape(-1, layers.shape[2]).sum(0)
        assert (areas > 0).all(), f"{stem}: empty mask"
        cover = layers.sum(2)
        if cover.max() > 1:                                       # any pixel claimed twice -> the scorer would double-count
            n_overlap += 1
            raise AssertionError(f"{stem}: overlapping masks (max multiplicity {cover.max()})")
    return {"rows": len(df), "images": int(stems.nunique()), "mean_per_image": float(len(df) / max(stems.nunique(), 1)),
            "overlapping_images": n_overlap}
