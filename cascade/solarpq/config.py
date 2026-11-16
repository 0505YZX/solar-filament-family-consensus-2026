"""Single source of truth for every hyper-parameter (the host asks that no micro-level decision is undocumented)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json


@dataclass
class Config:
    # ---- data / geometry -------------------------------------------------
    full_res: int = 2048            # native image size (fixed by the competition)
    work_res: int = 1024            # resolution the network sees (2x INTER_AREA down-sampling)
    disk_frac: float = 0.985        # pixels with r/R > disk_frac are treated as outside the disk (limb ring)
    n_folds: int = 4
    seed: int = 42

    # ---- physics-inspired channels (sigmas are in work_res pixels) -------
    local_mu_sigma: float = 16.0    # background estimate for local contrast
    local_sd_sigma: float = 32.0    # local standard deviation window
    ridge_sigmas: tuple = (1.5, 3.0, 6.0)

    # ---- network ---------------------------------------------------------
    base_width: int = 32
    depth: int = 5                  # number of resolution levels
    in_channels: int = 3
    aux_spine: bool = True          # auxiliary head supervised by the provided `spine` metadata (host-permitted)
    spine_weight: float = 0.3
    dice_weight: float = 0.5

    # ---- optimisation ----------------------------------------------------
    crop: int = 512
    batch_size: int = 8
    iters: int = 40000               # CAP on optimiser steps per fold (a safety ceiling; the real driver is the time budget,
                                       # computed from the session clock at run time -- see notebook section 4)
    lr: float = 2e-3
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    fg_crop_prob: float = 0.7       # probability a training crop is centred on annotated filament material
    val_every: int = 1000
    amp: bool = True
    num_workers: int = 0            # unused: crops are produced by threads (no multiprocessing / shared-memory issues)
    prefetch: int = 6

    # ---- post-processing (defaults; tuned on out-of-fold predictions) ----
    t_hi: float = 0.50              # hysteresis seed threshold
    t_lo: float = 0.35              # hysteresis extent threshold
    close_radius: int = 3           # morphological closing radius (work_res px) -- bridges fragments
    min_area: int = 40              # work_res px^2
    tau: float = 0.20               # keep component iff predicted expected matched IoU >= tau  (theory: PQ/2)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "Config":
        d = json.loads(s)
        for k in ("ridge_sigmas",):
            if k in d:
                d[k] = tuple(d[k])
        return Config(**d)
