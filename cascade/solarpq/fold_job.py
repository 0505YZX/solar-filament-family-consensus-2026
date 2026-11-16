"""Child process: trains ONE fold on ONE GPU (selected by CUDA_VISIBLE_DEVICES) -- `python -m solarpq.fold_job ...`."""
from __future__ import annotations
import argparse, json, time, traceback
import numpy as np
import torch
from .config import Config
from . import data as D
from . import train as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--fold", type=int, required=True); ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--safe", type=int, default=0)
    a = ap.parse_args()
    cfg = Config.from_json(open(f"{a.work}/cfg.json").read())
    if a.safe:                                                        # fallback profile after a crash: fp32, small batch
        cfg.amp, cfg.batch_size = False, min(cfg.batch_size, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    cache = D.open_cache(a.cache)
    fold_of = np.array(json.load(open(f"{a.work}/folds.json"))["fold_of"])
    tr, va = np.flatnonzero(fold_of != a.fold), np.flatnonzero(fold_of == a.fold)
    print(f"fold {a.fold}: {len(tr)} train / {len(va)} val observations | device {device} "
          f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} | safe={a.safe}", flush=True)
    t0 = time.time()
    bs, sec = T.autotune(cache, tr, cfg, device, log=lambda s: print(s, flush=True))
    iters = int(max(min(cfg.iters, 0.88 * a.budget / sec), 60))
    print(f"plan: batch {bs}, {sec:.3f} s/step -> {iters} iterations (cap {cfg.iters}) within {a.budget/60:.0f} min", flush=True)
    hist = T.train_fold(cache, tr, va, cfg, f"{a.work}/ckpt/fold{a.fold}.pt", device, fold=a.fold, bs=bs, iters=iters,
                        max_seconds=a.budget, log=lambda s: print(s, flush=True))
    best = min((h["val_bce"] for h in hist if "val_bce" in h), default=None)
    json.dump({"fold": a.fold, "batch": bs, "sec_per_step": sec, "iters_planned": iters, "iters_done": hist[-1]["iter"],
               "best_val_bce": best, "last": hist[-1], "seconds": time.time() - t0, "safe": a.safe},
              open(f"{a.work}/ckpt/result_fold{a.fold}.json", "w"))
    print(f"fold {a.fold} finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
