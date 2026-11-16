"""Fold training.

Crops are sampled on the CPU by a background *thread* (no DataLoader worker processes => no shared-memory failures on Kaggle)
and augmented on the GPU (D4 symmetry, photometric jitter, disk-validity mask).  AdamW + warm-up/cosine, AMP, EMA weights,
and an OOM-safe autotuner that measures the real step time to choose batch size and number of iterations for a time budget."""
from __future__ import annotations
import time, json, copy, threading, queue, gc
import numpy as np
import torch
import torch.nn as nn
from .config import Config
from . import model as Mdl
from . import geometry as G


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def is_oom(e: BaseException) -> bool:
    oom = getattr(torch.cuda, "OutOfMemoryError", ())
    return (oom and isinstance(e, oom)) or "out of memory" in str(e).lower()


# ------------------------------------------------------------------ sampling (CPU) and augmentation (GPU)
class CropSampler:
    """Draws raw crops (no augmentation) from the memory-mapped cache; single-threaded use."""

    def __init__(self, cache: dict, idx, cfg: Config, seed: int):
        self.c, self.idx, self.cfg = cache, np.asarray(idx), cfg
        self.rng = np.random.default_rng(seed)
        self.fg = {}
        for i in self.idx:                                        # foreground pixel coordinates (sub-sampled)
            ys, xs = np.nonzero(np.asarray(cache["soft"][i]) >= 128)
            if len(ys) > 3000:
                sel = np.random.default_rng(int(i)).choice(len(ys), 3000, replace=False); ys, xs = ys[sel], xs[sel]
            self.fg[int(i)] = np.stack([ys, xs], 1)

    def sample(self, bs: int):
        C, W, rng = self.cfg.crop, self.cfg.work_res, self.rng
        chans = np.empty((bs, 3, C, C), np.float16); soft = np.empty((bs, C, C), np.uint8); spn = np.empty((bs, C, C), np.uint8)
        disk = np.empty((bs, 3), np.float32); org = np.empty((bs, 2), np.int64)
        for b in range(bs):
            i = int(rng.choice(self.idx)); fg = self.fg[i]
            if len(fg) and rng.random() < self.cfg.fg_crop_prob:
                cy, cx = fg[rng.integers(len(fg))]
                y0 = int(np.clip(cy - C // 2 + rng.integers(-C // 3, C // 3 + 1), 0, W - C))
                x0 = int(np.clip(cx - C // 2 + rng.integers(-C // 3, C // 3 + 1), 0, W - C))
            else:
                y0, x0 = int(rng.integers(0, W - C + 1)), int(rng.integers(0, W - C + 1))
            chans[b] = self.c["chans"][i, :, y0:y0 + C, x0:x0 + C]
            soft[b] = self.c["soft"][i, y0:y0 + C, x0:x0 + C]
            spn[b] = self.c["spine"][i, y0:y0 + C, x0:x0 + C]
            disk[b] = self.c["disk"][i]; org[b] = (y0, x0)
        return chans, soft, spn, disk, org


class Prefetcher(threading.Thread):
    def __init__(self, sampler: CropSampler, bs: int, depth: int = 6):
        super().__init__(daemon=True)
        self.q, self.sampler, self.bs, self.stop_evt, self.err = queue.Queue(maxsize=depth), sampler, bs, threading.Event(), None
        self.start()

    def run(self):
        try:
            while not self.stop_evt.is_set():
                b = self.sampler.sample(self.bs)
                while not self.stop_evt.is_set():
                    try:
                        self.q.put(b, timeout=0.5); break
                    except queue.Full:
                        pass
        except BaseException as e:      # surfaced in get()
            self.err = e

    def get(self):
        while True:
            try:
                return self.q.get(timeout=1.0)
            except queue.Empty:
                if self.err is not None:
                    raise self.err

    def close(self):
        self.stop_evt.set()


def augment_gpu(chans, soft, spn, disk, org, cfg: Config, device, rng: np.random.Generator):
    """numpy crops -> (x, y=[soft, spine], valid) tensors. Image, targets and validity mask are stacked in ONE tensor so the
    D4 transform is applied to all of them identically (alignment is enforced by construction and unit-tested)."""
    C = chans.shape[-1]; B = chans.shape[0]
    x = torch.from_numpy(chans).to(device).float()
    soft_t = torch.from_numpy(soft).to(device).float().div_(255).unsqueeze(1)
    spn_t = torch.from_numpy(spn).to(device).float().div_(255).unsqueeze(1)
    dk = torch.from_numpy(disk).to(device); og = torch.from_numpy(org).to(device).float()
    ar = torch.arange(C, device=device, dtype=torch.float32)
    ys = og[:, 0:1] + ar[None, :]; xs = og[:, 1:2] + ar[None, :]
    r = torch.hypot(xs[:, None, :] - dk[:, 0, None, None], ys[:, :, None] - dk[:, 1, None, None]) / dk[:, 2, None, None]
    valid = (r < cfg.disk_frac).float().unsqueeze(1)
    stack = torch.cat([x, soft_t, spn_t, valid], 1)                       # (B, 6, C, C)
    out = []
    for b in range(B):
        k = int(rng.integers(4)); f = bool(rng.integers(2))
        t = torch.rot90(stack[b], k, (-2, -1))
        out.append(torch.flip(t, (-1,)) if f else t)
    stack = torch.stack(out)
    x, soft_t, spn_t, valid = stack[:, :3], stack[:, 3:4], stack[:, 4:5], stack[:, 5:6]
    gain = torch.empty(B, 3, 1, 1, device=device).uniform_(0.9, 1.1)
    noise = torch.randn(B, 3, 1, 1, device=device) * 0.03
    x = (x * gain + noise) * valid
    return x, torch.cat([soft_t, spn_t], 1), valid


# ------------------------------------------------------------------ one optimisation step
def train_step(model, opt, scaler, x, y, valid, cfg: Config, use_amp: bool, device_type: str):
    with torch.autocast(device_type=device_type, enabled=use_amp):
        out = model(x)
    loss, _, _ = Mdl.total_loss(out.float(), y[:, 0:1], y[:, 1:2], valid, cfg)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    return loss.detach()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def autotune(cache: dict, idx, cfg: Config, device, log=print, warm: int = 3, timed: int = 10):
    """Largest batch size (<= cfg.batch_size) that fits in memory, and the measured seconds per optimisation step."""
    use_amp = cfg.amp and device.type == "cuda"
    cands, b = [], cfg.batch_size                       # fine-grained halving ladder: bs, bs/2, bs/4, ... 1
    while b >= 1:
        cands.append(b); b //= 2
    if cands[-1] != 1:
        cands.append(1)
    rng = np.random.default_rng(0)
    for bs in cands:
        model = opt = sampler = None
        try:
            model = Mdl.build_model(cfg).to(device).train()
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            scaler = make_scaler(use_amp)
            sampler = CropSampler(cache, idx, cfg, seed=1)
            for step in range(warm + timed):
                if step == warm:
                    _sync(device); t0 = time.time()
                c, s, sp, dk, og = sampler.sample(bs)
                x, y, v = augment_gpu(c, s, sp, dk, og, cfg, device, rng)
                train_step(model, opt, scaler, x, y, v, cfg, use_amp, device.type)
            _sync(device)
            sec = (time.time() - t0) / timed
            log(f"autotune: batch {bs} fits, {sec:.3f} s/step")
            return bs, sec
        except Exception as e:
            if not is_oom(e):
                raise
            log(f"autotune: batch {bs} out of memory -> trying smaller")
        finally:
            del model, opt, sampler
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("even batch size 1 does not fit in memory")


# ------------------------------------------------------------------ validation
@torch.no_grad()
def validate(model, cache, idx, cfg: Config, device) -> dict:
    model.eval()
    W = cfg.work_res
    tot_bce = num = den = 0.0
    for i in idx:
        x = torch.from_numpy(np.asarray(cache["chans"][i], np.float32))[None].to(device)
        soft = torch.from_numpy(np.asarray(cache["soft"][i], np.float32) / 255.0)[None, None].to(device)
        cx, cy, R = cache["disk"][i]
        valid = torch.from_numpy((G.radius_map((W, W), cx, cy, R) < cfg.disk_frac).astype(np.float32))[None, None].to(device)
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            out = model(x * valid)
        lg = out[:, 0:1].float()
        tot_bce += float(Mdl.masked_bce(lg, soft, valid))
        p = torch.sigmoid(lg) * valid
        num += float(2 * (p * soft).sum()); den += float(p.sum() + (soft * valid).sum())
    return {"val_bce": tot_bce / max(len(idx), 1), "val_soft_dice": num / max(den, 1e-6)}


# ------------------------------------------------------------------ fold training
def train_fold(cache: dict, train_idx, val_idx, cfg: Config, ckpt_path: str, device, fold: int = 0,
               bs: int | None = None, iters: int | None = None, max_seconds: float | None = None, log=print) -> list:
    torch.manual_seed(cfg.seed + fold); np.random.seed(cfg.seed + fold)
    bs, iters = bs or cfg.batch_size, iters or cfg.iters
    val_every = max(min(cfg.val_every, iters // 8), 50)
    model = Mdl.build_model(cfg).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(int(0.03 * iters), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, 0.5 * (1 + np.cos(np.pi * min(s / max(iters, 1), 1.0))) * 0.98 + 0.02))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = make_scaler(use_amp)
    pf = Prefetcher(CropSampler(cache, train_idx, cfg, seed=cfg.seed * 100 + fold), bs, cfg.prefetch)
    rng_aug = np.random.default_rng(cfg.seed * 100 + fold + 1)
    hist, best, saved = [], 1e9, False
    run_loss, run_n, t0 = torch.zeros((), device=device), 0, time.time()
    try:
        for it in range(1, iters + 1):
            model.train()
            c, s, sp, dk, og = pf.get()
            x, y, v = augment_gpu(c, s, sp, dk, og, cfg, device, rng_aug)
            loss = train_step(model, opt, scaler, x, y, v, cfg, use_amp, device.type)
            sched.step()
            with torch.no_grad():                                   # EMA of weights
                d = min(cfg.ema_decay, (1 + it) / (10 + it))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            run_loss += loss; run_n += 1
            timeout = max_seconds is not None and (time.time() - t0) > max_seconds
            if it % val_every == 0 or it == iters or timeout:
                rec = {"fold": fold, "iter": it, "train_loss": float(run_loss) / max(run_n, 1), "lr": sched.get_last_lr()[0],
                       "sec": time.time() - t0}
                if len(val_idx):
                    rec.update(validate(ema, cache, val_idx, cfg, device))
                hist.append(rec); run_loss, run_n = torch.zeros((), device=device), 0
                log(json.dumps({k: (round(val, 4) if isinstance(val, float) else val) for k, val in rec.items()}))
                crit = rec.get("val_bce", rec["train_loss"])
                if crit <= best or it == iters or timeout or not saved:
                    best = min(best, crit); saved = True
                    torch.save({"state": ema.state_dict(), "cfg": cfg.to_json(), "fold": fold, "iter": it, "val": rec}, ckpt_path)
            if timeout:
                log(f"time budget reached at iteration {it}/{iters}: stopping (EMA checkpoint saved)")
                break
    finally:
        pf.close()
    return hist
