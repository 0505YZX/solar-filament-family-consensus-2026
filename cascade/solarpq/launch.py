"""Parent-side orchestration: folds run as independent subprocesses, one per GPU (two T4s => two folds at a time).
A crashed fold is retried once with a conservative profile (fp32, batch<=4)."""
from __future__ import annotations
import os, sys, json, math, subprocess, threading
from pathlib import Path


_PREFLIGHT = r"""
import time, torch
from solarpq.config import Config
from solarpq import model as Mdl, train as T
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda"
cfg = Config(); m = Mdl.build_model(cfg).to(dev).train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3); sc = T.make_scaler(amp)
S = 256 if amp else 64
x = torch.randn(2, 3, S, S, device=dev); y = (torch.rand(2, 2, S, S, device=dev) > 0.9).float(); v = torch.ones(2, 1, S, S, device=dev)
for i in range(4):                                   # full-size model, AMP, GradScaler, EMA-free training step
    if i == 1:
        if amp: torch.cuda.synchronize()
        t = time.time()
    T.train_step(m, opt, sc, x, y, v, cfg, amp, dev.type)
if amp: torch.cuda.synchronize()
sec = (time.time() - t) / 3
m.eval()
with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=amp):        # full-frame inference pass
    out = m(torch.randn(1, 3, 1024 if amp else 128, 1024 if amp else 128, device=dev))
assert torch.isfinite(out.float()).all()
mem = torch.cuda.max_memory_allocated() / 1e9 if amp else 0.0
print("PREFLIGHT_OK", torch.cuda.get_device_name(0) if amp else "cpu", f"{sec:.3f}s/step(b2@{S}px)", f"{mem:.2f}GB peak")
"""


def preflight(work, n_gpu: int, log=print) -> list:
    """Runs a real forward/backward (+ full-frame inference) on every visible GPU in a fresh process; returns usable GPU ids.
    Fails fast on environment problems (AMP, memory, driver) instead of after hours of work."""
    work = Path(work)
    ids = list(range(n_gpu)) if n_gpu > 0 else [None]
    ok = []
    for g in ids:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(work) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        if g is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(g)
        try:
            r = subprocess.run([sys.executable, "-c", _PREFLIGHT], cwd=str(work), env=env, capture_output=True, text=True, timeout=600)
            line = next((l for l in r.stdout.splitlines() if l.startswith("PREFLIGHT_OK")), None)
            if line:
                log(f"GPU {g}: {line}"); ok.append(g)
            else:
                log(f"!!! GPU {g} failed the preflight check:\n{(r.stdout + r.stderr)[-1500:]}")
        except Exception as e:
            log(f"!!! GPU {g} preflight error: {e!r}")
    return [g for g in ok if g is not None] if n_gpu > 0 else []


def _spawn(work, cache_dir, fold, gpu, budget, safe, use_gpu):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(work) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, "-m", "solarpq.fold_job", "--work", str(work), "--cache", str(cache_dir), "--fold", str(fold),
           "--budget", str(budget), "--safe", str(int(safe))]
    return subprocess.Popen(cmd, cwd=str(work), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)


def _pump(proc, tag, log):
    for line in proc.stdout:
        log(f"[{tag}] {line.rstrip()}")


def run_folds(work, cache_dir, folds: list, gpus: list, budget_sec_total: float, log=print) -> dict:
    """Runs every fold in `folds` on the given GPU ids (empty list = CPU, dry runs only); returns {fold: result_dict or None}."""
    work = Path(work)
    slots = max(len(gpus), 1)
    waves = math.ceil(len(folds) / slots)
    budget = budget_sec_total / waves
    log(f"launching {len(folds)} folds on {slots} slot(s): {waves} wave(s), {budget/60:.0f} min budget per fold")
    todo, safe_mode = list(folds), False
    for attempt in range(2):
        failed = []
        for w in range(0, len(todo), slots):
            wave = todo[w:w + slots]
            procs = []
            for j, f in enumerate(wave):
                p = _spawn(work, cache_dir, f, gpus[j % slots] if gpus else None, budget, safe_mode, bool(gpus))
                th = threading.Thread(target=_pump, args=(p, f"fold{f}|gpu{gpus[j % slots] if gpus else 'cpu'}", log), daemon=True); th.start()
                procs.append((f, p, th))
            for f, p, th in procs:
                p.wait(); th.join()
                if p.returncode != 0 or not (work / "ckpt" / f"fold{f}.pt").exists():
                    log(f"!!! fold {f} failed (return code {p.returncode})"); failed.append(f)
        if not failed:
            break
        todo, safe_mode = failed, True
        log(f"retrying folds {failed} with the conservative profile (fp32, batch<=4)")
    out = {}
    for f in folds:
        rp = work / "ckpt" / f"result_fold{f}.json"
        out[f] = json.load(open(rp)) if (rp.exists() and (work / "ckpt" / f"fold{f}.pt").exists()) else None
    return out
