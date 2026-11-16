"""D4 test-time augmentation and fold-ensembled probability maps."""
from __future__ import annotations
import numpy as np
import torch
from .config import Config
from . import model as Mdl
from . import geometry as G


def load_models(paths: list, device) -> tuple[list, Config]:
    models, cfg = [], None
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        cfg = Config.from_json(ck["cfg"])
        m = Mdl.build_model(cfg).to(device); m.load_state_dict(ck["state"]); m.eval()
        models.append(m)
    return models, cfg


_D4 = [(k, f) for k in range(4) for f in (False, True)]


def _apply(x: torch.Tensor, k: int, f: bool) -> torch.Tensor:
    x = torch.rot90(x, k, (-2, -1))
    return torch.flip(x, (-1,)) if f else x


def _invert(x: torch.Tensor, k: int, f: bool) -> torch.Tensor:
    x = torch.flip(x, (-1,)) if f else x
    return torch.rot90(x, -k, (-2, -1))


@torch.no_grad()
def predict_prob(models: list, chans: np.ndarray, disk, cfg: Config, device, tta: str = "d4") -> np.ndarray:
    """chans (3,H,W) float16/32 -> foreground probability (H,W) float32 = mean over models x TTA of sigmoid outputs."""
    H = chans.shape[-1]
    valid = torch.from_numpy((G.radius_map((H, H), *disk) < cfg.disk_frac).astype(np.float32))[None, None].to(device)
    x = torch.from_numpy(np.asarray(chans, np.float32))[None].to(device) * valid
    tfs = _D4 if tta == "d4" else ([(0, False), (0, True)] if tta == "flip" else [(0, False)])
    acc, n = torch.zeros(1, 1, H, H, device=device), 0
    for m in models:
        for k, f in tfs:
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                out = m(_apply(x, k, f))
            acc += _invert(torch.sigmoid(out[:, 0:1].float()), k, f); n += 1
    return (acc / n * valid)[0, 0].cpu().numpy().astype(np.float32)


def predict_cache(models, cache, idx, out_path: str, cfg: Config, device, tta: str = "d4", log=print) -> np.ndarray:
    """Write uint8 probability maps (prob*255) for cache indices `idx` into a memmap aligned with the cache order."""
    N, W = len(cache["stems"]), cfg.work_res
    mm = np.lib.format.open_memmap(out_path, "r+" if _exists(out_path) else "w+", np.uint8, (N, W, W))
    for j, i in enumerate(idx):
        p = predict_prob(models, cache["chans"][i], cache["disk"][i], cfg, device, tta)
        mm[i] = np.round(p * 255).astype(np.uint8)
        if j % 25 == 0:
            log(f"  predicted {j + 1}/{len(idx)}")
    mm.flush()
    return mm


def _exists(p):
    import os
    return os.path.exists(p)
