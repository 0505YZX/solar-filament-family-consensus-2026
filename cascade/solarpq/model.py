"""Residual U-Net (GroupNorm => independent of batch size and identical behaviour on 512 crops and 1024 full frames).

Outputs: channel 0 = filament-foreground logit (target: annotator-mean soft mask);
         channel 1 (optional) = spine logit (auxiliary target from the host-permitted `spine` metadata).
Trained from scratch by default: the 3 input channels are not RGB, so ImageNet weights are of limited use and this keeps the
solution fully reproducible without internet access."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from .config import Config


def _gn(c: int) -> nn.GroupNorm:
    g = max(d for d in range(1, min(8, c) + 1) if c % d == 0)       # largest divisor of c that is <= 8
    return nn.GroupNorm(g, c)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False); self.n1 = _gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False); self.n2 = _gn(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        h = Fn.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return Fn.gelu(h + self.skip(x))


class ResUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 32, depth: int = 5, n_out: int = 2):
        super().__init__()
        widths = [min(base * 2 ** i, base * 8) for i in range(depth)]
        self.depth = depth
        self.enc = nn.ModuleList()
        c_prev = in_channels
        for i, w in enumerate(widths):
            self.enc.append(nn.Sequential(ResBlock(c_prev, w), ResBlock(w, w)) if i else nn.Sequential(ResBlock(c_prev, w)))
            c_prev = w
        self.down = nn.ModuleList([nn.Conv2d(widths[i], widths[i], 2, stride=2) for i in range(depth - 1)])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up.append(nn.Conv2d(widths[i], widths[i - 1], 1))
            self.dec.append(ResBlock(2 * widths[i - 1], widths[i - 1]))
        self.head = nn.Conv2d(widths[0], n_out, 1)
        self.stride = 2 ** (depth - 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.stride, (-w) % self.stride
        if ph or pw:
            x = Fn.pad(x, (0, pw, 0, ph), mode="reflect")
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < self.depth - 1:
                skips.append(x)
                x = self.down[i](x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = Fn.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([s, up(x)], 1))
        out = self.head(x)
        return out[..., :h, :w]


def build_model(cfg: Config) -> ResUNet:
    return ResUNet(cfg.in_channels, cfg.base_width, cfg.depth, 2 if cfg.aux_spine else 1)


# ------------------------------------------------------------------ losses
def masked_bce(logits, target, valid):
    l = Fn.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (l * valid).sum() / valid.sum().clamp_min(1.0)


def soft_dice_loss(logits, target, valid, eps: float = 1.0):
    p = torch.sigmoid(logits) * valid
    t = target * valid
    return 1.0 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)


def total_loss(out, soft, spine, valid, cfg: Config):
    """Soft-target BCE (calibrated annotator-marginal probability) + soft Dice (recall on sparse foreground) [+ spine aux]."""
    main = masked_bce(out[:, 0:1], soft, valid) + cfg.dice_weight * soft_dice_loss(out[:, 0:1], soft, valid)
    if out.shape[1] > 1 and cfg.aux_spine:
        aux = 0.5 * masked_bce(out[:, 1:2], spine, valid) + soft_dice_loss(out[:, 1:2], spine, valid)
        return main + cfg.spine_weight * aux, main.detach(), aux.detach()
    return main, main.detach(), out.new_zeros(())
