"""Native-pixel, candidate-conditioned UNet++ refinement (our implementation).

The physical dense backbone is credited separately in NOTICE.md. This module
implements a genuine SMP nested UNet++, PCA-aligned crops, and inverse warps.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy import ndimage as ndi

CROP_W, CROP_H = 384, 128


def crop_matrix(seed_work: np.ndarray, margin: float = .35) -> np.ndarray:
    """Map native 2048 coordinates to a 384x128 long-axis crop.

    1024 pixel centres map to native centres as 2*x+0.5. PCA uses ALL
    candidate pixels, and the full extrema plus context determine the frame.
    """
    yy, xx = np.nonzero(seed_work)
    if len(xx) < 3:
        raise ValueError('candidate has fewer than three pixels')
    pts = np.stack([2 * xx + .5, 2 * yy + .5], axis=1).astype(np.float64)
    center = pts.mean(0)
    ev, vec = np.linalg.eigh(np.cov((pts - center).T))
    long = vec[:, np.argmax(ev)]
    if long[0] < 0:
        long = -long
    short = np.array([-long[1], long[0]])
    axes = np.stack([long, short])
    projected = (pts - center) @ axes.T
    low, high = projected.min(0), projected.max(0)
    center = center + axes.T @ ((low + high) / 2)
    span = np.maximum(high - low + 2, [64., 24.]) * (1 + 2 * margin)
    # Limit anisotropic distortion: context along the narrow axis remains useful.
    span[1] = max(span[1], span[0] / 5)
    scale = np.array([CROP_W - 1, CROP_H - 1]) / span
    linear = axes * scale[:, None]
    offset = np.array([(CROP_W - 1) / 2, (CROP_H - 1) / 2]) - linear @ center
    return np.c_[linear, offset].astype(np.float32)


def work_matrix(native_matrix: np.ndarray) -> np.ndarray:
    m = native_matrix.copy()
    m[:, 2] += .5 * m[:, :2].sum(1)
    m[:, :2] *= 2
    return m


def warp(array, matrix, nearest=False):
    return cv2.warpAffine(np.asarray(array, np.float32), matrix, (CROP_W, CROP_H),
                          flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def input_crop(native_flat, prob2, labels, label_id, matrix=None):
    seed = labels == label_id
    matrix = crop_matrix(seed) if matrix is None else matrix
    wm = work_matrix(matrix)
    channels = [warp(native_flat, matrix), warp(prob2[0], wm), warp(prob2[1], wm),
                warp(seed, wm, True), warp((labels > 0) & ~seed, wm, True)]
    x = np.stack(channels).astype(np.float32)
    assert x.shape == (5, CROP_H, CROP_W) and np.isfinite(x).all()
    return x, matrix


def paste_probability(crop_prob, matrix, size=2048):
    return cv2.warpAffine(np.asarray(crop_prob, np.float32), cv2.invertAffineTransform(matrix),
                          (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def paste_region(crop_prob, matrix, size=2048):
    inverse = cv2.invertAffineTransform(matrix)
    corners = np.array([[0, 0], [CROP_W-1, 0], [0, CROP_H-1], [CROP_W-1, CROP_H-1]], np.float32)
    corners = corners @ inverse[:, :2].T + inverse[:, 2]
    x0, y0 = np.maximum(np.floor(corners.min(0)-2).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(corners.max(0)+3).astype(int), size)
    assert x1 > x0 and y1 > y0
    local = inverse.copy(); local[:, 2] -= [x0, y0]
    p = cv2.warpAffine(np.asarray(crop_prob, np.float32), local, (x1-x0, y1-y0),
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (int(x0), int(y0), int(x1), int(y1)), p


def seed_connected(mask, seed):
    """Keep the best seed-overlapping connected component, never a remote blob."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    overlap = np.bincount(labels[seed.astype(bool)].ravel(), minlength=n)
    overlap[0] = 0
    if overlap.max() == 0:
        return np.zeros_like(mask, bool)
    chosen = int(overlap.argmax())
    return ndi.binary_fill_holes(labels == chosen)


def geometry_check():
    seed = np.zeros((1024, 1024), np.uint8)
    cv2.ellipse(seed, (520, 470), (130, 10), 35, 0, 360, 1, -1)
    matrix = crop_matrix(seed)
    crop = warp(seed, work_matrix(matrix), True)
    native = cv2.resize(seed, (2048, 2048), interpolation=cv2.INTER_NEAREST) > 0
    back = paste_probability(crop, matrix) >= .5
    iou = (back & native).sum() / (back | native).sum()
    assert iou > .94, f'warp round-trip IoU {iou}'
    probe = np.ones((2048, 2048), np.float32)
    x, _ = input_crop(probe, np.stack([seed, seed]), seed, 1, matrix)
    aligned_a, aligned_b = x[3] > .5, x[1] > .5
    channel_iou = (aligned_a & aligned_b).sum() / (aligned_a | aligned_b).sum()
    assert channel_iou > .97
    assert not x[4].any()
    return {'warp_roundtrip_iou': float(iou), 'seed_probability_alignment_iou': float(channel_iou),
            'shape': list(x.shape), 'native_pixel_center_mapping': '2*x+0.5'}


def build_refiner(pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet' if pretrained else None,
                           encoder_depth=5, in_channels=5, classes=1,
                           decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type='scse')


def crop_loss(logits, target):
    """Ordinary per-instance soft Dice+BCE and weak edge consistency.

    ResNet/SMP uses BatchNorm; this deliberately does not pretend microbatch
    two-pass Dice is equivalent to an exact full-batch objective.
    """
    import torch
    import torch.nn.functional as F
    logits, target = logits.float(), target.float()
    p = logits.sigmoid()
    dice = 1 - (2 * (p * target).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    edge = (F.l1_loss(p[..., 1:, :]-p[..., :-1, :], target[..., 1:, :]-target[..., :-1, :]) +
            F.l1_loss(p[..., :, 1:]-p[..., :, :-1], target[..., :, 1:]-target[..., :, :-1])) / 2
    return .50 * bce + .45 * dice.mean() + .05 * edge


def gpu_check(device, pretrained=True):
    import torch
    import time
    torch.manual_seed(2042)
    model = build_refiner(pretrained).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler('cuda')
    x = torch.randn(8, 5, CROP_H, CROP_W, device=device)
    y = torch.zeros(8, 1, CROP_H, CROP_W, device=device)
    y[:, :, 40:76, 50:300] = 1
    head = model.segmentation_head[0].weight.detach().clone()
    losses = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(x)
            loss = crop_loss(out, y)
        assert out.shape == y.shape and torch.isfinite(loss)
        scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(opt); scaler.update()
        assert scaler.get_scale() >= scale, 'AMP skipped preflight update'
        losses.append(float(loss.detach()))
    torch.cuda.synchronize(device)
    assert not torch.equal(head, model.segmentation_head[0].weight)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        p = model(x[:2]).sigmoid()
    assert torch.isfinite(p).all() and p.shape == (2, 1, CROP_H, CROP_W)
    result = {'model': 'SMP UnetPlusPlus ResNet34 + SCSE', 'parameters': sum(p.numel() for p in model.parameters()),
              'batch': 8, 'crop_wh': [CROP_W, CROP_H], 'five_update_losses': losses,
              'seconds_per_step': (time.time() - started) / 5,
              'peak_gb': torch.cuda.max_memory_allocated(device) / 1e9,
              'head_updated': True, 'finite_gradients': True}
    del model, opt, x, y
    torch.cuda.empty_cache()
    return result
