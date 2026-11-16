"""Reproducible evidence behind the design decisions (run in notebook section 2 on the training annotations only)."""
from __future__ import annotations
import numpy as np
import cv2
from pycocotools import mask as M
from . import data as D
from . import metrics as Mx


def human_vs_human(reg: D.Registry, size: int = 1024, max_obs: int | None = None) -> Mx.PQ:
    """One annotator plays 'the model', every OTHER annotator of the same image is the judge (official scoring logic).
    This is the honest ceiling for a predictor that behaves like a single expert."""
    parts, n = [], 0
    for stem, o in reg.obs.items():
        if len(o.records) < 2:
            continue
        rl = {r: D.instance_rles(reg.anns[r], size) for r in o.records}
        for a in o.records:
            for b in o.records:
                if a != b:
                    parts.append(Mx.score_matrix(Mx.iou_matrix(rl[b], rl[a]), len(rl[b]), len(rl[a])))
        n += 1
        if max_obs and n >= max_obs:
            break
    return Mx.pool(parts)


def touching_fraction(reg: D.Registry, n_records: int = 200, size: int = 1024, seed: int = 0) -> dict:
    """How often do instances of ONE annotator touch each other? If rarely, connected components of a foreground mask ARE the
    instances, and an oracle 'union mask -> components' recovers them (PQ vs. its own annotator)."""
    rng = np.random.default_rng(seed)
    rids = sorted(reg.anns)
    rids = [rids[i] for i in rng.permutation(len(rids))[:n_records]]
    total = touching = 0
    parts = []
    for r in rids:
        gts = D.instance_rles(reg.anns[r], size)
        u = M.decode(M.merge(gts)).astype(np.uint8)
        n, lab = cv2.connectedComponents(u, connectivity=8)
        comps = [Mx.encode(lab == k) for k in range(1, n)]
        per_cc = np.zeros(len(gts), int)
        for i, g in enumerate(gts):
            m = M.decode(g).astype(bool)
            per_cc[i] = len(np.unique(lab[m & (lab > 0)]))
        # instances that share a component with another instance
        cc_of = [set(np.unique(lab[M.decode(g).astype(bool)]).tolist()) - {0} for g in gts]
        for i in range(len(gts)):
            if any(cc_of[i] & cc_of[j] for j in range(len(gts)) if j != i):
                touching += 1
        total += len(gts)
        parts.append(Mx.score_matrix(Mx.iou_matrix(gts, comps), len(gts), len(comps)))
    pq = Mx.pool(parts)
    return {"records": len(rids), "instances": total, "touching": touching, "touching_fraction": touching / max(total, 1),
            "oracle_cc_pq": pq.pq}


def duplicate_loophole_demo(size: int = 256) -> dict:
    """The host scorer counts every (gt, pred) cell with IoU>0.5 as TP: duplicating a prediction doubles TP. We never do this."""
    a = np.zeros((size, size), np.uint8); a[40:100, 40:100] = 1
    gt = {"010101-20110101000000Ch": [Mx.encode(a)]}
    one, _ = Mx.evaluate(gt, {"20110101000000Ch": [Mx.encode(a)]})
    two, _ = Mx.evaluate(gt, {"20110101000000Ch": [Mx.encode(a), Mx.encode(a)]})
    return {"single_prediction_TP": one.tp, "duplicated_prediction_TP": two.tp}
