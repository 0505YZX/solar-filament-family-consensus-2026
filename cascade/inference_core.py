"""Exact selected-run inference functions extracted without training side effects."""
import numpy as np
import pandas as pd
import cv2
import torch
from pycocotools import mask as CM
from types import SimpleNamespace
import refiner_core as R
from solarpq import model as B, geometry as G, metrics as Mx, postprocess as P
S=SimpleNamespace(cfg=None,test_stems=None)
cfg=None


def load_public(path):
    # Author's val metrics contain NumPy scalar metadata. Permit only these
    # standard scalar/dtype constructors, keep the restricted tensor loader.
    scalar = np._core.multiarray.scalar
    permitted = {'numpy._core.multiarray.scalar', 'numpy.core.multiarray.scalar', 'numpy.dtype'}
    unexpected = set(torch.serialization.get_unsafe_globals_in_checkpoint(path)) - permitted
    assert not unexpected, f'unexpected checkpoint globals: {unexpected}'
    with torch.serialization.safe_globals([scalar, np.dtype,
            type(np.dtype(np.float64)), type(np.dtype(np.float32)), type(np.dtype(np.int64))]):
        return torch.load(path, map_location='cpu', weights_only=True)


def two_head_prob(model, chans, disk, device, d4=True):
    from solarpq.infer import _apply, _invert, _D4
    valid = (G.radius_map((1024, 1024), *disk) < cfg.disk_frac).astype(np.float32)
    x = torch.from_numpy(np.asarray(chans, np.float32).copy())[None].to(device)
    v = torch.from_numpy(valid)[None, None].to(device)
    x = x * v
    acc = torch.zeros(1, 2, 1024, 1024, device=device)
    transforms = _D4 if d4 else [(0, False)]
    with torch.no_grad():
        for k, flip in transforms:
            with torch.autocast('cuda', dtype=torch.float16):
                y = model(_apply(x, k, flip))
            assert y.shape == (1, 2, 1024, 1024) and torch.isfinite(y).all()
            acc += _invert(y.float().sigmoid(), k, flip)
    out = (acc / len(transforms) * v)[0].cpu().numpy()
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
    return out


@torch.no_grad()
def crop_predictions(model, xs, device):
    x = torch.from_numpy(np.stack(xs)).to(device)
    acc = torch.zeros(len(x),1,R.CROP_H,R.CROP_W, device=device)
    for dims in [(),(-1,),(-2,),(-1,-2)]:
        xx = x.flip(dims) if dims else x
        with torch.autocast('cuda', dtype=torch.float16):
            y = model(xx)
        y = y.float().sigmoid()
        acc += y.flip(dims) if dims else y
    return (acc/4)[:,0].cpu().numpy()


def refine_image(model, native, prob, oc, disk, device):
    """Disjoint geometry is fixed BEFORE expected-IoU fitting or evaluation."""
    if not oc.n_cand:
        return [], np.zeros((2048,2048), np.int32)
    crops, matrices=[],[]
    ids = oc.table['label'].astype(int).tolist()
    for label in ids:
        x, matrix = R.input_crop(native, prob, oc.labels, label)
        crops.append(x); matrices.append(matrix)
    preds = np.concatenate([crop_predictions(model,crops[j:j+16],device) for j in range(0,len(crops),16)])
    labels = np.zeros((2048,2048),np.int32)
    quality = np.zeros((2048,2048),np.float32)
    valid = G.radius_map((2048,2048),*(np.asarray(disk)*2)) < S.cfg.disk_frac
    full_seeds = cv2.resize(oc.labels.astype(np.int32),(2048,2048),interpolation=cv2.INTER_NEAREST)
    metadata=[]
    for j, label in enumerate(ids):
        mixed = .80*preds[j]+.20*crops[j][3]
        box, paste = R.paste_region(mixed, matrices[j])
        x0,y0,x1,y1=box
        seed=full_seeds[y0:y1,x0:x1]==label
        mask = R.seed_connected((paste>=.45)&valid[y0:y1,x0:x1],seed)
        fallback = not mask.any()
        # A rejected shape remains a scored base candidate; no fabricated geometry.
        if fallback:
            mask = seed
            paste = .25*seed.astype(np.float32)
        seed_area=max(int(seed.sum()),1)
        extra={'refined_area_ratio':float(mask.sum()/seed_area),
               'refined_seed_iou':float((mask&seed).sum()/max((mask|seed).sum(),1)),
               'refined_confidence':float(paste[mask].mean()) if mask.any() else 0.,
               'refined_spine_mean':float((crops[j][2]*preds[j]).sum()/max(preds[j].sum(),1)),
               'refined_area':float(mask.sum()),
               'refined_p90':float(np.percentile(paste[mask],90)) if mask.any() else 0.,
               'refined_changed_fraction':float((mask^seed).sum()/max((mask|seed).sum(),1)),
               'refined_empty_fallback':float(fallback)}
        region_q=quality[y0:y1,x0:x1]; region_l=labels[y0:y1,x0:x1]
        put=mask & (paste>region_q)
        region_q[put]=paste[put]; region_l[put]=label
        metadata.append(extra)
    rows=[]
    for j,label in enumerate(ids):
        m=labels==label
        if not m.any(): continue
        row={k:float(oc.table.iloc[j][k]) for k in P.FEATURE_COLUMNS}
        row.update(metadata[j]); row['refined_area']=float(m.sum())
        row.update({'label':label,'rle':Mx.to_counts_str(Mx.encode(m))})
        rows.append(row)
    return rows, labels


def fuse_test(fold_records,tau):
    """Fold consensus of matched instance shapes, then disjoint score painting.

    Each cluster has at most one candidate from each fold. The ensemble has
    no independent four-fold OOF analogue because public base models overlap
    in training images; final generalization is measured on public submission.
    """
    output={}
    for i,stem in enumerate(S.test_stems):
        candidates=[]
        for fold,records in enumerate(fold_records):
            assert records[i]['stem']==stem
            candidates.extend({**r,'fold':fold} for r in records[i]['candidates'] if r['score']>=.08)
        candidates.sort(key=lambda r:-r['score'])
        clusters=[]
        for row in candidates:
            rle=Mx.from_counts_str(row['rle'])
            options=[(j,c) for j,c in enumerate(clusters) if row['fold'] not in c['folds']]
            if options:
                overlaps=Mx.iou_matrix([rle],[c['rles'][0] for _,c in options]).ravel()
                hit=int(overlaps.argmax())
                if overlaps[hit]>.35:
                    cluster=options[hit][1]
                    cluster['rles'].append(rle); cluster['scores'].append(row['score']);cluster['folds'].add(row['fold'])
                    continue
            clusters.append({'rles':[rle],'scores':[row['score']],'folds':{row['fold']}})
        selected=[]
        for cluster in clusters:
            support=len(cluster['folds'])
            score=float(np.mean(cluster['scores'])*(.65+.35*support/4))
            if score<tau: continue
            layers=CM.decode(cluster['rles']).astype(np.float32)
            weights=np.array(cluster['scores'],np.float32)
            consensus=(layers*weights[None,None]).sum(2)/weights.sum()
            mask=consensus>=.5
            if mask.any(): selected.append((score,mask))
        owner=np.zeros((2048,2048),np.int32)
        # Higher-quality instances claim contested pixels first.
        for label,(_,mask) in enumerate(sorted(selected,key=lambda x:-x[0]),1):
            owner[mask & (owner==0)]=label
        output[stem]=[Mx.encode(owner==label) for label in range(1,len(selected)+1) if (owner==label).sum()>=16]
    return output
