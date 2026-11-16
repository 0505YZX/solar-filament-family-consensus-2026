"""One outer fold per GPU; all upstream features use THAT fold's backbone.

Unlike using a shared OOF bank from other backbone folds for refiner training,
this prevents indirect exposure of the outer validation labels through the
training features. Training features are consequently in-sample for the base
model, a domain shift documented in the final report.
"""
import os, json, time, math, argparse
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch
from pycocotools import mask as CM
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
import setup_check as S
import refiner_core as R
from solarpq import data as D, model as B, calibrate as C, metrics as Mx, postprocess as P, geometry as G

EXTRA = ['refined_area_ratio', 'refined_seed_iou', 'refined_confidence', 'refined_spine_mean',
         'refined_area', 'refined_p90', 'refined_changed_fraction', 'refined_empty_fallback']
FEATURES = P.FEATURE_COLUMNS + EXTRA
SMOKE = False
TRAIN_BATCH = 16


def predict_maps(fold, cache, output, device):
    ck = S.load_public(S.checkpoints[fold])
    model = B.build_model(S.cfg).to(device)
    model.load_state_dict(ck['state'], strict=True); model.eval()
    mm = np.lib.format.open_memmap(output, 'w+', np.uint8, (len(cache['stems']), 2, 1024, 1024))
    t = time.time()
    for i, stem in enumerate(cache['stems']):
        p = S.two_head_prob(model, cache['chans'][i], cache['disk'][i], device, d4=True)
        mm[i] = np.round(p * 255).astype(np.uint8)
        if i % 50 == 0 or i == len(cache['stems'])-1:
            print(f'[BASE MAPS] fold={fold} {i+1}/{len(cache["stems"])} seconds={time.time()-t:.1f}', flush=True)
    mm.flush()
    del model, ck
    torch.cuda.empty_cache()
    return mm


def candidates(i, maps, cache, ground_truth=False):
    stem = cache['stems'][i]
    gtw = D.gt_records_rles(S.reg, [stem], 1024) if ground_truth else None
    return C.build_candidates(stem, np.asarray(maps[i, 0], np.float32)/255,
                              np.asarray(cache['chans'][i]), cache['disk'][i], S.cfg,
                              gtw, keep_labels=True)


def gt_target(oc, candidate_index, matrix, decoded):
    target = np.zeros((R.CROP_H, R.CROP_W), np.float32)
    recs = S.reg.obs[oc.stem].records
    for rid in recs:
        mat = oc.mats[rid]
        if len(mat) == 0:
            continue
        j = int(mat[:, candidate_index].argmax())
        if mat[j, candidate_index] < .05:
            continue
        key = (rid, j)
        if key not in decoded:
            decoded[key] = CM.decode(D.instance_rles([S.reg.anns[rid][j]], 2048)[0])
        target += R.warp(decoded[key], matrix) / len(recs)
    return np.clip(target, 0, 1)


def build_bank(fold, maps, cache, native):
    root = S.TEMP / f'fold{fold}'
    root.mkdir(exist_ok=True)
    train_images = np.flatnonzero(S.fold_of != fold)
    specs, n = [], 0
    t = time.time()
    for k, i in enumerate(train_images):
        oc = candidates(i, maps, cache, True)
        specs.append((int(i), oc))
        n += oc.n_cand
    assert n > (12 if SMOKE else 1000)
    x = np.lib.format.open_memmap(root / 'crop_x.npy', 'w+', np.float16, (n, 5, R.CROP_H, R.CROP_W))
    y = np.lib.format.open_memmap(root / 'crop_y.npy', 'w+', np.uint8, (n, 1, R.CROP_H, R.CROP_W))
    cursor = 0
    for k, (i, oc) in enumerate(specs):
        decoded = {}
        prob = np.asarray(maps[i], np.float32)/255
        for j, label_id in enumerate(oc.table['label'].tolist()):
            crop, matrix = R.input_crop(native[i], prob, oc.labels, label_id)
            target = gt_target(oc, j, matrix, decoded)
            x[cursor] = crop; y[cursor, 0] = np.round(target*255).astype(np.uint8)
            cursor += 1
        if k % 50 == 0 or k == len(specs)-1:
            print(f'[CROP BANK] fold={fold} images={k+1}/{len(specs)} crops={cursor}/{n}', flush=True)
    assert cursor == n
    x.flush(); y.flush()
    del specs, x, y
    json.dump({'fold':fold, 'train_images':len(train_images), 'crops':n,
               'validation_images':int((S.fold_of==fold).sum()), 'seconds':time.time()-t,
               'upstream_model_fold':fold, 'outer_fold_excluded_from_all_stages':True},
              open(S.WORK / f'bank_fold{fold}.json', 'w'), indent=2)
    return root, n


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.x = np.load(root / 'crop_x.npy', mmap_mode='r')
        self.y = np.load(root / 'crop_y.npy', mmap_mode='r')
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(self.x[i], np.float32).copy()), torch.from_numpy(np.asarray(self.y[i], np.float32).copy()/255)


def train_refiner(fold, bank, device, init_path):
    torch.manual_seed(2042+fold); np.random.seed(2042+fold)
    # The T4 smoke run peaked at 12.4GB during exhaustive convolution tuning.
    # Fixed kernels avoid that one-time workspace spike and keep safe headroom.
    torch.backends.cudnn.benchmark = False
    model = R.build_refiner(False).to(device)
    model.load_state_dict(torch.load(init_path, map_location='cpu', weights_only=True), strict=True)
    opt = torch.optim.AdamW([{'params':model.encoder.parameters(), 'lr':1e-4},
                            {'params':list(model.decoder.parameters())+list(model.segmentation_head.parameters()), 'lr':4e-4}],
                           weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    ds = CropDataset(bank)
    loader = torch.utils.data.DataLoader(ds, batch_size=TRAIN_BATCH, shuffle=True, num_workers=0, pin_memory=True,
                                         drop_last=False, generator=torch.Generator().manual_seed(2042+fold))
    epochs = 2 if SMOKE else 24
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=2e-6)
    t = time.time(); history=[]; steps=0; skips=0
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs+1):
        model.train(); epoch_t=time.time(); losses=[]
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Synchronized flips across every conditioning channel and its target.
            for axis in (-1, -2):
                flip = torch.rand(len(x), device=device) < .5
                x[flip] = x[flip].flip((axis,)); y[flip] = y[flip].flip((axis,))
            x[:, :1] = x[:, :1] * (.90+.20*torch.rand(len(x),1,1,1,device=device))
            x[:, :1] += .04*torch.randn(len(x),1,1,1,device=device)
            x[:, 1:3] = (x[:, 1:3]+.01*torch.randn_like(x[:, 1:3])).clamp(0,1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(x)
                loss = R.crop_loss(logits, y)
            assert torch.isfinite(loss), 'nonfinite training loss'
            scale = scaler.get_scale()
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update()
            skipped = scaler.get_scale() < scale
            skips += int(skipped); steps += int(not skipped)
            losses.append(float(loss.detach()))
        sched.step()
        torch.cuda.synchronize(device)
        rec = {'fold':fold, 'epoch':epoch, 'loss':float(np.mean(losses)), 'epoch_seconds':time.time()-epoch_t,
               'total_seconds':time.time()-t, 'updates':steps, 'amp_skips':skips,
               'peak_gb':torch.cuda.max_memory_allocated(device)/1e9}
        history.append(rec)
        print('[REFINER TRAIN]', json.dumps(rec), flush=True)
        # Frequent resumable states: never lose an entire long run.
        state = {'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                 'fold':fold, 'epoch':epoch, 'train_image_count':int((S.fold_of!=fold).sum()),
                 'architecture':'UnetPlusPlus-resnet34-scse', 'channels':5, 'crop_wh':[384,128],
                 'updates':steps, 'amp_skips':skips, 'seed':2042+fold}
        torch.save(state, S.WORK / 'refiner_ckpt' / f'fold{fold}.pt')
        pd.DataFrame(history).to_csv(S.WORK / f'train_fold{fold}.csv', index=False)
        # Explicit cap for safe completion, chosen BEFORE viewing outer validation.
        if time.time()-t > 7200:
            print(f'[TIME CAP] fold {fold}: completed {epoch} epochs, fixed 2h cap reached', flush=True)
            break
    assert steps > (0 if SMOKE else 100) and skips/max(steps+skips,1) < .01
    del opt, scaler, loader, ds
    torch.cuda.empty_cache()
    model.eval()
    return model, history


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


def train_scorer_and_export(fold, model, maps, cache, native, device):
    train_rows, val_images=[],[]
    for i, stem in enumerate(S.stems):
        oc = candidates(i,maps,cache,False)
        rows, labels = refine_image(model,native[i],np.asarray(maps[i],np.float32)/255,oc,cache['disk'][i],device)
        rles=[Mx.from_counts_str(row['rle']) for row in rows]
        gts=D.gt_records_rles(S.reg,[stem],2048)
        u=np.zeros(len(rows))
        for gt in gts.values():
            mat=Mx.iou_matrix(gt,rles)
            u += (mat*(mat>.5)).sum(0)/len(gts)
        for j,row in enumerate(rows): row['target_u']=float(u[j]); row['stem']=stem
        if S.fold_of[i] == fold:
            val_images.append({'stem':stem,'candidates':rows})
        else:
            train_rows.extend({k:v for k,v in row.items() if k!='rle'} for row in rows)
        if i % 50 == 0 or i == len(S.stems)-1:
            print(f'[REFINER OOF] fold={fold} {i+1}/{len(S.stems)}',flush=True)
    table=pd.DataFrame(train_rows)
    assert not set(table.stem) & {r['stem'] for r in val_images}
    scorer=HistGradientBoostingRegressor(max_iter=250,learning_rate=.05,max_leaf_nodes=12,
                min_samples_leaf=25,l2_regularization=2.,random_state=2042+fold)
    scorer.fit(table[FEATURES].to_numpy(np.float64),table.target_u.to_numpy())
    table.to_csv(S.WORK/f'scorer_train_fold{fold}.csv',index=False)
    joblib.dump(scorer,S.WORK/f'scorer_fold{fold}.joblib')
    for rec in val_images:
        rows=rec['candidates']
        u=np.clip(scorer.predict(pd.DataFrame(rows)[FEATURES].to_numpy(np.float64)),0,1) if rows else []
        for row,score in zip(rows,u): row['score']=float(score)
    json.dump(val_images,open(S.WORK/f'refined_oof_fold{fold}.json','w'))
    return scorer


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True)
    ap.add_argument('--phase', choices=['base','refiner'], required=True)
    args=ap.parse_args(); fold=args.fold; dev=torch.device('cuda:0')
    start=time.time()
    cache=D.open_cache(str(S.TEMP/'train_cache')); test_cache=D.open_cache(str(S.TEMP/'test_cache'))
    native=np.load(S.TEMP/'train_native_flat.npy',mmap_mode='r')
    test_native=np.load(S.TEMP/'test_native_flat.npy',mmap_mode='r')
    if args.phase == 'base':
        predict_maps(fold,cache,S.TEMP/f'base_fold{fold}.npy',dev)
        predict_maps(fold,test_cache,S.TEMP/f'test_base_fold{fold}.npy',dev)
        json.dump({'fold':fold,'seconds':time.time()-start,'status':'base_maps_completed'},
                  open(S.WORK/f'base_fold{fold}_result.json','w'),indent=2)
        return
    maps=np.load(S.TEMP/f'base_fold{fold}.npy',mmap_mode='r')
    test_maps=np.load(S.TEMP/f'test_base_fold{fold}.npy',mmap_mode='r')
    bank,n=build_bank(fold,maps,cache,native)
    model,history=train_refiner(fold,bank,dev,S.WORK/'refiner_init.pt')
    scorer=train_scorer_and_export(fold,model,maps,cache,native,dev)
    test=[]
    for i,stem in enumerate(S.test_stems):
        oc=candidates(i,test_maps,test_cache,False)
        rows,_=refine_image(model,test_native[i],np.asarray(test_maps[i],np.float32)/255,oc,test_cache['disk'][i],dev)
        scores=np.clip(scorer.predict(pd.DataFrame(rows)[FEATURES].to_numpy(np.float64)),0,1) if rows else []
        for row,score in zip(rows,scores): row['score']=float(score)
        test.append({'stem':stem,'candidates':rows})
        if i % 30 == 0: print(f'[TEST REFINER] fold={fold} {i+1}/180',flush=True)
    json.dump(test,open(S.WORK/f'refined_test_fold{fold}.json','w'))
    result={'fold':fold,'seconds':time.time()-start,'epochs':len(history),'crops':n,
            'updates':history[-1]['updates'],'amp_skips':history[-1]['amp_skips'],
            'outer_validation_images':int((S.fold_of==fold).sum()),'upstream_model_fold':fold,
            'train_base_features':'in-sample from fold-specific base (outer fold excluded)',
            'test_images':len(test),'checkpoint_sha256':S.sha(S.WORK/'refiner_ckpt'/f'fold{fold}.pt')}
    json.dump(result,open(S.WORK/f'fold{fold}_result.json','w'),indent=2)
    print('[FOLD COMPLETE]',json.dumps(result),flush=True)

if __name__=='__main__': main()
