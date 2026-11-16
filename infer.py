"""Predict from grayscale images using frozen V4 dense/refiner/scorer weights."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'cascade'))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(2**20),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images',type=Path,required=True)
    ap.add_argument('--models',type=Path,default=ROOT/'weights')
    ap.add_argument('--output',type=Path,default=Path('inference_output'))
    ap.add_argument('--limit',type=int,default=0,help='Diagnostic only: first N images, no final-submission claim')
    ap.add_argument('--require-exact',action='store_true',help='Require the scored CSV byte hash after full inference')
    args=ap.parse_args()
    from verify_models import verify
    verify(args.models)
    import cv2
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from pycocotools import mask as CM
    import inference_core as K
    import refiner_core as R
    from solarpq import features as F, geometry as G, model as B, calibrate as C, postprocess as P
    from solarpq.config import Config
    from validate import validate

    assert torch.cuda.is_available(),'The selected scored recipe uses CUDA fp16; CPU parity is not established.'
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0')
    files=sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.jpeg','.jpg','.png'))
    stems=[p.stem for p in files]
    expected=json.loads((ROOT/'image_ids.json').read_text())
    if args.limit:
        assert args.limit>0
        files=files[:args.limit]; stems=stems[:args.limit]
    else:
        assert stems==expected,'Require exactly the original 180 test image IDs'
    assert len(stems)==len(set(stems)) and stems
    args.output.mkdir(parents=True,exist_ok=False)
    K.S.test_stems=stems
    fold_records=[]; started=time.time()
    features=P.FEATURE_COLUMNS+['refined_area_ratio','refined_seed_iou','refined_confidence',
        'refined_spine_mean','refined_area','refined_p90','refined_changed_fraction','refined_empty_fallback']
    for fold in range(4):
        checkpoint=K.load_public(args.models/'base'/f'fold{fold}.pt')
        assert int(checkpoint['fold'])==fold
        cfg=Config.from_json(checkpoint['cfg'])
        assert (cfg.work_res,cfg.full_res,cfg.in_channels,cfg.aux_spine,cfg.n_folds,cfg.seed)==(1024,2048,3,True,4,42)
        cfg.t_lo,cfg.t_hi,cfg.close_radius,cfg.min_area=.2,.4,3,40
        K.cfg=cfg; K.S.cfg=cfg
        base=B.build_model(cfg).to(device).eval()
        base.load_state_dict(checkpoint['state'],strict=True)
        del checkpoint
        refined=torch.load(args.models/'refiner'/f'fold{fold}.pt',map_location='cpu',weights_only=True)
        assert refined['fold']==fold and refined['epoch']==24
        refiner=R.build_refiner(False).to(device).eval()
        refiner.load_state_dict(refined['state'],strict=True)
        del refined
        scorer=joblib.load(args.models/'scorer'/f'fold{fold}.joblib')
        records=[]
        for i,(path,stem) in enumerate(zip(files,stems)):
            image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            assert image is not None and image.shape==(2048,2048)
            channels=F.compute_channels(image,cfg)
            disk=np.asarray(channels.disk,np.float32)
            maps=np.round(K.two_head_prob(base,channels.chans,disk,device,d4=True)*255).astype(np.uint8)
            oc=C.build_candidates(stem,maps[0].astype(np.float32)/255,
                np.asarray(channels.chans),disk,cfg,None,keep_labels=True)
            rmap=G.radius_map((2048,2048),*(disk*2))
            valid=rmap<cfg.disk_frac
            profile=G.radial_median_profile(image.astype(np.float32),rmap,valid)
            native=(np.clip((image.astype(np.float32)/np.maximum(profile,1)-1)/.25,-3,3)*valid).astype(np.float16)
            rows,_=K.refine_image(refiner,native,maps.astype(np.float32)/255,oc,disk,device)
            scores=np.clip(scorer.predict(pd.DataFrame(rows)[features].to_numpy(np.float64)),0,1) if rows else []
            for row,score in zip(rows,scores): row['score']=float(score)
            records.append({'stem':stem,'candidates':rows})
            print(f'fold={fold} images={i+1}/{len(stems)}',flush=True)
        fold_records.append(records)
        (args.output/f'refined_test_fold{fold}.json').write_text(json.dumps(records))
        del base,refiner,scorer
        torch.cuda.empty_cache()
    predictions=K.fuse_test(fold_records,.28)
    rows=[]
    for stem in stems:
        masks=predictions[stem]
        if not masks: masks=[CM.encode(np.asfortranarray(np.zeros((2048,2048),np.uint8)))]
        for i,rle in enumerate(masks,1):
            counts=rle['counts'].decode('ascii') if isinstance(rle['counts'],bytes) else rle['counts']
            rows.append({'filament_id':f'{stem}_{i}','segmentation_rle':counts})
    output=args.output/'submission.csv'
    pd.DataFrame(rows).to_csv(output,index=False,lineterminator='\n')
    result=validate(output,expected_ids=stems,require_scored_hash=args.require_exact and not args.limit)
    result.update({'source_version':357558453,'formal_training_updates':0,'seconds':time.time()-started,
        'diagnostic_subset':bool(args.limit),'scored_reference_sha256':'85db516f7c7490e7d48a6dd953f408df7b28d4b6d2a47afb9207ed1528c9668d',
        'exact_scored_file_match':digest(output)=='85db516f7c7490e7d48a6dd953f408df7b28d4b6d2a47afb9207ed1528c9668d'})
    (args.output/'inference_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
