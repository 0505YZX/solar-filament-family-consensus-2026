"""Cache, four independent fold workers, native OOF scoring and legal submission.

This is a full recipe comparison. 707-image OOF is not the old 107-image dev set.
Threshold selection uses these OOF data, so it is reported as development CV.
"""
import os, sys, json, time, math, threading, subprocess
from pathlib import Path
from dataclasses import asdict
import numpy as np, pandas as pd, cv2, torch
from pycocotools import mask as CM
import joblib
import setup_check as S
import refiner_core as R
from solarpq import data as D, geometry as G, calibrate as C, metrics as Mx, postprocess as P, pipeline as Pl


def stage(name): print(f'\n[CASCADE STAGE] {name}',flush=True)


def native_flat_cache(stems, image_dir, cache, output):
    mm=np.lib.format.open_memmap(output,'w+',np.float16,(len(stems),2048,2048))
    for i,stem in enumerate(stems):
        img=cv2.imread(D.find_image(stem,[str(image_dir)]),0).astype(np.float32)
        disk=cache['disk'][i]*2
        rmap=G.radius_map((2048,2048),*disk); valid=rmap<S.cfg.disk_frac
        profile=G.radial_median_profile(img,rmap,valid)
        flat=(img/np.maximum(profile,1)-1)/.25
        mm[i]=np.clip(flat,-3,3)*valid
        if i%50==0 or i==len(stems)-1: print(f'[NATIVE CACHE] {i+1}/{len(stems)}',flush=True)
    mm.flush(); del mm


def run_workers(phase):
    n_gpu=torch.cuda.device_count()
    slots=min(n_gpu,2)
    print(f'[FOLD PLAN] phase={phase}: 4 folds, {slots} independent GPU workers, batch16, 24 epochs each; 2h training cap/fold',flush=True)
    def pump(p,fold):
        with open(S.WORK/f'{phase}_worker_fold{fold}.log','w') as log:
            for line in p.stdout:
                log.write(line); log.flush()
                print(f'[fold{fold}] {line.rstrip()}',flush=True)
    for wave in range(0,4,slots):
        running=[]
        for g,fold in enumerate(range(wave,min(wave+slots,4))):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),PYTHONUNBUFFERED='1',
                     OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(S.WORK))
            p=subprocess.Popen([sys.executable,str(S.WORK/'cascade_worker.py'),'--fold',str(fold),'--phase',phase],
                       cwd=str(S.WORK),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            th=threading.Thread(target=pump,args=(p,fold),daemon=True); th.start(); running.append((fold,p,th))
        failures=[]
        for fold,p,th in running:
            p.wait(); th.join()
            result_file = f'base_fold{fold}_result.json' if phase == 'base' else f'fold{fold}_result.json'
            if p.returncode!=0 or not (S.WORK/result_file).exists(): failures.append((fold,p.returncode))
        assert not failures,f'fold workers failed: {failures}; saved checkpoints and logs retained'


def native_pooled(gt, images, tau):
    parts=[]; per_image=[]
    for rec in images:
        stem=rec['stem']; rows=rec['candidates']
        chosen=[r for r in rows if r['score']>=tau]
        preds=[Mx.from_counts_str(r['rle']) for r in chosen]
        p=[]
        for rid in S.reg.obs[stem].records:
            gts=gt[rid]
            p.append(Mx.score_matrix(Mx.iou_matrix(gts,preds),len(gts),len(preds)))
        block=tuple(sum(x[k] for x in p) for k in range(4))
        parts.append(block); per_image.append({'stem':stem,'sum_iou':block[0],'tp':block[1],'fp':block[2],'fn':block[3]})
    return Mx.pool(parts),pd.DataFrame(per_image)


def tune_refined(gt, images):
    # Precompute native IoU matrices once; no resampling at score time.
    parts=[]
    for rec in images:
        rows=rec['candidates']; rles=[Mx.from_counts_str(r['rle']) for r in rows]
        scores=np.array([r['score'] for r in rows])
        mats=[Mx.iou_matrix(gt[rid],rles) for rid in S.reg.obs[rec['stem']].records]
        parts.append((scores,mats))
    curve=[]
    for tau in np.round(np.arange(.10,.421,.02),3):
        stats=[]
        for scores,mats in parts:
            keep=scores>=tau
            for mat in mats: stats.append(Mx.score_matrix(mat[:,keep],mat.shape[0],int(keep.sum())))
        pq=Mx.pool(stats)
        curve.append({'tau':float(tau),**asdict(pq)})
    table=pd.DataFrame(curve); best=table.iloc[table.pq.argmax()]
    table.to_csv(S.WORK/'refiner_tau_curve.csv',index=False)
    return float(best.tau),table


def source_control(cache,test_cache,gt):
    stage('PUBLIC BASE OOF REPRODUCTION AND LEARNED COMPONENT SCORER')
    maps=[np.load(S.TEMP/f'base_fold{f}.npy',mmap_mode='r') for f in range(4)]
    gtw={s:D.gt_records_rles(S.reg,[s],1024) for s in S.stems}
    prob_of=lambda i:np.asarray(maps[S.fold_of[i]][i,0],np.float32)/255
    cands=C.build_all_candidates(np.arange(707),S.stems,prob_of,cache,S.cfg,gtw,n_jobs=2,keep_labels=True)
    scores,scorer,_=C.crossfit_scores(cands,n_splits=5,seed=42)
    by=C.split_by_obs(scores,cands)
    tau,pq_work,curve=C.tune_tau(cands,by)
    curve.to_csv(S.WORK/'base_tau_curve.csv',index=False)
    joblib.dump(scorer,S.WORK/'base_component_scorer.joblib')
    images=[]
    for s,c,u in zip(S.stems,cands,by):
        ids=Pl.select_ids(c.table,u,tau,min_per_image=1) if c.n_cand else []
        rles=P.labels_to_rles(c.labels,ids,2048) if ids else []
        images.append({'stem':s,'candidates':[{'score':1.,'rle':Mx.to_counts_str(r)} for r in rles]})
    pq,per=native_pooled(gt,images,.5)
    per.to_csv(S.WORK/'base_oof_per_image.csv',index=False)
    json.dump({'pq':asdict(pq),'work_pq':pq_work,'tau':tau,'author_reference_pq':.43308033200286,
               'same_source_weights':True,'fold_assignment_verified':True,
               'scorer_protocol':'source-style five-group crossfit on base OOF candidates',
               'min_per_image':1},open(S.WORK/'base_reproduction.json','w'),indent=2)
    assert abs(pq.pq-.43308033200286)<.018, f'base OOF reproduction deviates unexpectedly: {pq}'
    test_maps=[np.load(S.TEMP/f'test_base_fold{f}.npy',mmap_mode='r') for f in range(4)]
    preds={}
    for i,s in enumerate(S.test_stems):
        prob=sum(np.asarray(m[i,0],np.float32)/255 for m in test_maps)/4
        preds[s],_=Pl.predict_instances(s,prob,np.asarray(test_cache['chans'][i]),test_cache['disk'][i],
                                        S.cfg,scorer,tau,min_per_image=1)
    rep=write_checked(preds,'public_base_control_submission.csv')
    print('[BASE REPRODUCED]',json.dumps({'pq':asdict(pq),'tau':tau,'submission':rep}),flush=True)
    return pq,per,preds


def write_checked(preds,filename):
    assert set(preds)==set(S.test_stems)
    rows=[]; empties=0
    for stem in S.test_stems:
        masks=preds[stem]
        if not masks:
            masks=[Mx.encode(np.zeros((2048,2048),np.uint8))]; empties+=1
        coverage=np.zeros((2048,2048),np.uint8)
        for k,r in enumerate(masks,1):
            m=CM.decode(r)
            assert m.shape==(2048,2048) and not (coverage&m).any()
            coverage |= m
            counts=Mx.to_counts_str(r)
            assert np.array_equal(m,CM.decode(Mx.from_counts_str(counts)))
            rows.append({'filament_id':f'{stem}_{k}','segmentation_rle':counts})
    frame=pd.DataFrame(rows)
    assert frame.filament_id.is_unique and not frame.segmentation_rle.isna().any()
    path=S.WORK/filename; frame.to_csv(path,index=False)
    back=pd.read_csv(path,keep_default_na=False)
    assert frame.equals(back)
    assert set(back.filament_id.str.rsplit('_',n=1).str[0])==set(S.test_stems)
    return {'file':filename,'rows':len(frame),'images':len(S.test_stems),'empty_images':empties,'disjoint':True,
            'rle_roundtrip':True,'csv_roundtrip':True,'sha256':S.sha(path)}


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


def paired_bootstrap(base,new):
    a=base.set_index('stem').loc[S.stems][['sum_iou','tp','fp','fn']].to_numpy()
    b=new.set_index('stem').loc[S.stems][['sum_iou','tp','fp','fn']].to_numpy()
    rng=np.random.default_rng(2042); delta=[]
    for _ in range(2000):
        idx=rng.integers(0,707,707); aa=a[idx].sum(0);bb=b[idx].sum(0)
        p=lambda v:v[0]/max(v[1]+.5*(v[2]+v[3]),1)
        delta.append(p(bb)-p(aa))
    return {'native_pq_delta':float((b.sum(0)[0]/(b.sum(0)[1]+.5*(b.sum(0)[2]+b.sum(0)[3]))) -
                                  (a.sum(0)[0]/(a.sum(0)[1]+.5*(a.sum(0)[2]+a.sum(0)[3])))),
            'image_bootstrap_95_ci':np.quantile(delta,[.025,.975]).tolist(),
            'thresholds_tuned_on_these_oof_data':True,'not_an_independent_test':True}


def run_smoke():
    """Exercise actual targets, optimizer, paste-back, scorer and CSV before hours of training."""
    import cascade_worker as W
    names=['WORK','TEMP','stems','test_stems','fold_of']
    original={name:getattr(S,name) for name in names}
    t=time.time()
    try:
        W.SMOKE=True
        S.WORK=original['WORK']/'smoke';S.WORK.mkdir(exist_ok=True)
        S.TEMP=original['TEMP']/'smoke';S.TEMP.mkdir(exist_ok=True)
        (S.WORK/'refiner_ckpt').mkdir(exist_ok=True)
        idx=np.r_[np.flatnonzero(original['fold_of']!=0)[:6],np.flatnonzero(original['fold_of']==0)[:2]]
        S.stems=[original['stems'][i] for i in idx]
        S.fold_of=original['fold_of'][idx]
        S.test_stems=original['test_stems'][:2]
        D.build_cache(S.reg,S.stems,[str(S.TRAIN_DIR)],S.cfg,str(S.TEMP/'train_cache'),with_targets=False)
        test_reg=D.Registry(obs={s:D.Observation(s,s[-2:],s[:-2]) for s in S.test_stems},anns={})
        D.build_cache(test_reg,S.test_stems,[str(S.TEST_DIR)],S.cfg,str(S.TEMP/'test_cache'),with_targets=False)
        cache=D.open_cache(str(S.TEMP/'train_cache'));test_cache=D.open_cache(str(S.TEMP/'test_cache'))
        native_flat_cache(S.stems,S.TRAIN_DIR,cache,S.TEMP/'train_native_flat.npy')
        native_flat_cache(S.test_stems,S.TEST_DIR,test_cache,S.TEMP/'test_native_flat.npy')
        native=np.load(S.TEMP/'train_native_flat.npy',mmap_mode='r')
        test_native=np.load(S.TEMP/'test_native_flat.npy',mmap_mode='r')
        dev=torch.device('cuda:0')
        maps=W.predict_maps(0,cache,S.TEMP/'smoke_maps.npy',dev)
        test_maps=W.predict_maps(0,test_cache,S.TEMP/'smoke_test_maps.npy',dev)
        bank,n=W.build_bank(0,maps,cache,native)
        torch.manual_seed(2042);init=R.build_refiner(True)
        torch.save(init.state_dict(),S.WORK/'refiner_init.pt');del init
        model,history=W.train_refiner(0,bank,dev,S.WORK/'refiner_init.pt')
        scorer=W.train_scorer_and_export(0,model,maps,cache,native,dev)
        test=[]
        for i,s in enumerate(S.test_stems):
            oc=W.candidates(i,test_maps,test_cache,False)
            rows,_=W.refine_image(model,test_native[i],np.asarray(test_maps[i],np.float32)/255,oc,test_cache['disk'][i],dev)
            scores=np.clip(scorer.predict(pd.DataFrame(rows)[W.FEATURES].to_numpy(np.float64)),0,1) if rows else []
            for row,score in zip(rows,scores):row['score']=float(score)
            test.append({'stem':s,'candidates':rows})
        fused=fuse_test([test,test,test,test],.20)
        rep=write_checked(fused,'smoke_submission.csv')
        gt=D.gt_records_rles(S.reg,S.stems,2048)
        oof=json.load(open(S.WORK/'refined_oof_fold0.json'))
        pq,_=native_pooled(gt,oof,.20)
        result={'status':'end_to_end_smoke_passed','seconds':time.time()-t,'train_images':6,'validation_images':2,
                'test_images':2,'crops':n,'epochs':2,'train_batch':W.TRAIN_BATCH,
                'updates':history[-1]['updates'],'amp_skips':history[-1]['amp_skips'],
                'peak_gb':max(r['peak_gb'] for r in history),'submission':rep,
                'smoke_pq_is_not_an_accuracy_estimate':True,'smoke_pq':asdict(pq)}
        json.dump(result,open(original['WORK']/'cascade_smoke.json','w'),indent=2)
        print('[CASCADE SMOKE PASSED]',json.dumps(result),flush=True)
        del model
        torch.cuda.empty_cache()
    finally:
        for name,value in original.items():setattr(S,name,value)
        W.SMOKE=False


def main():
    start=time.time()
    stage('REAL GPU PREFLIGHT')
    S.run_preflight()
    stage('END-TO-END REAL-DATA SMOKE TEST')
    run_smoke()
    stage('PHYSICAL FEATURES AND NATIVE FLAT-FIELD CACHE')
    D.build_cache(S.reg,S.stems,[str(S.TRAIN_DIR)],S.cfg,str(S.TEMP/'train_cache'),with_targets=False)
    test_reg=D.Registry(obs={s:D.Observation(s,s[-2:],s[:-2]) for s in S.test_stems},anns={})
    D.build_cache(test_reg,S.test_stems,[str(S.TEST_DIR)],S.cfg,str(S.TEMP/'test_cache'),with_targets=False)
    cache=D.open_cache(str(S.TEMP/'train_cache'));test_cache=D.open_cache(str(S.TEMP/'test_cache'))
    native_flat_cache(S.stems,S.TRAIN_DIR,cache,S.TEMP/'train_native_flat.npy')
    native_flat_cache(S.test_stems,S.TEST_DIR,test_cache,S.TEMP/'test_native_flat.npy')
    torch.manual_seed(2042)
    init=R.build_refiner(pretrained=True)
    torch.save(init.state_dict(),S.WORK/'refiner_init.pt'); del init
    (S.WORK/'refiner_ckpt').mkdir(exist_ok=True)
    stage('FOUR-FOLD BASE MAPS; OUTER FOLD EXCLUDED FROM ALL TRAINING FEATURES')
    run_workers('base')
    gt=D.gt_records_rles(S.reg,S.stems,2048)
    base_pq,base_per,base_preds=source_control(cache,test_cache,gt)
    stage('FOUR-FOLD UNET++ INSTANCE REFINER TRAINING')
    run_workers('refiner')
    stage('OFFICIAL-EQUIVALENT NATIVE OOF PQ AND RECIPE SELECTION')
    refined=sum([json.load(open(S.WORK/f'refined_oof_fold{f}.json')) for f in range(4)],[])
    assert len(refined)==707 and {r['stem'] for r in refined}==set(S.stems)
    tau,curve=tune_refined(gt,refined)
    refined_pq,refined_per=native_pooled(gt,refined,tau)
    refined_per.to_csv(S.WORK/'refined_oof_per_image.csv',index=False)
    comparison=paired_bootstrap(base_per,refined_per)
    pd.DataFrame([{'recipe':'public_base_source_style_oof',**asdict(base_pq)},
                  {'recipe':'strict_outer_fold_unetplusplus_cascade',**asdict(refined_pq)}]).to_csv(S.WORK/'cascade_comparison.csv',index=False)
    stage('FOUR-FOLD INSTANCE FUSION AND SUBMISSION CHECKS')
    records=[json.load(open(S.WORK/f'refined_test_fold{f}.json')) for f in range(4)]
    fused=fuse_test(records,tau)
    refined_report=write_checked(fused,'unetplusplus_cascade_submission.csv')
    # Preserve both candidates; avoid replacing a stronger recipe based on wishful thinking.
    select_refined=refined_pq.pq>=base_pq.pq+.003
    chosen=fused if select_refined else base_preds
    final_report=write_checked(chosen,'submission.csv')
    result={'status':'completed','total_seconds':time.time()-start,'gpus':torch.cuda.device_count(),
            'base_oof_pq':asdict(base_pq),'refined_oof_pq':asdict(refined_pq),'tau':tau,
            'comparison':comparison,'selected_recipe':'unetplusplus_cascade' if select_refined else 'public_base_control',
            'refined_submission':refined_report,'submission':final_report,
            'fold_results':[json.load(open(S.WORK/f'fold{f}_result.json')) for f in range(4)],
            'old_public_best':.35,'public_score':None,'public_rank':None,'top10_achieved':False,
            'validation_limits':['OOF used for threshold and recipe selection; not independent test',
               'strict fold-specific base/refiner/scorer validation; training upstream features are in-sample',
               'four-fold test fusion cannot be independently validated with these overlapping public weights',
               '707-image folds differ from our older 107-image developer set; do not directly compare their PQ']}
    json.dump(result,open(S.WORK/'cascade_summary.json','w'),indent=2)
    print('[CASCADE FINAL REPORT]',json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
