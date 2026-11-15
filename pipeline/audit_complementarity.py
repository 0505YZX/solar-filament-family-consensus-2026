"""Prediction agreement audit. No labels or accuracy estimates are used."""
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parent

import numpy as np,pandas as pd
from pycocotools import mask as CM
SIZE=2048*2048
def read(path):
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    assert list(df.columns)==['filament_id','segmentation_rle'] and df.filament_id.is_unique
    return {s:list(g.segmentation_rle) for s,g in df.groupby(df.filament_id.str.rsplit('_',n=1).str[0])}
def px(r):
    m=CM.decode(dict(size=[2048,2048],counts=r.encode('ascii')))
    return np.flatnonzero(m.ravel(order='F'))
def labels(rows):
    pp=[px(r) for r in rows]; lab=np.zeros(SIZE,np.int16)
    for i,p in enumerate(pp,1):
        assert not lab[p].any()
        lab[p]=i
    return pp,lab,np.array([len(p) for p in pp])
def match(p,lab,areas):
    c=np.bincount(lab[p],minlength=len(areas)+1)[1:]
    u=len(p)+areas-c
    a=np.divide(c,u,out=np.zeros(len(c),float),where=u>0)
    return float(a.max()) if len(a) else 0.
def run():
    old=read(R/'original-039.csv'); new=read(R/'submission.csv')
    assert set(old)==set(new) and len(old)==180
    records=[]; common=0; empty_old=0; old_total=0
    for stem in sorted(old):
        masks,lab,areas=labels(new[stem]); op,_,_=labels(old[stem])
        for i,p in enumerate(op):
            if not len(p): empty_old+=1; continue
            old_total+=1
            iou=match(p,lab,areas); overlap=float((lab[p]>0).mean())
            if iou>.5: common+=1
            else: records.append(dict(stem=stem,index=i,area=len(p),best_iou=iou,overlap=overlap))
    value=dict(old_nonempty=old_total,old_empty=empty_old,new_rows=sum(map(len,new.values())),
        matched_old_iou_gt_05=common,unmatched_old=len(records),
        low_overlap_old=sum(r['overlap']<.1 for r in records),unmatched=records,
        limitation='Agreement audit on unlabeled test predictions, not held validation or an accuracy estimate')
    (R/'complementarity-audit.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in value.items() if k!='unmatched'}))
if __name__=='__main__':run()
