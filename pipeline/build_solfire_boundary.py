"""New source geometry inside one existing family, with frozen drift guards."""
from audit_complementarity import R,read,px,labels,CM,SIZE
from build_boundary_consensus import FILES,enc,iou,components,rles,family_score
from build_updated_direct import sha,EXPECTED
from collections import defaultdict,Counter
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import json

def main():
    protocol=json.loads((R/'solfire-boundary-protocol.json').read_text())
    for f,h in [('base_file','base_sha256'),('anchor_file','anchor_sha256'),('new_source_file','new_source_sha256')]:assert sha(R/protocol[f])==protocol[h]
    files={('hdjojo-latest-369397665.csv' if f.startswith('hdjojo-') else f):fam for f,fam in FILES.items()}
    for f in files:assert sha(R/f)==EXPECTED[f]
    files[protocol['new_source_file']]='unet'
    src={f:read(R/f) for f in files};base=read(R/protocol['base_file']);anchor=read(R/protocol['anchor_file'])
    out=[];records=[]
    for n,stem in enumerate(sorted(base),1):
        bp,blab,_=labels(base[stem]);ar=anchor[stem];ap=[px(r) for r in ar]
        groups=[defaultdict(list) for _ in ap];support=[defaultdict(list) for _ in ap];newmatch={}
        for f in sorted(files):
            rr=src[f].get(stem,[])
            if not rr:continue
            mat=np.asarray(CM.iou(rles(ar),rles(rr),[0]*len(rr)))
            for b,a in enumerate(ap):
                j=int(mat[b].argmax());v=float(mat[b,j])
                if v<=.5 or int(mat[:,j].argmax())!=b:continue
                if f==protocol['new_source_file'] and v<.6:continue
                groups[b][files[f]].append(px(rr[j]));support[b][files[f]].append(v)
                if f==protocol['new_source_file']:newmatch[b]=v
        proposals=list(bp);image_records=[]
        for b,p in enumerate(bp):
            rec=dict(stem=stem,index=b+1);image_records.append(rec)
            if b>=len(ap):rec['decision']='preserved_recall_addition';continue
            if b not in newmatch:rec['decision']='no_new_source_correspondence';continue
            g=groups[b];a=ap[b]
            if len(g)<3:rec['decision']='insufficient_families';continue
            if len(g)==3 and min(max(v) for v in support[b].values())<.6:rec['decision']='three_family_agreement';continue
            union=np.unique(np.concatenate([a,*[q for f in sorted(g) for q in g[f]]]))
            votes=np.zeros(len(union),float)
            for f in sorted(g):
                for q in g[f]:votes[np.searchsorted(union,q)]+=1/len(g[f])
            half=len(g)/2
            q=union[(votes>half+1e-9)|((abs(votes-half)<=1e-9)&np.isin(union,a,assume_unique=True))]
            q=q[(blab[q]==0)|(blab[q]==b+1)]
            if np.array_equal(p,q):rec['decision']='unchanged';continue
            if not len(q):rec['decision']='empty_guard';continue
            sim=iou(p,q);anchor_sim=iou(a,q)
            if min(sim,anchor_sim)<.8 or not .8<=len(q)/len(p)<=1.25 or not .8<=len(q)/len(a)<=1.25:rec['decision']='geometry_guard';continue
            if components(q)!=components(p) or components(q)!=components(a):rec['decision']='topology_guard';continue
            before,hits=family_score(p,g);after,newhits=family_score(q,g)
            if after<=before+1e-9 or newhits<hits:rec['decision']='agreement_guard';continue
            proposals[b]=q
            rec.update(decision='changed',area_before=len(p),area_after=len(q),iou_to_current=sim,iou_to_anchor=anchor_sim,
                family_mean_iou_before=before,family_mean_iou_after=after,solfire_anchor_iou=newmatch[b],families=len(g))
        occupancy=np.zeros(SIZE,np.uint16)
        for q in proposals:occupancy[q]+=1
        for b,q in enumerate(proposals):
            if np.any(occupancy[q]>1):
                assert image_records[b]['decision']=='changed'
                proposals[b]=bp[b];image_records[b]['decision']='expansion_conflict'
        rows=[enc(q) if image_records[b]['decision']=='changed' else base[stem][b] for b,q in enumerate(proposals)]
        labels(rows)
        assert rows[len(ap):]==base[stem][len(ap):]
        for b,r in enumerate(rows,1):out.append((f'{stem}_{b}',r))
        records.extend(image_records)
        if n%30==0:print(n,dict(Counter(r['decision'] for r in records)),flush=True)
    changed=[r for r in records if r['decision']=='changed']
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol,stage='no_change_do_not_submit',
        images=180,rows=1212,changed_instances=len(changed),changed_images=len({r['stem'] for r in changed}),
        decisions=dict(Counter(r['decision'] for r in records)),instances=records,source_sha256=sha(R/'build_solfire_boundary.py'),
        input_sha256={f:sha(R/f) for f in [protocol['base_file'],protocol['anchor_file'],*files]},
        formal_training_updates=0,independent_held_validation=False,accuracy_estimate=None,public_score=None,submission_status='not_submitted')
    if changed:
        frame=pd.DataFrame(out,columns=['filament_id','segmentation_rle']);path=R/'submission-solfire-boundary-v1.csv'
        original=pd.read_csv(R/protocol['base_file'],dtype=str,keep_default_na=False)
        assert frame.filament_id.tolist()==original.filament_id.tolist() and len(frame)==1212
        frame.to_csv(path,index=False,lineterminator='\n')
        assert frame.filament_id.is_unique and pd.read_csv(path,dtype=str,keep_default_na=False).equals(frame)
        actual=read(path);assert set(actual)==set(base) and len(actual)==180
        for stem in actual:
            labels(actual[stem])
            for r in actual[stem]:assert enc(px(r))==r
        checksum=sha(path);assert checksum not in {sha(p) for p in R.glob('submission*.csv') if p!=path}
        result.update(stage='candidate_validated',output_file=path.name,output_sha256=checksum,
            minimum_current_iou=min(r['iou_to_current'] for r in changed),minimum_anchor_iou=min(r['iou_to_anchor'] for r in changed),
            validation='1212 IDs/order exact,180 images,13 recall masks preserved,canonical RLE,no overlaps,CSV readback,new SHA passed')
    (R/'solfire-boundary-v1-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('protocol','instances','input_sha256')}),flush=True)

if __name__=='__main__':main()
