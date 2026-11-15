"""Prespecified supplement to scored0.41; no training or threshold search.

Public model/prediction credits remain in README.md. Uses equal-family
consensus from the preceding scored experiment only on new candidates.
"""
from audit_complementarity import R, CM, read, labels, px, SIZE, match
from build_boundary_consensus import enc, iou, components, rles, FILES
from build_updated_direct import EXPECTED, sha
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

DIRECT='hdjojo-latest-369397665.csv'

def best_index(p, masks):
    if not masks: return None,0.
    values=[iou(p,q) for q in masks]
    j=int(np.argmax(values))
    return j,float(values[j])

def main():
    protocol=json.loads((R/'new-direct-recall-protocol.json').read_text())
    assert sha(R/protocol['base_file'])==protocol['base_sha256']
    assert sha(R/DIRECT)==protocol['candidate_source_sha256']
    files={f:family for f,family in FILES.items() if family!='direct_yolo'}
    for f in files: assert sha(R/f)==EXPECTED[f],f
    base=read(R/protocol['base_file']); direct=read(R/DIRECT)
    sources={f:read(R/f) for f in files}
    assert len(base)==180 and set(direct)<=set(base)
    additions=[]; diagnostics=[]; output=[]; reasons=Counter()
    for number,stem in enumerate(sorted(base),1):
        bp,blab,ba=labels(base[stem]);dp,_,_=labels(direct.get(stem,[]))
        other={f:[px(r) for r in sources[f].get(stem,[])] for f in files}
        candidates=[]
        for index,p in enumerate(dp):
            rec=dict(stem=stem,source_index=index+1,direct_area=len(p))
            diagnostics.append(rec)
            if not len(p): rec['decision']='empty_source';continue
            base_iou=match(p,blab,ba);overlap=float((blab[p]>0).mean())
            rec.update(best_base_iou=base_iou,base_overlap_fraction=overlap)
            if base_iou>.5 or overlap>.1: rec['decision']='existing_or_overlap';continue
            groups=defaultdict(list);support=defaultdict(list)
            groups['direct_yolo'].append(p);support['direct_yolo'].append(1.)
            for f in sorted(other):
                j,value=best_index(p,other[f])
                if value<.6: continue
                back,_=best_index(other[f][j],dp)
                if back!=index: continue
                groups[files[f]].append(other[f][j]);support[files[f]].append(value)
            rec.update(families=sorted(groups),support={f:max(v) for f,v in support.items()})
            if len(groups)<3: rec['decision']='insufficient_families';continue
            union=np.unique(np.concatenate([m for f in sorted(groups) for m in groups[f]]))
            votes=np.zeros(len(union),np.float64)
            for f in sorted(groups):
                for m in groups[f]:votes[np.searchsorted(union,m)]+=1/len(groups[f])
            half=len(groups)/2
            old_here=np.isin(union,p,assume_unique=True)
            q=union[(votes>half+1e-9)|((np.abs(votes-half)<=1e-9)&old_here)]
            if not len(q):rec['decision']='empty_consensus';continue
            similarity=iou(p,q);ratio=len(q)/len(p)
            rec.update(boundary_iou_to_direct=similarity,boundary_area_ratio=ratio)
            if similarity<.8 or not .8<=ratio<=1.25:rec['decision']='geometry_guard';continue
            if components(p)!=components(q):rec['decision']='topology_guard';continue
            clipped=q[blab[q]==0]
            retained=len(clipped)/len(q)
            if retained<.9 or not len(clipped):rec['decision']='remaining_area_guard';continue
            if components(clipped)!=components(p):rec['decision']='clipped_topology_guard';continue
            after={f:max(iou(clipped,m) for m in groups[f]) for f in sorted(groups)}
            rec.update(retained_fraction=retained,support_after=after,final_area=len(clipped))
            if sum(value>=.6 for value in after.values())<3:rec['decision']='after_support_guard';continue
            rec['decision']='accepted'
            candidates.append((clipped,rec))
        occupancy=np.zeros(SIZE,np.uint16)
        for q,_ in candidates:occupancy[q]+=1
        valid=[]
        for q,rec in candidates:
            if np.any(occupancy[q]>1):rec['decision']='supplement_conflict';continue
            valid.append((q,rec))
        rows=list(base[stem])
        for q,rec in valid:
            assert not blab[q].any()
            row=enc(q);assert np.array_equal(px(row),q)
            rows.append(row);additions.append(rec)
        labels(rows)
        for index,row in enumerate(rows,1):output.append((f'{stem}_{index}',row))
        assert rows[:len(base[stem])]==base[stem]
        if number%30==0:print(number,'qualifying supplements',len(additions),flush=True)
    reasons.update(rec['decision'] for rec in diagnostics)
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol,
        stage='candidate_checks_completed',images=180,base_instances=1199,added_instances=len(additions),
        changed_images=len({rec['stem'] for rec in additions}),rows=len(output),
        decisions=dict(reasons),additions=additions,diagnostics=diagnostics,
        input_sha256={f:sha(R/f) for f in [protocol['base_file'],DIRECT,*files]},
        source_sha256=sha(R/'build_new_direct_recall.py'),formal_training_updates=0,
        independent_held_validation=False,accuracy_estimate=None,public_score=None,submission_status='not_submitted')
    if additions:
        frame=pd.DataFrame(output,columns=['filament_id','segmentation_rle'])
        path=R/'submission-new-direct-recall-v1.csv'
        frame.to_csv(path,index=False,lineterminator='\n')
        assert frame.filament_id.is_unique and pd.read_csv(path,dtype=str,keep_default_na=False).equals(frame)
        new=read(path);assert set(new)==set(base)
        for stem in new:
            labels(new[stem]);assert new[stem][:len(base[stem])]==base[stem]
            for row in new[stem]:assert enc(px(row))==row
        checksum=sha(path)
        assert checksum not in {sha(R/f) for f in ['submission.csv','submission-boundary-consensus-v1.csv',protocol['base_file']]}
        result.update(stage='candidate_validated',output_file=path.name,output_sha256=checksum,
            validation='180ID coverage,1199 base masks and IDs exact,canonical RLE roundtrip,disjointness,CSV readback,new SHA passed')
    else:result['stage']='no_qualifying_candidates_do_not_submit'
    (R/'new-direct-recall-v1-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('protocol','diagnostics','input_sha256')}),flush=True)

if __name__=='__main__':main()
