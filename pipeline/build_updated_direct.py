"""Attributed prvsiyan five-family vote, replacing only HDJoJo's scored input.

Upstream voting functions and rules: prvsiyan V18, Apache2.0.
Predictions: tushhaaarrrr, Dariush Afshar, HDJoJo, Muhammad Shoaib Altaf,
Tim Gluhkikh, Fususu/phuongncn and Nomannic19. No training or labels.
"""
from audit_complementarity import R, CM, px, labels, read
import ast, hashlib,json
from datetime import datetime,timezone
import numpy as np,pandas as pd

EXPECTED={
 'tush040-v354107337.csv':'91283f40c13fa18c0222fdb8d22f02f1939144829e01ed69d534498b804916d0',
 'dariush-v2-submission.csv':'b9c62bf52ef47c61830fe7ec9c88443d092bfe5872ea2447a57477a9383b604a',
 'hdjojo-v1-367892109.csv':'035d5da13bc5898e8c331fc0e5ee21606e3e39e4419708a72529147f0a1ed9b6',
 'hdjojo-latest-369397665.csv':'37e14844b57a00bd6f2138936dd098806d9654336045872c6b2425c4db25517d',
 'tush037-v351734356.csv':'5ec22a6c3849b07078d2427f59385be8ce61411df944022557ca2c35b5e44429',
 'magfilo-v362037449.csv':'07752516547a93f0b901c11ca43ef8ab329a9961944d13c935803ddecf35ce7f',
 'tim-v342770268.csv':'b46c72a265f665a0415154e90a8892845e2f3d7ba4a9e8d3c5d646dfe8e044d3',
 'maskrcnn-thr070.csv':'8c064a0da930de9add44e63eef1b0cb4ca85325ed74f224403938232f4beb2dc'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def vote(direct_file):
    files=['tush040-v354107337.csv','dariush-v2-submission.csv',direct_file,
           'tush037-v351734356.csv','magfilo-v362037449.csv','tim-v342770268.csv','maskrcnn-thr070.csv']
    families=['unet','unet','direct_yolo','unet','yolo26_crop','yolo11_crop','maskrcnn']
    sources={name:read(R/name) for name in files};family=dict(zip(files,families))
    # Reuse original functions verbatim; no reimplementation of vote thresholds.
    original=(R/'upstream-cell-08.txt').read_text(encoding='utf-8')
    nodes=ast.parse(original).body
    code='\n\n'.join(ast.get_source_segment(original,n) for n in nodes if isinstance(n,ast.FunctionDef))
    env=dict(np=np,mask_utils=CM,SIZE=2048**2,sources=sources,family=family,
             IOU_MATCH=.5,MIN_FAMILIES=3,STRICT_IOU=.6,MIN_KEEP=.5)
    exec(code,env)
    stems=json.loads((R/'image_ids.json').read_text());kept=[];taken_from={name:0 for name in files};clipped=0;backfilled=0
    for stem in stems:
        voters=env['label_voters'](stem);taken=np.zeros(2048**2,bool);this=[];runner=None
        for name,(masks,_,_) in voters.items():
            for i,p in enumerate(masks):
                found=env['families_drawing'](p,family[name],voters)
                if runner is None or len(found)>runner[0]:runner=(len(found),name,i)
                if not env['kept_by_vote'](found):continue
                free=~taken[p];left=int(free.sum())
                if left<.5*p.size:continue
                if left==p.size:rle=sources[name][stem][i]
                else:p=p[free];rle=env['encode'](p);clipped+=1
                taken[p]=True;taken_from[name]+=1;this.append(rle)
        if not this and runner is not None:
            _,name,i=runner;this.append(sources[name][stem][i]);taken_from[name]+=1;backfilled+=1
        kept.extend((f'{stem}_{i}',rle) for i,rle in enumerate(this,1))
    return pd.DataFrame(kept,columns=['filament_id','segmentation_rle']),dict(taken_from=taken_from,clipped=clipped,backfilled=backfilled)

def main():
    for f,h in EXPECTED.items():assert sha(R/f)==h,f
    protocol=json.loads((R/'updated-direct-protocol.json').read_text())
    control,_=vote('hdjojo-v1-367892109.csv')
    control_bytes=control.to_csv(index=False,lineterminator='\n').encode()
    assert hashlib.md5(control_bytes).hexdigest()==protocol['must_first_reproduce_scored_base_md5']
    assert control_bytes==(R/'submission.csv').read_bytes()
    print('Exact original1199-row scored control reproduced',flush=True)
    candidate,report=vote('hdjojo-latest-369397665.csv')
    path=R/'submission-updated-direct-v1.csv';candidate.to_csv(path,index=False,lineterminator='\n')
    assert pd.read_csv(path,dtype=str,keep_default_na=False).equals(candidate)
    old=read(R/'submission.csv');new=read(path);assert set(new)==set(old) and len(new)==180
    assert candidate.filament_id.is_unique and list(candidate.columns)==['filament_id','segmentation_rle']
    changes=[];empty=0
    for stem in sorted(new):
        masks,_,_=labels(new[stem])
        for r,p in zip(new[stem],masks):
            if len(p)==0:empty+=1;assert len(new[stem])==1
            m=CM.decode(dict(size=[2048,2048],counts=r.encode()))
            assert CM.encode(np.asfortranarray(m))['counts'].decode()==r
        removed=list(set(old[stem])-set(new[stem]));added=list(set(new[stem])-set(old[stem]))
        if removed or added:changes.append(dict(stem=stem,removed=len(removed),added=len(added)))
    h=sha(path);assert h not in {sha(R/'submission.csv'),sha(R/'submission-complement-v1.csv'),sha(R/'original-039.csv')}
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),stage='candidate_validated',
      protocol=protocol,input_sha256=EXPECTED,source_script_version=369397665,source_public_score=.40,
      baseline_exact_reproduction=True,rows=len(candidate),images=len(new),empty_masks=empty,
      changed_images=len(changes),removed_exact_masks=sum(c['removed'] for c in changes),
      added_exact_masks=sum(c['added'] for c in changes),changes=changes,report=report,
      output_sha256=h,output_file=path.name,validation='unique IDs, exact coverage, RLE roundtrip, pixel disjointness, CSV readback passed',
      independent_held_validation=False,public_score=None,submission_status='not_submitted')
    (R/'updated-direct-v1-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('protocol','input_sha256','changes')}))

if __name__=='__main__':main()
