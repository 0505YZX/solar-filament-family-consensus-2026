"""Validate native instance masks and identify the actually scored artifact."""
from pathlib import Path
import hashlib
import json
import sys

SCORED_SHA='85db516f7c7490e7d48a6dd953f408df7b28d4b6d2a47afb9207ed1528c9668d'

def validate(path,expected_ids=None,require_scored_hash=True):
    import numpy as np
    import pandas as pd
    from pycocotools import mask as CM
    path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if require_scored_hash: assert digest==SCORED_SHA,'Different file from actual scored CSV'
    frame=pd.read_csv(path,dtype=str,keep_default_na=False)
    assert list(frame.columns)==['filament_id','segmentation_rle'] and frame.filament_id.is_unique
    ids=frame.filament_id.str.rsplit('_',n=1).str[0]
    expected_ids=expected_ids or json.loads((Path(__file__).parent/'image_ids.json').read_text())
    assert sorted(set(ids))==sorted(expected_ids)
    empty=0
    for stem,group in frame.groupby(ids,sort=True):
        occupied=np.zeros((2048,2048),bool)
        assert group.filament_id.tolist()==[f'{stem}_{i}' for i in range(1,len(group)+1)]
        for counts in group.segmentation_rle:
            mask=CM.decode({'size':[2048,2048],'counts':counts.encode('ascii')})
            assert mask.shape==(2048,2048) and not np.any(occupied&(mask>0))
            assert CM.encode(np.asfortranarray(mask))['counts'].decode('ascii')==counts
            occupied|=mask>0
            if not mask.any(): empty+=1; assert len(group)==1
    assert pd.read_csv(path,dtype=str,keep_default_na=False).equals(frame)
    result={'sha256':digest,'rows':len(frame),'images':len(expected_ids),'legal_empty_images':empty,
        'canonical_rle':True,'disjoint':True,'unique_ids':True,'csv_readback':True}
    if require_scored_hash: assert len(frame)==1281 and empty==2
    return result

if __name__=='__main__':
    result=validate(sys.argv[1]); print(json.dumps(result,indent=2))
    Path(sys.argv[1]).with_suffix('.validation.json').write_text(json.dumps(result,indent=2))
