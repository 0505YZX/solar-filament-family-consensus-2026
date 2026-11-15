"""Validate the exact final artifact including all decoded masks."""
from pathlib import Path
import sys,hashlib,json
import numpy as np,pandas as pd
from pycocotools import mask as cm

def validate(path):
    path=Path(path)
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    assert checksum=='68b26327bd9770d621753144ec29220af0e2b40094ee5756953af41409c94a09'
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    assert list(df.columns)==['filament_id','segmentation_rle']
    assert len(df)==1212 and df.filament_id.is_unique
    ids=df.filament_id.str.rsplit('_',n=1).str[0]
    expected=json.loads((Path(__file__).parent/'pipeline/image_ids.json').read_text())
    assert sorted(set(ids))==expected and len(expected)==180
    empty=0
    for stem,g in df.groupby(ids,sort=True):
        occupied=np.zeros((2048,2048),bool)
        assert g.filament_id.tolist()==[f'{stem}_{i}' for i in range(1,len(g)+1)]
        for r in g.segmentation_rle:
            mask=cm.decode(dict(size=[2048,2048],counts=r.encode('ascii')))
            assert mask.shape==(2048,2048) and not np.any(occupied & (mask>0))
            assert cm.encode(np.asfortranarray(mask))['counts'].decode('ascii')==r
            occupied|=mask>0
            if not mask.any(): empty+=1; assert len(g)==1
    assert pd.read_csv(path,dtype=str,keep_default_na=False).equals(df)
    result=dict(sha256=checksum,rows=len(df),images=len(expected),legal_empty_images=empty,canonical_rle=True,disjoint=True,unique_ids=True,csv_readback=True)
    print(json.dumps(result,indent=2))
    path.with_suffix('.validation.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__': validate(sys.argv[1])
