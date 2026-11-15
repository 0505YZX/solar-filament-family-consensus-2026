"""Equal-family mask geometry experiment; no training or hidden test labels.

Sources and detection set: attributed prvsiyan V18 (Apache2.0).
Prediction credits and limitations are retained in README.md.
"""
from audit_complementarity import R, CM, px, labels, read, SIZE
from build_updated_direct import EXPECTED, sha
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from scipy.ndimage import label

FILES = {
    'tush040-v354107337.csv': 'unet',
    'dariush-v2-submission.csv': 'unet',
    'tush037-v351734356.csv': 'unet',
    'hdjojo-v1-367892109.csv': 'direct_yolo',
    'magfilo-v362037449.csv': 'yolo26_crop',
    'tim-v342770268.csv': 'yolo11_crop',
    'maskrcnn-thr070.csv': 'maskrcnn',
}

def enc(p):
    m = np.zeros(SIZE, np.uint8)
    m[p] = 1
    return CM.encode(np.asfortranarray(m.reshape((2048, 2048), order='F')))['counts'].decode()

def rles(rows):
    return [dict(size=[2048, 2048], counts=r.encode()) for r in rows]

def iou(a, b):
    hit = np.intersect1d(a, b, assume_unique=True).size
    return hit / max(1, len(a) + len(b) - hit)

def components(p):
    y, x = p % 2048, p // 2048
    m = np.zeros((int(y.max()-y.min()+1), int(x.max()-x.min()+1)), bool)
    m[y-y.min(), x-x.min()] = True
    return int(label(m, structure=np.ones((3, 3)))[1])

def family_score(p, groups):
    values = [float(np.mean([iou(p, q) for q in groups[f]])) for f in sorted(groups)]
    return float(np.mean(values)), sum(v > .5 for v in values)

def main():
    protocol = json.loads((R/'boundary-consensus-protocol.json').read_text())
    assert sha(R/protocol['base_file']) == protocol['base_sha256']
    for f in FILES:
        assert sha(R/f) == EXPECTED[f], f
    base = read(R/'submission.csv')
    source = {f: read(R/f) for f in FILES}
    decisions, reasons, output = [], Counter(), []
    for image_no, stem in enumerate(sorted(base), 1):
        bp, base_lab, _ = labels(base[stem])
        groups = [defaultdict(list) for _ in bp]
        support = [defaultdict(list) for _ in bp]
        for f in sorted(FILES):
            rows = source[f].get(stem, [])
            if not rows:
                continue
            mat = np.asarray(CM.iou(rles(base[stem]), rles(rows), [0]*len(rows)))
            for b in range(len(bp)):
                s = int(mat[b].argmax())
                if mat[b, s] <= .5 or int(mat[:, s].argmax()) != b:
                    continue
                groups[b][FILES[f]].append(px(rows[s]))
                support[b][FILES[f]].append(float(mat[b,s]))
        proposals = list(bp)
        image_records = []
        for b, p in enumerate(bp):
            rec = dict(stem=stem, index=b+1, base_area=len(p), families=len(groups[b]))
            image_records.append(rec)
            if not len(p):
                rec['decision'] = 'legal_empty'; continue
            if len(groups[b]) < 3:
                rec['decision'] = 'insufficient_families'; continue
            if len(groups[b]) == 3 and min(max(s) for s in support[b].values()) < .6:
                rec['decision'] = 'three_family_agreement'; continue
            union = np.unique(np.concatenate([q for f in sorted(groups[b]) for q in groups[b][f]]+[p]))
            votes = np.zeros(len(union), np.float64)
            for f in sorted(groups[b]):
                for q in groups[b][f]:
                    votes[np.searchsorted(union, q)] += 1/len(groups[b][f])
            half = len(groups[b])/2
            old_here = np.isin(union, p, assume_unique=True)
            chosen = (votes > half+1e-9) | (np.abs(votes-half)<=1e-9) & old_here
            allowed = (base_lab[union] == 0) | (base_lab[union] == b+1)
            q = union[chosen & allowed]
            if np.array_equal(q, p):
                rec['decision'] = 'unchanged'; continue
            similarity = iou(p,q); ratio = len(q)/len(p)
            rec.update(candidate_area=len(q), base_iou=similarity, area_ratio=ratio)
            if similarity < .8 or not .8 <= ratio <= 1.25:
                rec['decision'] = 'geometry_guard'; continue
            if components(p) != components(q):
                rec['decision'] = 'topology_guard'; continue
            before, before_hits = family_score(p, groups[b])
            after, after_hits = family_score(q, groups[b])
            rec.update(family_mean_iou_before=before, family_mean_iou_after=after)
            if after <= before+1e-9 or after_hits < before_hits:
                rec['decision'] = 'agreement_guard'; continue
            proposals[b] = q
            rec['decision'] = 'changed'
        occupancy = np.zeros(SIZE,np.uint16)
        for q in proposals:
            occupancy[q] += 1
        for b, q in enumerate(proposals):
            if np.any(occupancy[q] > 1):
                assert image_records[b]['decision']=='changed'
                proposals[b] = bp[b]
                image_records[b]['decision'] = 'expansion_conflict_guard'
        check = np.zeros(SIZE,bool)
        for b,q in enumerate(proposals):
            assert not check[q].any(), (stem,b)
            check[q]=True
            row = enc(q) if image_records[b]['decision']=='changed' else base[stem][b]
            assert np.array_equal(px(row),q)
            output.append((f'{stem}_{b+1}',row))
        decisions.extend(image_records)
        reasons.update(r['decision'] for r in image_records)
        if image_no%30==0: print(image_no, dict(reasons),flush=True)
    frame = pd.DataFrame(output,columns=['filament_id','segmentation_rle'])
    out = R/'submission-boundary-consensus-v1.csv'
    frame.to_csv(out,index=False,lineterminator='\n')
    assert pd.read_csv(out,dtype=str,keep_default_na=False).equals(frame)
    assert frame.filament_id.is_unique
    assert frame.filament_id.tolist() == pd.read_csv(R/'submission.csv').filament_id.tolist()
    actual=read(out)
    assert set(actual)==set(base) and len(actual)==180
    assert len(frame)==1199
    for stem in actual:
        labels(actual[stem])
        for row in actual[stem]: assert enc(px(row))==row
    checksum = sha(out)
    past = [sha(R/f) for f in ('submission.csv','submission-complement-v1.csv','submission-updated-direct-v1.csv','original-039.csv')]
    assert checksum not in past
    changed=[r for r in decisions if r['decision']=='changed']
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),stage='candidate_geometry_validated',
        protocol=protocol, input_sha256={f:sha(R/f) for f in FILES}, source_sha256=sha(R/'build_boundary_consensus.py'),
        images=180,rows=len(frame),changed_instances=len(changed),changed_images=len(set(r['stem'] for r in changed)),
        decisions=dict(reasons), output_sha256=checksum, output_file=out.name,
        validation='same1199 IDs/count/order;180 images;no overlap;canonical RLE;CSV readback;topology and area guards',
        independent_held_validation=False, public_score=None, submission_status='not_submitted',
        minimum_base_iou=min(r['base_iou'] for r in changed),
        mean_family_agreement_gain=float(np.mean([r['family_mean_iou_after']-r['family_mean_iou_before'] for r in changed])),
        instances=decisions)
    (R/'boundary-consensus-v1-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('protocol','input_sha256','instances')}),flush=True)

if __name__=='__main__': main()
