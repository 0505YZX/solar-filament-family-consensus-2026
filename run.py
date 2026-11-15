"""Replay the attributed prediction-fusion pipeline; no model training."""
from pathlib import Path
import argparse, shutil, subprocess, sys, hashlib, json

EXPECTED={
 'submission.csv':'6d7c830c29c31a2ad066c9bebb1684bc511dcc1c53bb7a168b5c0fbe1a91f33b',
 'submission-boundary-consensus-v1.csv':'c435252d6b59a67bf249e945383323549694068b446bd6ef53f0a2cb52fa6c25',
 'submission-boundary-updated-source-v1.csv':'a23998a106007f9bdd7d4e896786d3fa3813e81fa35b8c25d2b96d2c7bce58f2',
 'submission-new-direct-recall-v1.csv':'9546cb5e40df266fe458e94fcf1ea981268ce99926e9e8c36cb00dc7e52f07d7',
 'submission-solfire-boundary-v1.csv':'68b26327bd9770d621753144ec29220af0e2b40094ee5756953af41409c94a09'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='replay_output')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    work=Path(args.output).resolve()
    if work.exists(): raise FileExistsError('Use a new output directory: '+str(work))
    shutil.copytree(root/'pipeline',work)
    for k,v in json.loads((root/'input_manifest.json').read_text()).items():
        assert hashlib.sha256((root/k).read_bytes()).hexdigest()==v,k
    driver=r"""
from build_updated_direct import vote,sha
from audit_complementarity import R
from build_boundary_consensus import main as boundary,FILES
from build_new_direct_recall import main as recall
from build_solfire_boundary import main as solfire
import json
expected=json.loads((R/'expected_outputs.json').read_text())
def check(name):
    assert sha(R/name)==expected[name],name
    print('SHA256 MATCH',name,sha(R/name),flush=True)
base,_=vote('hdjojo-v1-367892109.csv')
base.to_csv(R/'submission.csv',index=False,lineterminator='\n')
check('submission.csv')
boundary()
check('submission-boundary-consensus-v1.csv')
files={f:family for f,family in FILES.items() if family!='direct_yolo'}
files['hdjojo-latest-369397665.csv']='direct_yolo'
boundary(files=files,protocol_name='boundary-updated-source-protocol.json',output_name='submission-boundary-updated-source-v1.csv',result_name='boundary-updated-source-v1-result.json')
check('submission-boundary-updated-source-v1.csv')
recall()
check('submission-new-direct-recall-v1.csv')
solfire()
check('submission-solfire-boundary-v1.csv')
"""
    (work/'expected_outputs.json').write_text(json.dumps(EXPECTED,indent=2))
    (work/'replay_driver.py').write_text(driver)
    subprocess.run([sys.executable,str(work/'replay_driver.py')],cwd=work,check=True)
    shutil.copy2(work/'submission-solfire-boundary-v1.csv',work/'final_submission.csv')
    subprocess.run([sys.executable,str(root/'validate.py'),str(work/'final_submission.csv')],check=True)
    print('Fusion replay complete. This does not reconstruct upstream models from weights.')

if __name__=='__main__': main()
