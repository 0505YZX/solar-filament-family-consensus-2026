"""Reassemble any repository-stored weight parts and verify every required byte."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(2**20),b''): h.update(chunk)
    return h.hexdigest()

def verify(root=ROOT/'weights'):
    root=Path(root)
    manifest=json.loads((ROOT/'model_manifest.json').read_text())
    assert manifest['complete'],'Model package is not complete'
    for entry in manifest['files']:
        path=root/entry['path']
        if not path.exists() and entry.get('parts'):
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix(path.suffix+'.tmp')
            with temporary.open('wb') as target:
                for part in entry['parts']:
                    source=root/part['path']
                    assert digest(source)==part['sha256']
                    with source.open('rb') as handle:
                        for chunk in iter(lambda:handle.read(2**20),b''): target.write(chunk)
            assert digest(temporary)==entry['sha256']
            temporary.replace(path)
        assert path.is_file() and digest(path)==entry['sha256'],f'Missing or mismatched model file: {entry["path"]}'
    return manifest

if __name__=='__main__':
    result=verify(); print('Verified required model files:',len(result['files']))
