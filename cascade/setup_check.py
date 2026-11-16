"""Fail-fast checks shared by the short preflight and full training pipeline."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import sys, json, time, hashlib, platform, subprocess, importlib.util
from pathlib import Path

WORK = Path('/kaggle/working')
TEMP = Path('/kaggle/temp/pq_cascade')
TEMP.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))

if importlib.util.find_spec('segmentation_models_pytorch') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                    'segmentation-models-pytorch==0.5.0'], check=True)
if importlib.util.find_spec('pycocotools') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                    'pycocotools==2.0.11'], check=True)

import numpy as np, cv2, torch
import segmentation_models_pytorch as smp
from solarpq import data as D, features as F, geometry as G, model as B, metrics as Mx
from solarpq.config import Config
import refiner_core as R
torch.set_num_threads(2)
cv2.setNumThreads(1)
assert torch.cuda.is_available(), 'GPU accelerator required'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_public(path):
    # Author's val metrics contain NumPy scalar metadata. Permit only these
    # standard scalar/dtype constructors, keep the restricted tensor loader.
    scalar = np._core.multiarray.scalar
    permitted = {'numpy._core.multiarray.scalar', 'numpy.core.multiarray.scalar', 'numpy.dtype'}
    unexpected = set(torch.serialization.get_unsafe_globals_in_checkpoint(path)) - permitted
    assert not unexpected, f'unexpected checkpoint globals: {unexpected}'
    with torch.serialization.safe_globals([scalar, np.dtype,
            type(np.dtype(np.float64)), type(np.dtype(np.float32)), type(np.dtype(np.int64))]):
        return torch.load(path, map_location='cpu', weights_only=True)


roots = [p.parent.parent for p in Path('/kaggle/input').rglob('fold0.pt')
         if p.parent.name == 'ckpt' and all((p.parent / f'fold{f}.pt').exists() for f in range(4))]
assert len(roots) == 1, f'expected exactly one public four-fold source, found {roots}'
PUBLIC = roots[0]
checkpoints = [PUBLIC / 'ckpt' / f'fold{f}.pt' for f in range(4)]
cfgs = []
hashes = {}
for f, path in enumerate(checkpoints):
    ck = load_public(path)
    assert set(['state', 'cfg', 'fold']).issubset(ck)
    assert int(ck['fold']) == f
    cfgs.append(Config.from_json(ck['cfg']))
    model = B.build_model(cfgs[-1])
    model.load_state_dict(ck['state'], strict=True)
    assert all(torch.isfinite(t).all() for t in ck['state'].values())
    hashes[path.name] = sha(path)
    del ck, model
assert len({c.to_json() for c in cfgs}) == 1
cfg = cfgs[0]
assert (cfg.work_res, cfg.full_res, cfg.in_channels, cfg.aux_spine, cfg.n_folds, cfg.seed) == (1024, 2048, 3, True, 4, 42)
cfg.t_lo, cfg.t_hi, cfg.close_radius, cfg.min_area = .2, .4, 3, 40
ann_paths = list(Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
assert len(ann_paths) == 1
ANN = ann_paths[0]
TRAIN_DIR = ANN.parent / 'train_images'
TEST_DIR = ANN.parent.parent / 'test' / 'test_images'
reg = D.load_registry(str(ANN))
stems = reg.stems()
test_files = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in ['.jpeg', '.jpg', '.png'])
test_stems = [p.stem for p in test_files]
assert len(stems) == 707 and len(test_stems) == 180
folds = D.make_folds(reg, 4, 42, stems)
fold_of = np.array([folds[s] for s in stems])
original_fold_of = np.array(json.loads((PUBLIC / 'folds.json').read_text())['fold_of'])
assert np.array_equal(fold_of, original_fold_of), 'public fold assignment mismatch'
json.dump({'stems': stems, 'fold_of': fold_of.tolist(), 'records': {s: reg.obs[s].records for s in stems}},
          open(WORK / 'cascade_folds.json', 'w'), indent=2)
json.dump(hashes, open(WORK / 'public_checkpoint_hashes.json', 'w'), indent=2)
print('[INPUT VERIFIED]', json.dumps({'source': str(PUBLIC), 'train': len(stems), 'test': len(test_stems),
      'folds': np.bincount(fold_of).tolist(), 'weights_only': True, 'hashes': hashes}), flush=True)


def two_head_prob(model, chans, disk, device, d4=True):
    from solarpq.infer import _apply, _invert, _D4
    valid = (G.radius_map((1024, 1024), *disk) < cfg.disk_frac).astype(np.float32)
    x = torch.from_numpy(np.asarray(chans, np.float32).copy())[None].to(device)
    v = torch.from_numpy(valid)[None, None].to(device)
    x = x * v
    acc = torch.zeros(1, 2, 1024, 1024, device=device)
    transforms = _D4 if d4 else [(0, False)]
    with torch.no_grad():
        for k, flip in transforms:
            with torch.autocast('cuda', dtype=torch.float16):
                y = model(_apply(x, k, flip))
            assert y.shape == (1, 2, 1024, 1024) and torch.isfinite(y).all()
            acc += _invert(y.float().sigmoid(), k, flip)
    out = (acc / len(transforms) * v)[0].cpu().numpy()
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
    return out


def run_preflight():
    start = time.time()
    geo = R.geometry_check()
    img = cv2.imread(D.find_image(stems[0], [str(TRAIN_DIR)]), 0)
    ch = F.compute_channels(img, cfg)
    base = []
    for f, path in enumerate(checkpoints):
        dev = torch.device(f'cuda:{f % torch.cuda.device_count()}')
        ck = load_public(path)
        model = B.build_model(cfg).to(dev)
        model.load_state_dict(ck['state'], strict=True); model.eval()
        t = time.time()
        p = two_head_prob(model, ch.chans, ch.disk, dev, d4=False)
        base.append({'fold': f, 'shape': list(p.shape), 'seconds': time.time()-t, 'max_fg': float(p[0].max())})
        del model, ck
        torch.cuda.empty_cache()
    refiner_checks = [R.gpu_check(torch.device(f'cuda:{g}'), pretrained=True) for g in range(torch.cuda.device_count())]
    # Strict >0.5 boundary, native RLE round-trip, and separate annotator pooling.
    a = np.zeros((16, 16), np.uint8); a[2:10, 2:10] = 1
    rle = Mx.encode(a)
    from pycocotools import mask as CM
    assert np.array_equal(CM.decode(rle), a)
    assert Mx.score_matrix(np.array([[.5]]), 1, 1) == (0., 0, 1, 1)
    assert Mx.score_matrix(np.array([[.50001]]), 1, 1)[1:] == (1, 0, 0)
    frozen = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout
    (WORK / 'requirements_frozen.txt').write_text(frozen)
    result = {'status': 'preflight_passed', 'seconds': time.time()-start,
              'python': platform.python_version(), 'torch': torch.__version__, 'smp': smp.__version__,
              'gpus': [torch.cuda.get_device_name(g) for g in range(torch.cuda.device_count())],
              'geometry': geo, 'base': base, 'refiner': refiner_checks, 'weights': hashes,
              'folds_exact_match': True, 'metric_checks': True, 'training_started': False}
    json.dump(result, open(WORK / 'cascade_preflight.json', 'w'), indent=2)
    print('[CASCADE PREFLIGHT]', json.dumps(result, indent=2), flush=True)
    return result
