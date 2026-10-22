"""Frozen two-stage hard-mask fusion. Never reads GT."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'refiner'))
from residual_model import Residual
from gray_input import prepare_gray

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def labels(p):
    # Route order: Al3Ni, Primary Si, Al, Eutectic Si.
    # Preserve original team class priority at exact ties.
    return np.array([2,0,1,3], dtype=np.uint8)[p[[2,0,1,3]].argmax(0)]

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--input', type=Path, required=True)
    a.add_argument('--release-labels', type=Path, required=True)
    a.add_argument('--expert-probabilities', type=Path)
    a.add_argument('--b4-archive', type=Path)
    a.add_argument('--hr1', type=Path)
    a.add_argument('--hr2', type=Path)
    a.add_argument('--refiner', type=Path, help='stage-2 residual refiner checkpoint (omit with --stage1-only)')
    a.add_argument('--stage1-only', action='store_true', help='recommended: R + E Al3Ni/Al override only, no refiner')
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--device', default='cpu')
    a.add_argument('--strict-valid', action='store_true')
    args = a.parse_args()
    assert args.expert_probabilities is not None or all([args.b4_archive,args.hr1,args.hr2])
    assert not args.output.exists(), 'Output must be a new directory'
    assert args.stage1_only or args.refiner is not None, '--refiner is required unless --stage1-only'
    torch.set_num_threads(2)
    model = None
    if not args.stage1_only:
        assert sha(args.refiner) == '932606f8a4764eb6125ccbf80f3fe6b4fb0fc176e884a156c012688aff830926'
        model = Residual().to(args.device).eval()
        model.load_state_dict(torch.load(args.refiner, map_location='cpu', weights_only=False)['state_dict'])
    import yaml
    palette = np.asarray(yaml.safe_load((ROOT/'Model349_exh14/configs/model349_lite.yaml').read_text())['palette'],dtype=np.uint8)
    expected = {r['name']:r['sha256'] for r in json.loads((ROOT/'config/input_images.json').read_text())}
    images = sorted(p for p in args.input.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg','.tif','.tiff','.bmp'])
    names = [p.stem.removesuffix('_image') for p in images]
    assert images and len(names)==len(set(names))
    if args.strict_valid: assert set(names)==set(expected)
    for kind in ['predicted_labels','predicted_masks']:(args.output/kind).mkdir(parents=True,exist_ok=False)
    archive = np.load(args.b4_archive,allow_pickle=False) if args.b4_archive else None
    rows = []
    try:
        for path,name in zip(images,names):
            if args.strict_valid: assert sha(path)==expected[name], name
            if args.expert_probabilities:
                p = np.load(args.expert_probabilities/f'{name}.npy').astype(np.float32)
            else:
                parts = [archive[name].astype(np.float32)]
                for folder in [args.hr1,args.hr2]:
                    with np.load(folder/f'{name}.npz',allow_pickle=False) as z: parts.append(z[name].astype(np.float32))
                p = (.5*parts[0]+.25*parts[1]+.25*parts[2])[[1,2,0,3]]
            gray = prepare_gray(np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255.)
            parent = np.asarray(Image.open(args.release_labels/f'{name}.png'))
            assert p.shape==(4,*gray.shape) and parent.shape==gray.shape and np.all(parent<4)
            assert np.isfinite(p).all() and np.max(np.abs(p.sum(0)-1))<.03
            expert = labels(p)
            stage1 = np.where(np.isin(expert,[0,2]),expert,parent)
            if model is None:
                pred = stage1.astype(np.uint8)
            else:
                with torch.no_grad():
                    value,delta = model(torch.from_numpy(np.log(np.maximum(p,1e-7))[None]).to(args.device),torch.from_numpy(gray[None,None]).to(args.device))
                assert torch.isfinite(value).all() and delta.abs().max()<=1
                refined = labels(value.cpu().numpy()[0])
                pred = np.where(refined==2,refined,stage1).astype(np.uint8)
            Image.fromarray(pred).save(args.output/f'predicted_labels/{name}.png')
            Image.fromarray(palette[pred]).save(args.output/f'predicted_masks/{name}_mask.png')
            rows.append({'name':name,'sha256':sha(path),'prediction_sha256':sha(args.output/f'predicted_labels/{name}.png')})
            print(name,flush=True)
    finally:
        if archive is not None: archive.close()
    (args.output/'completion.json').write_text(json.dumps({'images':rows,'gt_read':False,'stage1_only':args.stage1_only,'refiner_sha256':None if args.stage1_only else sha(args.refiner),'parent_outputs_supplied_externally':True},indent=2)+'\n')

if __name__=='__main__':main()
