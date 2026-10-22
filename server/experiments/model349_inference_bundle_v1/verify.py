"""Verify bundle hashes, imports and optionally strict checkpoint loading (CPU)."""
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from tqdm import tqdm


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--load-models', action='store_true', help='Strict-load all 23 weights on CPU')
    a = p.parse_args()
    for line in tqdm((ROOT / 'SHA256SUMS.txt').read_text().splitlines(), desc='Bundle SHA256', unit='file'):
        expected, relative = line.split('  ', 1)
        assert digest(ROOT / relative) == expected, relative
    manifest = json.loads((ROOT / 'provenance/source_manifest.json').read_text())
    for row in tqdm(manifest, desc='Import inference modules', unit='module'):
        importlib.import_module(row['module'])
    if a.load_models:
        import torch
        import b4_gray, b4_zcdf
        import calibrate_al
        from model183_metric_calibration import load_aspp, load_rep
        from model253_highres_scale_finetune import Stride1NativeStemASPP
        from komap_seg.checkpoint import load_model
        torch.set_num_threads(2)
        rows = json.loads((ROOT / 'provenance/weights_manifest.json').read_text())
        for row in tqdm(rows, desc='Strict-load all checkpoints', unit='checkpoint'):
            path = ROOT / row['path']
            name = str(path.relative_to(ROOT / 'weights'))
            if '/fold' in name:
                module = b4_gray if name.startswith('model322') else b4_zcdf
                model = module.load_member(str(path.parent), 'cpu')['model']
            elif name.startswith(('model336', 'model347')):
                model = load_model(path, torch.device('cpu'))
            elif name.startswith('model269'):
                model = Stride1NativeStemASPP()
                model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
            elif name.startswith(('model97_', 'model98_')):
                model = calibrate_al.load_model(str(path), torch.device('cpu'))
            elif name.startswith('model181'):
                model = load_rep(path, torch.device('cpu'))
            else:
                model = load_aspp(path, torch.device('cpu'))
            del model
            gc.collect()
    print('PASS: bundle hashes/imports' + (' and all 23 checkpoint loads' if a.load_models else ''))


if __name__ == '__main__':
    main()
