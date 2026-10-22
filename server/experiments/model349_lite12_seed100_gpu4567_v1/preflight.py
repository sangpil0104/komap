"""CPU-only preflight; no training, CUDA forward, or TENT adaptation."""
from __future__ import annotations
import ast
import importlib.metadata
from pathlib import Path
import subprocess
import sys
from tqdm import tqdm
from common import ROOT, BASE, read_json, sha, validate_plan

def inventory():
    output = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used',
                                      '--format=csv,noheader,nounits'],text=True)
    result = {}
    for line in output.splitlines():
        index,uuid,name,total,used = [x.strip() for x in line.split(',')]
        result[int(index)] = dict(uuid=uuid,name=name,memory_mib=int(total),used_mib=int(used))
    return result

def check(gpus):
    if Path(sys.prefix).name!='komap': raise RuntimeError(f'Conda komap required: {sys.prefix}')
    validate_plan(read_json(ROOT/'plan.json'))
    for relative,expected in read_json(ROOT/'code_manifest.json').items():
        path = ROOT/relative
        if sha(path)!=expected: raise ValueError(f'Experiment source changed: {relative}')
        if path.suffix=='.py': ast.parse(path.read_text(),filename=str(path))
    for path,expected in tqdm(read_json(ROOT/'input_manifest.json').items(),desc='Verify immutable inputs',unit='file',mininterval=5):
        if sha(path)!=expected: raise ValueError(f'Input changed: {path}')
    weights = read_json(ROOT/'local_weights.json')['checkpoints']
    if len(weights)!=12: raise ValueError('lite12 requires twelve checkpoints')
    for row in tqdm(weights,desc='Verify FP16 weights',unit='file'):
        if sha(ROOT/row['relative'])!=row['sha256']: raise ValueError(f'FP16 input changed: {row["relative"]}')
    data = read_json(ROOT/'data_manifest.json')
    if len(data['images'])!=20 or data['train_count']!=70: raise ValueError('Expected train70 / valid20')
    devices = inventory()
    if gpus!=[4,5,6,7]: raise ValueError('This experiment is fixed to physical GPUs 4,5,6,7')
    if any(g not in devices for g in gpus): raise ValueError('Required physical GPU missing')
    versions = {name:importlib.metadata.version(name) for name in
                ['torch','torchvision','segmentation-models-pytorch','timm','albumentations','numpy','tqdm']}
    import torch
    import segmentation_models_pytorch
    import cv2
    from engine import frozen
    from komap_seg.checkpoint import load_model
    import b4_zcdf
    return dict(training_started=False,gpu_inference_started=False,python=sys.executable,versions=versions,
                gpus={str(g):devices[g] for g in gpus},training_count=100,candidate_count=84,seeds=[3407,3506],
                baseline='local preserved weights converted to FP16; measured only after user starts',
                upstream_weights_supplied=False,high_route_counts=data['high_route_counts'])

def verify_completed(run):
    """Reject changed/missing artifacts on resume rather than silently skipping them."""
    records = [read_json(p) for p in (run/'completed').glob('*.json')]
    expected = {t['id']:t for t in read_json(run/'plan.json')['tasks']}
    for record in tqdm(records,desc='Verify completed jobs',unit='job',mininterval=5):
        if record['id'] not in expected: raise ValueError('Unknown completed task')
        for field,value in expected[record['id']].items():
            if record.get(field)!=value: raise ValueError(f'Completed task differs: {record["id"]}/{field}')
        for path,digest in record['artifacts'].items():
            if sha(path)!=digest: raise ValueError(f'Completed artifact changed: {path}')
