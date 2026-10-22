"""CPU-only checks. No optimizer steps, GPU inference, or training processes."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
from tqdm import tqdm
from common import ROOT, BASE, read_json, write_json, sha, make_plan, validate_plan

def inventory():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used',
                                    '--format=csv,noheader,nounits'],text=True)
    values={}
    for line in output.splitlines():
        index,uuid,name,total,used=[s.strip() for s in line.split(',')]
        values[int(index)]=dict(uuid=uuid,name=name,memory_mib=int(total),used_mib=int(used))
    return values

def code_files():
    return [p for p in sorted(ROOT.rglob('*')) if p.is_file() and
            (p.parent==ROOT and p.suffix in ['.py','.sh','.json','.md','.diff'] or
             p.is_relative_to(ROOT/'source') or p.is_relative_to(ROOT/'frozen'))
            and '__pycache__' not in p.parts and p.name!='code_manifest.json']

def check_code():
    for relative,expected in read_json(ROOT/'code_manifest.json').items():
        if sha(ROOT/relative)!=expected: raise ValueError(f'Experiment code changed: {relative}')
    for p in code_files():
        if p.suffix=='.py': ast.parse(p.read_text(),filename=str(p))

def verify_inputs():
    manifest=read_json(ROOT/'input_manifest.json')
    for path,expected in tqdm(manifest.items(),desc='Verify original inputs',unit='file',mininterval=5):
        if sha(path)!=expected: raise ValueError(f'Original input changed: {path}')

def check(gpus, prepare=True):
    if Path(sys.prefix).name!='komap': raise RuntimeError(f'Conda komap required; got {sys.prefix}')
    validate_plan(read_json(ROOT/'plan.json')); check_code(); verify_inputs()
    devices=inventory()
    if len(gpus)!=4 or len(set(gpus))!=4 or any(g not in devices for g in gpus):
        raise ValueError(f'Expected four distinct physical GPUs: {gpus}')
    import torch
    versions={name:importlib.metadata.version(name) for name in
              ['torch','torchvision','segmentation-models-pytorch','timm','albumentations','numpy','tqdm']}
    result=dict(training_started=False,python=sys.executable,versions=versions,
                gpus={str(g):devices[g] for g in gpus},training_count=100,seeds=[3407,3506])
    if prepare:
        from engine import prepare_cache, baseline_cache
        cached=prepare_cache()
        for relative,expected in tqdm(cached['files'].items(),desc='Verify evaluation cache',unit='file',mininterval=5):
            if sha(baseline_cache()/relative)!=expected: raise ValueError(f'Evaluation cache changed: {relative}')
        result['baseline_image_miou']=cached['image_miou']
    return result

if __name__=='__main__':
    print(check([4,5,6,7]))
