"""Snapshot and package integration with known checkpoint fixtures; never train."""
import gc
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import ROOT, REPO, read_json, write_json, sha, environment
from run import snapshot
from scheduler import render
from package_model import package

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(2)
from tqdm import tqdm

with tempfile.TemporaryDirectory(prefix='exh14_runtime_') as directory:
    run=Path(directory)/'run';run.mkdir()
    implementation=snapshot(run)
    for name,digest in read_json(ROOT/'code_manifest.json').items():
        assert sha(implementation/name)==digest
    assert (implementation/'folds/train_fold1/images').is_dir()
    # Stand in for training outputs with existing original weights, only inside this temporary test.
    original=REPO/'results/model349_inference_bundle_20261007_1'
    for task in read_json(ROOT/'plan.json')['tasks']:
        target=Path(render([task['checkpoint']],implementation/'source',run,4)[0])
        relative=target.relative_to(run/'outputs')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(original/'weights'/relative)
        if '_fold' in task['id']:
            (target.parent/'args.json').symlink_to((original/'weights'/relative).parent/'args.json')
    bundle=package(run)
    env=environment(implementation);env['CUDA_VISIBLE_DEVICES']='';env.pop('PYTHONPATH',None)
    subprocess.run([sys.executable,'-B',str(bundle/'verify_bundle.py')],env=env,check=True)
    for path in [bundle/'infer_lite.py',*bundle.joinpath('src').rglob('*.py'),*bundle.joinpath('configs').glob('*')]:
        assert sha(path)==sha(ROOT/'frozen'/path.relative_to(bundle))
    manifest=read_json(bundle/'checkpoint_manifest.json')
    assert len(manifest['checkpoints'])==14 and not (bundle/'results').exists()
    sys.path.insert(0,str(bundle/'src'))
    from komap_seg.checkpoint import load_model
    from b4_gray import load_member as load_gray
    from b4_zcdf import load_member as load_zcdf
    from tent_style_search import load_aspp
    from model253_highres_scale_finetune import Stride1NativeStemASPP
    for row in tqdm(manifest['checkpoints'],desc='Strict CPU load of packaged weights',unit='model'):
        path=bundle/row['path']
        if 'fold' in path.parent.name:
            loader=load_gray if 'model322' in str(path) else load_zcdf
            model=loader(str(path.parent),'cpu')['model']
        elif 'hrnet' in str(path):
            model=load_model(path,torch.device('cpu'))
        elif 'model269' in str(path):
            model=Stride1NativeStemASPP()
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        else:
            model=load_aspp(path,torch.device('cpu'))
        assert next(model.parameters()).dtype==torch.float32
        del model;gc.collect()
    subprocess.run([sys.executable,'-B',str(bundle/'infer_lite.py'),'--help'],env=env,check=True,stdout=subprocess.DEVNULL)
write_json(ROOT/'checks/runtime_cpu.json',dict(ok=True,training_started=False,inference_started=False,
    fixture_weights_only=True,snapshot_hashes_verified=True,packaged_checkpoints=14,
    strict_cpu_loaded=14,original_inference_code_and_configs_unchanged=True,forward_calls=0))
print('PASS: runtime source snapshot, full FP32 bundle export, new hashes, and strict CPU loading of 14 fixture models.')
