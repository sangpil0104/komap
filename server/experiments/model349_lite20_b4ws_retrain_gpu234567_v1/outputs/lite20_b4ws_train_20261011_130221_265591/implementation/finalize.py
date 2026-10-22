"""Evaluate freshly trained R/E/F and save both fixed fusion stages."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time
import numpy as np
from tqdm import tqdm
from experiment import ROOT,read_json,write_json
from package_fusion import package
from features import blend


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpus',required=True)
    a=p.parse_args();run=a.run.resolve()
    status=read_json(run/'status.json')
    plan=read_json(run/'plan.json')
    if status['phase']!='complete' or set(status['completed'])!={t['id'] for t in plan['tasks']}:
        raise ValueError('All 51 stages must finish before final export.')
    started=time.monotonic();model=package(run)
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    def call(script,*arguments):
        command=[sys.executable,'-B','-u',str(model/script),*map(str,arguments)]
        with (run/'finalization_commands.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps(command)+'\n')
        subprocess.run(command,env=env,cwd=model,check=True)
    call('verify_model.py')
    call('Model349_exh14/infer_lite.py','--input',ROOT/'data/valid/images','--gt',ROOT/'data/valid/masks',
        '--output',run/'release_inference','--gpus',a.gpus,'--keep-probabilities',
        '--config',model/'lite20/selected_R_lite20.yaml')
    arrays=[np.load(run/f'expert_valid/{family}/probabilities.npz',allow_pickle=False) for family in ['b4','hr1','hr2']]
    expert=run/'expert_probabilities';expert.mkdir()
    names=[r['name'] for r in read_json(run/'release_inference/input_manifest.json')['images']]
    try:
        if len(names)!=20 or any(set(z.files)!=set(names) for z in arrays):raise ValueError('Expert/release valid20 mismatch.')
        for name in tqdm(names,desc='Blend expert families',unit='image'):
            np.save(expert/f'{name}.npy',blend(*(z[name] for z in arrays)))
    finally:
        for z in arrays:z.close()
    call('infer_stage1.py','--release',run/'release_inference','--expert',expert,'--output',run/'stage1')
    call('infer_fusion.py','--input',ROOT/'data/valid/images','--release-labels',run/'release_inference/predicted_labels',
        '--expert-probabilities',expert,'--refiner',model/'assets/refiner/epoch8.pt','--output',run/'stage2','--device','cpu')
    for stage in ['stage1','stage2']:
        call('score_masks.py','--labels',run/stage/'predicted_labels','--gt',ROOT/'data/valid/masks','--output',run/stage/'metrics.json')
    # Same fixed 20 weights for image-only test10; no test score without ground truth.
    call('lite20/infer_compact.py','--config',model/'lite20/selected_config_lite20.json',
         '--parent-package',model,'--assets',model/'assets','--input',ROOT/'data/test/images',
         '--output',run/'test_inference','--gpus',a.gpus)
    from export_results import export
    result=export(run)
    write_json(run/'finished.json',dict(result=str(result),trained=35,deployed_weights=20,finalization_seconds=time.monotonic()-started))
    print(f'RESULT: {result}',flush=True)


if __name__=='__main__':main()
