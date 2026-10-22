"""Check snapshot and 27-weight packaging using explicitly synthetic fixtures; no training."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import ROOT,REPO,read_json,write_json,sha,environment
from run import snapshot
from scheduler import render
from package_fusion import package
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(2)

with tempfile.TemporaryDirectory(prefix='runtime_fixture_',dir=ROOT/'checks') as directory:
    run=Path(directory);implementation=snapshot(run)
    for name,digest in read_json(ROOT/'code_manifest.json').items():
        assert sha(implementation/name)==digest
    assert (implementation/'expert_splits/model329_cv/train_fold1/images').is_dir()
    original=REPO/'results/model349_inference_bundle_20261007_1/weights'
    hr=run/'synthetic_hr.pt'
    payload=torch.load(original/'model336_hrnet_best/last.pt',map_location='cpu',weights_only=False)
    payload['preprocessing_config']=dict(patch_size=512,input_scale=2.0,normalize_width=True,target_width=1230)
    torch.save(payload,hr);del payload
    sys.path.insert(0,str(ROOT/'fusion_source/refiner'))
    from residual_model import Residual
    for task in read_json(ROOT/'plan.json')['tasks']:
        if task['role']!='train':continue
        target=Path(render([task['checkpoint']],implementation/'source',run,4)[0])
        target.parent.mkdir(parents=True,exist_ok=True)
        if task['family']=='release':
            source=original/target.relative_to(run/'outputs')
            target.symlink_to(source)
            if '_fold' in task['id']:(target.parent/'args.json').symlink_to(source.parent/'args.json')
        elif task['family']=='expert':
            if 'model329' in str(target):
                source=original/'model329_fullstats_width4000_uniform_group_oof/fold1/last.pt'
                target.symlink_to(source)
                (target.parent/'args.json').symlink_to(source.parent/'args.json')
            else:
                target.symlink_to(hr)
                write_json(target.parent/'args.json',dict(synthetic_fixture=True))
        else:
            torch.save(dict(state_dict=Residual().state_dict()),target)
            for name in ['completion.json','recipe.json']:write_json(target.parent/name,dict(synthetic_fixture=True,optimizer_updates=0))
    (run/'refiner_features/train').mkdir(parents=True)
    write_json(run/'refiner_features/train/manifest.json',dict(synthetic_fixture=True))
    bundle=package(run)
    env=environment(implementation);env['CUDA_VISIBLE_DEVICES']='';env.pop('PYTHONPATH',None)
    subprocess.run([sys.executable,'-B',str(bundle/'verify_model.py')],env=env,check=True)
    subprocess.run([sys.executable,'-B',str(bundle/'Model349_exh14/verify_bundle.py')],env=env,check=True)
    code='''
import sys,gc,json
from pathlib import Path
import torch
root=Path(sys.argv[1]);torch.set_num_threads(2)
sys.path[:0]=[str(root/'model397_reproduction_bundle/model329'),str(root/'model397_reproduction_bundle/model392'),str(root/'refiner')]
from common import load_member
from komap_seg.checkpoint import load_model_and_preprocessing,resolve_preprocessing
from residual_model import Residual
for record in json.loads((root/'config/auxiliary_assets.json').read_text()):
    path=root/'assets'/record['path']
    if 'model329' in record['path']:model=load_member(path.parent,'cpu','last.pt')['model']
    elif 'model392' in record['path']:
        model,saved=load_model_and_preprocessing(path,torch.device('cpu'))
        prep=resolve_preprocessing(saved)
        assert prep==dict(patch_size=512,input_scale=2.0,normalize_width=True,target_width=1230)
    else:
        model=Residual();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'])
    assert next(model.parameters()).dtype==torch.float32
    del model;gc.collect()
print('Strict CPU load passed for 12 expert fixtures + one untrained refiner fixture.')
'''
    subprocess.run([sys.executable,'-B','-c',code,str(bundle)],env=env,check=True)
    for script in ['infer_model401.py','infer_stage1.py','infer_fusion.py']:
        subprocess.run([sys.executable,'-B',str(bundle/script),'--help'],env=env,check=True,stdout=subprocess.DEVNULL)
    assert len(list(bundle.rglob('*.pt')))==27
write_json(ROOT/'checks/runtime_cpu.json',dict(ok=True,training_started=False,inference_started=False,
    fixture_weights_only=True,snapshot_hashes_verified=True,packaged_checkpoints=27,
    strict_cpu_loaded_expert_refiner=13,hrnet_preprocessing_preserved=True,forward_calls=0))
print('PASS: immutable snapshot, 27-model packaging and hashes, E/F loading; fixture weights only.')
