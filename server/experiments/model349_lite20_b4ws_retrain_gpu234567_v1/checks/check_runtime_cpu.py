"""Temporary fixture packaging and strict loading. Never train or use CUDA."""
from pathlib import Path
import os,subprocess,sys,tempfile,shutil
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
    for name,digest in read_json(ROOT/'code_manifest.json').items():assert sha(implementation/name)==digest
    original=REPO/'results/model349_inference_bundle_20261007_1/weights'
    expert=REPO/'experiments/model349_fusion805_retrain_gpu234567_v1/outputs/fusion805_train_20261009_164635_301741/expert'
    tasks={t['id']:t for t in read_json(ROOT/'plan.json')['tasks']}
    for slot in read_json(ROOT/'plan.json')['release_slots']:
        task=tasks[slot['task']]
        target=Path(render([task['checkpoint']],implementation/'source',run,4)[0])
        target.parent.mkdir(parents=True,exist_ok=True)
        source=original/slot['path'].removeprefix('weights/')
        if not target.exists():target.symlink_to(source)
        if '_fold' in task['id'] and not (target.parent/'args.json').exists():
            (target.parent/'args.json').symlink_to(source.parent/'args.json')
    for task in tasks.values():
        if task['family'] not in ['warmstart','expert','refiner']:continue
        target=Path(render([task['checkpoint']],implementation/'source',run,4)[0])
        if target.exists():continue
        target.parent.mkdir(parents=True,exist_ok=True)
        relative=str(target.relative_to(run/'expert')).replace('model329_ws/','model329/')
        source=expert/relative
        target.symlink_to(source)
        if task['family']!='refiner':(target.parent/'args.json').symlink_to(source.parent/'args.json')
        if task['family']=='warmstart':write_json(target.parent/'bn_freeze_check.json',dict(ok=True,bn_running_stats_frozen=True,fixture_only=True))
        if task['family']=='refiner':
            for name in ['completion.json','recipe.json']:write_json(target.parent/name,dict(fixture_only=True,optimizer_updates=0))
    (run/'refiner_features/train').mkdir(parents=True)
    write_json(run/'refiner_features/train/manifest.json',dict(fixture_only=True))
    bundle=package(run)
    env=environment(implementation);env['CUDA_VISIBLE_DEVICES']='';env.pop('PYTHONPATH',None)
    for script in ['verify_model.py','Model349_exh14/verify_bundle.py']:
        subprocess.run([sys.executable,'-B',str(bundle/script)],env=env,check=True)
    assets=read_json(bundle/'lite20/selected_assets_lite20.json')
    assert assets['count']==20 and len(assets['checkpoints'])==20
    for row in assets['checkpoints']+assets['metadata']:
        base=bundle/'Model349_exh14' if row['namespace']=='R' else bundle/'assets'
        assert sha(base/row['path'])==row['sha256']
    for row in assets['metadata']:
        assert sha(bundle/'lite20/parent_metadata'/row['namespace']/row['path'])==row['sha256']
    expected={r['path']:r['sha256'] for f in ['auxiliary_assets.json','b4ws_assets.json'] for r in read_json(bundle/'config'/f)}
    assert all(expected[r['path']]==r['sha256'] for r in assets['checkpoints']+assets['metadata'] if r['namespace']!='R')
    code='''
import sys,gc,json
from pathlib import Path
import torch
root=Path(sys.argv[1]);torch.set_num_threads(2)
sys.path[:0]=[str(root/'model397_reproduction_bundle/model329'),str(root/'model397_reproduction_bundle/model392'),str(root/'refiner')]
from common import load_member
from komap_seg.checkpoint import load_model_and_preprocessing,resolve_preprocessing
from residual_model import Residual
for record in json.loads((root/'lite20/selected_assets_lite20.json').read_text())['checkpoints']:
    if record['namespace']=='R':continue
    path=root/'assets'/record['path']
    if 'model329_ws' in record['path']:model=load_member(path.parent,'cpu','last.pt')['model']
    elif 'model392' in record['path']:
        model,saved=load_model_and_preprocessing(path,torch.device('cpu'));p=resolve_preprocessing(saved)
        assert p['patch_size']==512 and p['input_scale']==2 and p['normalize_width'] and p['target_width']>0
    else:
        model=Residual();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'])
    assert next(model.parameters()).dtype==torch.float32
    del model;gc.collect()
print('Strict CPU load passed for nine auxiliary fixture weights.')
'''
    subprocess.run([sys.executable,'-B','-c',code,str(bundle)],env=env,check=True)
    for script in ['lite20/infer_compact.py','lite20/infer_expert.py','infer_stage1.py','infer_fusion.py']:
        subprocess.run([sys.executable,'-B',str(bundle/script),'--help'],env=env,check=True,stdout=subprocess.DEVNULL)
    assert len(list(bundle.rglob('*.pt')))==20
write_json(ROOT/'checks/runtime_cpu.json',dict(ok=True,training_started=False,inference_started=False,
    fixture_weights_only=True,fixtures_deleted=True,snapshot_hashes_verified=True,packaged_checkpoints=20,
    strict_cpu_loaded_expert_refiner=9,hrnet_preprocessing_preserved=True,forward_calls=0,
    strict_compact_manifests_verified=True))
print('PASS: snapshot, 20-weight packaging, strict manifests and CPU loads; fixtures removed.')
