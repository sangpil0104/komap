"""Export exactly the requested 23 trained weights with fresh strict manifests."""
from pathlib import Path
import shutil
from tqdm import tqdm
from experiment import read_json,write_json,sha
from package_model import package as package_release
from scheduler import render


def package(run):
    import torch
    torch.set_num_threads(2)
    run=Path(run);implementation=run/'implementation'
    release=package_release(run);dest=run/'retrained_model'
    shutil.copytree(implementation/'fusion_source',dest,
                    ignore=shutil.ignore_patterns('Model349_exh14','results','README.md','__pycache__','*.pyc'))
    shutil.copytree(release,dest/'Model349_exh14')
    (dest/'lite23').mkdir()
    for name in ['infer_compact.py','infer_expert.py','selected_config_b4ws.json',
                 'selected_R.yaml','selected_assets_b4ws.json']:
        shutil.copy2(implementation/'compact_source'/name,dest/'lite23'/name)
    plan=read_json(run/'plan.json');tasks={t['id']:t for t in plan['tasks']}
    selected=[tasks[f'train_b4ws_fold{f}'] for f in range(1,5)]
    selected += [tasks[f'train_expert_hr{s}_fold{f}'] for s,folds in plan['expert_hr_folds'].items() for f in folds]
    selected += [tasks['train_refiner']]
    records=[]
    for task in tqdm(selected,desc='Package 9 lite23 auxiliary weights',unit='model'):
        checkpoint=Path(render([task['checkpoint']],implementation/'source',run,4)[0])
        relative=str(checkpoint.relative_to(run/'expert'))
        target=dest/'assets'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state=payload.get('state_dict',payload.get('model',payload))
        floating=[x for x in state.values() if torch.is_tensor(x) and torch.is_floating_point(x)]
        if not floating or not all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in floating):
            raise ValueError(f'Invalid FP32 checkpoint: {checkpoint}')
        shutil.copy2(checkpoint,target)
        if task['family']!='refiner':shutil.copy2(checkpoint.parent/'args.json',target.parent/'args.json')
        if task['family']=='warmstart':
            receipt=read_json(checkpoint.parent/'bn_freeze_check.json')
            if not receipt['ok'] or not receipt['bn_running_stats_frozen']:raise ValueError('BN freeze failed')
            shutil.copy2(checkpoint.parent/'bn_freeze_check.json',target.parent/'bn_freeze_check.json')
        records.append(dict(path=relative,sha256=sha(target),bytes=target.stat().st_size,
                            source_checkpoint=str(checkpoint),task=task['id'],seed=task['seed']))
        del payload,state,floating
    if len(records)!=9:raise ValueError('Expected four B4ws, four HRNet and one F')
    write_json(dest/'config/auxiliary_assets.json',[r for r in records if not r['path'].startswith('model329_ws/')])
    warmstart=[r for r in records if r['path'].startswith('model329_ws/')]
    warmstart += [dict(path=f'model329_ws/fold{f}/args.json',sha256=sha(dest/f'assets/model329_ws/fold{f}/args.json')) for f in range(1,5)]
    write_json(dest/'config/b4ws_assets.json',warmstart)
    # The strict compact runner also validates args and calibration: replace its published
    # manifests and embedded metadata with this run's actual assets, retaining originals.
    compact=dest/'lite23';selected_assets=read_json(compact/'selected_assets_b4ws.json')
    shutil.copy2(compact/'selected_assets_b4ws.json',compact/'upstream_selected_assets_b4ws.json')
    for section in ['checkpoints','metadata']:
        for row in selected_assets[section]:
            base=dest/'Model349_exh14' if row['namespace']=='R' else dest/'assets'
            path=base/row['path']
            row.update(sha256=sha(path),bytes=path.stat().st_size)
            if section=='metadata':
                copy=compact/'parent_metadata'/row['namespace']/row['path']
                copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,copy)
    selected_assets.update(count=23,bytes=sum(r['bytes'] for r in selected_assets['checkpoints']),
                           source='New local S=0 retraining; published hashes retained in upstream manifest')
    write_json(compact/'selected_assets_b4ws.json',selected_assets)
    for f in ['completion.json','recipe.json']:
        shutil.copy2(run/'expert/refiner'/f,dest/'refiner'/('training_'+f))
    shutil.copy2(run/'refiner_features/train/manifest.json',dest/'refiner/training_features_manifest.json')
    shutil.copy2(implementation/'verify_fusion.py',dest/'verify_model.py')
    (dest/'README.md').write_text(
        '# Model349 fusion805 lite23 B4ws — local retrained model\n\n'
        '23 weights: R14 + E B4ws4 + HRNet4 + F1. Fixed stage2 result; no published score is claimed.\n'
        'Training used S=0, all ancestor stages, all 12 E OOF branches, and four BN-frozen warmstarts.\n'
        'Use conda komap. Verify all local hashes: `python -B verify_model.py`.\n\n'
        'From this folder, fresh image-only inference:\n\n'
        '```bash\n'
        'python -B -u lite23/infer_compact.py --config lite23/selected_config_b4ws.json '
        '--parent-package . --assets assets --input /path/to/images --output /new/output --gpus 2,3,4,5,6,7\n'
        '```\n\nStrict checks use this run\'s hashes; no --allow-retrained bypass is necessary.\n')
    write_json(dest/'model_manifest.json',dict(checkpoint_count=23,
        files_sha256={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}))
    return dest
