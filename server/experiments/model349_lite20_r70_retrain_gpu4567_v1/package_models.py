"""Build strict 17/20-weight packages, changing only the three R B4 paths."""
import copy,shutil
from pathlib import Path
from common import ROOT,MODEL,B4_DIRS,read_json,write_json,sha
from verify_model import verify

def package_one(destination,variant,trained):
    import torch
    torch.set_num_threads(2)
    destination=Path(destination)
    shutil.copytree(ROOT/'fusion_source',destination,
        ignore=shutil.ignore_patterns('Model349_exh14','results','README.md','__pycache__','*.pyc','SOURCE_SHA256.json'))
    shutil.copytree(ROOT/'frozen',destination/'Model349_exh14',
        ignore=shutil.ignore_patterns('checkpoint_manifest.json','recipe_manifest.json','verify_bundle.py','README.md','weights'))
    compact=destination/'compact';compact.mkdir()
    for name in ['infer_compact.py','infer_expert.py']:
        shutil.copy2(ROOT/'compact_source'/name,compact/name)
    cfg=read_json(ROOT/f'compact_source/selected_config_{variant}.json')
    cfg.update(R_config_file='R_config.yaml',assets_manifest_file='assets_manifest.json',
               status=f'{variant}: local paired experiment; all deployed checkpoints and metadata SHA256 checked')
    write_json(compact/'config.json',cfg)
    shutil.copy2(ROOT/f'compact_source/selected_R_{variant}.yaml',compact/'R_config.yaml')
    shutil.copy2(compact/'R_config.yaml',destination/'Model349_exh14/configs/model349_lite.yaml')
    selected=read_json(MODEL/'lite20/selected_assets_lite20.json')
    checkpoints=[];metadata={};full70=variant=='lite20_r70'
    for row in selected['checkpoints']:
        if full70 and row['namespace']=='R' and '/fold' in row['path']:continue
        origin=(MODEL/'Model349_exh14' if row['namespace']=='R' else MODEL/'assets')/row['path']
        target=(destination/'Model349_exh14' if row['namespace']=='R' else destination/'assets')/row['path']
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,target)
        entry=dict(namespace=row['namespace'],path=row['path'],sha256=sha(target),bytes=target.stat().st_size,source_checkpoint=str(origin))
        assert entry['sha256']==row['sha256'];checkpoints.append(entry)
        if origin.name=='last.pt' and (origin.parent/'args.json').exists():
            shutil.copy2(origin.parent/'args.json',target.parent/'args.json')
            key=(row['namespace'],str(Path(row['path']).parent/'args.json'));metadata[key]=target.parent/'args.json'
        if 'model329_ws/' in row['path']:
            shutil.copy2(origin.parent/'bn_freeze_check.json',target.parent/'bn_freeze_check.json')
    if full70:
        for component,folder in B4_DIRS.items():
            original=Path(trained[component]);payload=torch.load(original,map_location='cpu',weights_only=False)
            state=payload.get('model',payload.get('state_dict',payload))
            floating=[v for v in state.values() if torch.is_tensor(v) and torch.is_floating_point(v)]
            assert floating and all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in floating),original
            relative=f'weights/{folder}_full70/last.pt';target=destination/'Model349_exh14'/relative
            target.parent.mkdir(parents=True);shutil.copy2(original,target)
            shutil.copy2(original.parent/'args.json',target.parent/'args.json')
            checkpoints.append(dict(namespace='R',path=relative,sha256=sha(target),bytes=target.stat().st_size,source_checkpoint=str(original)))
            metadata[('R',f'weights/{folder}_full70/args.json')]=target.parent/'args.json'
            del payload,state,floating
    metadata[('R','configs/calibration.json')]=destination/'Model349_exh14/configs/calibration.json'
    meta=[]
    for (ns,relative),path in sorted(metadata.items()):
        row=dict(namespace=ns,path=relative,sha256=sha(path),bytes=path.stat().st_size);meta.append(row)
        copy_path=compact/'parent_metadata'/ns/relative;copy_path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,copy_path)
    expected=17 if full70 else 20
    assert len(checkpoints)==expected
    write_json(compact/'assets_manifest.json',dict(count=expected,checkpoints=checkpoints,metadata=meta,
        bytes=sum(r['bytes'] for r in checkpoints),unchecked_checkpoints=[]))
    auxiliary=[{k:v for k,v in r.items() if k!='namespace'} for r in checkpoints if r['namespace']=='E' and not r['path'].startswith('model329_ws/')]
    b4ws=[{k:v for k,v in r.items() if k!='namespace'} for r in checkpoints+meta if r['namespace']=='E' and r['path'].startswith('model329_ws/')]
    write_json(destination/'config/auxiliary_assets.json',auxiliary);write_json(destination/'config/b4ws_assets.json',b4ws)
    rrows=[{k:v for k,v in r.items() if k!='namespace'} for r in checkpoints if r['namespace']=='R']
    write_json(destination/'Model349_exh14/checkpoint_manifest.json',dict(count=len(rrows),checkpoints=rrows,bytes=sum(r['bytes'] for r in rrows)))
    recipe={str(p.relative_to(destination/'Model349_exh14')):sha(p) for p in (destination/'Model349_exh14').rglob('*')
            if p.is_file() and 'weights' not in p.relative_to(destination/'Model349_exh14').parts}
    write_json(destination/'Model349_exh14/recipe_manifest.json',dict(files_sha256=recipe))
    shutil.copy2(ROOT/'verify_model.py',destination/'verify_model.py')
    write_json(destination/'provenance.json',read_json(ROOT/'provenance.json'))
    (destination/'README.md').write_text(f'# {variant} — local paired experiment\n\n'
        f'{expected} checkpoints; all SHA256 values verified, including newly trained R B4. Use conda komap.\n\n'
        '```bash\npython -B verify_model.py\n'
        'python -B -u compact/infer_compact.py --config compact/config.json --parent-package . --assets assets '
        '--input /path/to/images --output /new/output --gpus 4,5,6,7\n```\n')
    write_json(destination/'model_manifest.json',dict(variant=variant,checkpoint_count=expected,
        files_sha256={str(p.relative_to(destination)):sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}))
    verify(destination);return destination

def package_pair(output,trained):
    paths={v:package_one(output/v,v,trained) for v in ['lite20_r70','lite20']}
    manifests={v:read_json(p/'compact/assets_manifest.json') for v,p in paths.items()}
    unchanged=[]
    control={(r['namespace'],r['path']):r['sha256'] for r in manifests['lite20']['checkpoints']}
    for row in manifests['lite20_r70']['checkpoints']:
        key=(row['namespace'],row['path'])
        if '_full70/' not in row['path']:
            assert control[key]==row['sha256'];unchanged.append('/'.join(key))
    assert len(unchanged)==14
    write_json(output/'paired_weights.json',dict(unchanged_checkpoints=unchanged,unchanged_count=14,
        new_R_B4=3,r70_checkpoint_count=17,control_checkpoint_count=20))
    return {v:str(p) for v,p in paths.items()}
