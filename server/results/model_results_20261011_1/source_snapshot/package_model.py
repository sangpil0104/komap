"""Package all newly trained checkpoints in the original exh14 FP32 format."""
from pathlib import Path
import shutil

from tqdm import tqdm

from experiment import ROOT, read_json, write_json, sha
from scheduler import render


def package(run):
    import torch
    torch.set_num_threads(2)
    run = Path(run)
    implementation = run / 'implementation'
    destination = run / 'retrained_release'
    if destination.exists():
        raise FileExistsError(destination)
    # Do not carry the original published validation scores into the new model.
    shutil.copytree(implementation / 'frozen', destination,
                    ignore=shutil.ignore_patterns('results', 'search', 'README.md'))
    for name in ['checkpoint_manifest.json', 'recipe_manifest.json']:
        (destination / name).rename(destination / ('upstream_' + name))
    shutil.copy2(implementation / 'verify_retrained.py', destination / 'verify_bundle.py')
    expected = {row['path'] for row in read_json(destination / 'upstream_checkpoint_manifest.json')['checkpoints']}
    plan=read_json(run/'plan.json')
    tasks={t['id']:t for t in plan['tasks']}
    records = []
    for slot in tqdm(plan['release_slots'], desc='Package 11 retrained FP32 checkpoints', unit='model'):
        task=tasks[slot['task']]
        original = Path(render([task['checkpoint']], implementation / 'source', run, 4)[0])
        relative = slot['path']
        if relative not in expected:
            raise ValueError(f'Unexpected checkpoint: {relative}')
        payload = torch.load(original, map_location='cpu', weights_only=False)
        if 'state_dict' in payload:
            state = payload['state_dict']
            packaged = dict(state_dict=state, model_config=payload['model_config'])
        elif 'model' in payload:
            state = payload['model']
            packaged = dict(model=state)
        else:
            state = payload
            packaged = state
        floating = [x for x in state.values() if torch.is_tensor(x) and torch.is_floating_point(x)]
        if not floating or not all(x.dtype == torch.float32 and torch.isfinite(x).all() for x in floating):
            raise ValueError(f'Expected finite FP32 checkpoint parameters: {original}')
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            torch.save(packaged, stream)
        if '_fold' in task['id']:
            shutil.copy2(original.parent / 'args.json', target.parent / 'args.json')
        records.append(dict(path=relative, bytes=target.stat().st_size, sha256=sha(target),
                            source_checkpoint=str(original), source_sha256=sha(original),
                            task=task['id'], seed=task['seed']))
        del payload, state, packaged, floating
    if {r['path'] for r in records} != expected or len(records) != 11:
        raise ValueError('Must replace all eleven lite20 weights with new checkpoints.')
    write_json(destination / 'checkpoint_manifest.json', dict(count=11, bytes=sum(r['bytes'] for r in records),
               source='New exh14 component retraining; original FP32 tensor values retained', checkpoints=records))
    (destination / 'README.md').write_text(
        '# Model349 lite20 R — newly retrained weights\n\n'
        'All 11 weights were retrained. The lite20 inference code, calibration, fold subset and fusion are fixed.\n'
        'See ../ for the measured valid20 result. No published upstream score is claimed for these weights.\n\n'
        'Verify: `python verify_bundle.py`\n\n'
        'Inference: `python infer_lite.py --input <images> --output <new_directory> --gpus 2,3,4,5,6,7`\n')
    files = {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*'))
             if p.is_file() and 'weights' not in p.relative_to(destination).parts and p.name != 'recipe_manifest.json'}
    write_json(destination / 'recipe_manifest.json', dict(version='model349-exh14-retrained-local-v1', files_sha256=files))
    return destination
