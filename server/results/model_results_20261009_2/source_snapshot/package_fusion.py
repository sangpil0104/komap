"""Freeze all 27 new weights, preserving preprocessing and recording fresh hashes."""
from pathlib import Path
import shutil
from tqdm import tqdm
from experiment import ROOT,read_json,write_json,sha
from package_model import package as package_release
from scheduler import render


def package(run):
    run=Path(run);implementation=run/'implementation'
    release=package_release(run)
    dest=run/'retrained_model'
    shutil.copytree(implementation/'fusion_source',dest,
        ignore=shutil.ignore_patterns('Model349_exh14','results','search','README.md','SOURCE_SHA256.json'))
    shutil.copytree(release,dest/'Model349_exh14')
    (dest/'config/auxiliary_assets.json').rename(dest/'config/upstream_auxiliary_assets.json')
    records=[]
    import torch
    torch.set_num_threads(2)
    tasks=[t for t in read_json(run/'plan.json')['tasks'] if t.get('family') in ['expert','refiner']]
    for task in tqdm(tasks,desc='Package 13 expert/refiner weights',unit='model'):
        checkpoint=Path(render([task['checkpoint']],implementation/'source',run,4)[0])
        relative=str(checkpoint.relative_to(run/'expert'))
        target=dest/'assets'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state=payload.get('state_dict',payload.get('model',payload))
        floating=[x for x in state.values() if torch.is_tensor(x) and torch.is_floating_point(x)]
        if not floating or not all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in floating):
            raise ValueError(f'Invalid FP32 expert/refiner checkpoint: {checkpoint}')
        # Preserve full HRNet preprocessing metadata including modal target width.
        shutil.copy2(checkpoint,target)
        if task['family']=='expert':shutil.copy2(checkpoint.parent/'args.json',target.parent/'args.json')
        records.append(dict(path=relative,sha256=sha(target),bytes=target.stat().st_size,
                            source_checkpoint=str(checkpoint),task=task['id'],seed=task['seed']))
        del payload,state,floating
    if len(records)!=13:raise ValueError('Expected 12 expert weights and one trained refiner.')
    write_json(dest/'config/auxiliary_assets.json',records)
    shutil.copy2(run/'expert/refiner/completion.json',dest/'refiner/training_receipt.json')
    shutil.copy2(run/'expert/refiner/recipe.json',dest/'refiner/training_recipe.json')
    shutil.copy2(run/'refiner_features/train/manifest.json',dest/'refiner/training_features_manifest.json')
    shutil.copy2(implementation/'verify_fusion.py',dest/'verify_model.py')
    (dest/'README.md').write_text('# Model349 fusion805 — local retrained model\n\n'
        'All 27 component weights were newly trained. Main candidate: stage1 R+E (no F); stage2 is retained for comparison.\n'
        'No upstream score is claimed for these new weights. See the containing result report for measured image mIoU.\n\n'
        'Use conda komap. Check hashes: `python verify_model.py`.\n'
        'R: `python Model349_exh14/infer_lite.py --input <images> --output <new_R> --gpus 2,3,4,5,6,7`\n'
        'E: `CUDA_VISIBLE_DEVICES=2 python infer_model401.py --input <images> --assets assets --output <new_E>`\n'
        'Stage1: `python infer_stage1.py --release <new_R> --expert <new_E> --output <new_stage1>`\n'
        'Stage2: `python infer_fusion.py --input <images> --release-labels <new_R>/predicted_labels --expert-probabilities <new_E> --refiner assets/refiner/epoch8.pt --output <new_stage2> --device cpu`\n')
    write_json(dest/'model_manifest.json',dict(checkpoint_count=27,
        files_sha256={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}))
    return dest
