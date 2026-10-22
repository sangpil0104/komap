"""One-time CPU packaging performed during build; does not train or run inference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from collections import Counter
from types import SimpleNamespace
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch

if (ROOT/'local_weights.json').exists(): raise FileExistsError('Build artifacts already prepared')
torch.set_num_threads(2)
upstream = read_json(ROOT/'upstream/checkpoint_manifest.json')
rows = []
for row in tqdm(upstream['checkpoints'],desc='Prepare local FP16 copies',unit='weight'):
    relative = row['path']
    result = fp16_checkpoint(ORIGINAL_BUNDLE/relative,ROOT/relative)
    result.update(relative=relative,upstream_sha256=row['sha256'],upstream_hash_match=result['sha256']==row['sha256'])
    rows.append(result)
write_json(ROOT/'local_weights.json',dict(count=12,source=str(ORIGINAL_BUNDLE),checkpoints=rows,
           official_upstream_weight_files_supplied=False,bytes=sum(r['bytes'] for r in rows)))

sys.path.insert(0,str(ROOT/'frozen/src'))
from highres1024_search import choose_source
routes = SimpleNamespace(source_model109='109',source_restart51='restart51',source_seed456='restart59',
                         source_pairwise='restart61',source_ema='restart51')
images = []
for path in sorted((BASE/'data/valid/images').glob('*_image.png')):
    name = path.stem.removesuffix('_image')
    with Image.open(path) as im: raw=np.array(im.convert('RGB')); size=list(im.size)
    mean = float(raw[...,0].astype(np.float32).mean())
    target = BASE/'data/valid/masks'/f'{name}_mask.png'
    images.append(dict(name=name,path=str(path),gt=str(target),size=size,sha256=sha(path),
                       gt_sha256=sha(target),mean=mean,high_family=choose_source(mean,routes)))
train_count = len(list((BASE/'data/train/images').glob('*_image.png')))
assert len(images)==20 and train_count==70
counts = dict(Counter(r['high_family'] for r in images)); assert set(counts)==set(HIGH)
write_json(ROOT/'data_manifest.json',dict(train_count=train_count,images=images,high_route_counts=counts))
write_json(ROOT/'plan.json',validate_plan(make_plan()))

inputs=set()
for split in ['train','valid']:
    inputs.update((BASE/'data'/split).rglob('*.png'))
inputs.update((BASE/'data_model322_group_cv').rglob('*.png'))
recipe = read_json(ROOT/'original_recipe.json')
from worker import command_for
for task in make_plan()['tasks']:
    if task['kind']!='train': continue
    command=command_for(task,Path('/tmp/lite12_build_no_execution')/task['id'])
    for flag in ['--checkpoint','--init-checkpoint']:
        if flag in command: inputs.add(Path(command[command.index(flag)+1]))
inputs.update(ORIGINAL_BUNDLE/r['path'] for r in upstream['checkpoints'])
for path in Path('/home/sppark/.cache/huggingface/hub').rglob('*'):
    if path.is_file() and ('snapshots' in path.parts or 'refs' in path.parts): inputs.add(path)
inputs.update(Path('/home/sppark/.cache/torch/hub/checkpoints').glob('*'))
write_json(ROOT/'input_manifest.json',{str(p):sha(p) for p in tqdm(sorted(inputs),desc='Hash fixed input files',unit='file',mininterval=5)})
write_json(ROOT/'provenance.json',dict(repository='https://github.com/CNUKoMaP/lhs',
    commit='f6456f54e5fc6517a89a3fdb679645a0520bbf61',upstream_path='Model349_lite12_base',
    archive_sha256=sha(ROOT/'downloads/lhs_f6456f54.tar.gz'),
    upstream_files={str(p.relative_to(ROOT/'upstream')):sha(p) for p in sorted((ROOT/'upstream').rglob('*')) if p.is_file()},
    training_source_copied_from=str(REPO/'experiments/model349_seed100_gpu4567_v1/source'),
    training_recipe_copied_from=str(REPO/'experiments/model349_seed100_gpu4567_v1/original_recipe.json'),
    baseline_source=str(ORIGINAL_BUNDLE),upstream_weights_supplied=False,
    original_models_and_results_modified=False,training_started=False))
print('Prepared',len(rows),'FP16 files; routes:',counts)
