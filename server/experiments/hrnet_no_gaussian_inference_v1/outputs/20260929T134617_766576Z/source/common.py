"""Immutable inference artifacts; the official score always uses empty IoU=0."""
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from metrics import PALETTE, CLASSES, mask_ids, metrics, summarize
from panels import comparison_panel

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASELINE = json.loads((ROOT/'baseline.json').read_text())
DEADLINE = datetime.fromisoformat(BASELINE['deadline_utc']).timestamp()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def require_environment():
    if Path(sys.prefix).name != 'komap': raise RuntimeError('Use conda komap')
    for p,digest in json.loads((ROOT/'preservation.json').read_text()).items():
        if sha(p)!=digest: raise RuntimeError(f'Preserved artifact changed: {p}')


def data_rows():
    paths=sorted((PROJECT/'dataset/valid/images').glob('*_image.png'))
    if len(paths)!=20: raise RuntimeError('Expected official valid20')
    return [dict(image_id=p.name.removesuffix('_image.png'),image=p,
                 mask=PROJECT/'dataset/valid/masks'/p.name.replace('_image.png','_mask.png')) for p in paths]


def reserve_result():
    tag=datetime.now(timezone.utc).strftime('%m%d')
    for i in range(1,100000):
        p=PROJECT/'results'/f'model_results_{tag}_{i}'
        try: p.mkdir(); return p
        except FileExistsError: pass
    raise RuntimeError('No result directory available')


def score_image(prob, row, directory, panels=True):
    pred=np.asarray(prob).argmax(0).astype(np.uint8)
    target=mask_ids(np.asarray(Image.open(row['mask']).convert('RGB')))
    result=dict(image_id=row['image_id'], **metrics(target,pred))
    directory=Path(directory)
    (directory/'predictions').mkdir(exist_ok=True)
    Image.fromarray(PALETTE[pred]).save(directory/'predictions'/f"{row['image_id']}.png")
    if panels:
        (directory/'valid_mask_comparison').mkdir(exist_ok=True)
        comparison_panel(Image.open(row['image']),target,pred,result['image_miou'],row['image_id']).save(
            directory/'valid_mask_comparison'/f"{row['image_id']}.png",compress_level=2)
    save_json(directory/'per_image'/f"{row['image_id']}.json",result)
    return result


def finalize(directory, arm, seconds, extra=None):
    directory=Path(directory)
    rows=[json.loads((directory/'per_image'/f"{r['image_id']}.json").read_text()) for r in data_rows()]
    summary=summarize(rows)
    summary.update(model='HRNet_taeho_no_gaussian',arm=arm,seconds=seconds,
                   checkpoint=BASELINE['checkpoint'],checkpoint_sha256=sha(BASELINE['checkpoint']),
                   result=str(directory),training=False,validation_tuning=True,
                   generalization_not_established=True,**(extra or {}))
    save_json(directory/'summary.json',summary)
    with (directory/'per_image.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image_id','image_miou',*CLASSES])
        w.writerows([r['image_id'],r['image_miou'],*r['class_iou']] for r in rows)
    lines=[f"# {arm['id']} — HRNet_taeho_no_gaussian inference",'',
           f"image mIoU: **{100*summary['image_miou']:.6f}%** (valid20, empty union=0)",'',
           f"Control: `{arm.get('contrast','original')}`",f"Change: `{arm.get('change',{})}`",'',
           'Fixed trained checkpoint. Inference only; no gradient updates. Validation-guided exploration; not an independent test result.',
           f"Checkpoint: `{BASELINE['checkpoint']}`",'', '```json',json.dumps(arm,ensure_ascii=False,indent=2),'```','',
           'Panels: original top-left, errors in ground-truth colors top-right (correct pixels white), prediction bottom-left, ground truth bottom-right.',
           'The header reports image mIoU and 100%−image mIoU; the latter is not pixel error rate.','',
           '| Image | image mIoU | Comparison |','|---|---:|---|']
    for r in rows:
        name=r['image_id'];lines.append(f"| {name} | {100*r['image_miou']:.4f}% | [panel](valid_mask_comparison/{name}.png) |")
    (directory/(directory.name+'.md')).write_text('\n'.join(lines)+'\n')
    save_json(directory/'complete.json',dict(num_images=20,summary_sha256=sha(directory/'summary.json'),
              files={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file() and p.name!='complete.json'}))
    return summary


def update_comparison():
    entries=[]
    for p in sorted((ROOT/'outputs').glob('*/arms/*/complete.json')):
        entry=json.loads(p.read_text());s=json.loads((Path(entry['result'])/'summary.json').read_text());entries.append(s)
    if not entries:return
    by_id={s['arm']['id']:s for s in entries}
    lines=['# HRNet no-gaussian inference ablation','',
           'Fixed checkpoint; official valid20 image mIoU with empty class IoU=0. All conditions are selected/explored on this validation set.',
           f"Original saved-mask reference: {BASELINE['expected_empty0']*100:.6f}%",'',
           '| ID | Control | Change | image mIoU | Delta vs control (pp) | Result |','|---|---|---|---:|---:|---|']
    for s in entries:
        a=s['arm'];control=by_id.get(a.get('contrast'));ref=control['image_miou'] if control else BASELINE['expected_empty0']
        lines.append(f"| {a['id']} | {a.get('contrast','original')} | {a.get('change',{})} | {100*s['image_miou']:.6f}% | {100*(s['image_miou']-ref):+.6f} | [{Path(s['result']).name}]({s['result']}/{Path(s['result']).name}.md) |")
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    save_json(ROOT/'comparison.json',entries)
