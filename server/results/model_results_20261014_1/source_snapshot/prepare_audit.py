"""Capture and verify the completed parent run; never mutate any parent file."""
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from common import ROOT,read,write,sha,now
from prediction_checks import validate_prediction_inventory
from export_results import PALETTE,decode
from report_utils import class_ious

def main():
    parent=Path(read(ROOT/'parent.json')['parent_run']);files={};artifacts={}
    def add(p):
        p=Path(p);digest=sha(p);files[str(p)]=digest;return digest
    rows={p.stem:read(p) for p in (parent/'stages').glob('*.json')}
    expected={r['name'] for r in read(ROOT/'plan.json')['tasks']}
    if set(rows)!=expected or len(rows)!=48:raise ValueError('Incomplete parent stage inventory')
    for name,row in tqdm(rows.items(),desc='Verify completed artifacts',unit='stage',mininterval=5):
        if row['status']!='done' or row['returncode']!=0:raise ValueError(name)
        path=Path(row['artifact'])
        if not path.resolve().is_relative_to(parent):raise ValueError(f'Unexpected artifact location {path}')
        digest=add(path)
        if digest!=row['artifact_sha256']:raise ValueError(f'Artifact changed: {name}')
        artifacts[str(path.resolve())]=digest;add(parent/f'stages/{name}.json')
    train=[r for r in rows.values() if Path(r['artifact']).suffix=='.pt']
    if len(train)!=33:raise ValueError('Expected 33 completed training jobs')
    deployed=[]
    for p in sorted((parent/'model/weights').rglob('*')):
        if not p.is_file():continue
        digest=add(p);add(p.resolve())
        if p.suffix=='.pt':
            if artifacts.get(str(p.resolve()))!=digest:raise ValueError(f'Unverified deployment weight {p}')
            deployed.append(dict(path=str(p.relative_to(parent/'model/weights')),source=str(p.resolve()),sha256=digest))
    if len(deployed)!=13:raise ValueError('Expected 13 deployment checkpoints')
    for p in (parent/'implementation').rglob('*'):
        if p.is_file() and 'data' not in p.relative_to(parent/'implementation').parts:add(p)
    for name in ['status.json','preflight.json','environment.freeze.txt']:add(parent/name)
    for name,digest in tqdm(read(ROOT/'data_manifest.json').items(),desc='Verify data',unit='file',mininterval=5):
        if add((ROOT/name).resolve())!=digest:raise ValueError(name)
    inventory=validate_prediction_inventory(ROOT/'data/valid/images',parent/'pred_valid',20)
    scores=[]
    for p in tqdm(sorted((ROOT/'data/valid/images').glob('*_image.png')),desc='Audit saved valid20',unit='image',mininterval=5):
        name=p.stem.removesuffix('_image');label=parent/f'pred_valid/fusion/predicted_labels/{name}.png';mask=parent/f'pred_valid/fusion/predicted_masks/{name}_mask.png'
        with Image.open(label) as im:prediction=np.asarray(im)
        with Image.open(mask) as im:rgb=np.asarray(im.convert('RGB'))
        if not np.isin(prediction,range(4)).all() or not np.array_equal(rgb,PALETTE[prediction]):raise ValueError(f'Color/label mismatch {name}')
        gt=decode(ROOT/f'data/valid/masks/{name}_mask.png');ious=class_ious(prediction,gt)
        scores.append(dict(image=name,class_iou=ious,image_miou=float(np.mean(ious))));add(label);add(mask)
    for name in ['completion.json','phase_timings.json','fusion/completion.json']:add(parent/'pred_valid'/name)
    write(ROOT/'input_manifest.json',files)
    write(ROOT/'checks/parent_audit.json',dict(ok=True,audited_at=now(),parent_run=str(parent),completed_stages=48,completed_training=33,
          deployed_weights=deployed,inventory=inventory,valid_prediction_image_miou=float(np.mean([r['image_miou'] for r in scores])),
          old_stage_timings=read(parent/'pred_valid/phase_timings.json'),old_end_to_end_time_available=False,checked_files=len(files),training_started=False,inference_started=False))
    write(ROOT/'checks/existing_valid20_metrics.json',dict(count=20,image_miou=float(np.mean([r['image_miou'] for r in scores])),images=scores,source='existing parent predictions; no new inference'))
    print('PASS: all 33 training jobs and 13 deployment weights verified; saved valid20 complete.')

if __name__=='__main__':main()
