"""Synthetic labels/placeholder bytes validate the exporter; never evaluated model weights."""
from pathlib import Path
import os, shutil, subprocess, sys, tempfile
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from common import read,write
from export_results import export
with tempfile.TemporaryDirectory(prefix='export_fixture_',dir=ROOT/'checks') as tmp:
    base=Path(tmp);run=base/'run';run.mkdir();impl=run/'implementation';impl.mkdir()
    shutil.copytree(ROOT/'source',impl/'source');shutil.copy2(ROOT/'plan.json',impl/'plan.json')
    assets=read(ROOT/'source/inference/selected_assets.json')
    for r in assets['checkpoints']+assets['metadata']:
        path=r['path']
        if r['namespace']=='R':
            if not path.startswith('weights/'):continue
            path=path.removeprefix('weights/')
        target=run/'model/weights'/r['namespace']/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('TEST ONLY: no model weights\n')
    plan=read(ROOT/'plan.json')
    for name in [r['name'] for r in plan['tasks']]+['score_valid']:write(run/f'stages/{name}.json',dict(status='done'))
    for split,count in [('valid',20),('test',10)]:
        output=run/f'pred_{split}'
        for folder in ['predicted_labels','predicted_masks']:(output/'fusion'/folder).mkdir(parents=True)
        for p in (ROOT/f'data/{split}/images').glob('*_image.png'):
            with Image.open(p) as im:w,h=im.size
            label=Image.fromarray(np.zeros((h,w),np.uint8));name=p.stem.removesuffix('_image')+'.png'
            label.save(output/'fusion/predicted_labels'/name);label.convert('RGB').save(output/'fusion/predicted_masks'/name)
        write(output/'inference_timing.json',dict(images=count,wall_seconds=float(count)*2,seconds_per_image=2.,images_per_second=.5,synthetic_fixture=True))
    subprocess.run([sys.executable,str(ROOT/'source/tools/score_masks.py'),'--labels',str(run/'pred_valid/fusion/predicted_labels'),'--gt',str(ROOT/'data/valid/masks'),'--output',str(run/'pred_valid/metrics.json')],check=True)
    write(run/'pipeline_completion.json',dict(ok=True,wall_seconds=123.))
    write(run/'preflight.json',dict(synthetic_fixture=True));(run/'environment.freeze.txt').write_text('TEST ONLY\n')
    destination=export(run,base/'test_results')
    assert len(list((destination/'valid_mask_comparison').glob('*.png')))==20
    assert len(list((destination/'selected_model/weights').rglob('*.pt')))==13
    assert len(list((destination/'test_predictions/predicted_masks').glob('*.png')))==10
    assert read(destination/'metrics.json')['count']==20
    text=(destination/f'{destination.name}.md').read_text()
    assert '| valid | 20 | 40.000 | 2.000 |' in text
    assert '| test | 10 | 20.000 | 2.000 |' in text
    subprocess.run(['bash','-n',str(destination/'selected_model/predict.sh')],check=True)
write(ROOT/'checks/export_cpu.json',dict(ok=True,synthetic_fixture=True,model_inference=False,panels=20,weights_inventory=13,test_masks=10,report_timing_rows=2))
print('PASS: full synthetic export (not model performance), 20 panels, 13 weight slots, measured-time report fields.')
