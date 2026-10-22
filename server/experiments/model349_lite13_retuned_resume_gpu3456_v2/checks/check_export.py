"""Export real prior valid predictions and synthetic test outputs to a temporary directory."""
from pathlib import Path
import shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from common import read,write
from export_results import export
parent=Path(read(ROOT/'parent.json')['parent_run'])
with tempfile.TemporaryDirectory(prefix='export_fixture_',dir=ROOT/'checks') as tmp:
    base=Path(tmp);run=base/'run';run.mkdir();impl=run/'implementation';impl.mkdir()
    shutil.copytree(ROOT/'source',impl/'source');shutil.copy2(ROOT/'plan.json',impl/'plan.json')
    for r in read(ROOT/'source/inference/selected_assets.json')['checkpoints']:
        rel=r['path'].removeprefix('weights/') if r['namespace']=='R' else r['path']
        p=run/'model/weights'/r['namespace']/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('TEST PLACEHOLDER, NOT WEIGHTS\n')
    shutil.copytree(parent/'stages',run/'stages');write(run/'stages/score_valid.json',dict(status='done'))
    for split,n in [('valid',20),('test',10)]:
        output=run/f'pred_{split}'
        if split=='valid':shutil.copytree(parent/'pred_valid/fusion',output/'fusion')
        else:
            for folder in ['predicted_labels','predicted_masks']:(output/'fusion'/folder).mkdir(parents=True)
            for p in (ROOT/'data/test/images').glob('*_image.png'):
                name=p.stem.removesuffix('_image')
                (output/f'fusion/predicted_labels/{name}.png').touch()
                (output/f'fusion/predicted_masks/{name}_mask.png').touch()
        write(output/'completion.json',dict(synthetic_fixture=True))
        write(output/'inference_timing.json',dict(images=n,wall_seconds=float(n)*2,seconds_per_image=2.,images_per_second=.5,inventory_verified=True,synthetic_fixture=True))
    subprocess.run([sys.executable,str(ROOT/'source/tools/score_masks.py'),'--labels',str(run/'pred_valid/fusion/predicted_labels'),'--gt',str(ROOT/'data/valid/masks'),'--output',str(run/'pred_valid/metrics.json')],check=True)
    write(run/'pipeline_completion.json',dict(ok=True,wall_seconds=123.,parent_attempt_wall_seconds=100.,total_active_wall_seconds=223.))
    write(run/'resume_provenance.json',read(ROOT/'parent.json'))
    write(run/'preflight.json',dict(synthetic_fixture=True));(run/'environment.freeze.txt').write_text('TEST ONLY\n')
    destination=export(run,base/'temporary_results')
    assert len(list((destination/'valid_mask_comparison').glob('*.png')))==20
    assert len(list((destination/'selected_model/weights').rglob('*.pt')))==13
    assert len(list((destination/'test_predictions/predicted_masks').glob('*_mask.png')))==10
    report=(destination/f'{destination.name}.md').read_text()
    assert '| valid | 20 | 40.000 | 2.000 |' in report
    assert '| test | 10 | 20.000 | 2.000 |' in report
    assert '재학습은 하지 않았습니다' in report
    actual_score=read(destination/'metrics.json')['image_miou']
write(ROOT/'checks/export_check.json',dict(ok=True,real_saved_valid_predictions=True,test_masks_and_timings_synthetic=True,new_inference=False,panels=20,model_weight_slots=13,report_saved_valid_image_miou=actual_score))
print('PASS: real valid20 naming/color output accepted through full export; temporary fixture removed.')
