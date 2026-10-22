"""CPU export fixture using existing predictions on BOTH sides; no r70 training."""
from pathlib import Path
import sys,shutil
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from common import BASE,read_json,write_json,stamp
from export_results import export
from verify_model import verify
fixture=read_json(root/'checks/package_fixture_latest.json')
run=root/'checks'/('export_fixture_'+stamp());run.mkdir();attempt=run/'attempt';attempt.mkdir()
valid=run/'baseline_prediction_fixture';valid.mkdir()
for name,path in [('release',BASE/'release_inference'),('expert',BASE/'expert_probabilities'),('fusion',BASE/'stage2')]:
    (valid/name).symlink_to(path,target_is_directory=True)
done={'package':{'models':fixture['models']}}
for v in ['lite20_r70','lite20']:
    done[f'{v}_valid']={'inference':str(valid)}
    done[f'{v}_test']={'inference':str(BASE/'test_inference')}
for name in ['plan.json','provenance.json']:shutil.copy2(root/name,run/name)
(run/'environment.freeze.txt').write_text('CPU-only validation fixture, not a training run\n')
(run/'events.jsonl').write_text('')
write_json(run/'preflight.json',dict(fixture=True,training_started=False))
write_json(run/'implementation_manifest.json',{})
result=export(run,attempt,done,results_root=run/'results_fixture')
summary=read_json(result/'paired_comparison.json')
for stage in ['R','stage1','stage2']:
    assert summary['metrics'][stage]['delta_pp']==0
    assert summary['metrics'][stage]['specimen_bootstrap95_pp']==[0.,0.]
assert abs(read_json(result/'metrics.json')['image_miou']-.8025584552762022)<1e-12
assert abs(read_json(result/'r_only/metrics.json')['image_miou']-.8001013505955459)<1e-12
assert summary['test_mask_difference']['changed_pixels']==0
assert len(list(result.glob('**/valid_mask_comparison/*.png')))==120
assert len(list((result/'test_predictions/predicted_masks').glob('*.png')))==10
verify(result/'selected_model')
(run/'README.md').write_text('Validation-only fixture. Both sides use old lite20 predictions. These are NOT r70 training results.\n')
write_json(root/'checks/export_validation.json',dict(ok=True,fixture_result=str(result),panels=120,
    baseline_R_image_miou=.8001013505955459,baseline_REF_image_miou=.8025584552762022,
    training_started=False,cuda_used=False,fixture_not_r70_results=True))
print('EXPORT FIXTURE PASS: report + 120 panels + metrics + 17 weights + test10; NO TRAINING')
