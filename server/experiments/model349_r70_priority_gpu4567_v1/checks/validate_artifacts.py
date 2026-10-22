"""CPU-only package/report fixture. Never trains, launches CUDA or writes real results."""
import shutil, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from priority_common import R70, SEED_RUN, IDS, read, write, sha, stamp
from evaluate_seeds import replace_hr336, load_report_tools, export

work=root/'checks'/('artifact_fixture_'+stamp());work.mkdir()
fixture=read(R70/'checks/package_fixture_latest.json')
assert fixture['training_started'] is False and fixture['fixture_weights_are_old_fold_models']
source=Path(fixture['models']['lite20_r70'])
verify,_,_=load_report_tools(R70)
before_manifest=sha(source/'model_manifest.json')
before_weight=sha(source/'Model349_exh14/weights/model336_hrnet_best/last.pt')
candidate=read(SEED_RUN/'completed/336_00.json');package=work/'package_fixture'
replace_hr336(source,package,candidate,verify)
assert sha(source/'model_manifest.json')==before_manifest
assert sha(source/'Model349_exh14/weights/model336_hrnet_best/last.pt')==before_weight
assert sha(package/'Model349_exh14/weights/model336_hrnet_best/last.pt')==candidate['artifacts'][candidate['checkpoint']]
assert verify(package)['checkpoint_count']==17
old={(r['namespace'],r['path']):r['sha256'] for r in read(source/'compact/assets_manifest.json')['checkpoints']}
new={(r['namespace'],r['path']):r['sha256'] for r in read(package/'compact/assets_manifest.json')['checkpoints']}
assert [key for key in old if old[key]!=new[key]]==[('R','weights/model336_hrnet_best/last.pt')]

# Reuse the previously verified baseline prediction fixture for export mechanics only.
old_export=Path(read(R70/'checks/export_validation.json')['fixture_result'])
scored=old_export.parents[1]/'attempt/scored/lite20_r70'
assert all((scored/s/'metrics.json').is_file() for s in ['R','stage1','stage2'])
r70=work/'r70_fixture';r70.mkdir();(r70/'implementation').symlink_to(R70,target_is_directory=True)
write(work/'state.json',dict(r70_run=str(r70)))
(work/'implementation').mkdir()
for p in root.iterdir():
    if p.is_file() and p.suffix in ['.py','.sh','.md']:shutil.copy2(p,work/'implementation'/p.name)
metrics={s:read(scored/s/'metrics.json') for s in ['R','stage1','stage2']}
for identifier in IDS:
    record=read(SEED_RUN/f'completed/{identifier}.json')
    write(work/f'evaluated/{identifier}.json',dict(id=identifier,seed=record['seed'],package=str(package),
        scored=str(scored),metrics=metrics,original_REF=record['image_miou'],attempt=str(work),files_sha256={}))
result=export(work,results_root=work/'fixture_results')
summary=read(result/'summary.json')
assert len(summary['candidates'])==8
assert len(list(result.glob('**/valid_mask_comparison/*.png')))==500
assert len(list((result/'valid_mask_comparison').glob('*.png')))==20
assert (result/(result.name+'.md')).is_file()
assert verify(result/'selected_model')['checkpoint_count']==17
for row in summary['candidates']:
    assert abs(row['delta_REF_pp']-100*(row['metrics']['stage2']['image_miou']-row['original_REF']))<1e-12
assert not (result/'INCOMPLETE').exists()
(work/'README.md').write_text('CPU test fixture only. Full70 slots contain old fold weights; predictions are old baseline predictions. Not a new r70 experiment result.\n')
write(root/'checks/artifact_validation.json',dict(ok=True,training_started=False,cuda_used=False,
    fixture_only=True,fixture=str(work),report=str(result),strict_checkpoint_count=17,
    only_HR336_replaced=True,source_unchanged=True,seed_rows=8,panels=500))
print('ARTIFACT FIXTURE PASS: 17 strict weights, only HR336 changed, 8 seed report rows, 500 panels. NO TRAINING / CUDA.')
