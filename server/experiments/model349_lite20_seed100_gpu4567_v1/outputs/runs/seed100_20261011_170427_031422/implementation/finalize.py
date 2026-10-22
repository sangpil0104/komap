"""Package selected 20 weights, verify fresh inference, then export one result."""
import copy,csv,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
from common import ROOT,REPO,BASE,EXPECTED_SCORE,read_json,write_json,sha,baseline_weights,records
from report_utils import allocate_result

def packaging_view(run,attempt,selected):
    view=attempt/'package_run';view.mkdir();(view/'implementation').symlink_to(ROOT,target_is_directory=True)
    plan=read_json(ROOT/'original_recipe.json');tasks={t['id']:t for t in plan['tasks']}
    weights=baseline_weights();weights.update(selected['context']['weights']);completed=records(run)
    seeds={r.get('checkpoint'):r.get('seed') for r in completed.values() if r.get('checkpoint')}
    for slot in plan['release_slots']:
        path=weights['R/'+slot['path']];tasks[slot['task']]['checkpoint']=path
        tasks[slot['task']]['seed']=seeds.get(path,tasks[slot['task']]['seed'])
    for fold in range(1,5):tasks[f'train_b4ws_fold{fold}']['checkpoint']=weights[f'E/model329_ws/fold{fold}/last.pt']
    for s,folds in plan['expert_hr_folds'].items():
        for fold in folds:tasks[f'train_expert_hr{s}_fold{fold}']['checkpoint']=weights[f'E/model392/seed{s}/fold{fold}/last.pt']
    tasks['train_refiner']['checkpoint']=weights['E/refiner/epoch8.pt']
    # package_fusion derives assets/<relative> from this view's expert directory.
    for key,path in weights.items():
        if not key.startswith('E/'):continue
        destination=view/'expert'/key.removeprefix('E/');destination.parent.mkdir(parents=True,exist_ok=True)
        destination.symlink_to(path)
        if destination.name=='last.pt':
            (destination.parent/'args.json').symlink_to(Path(path).parent/'args.json')
            if 'model329_ws' in key:(destination.parent/'bn_freeze_check.json').symlink_to(Path(path).parent/'bn_freeze_check.json')
    for t in tasks.values():
        checkpoint=t.get('checkpoint','')
        if checkpoint in weights.values() and not any(t['id']==slot['task'] for slot in plan['release_slots']):
            key=next(k for k,v in weights.items() if k.startswith('E/') and v==checkpoint)
            t['checkpoint']=str(view/'expert'/key.removeprefix('E/'));t['seed']=seeds.get(checkpoint,t['seed'])
    f_id=selected['context']['choices'].get('F')
    f_dir=Path(completed[f_id]['training_dir']) if f_id else BASE/'expert/refiner'
    for name in ['completion.json','recipe.json']:(view/'expert/refiner'/name).symlink_to(f_dir/name)
    features=Path(completed['features']['features']) if f_id else BASE/'refiner_features/train'
    (view/'refiner_features').mkdir();(view/'refiner_features/train').symlink_to(features,target_is_directory=True)
    write_json(view/'plan.json',plan);return view

def fresh_inference(model,split,out):
    # UUID comes from the scheduler's physical 4..7 mapping; exactly one GPU here.
    subprocess.run([sys.executable,'-B','-u',str(model/'lite20/infer_compact.py'),
        '--config',str(model/'lite20/selected_config_lite20.json'),'--parent-package',str(model),
        '--assets',str(model/'assets'),'--input',str(ROOT/f'data/{split}/images'),
        '--output',str(out),'--gpus',os.environ['CUDA_VISIBLE_DEVICES']],check=True)

def export(run,attempt,selected):
    from package_fusion import package
    view=packaging_view(run,attempt,selected);model=package(view)
    (model/'README.md').write_text('# Model349 lite20 — selected seed-search model\n\n'
        '20 deployed checkpoints: R11 + E8 + F1. Fixed lite20 recipe; selected from 100 training jobs.\n'
        'Validation-selected development score, not an independent generalization estimate.\n'
        'Use conda komap. `python -B verify_model.py` checks all files.\n\n'
        '```bash\npython -B -u lite20/infer_compact.py --config lite20/selected_config_lite20.json '
        '--parent-package . --assets assets --input /path/to/images --output /new/output --gpus 4,5,6,7\n```\n')
    write_json(model/'selection.json',selected)
    write_json(model/'model_manifest.json',dict(checkpoint_count=20,files_sha256={str(p.relative_to(model)):sha(p)
        for p in sorted(model.rglob('*')) if p.is_file() and p.name!='model_manifest.json'}))
    subprocess.run([sys.executable,'-B',str(model/'verify_model.py')],check=True)
    valid=attempt/'fresh_valid';fresh_inference(model,'valid',valid)
    cached=Path(selected['evaluation'])
    for row in selected['images']:
        name=row['image'];actual=np.asarray(Image.open(valid/f'fusion/predicted_labels/{name}.png'))
        expected=np.asarray(Image.open(cached/f'predicted_labels/{name}.png'))
        if not np.array_equal(actual,expected):raise ValueError(f'Fresh package disagrees with search: {name}')
    test=attempt/'final_test';fresh_inference(model,'test',test)
    assert len(list((test/'fusion/predicted_labels').glob('*.png')))==10
    result=allocate_result(REPO/'results')
    shutil.copytree(cached/'valid_mask_comparison',result/'valid_mask_comparison')
    shutil.copytree(cached/'predicted_labels',result/'predicted_labels')
    shutil.copytree(test/'fusion',result/'test_predictions')
    # Move only freshly created package files; old runs and models are untouched.
    shutil.move(str(model),result/'selected_model')
    shutil.copytree(ROOT,result/'source_snapshot',ignore=shutil.ignore_patterns('data','folds','expert_splits','outputs','logs','checks','__pycache__','*.pyc'))
    for name in ['plan.json','environment.freeze.txt','implementation_manifest.json']:
        shutil.copy2(run/name,result/name)
    write_json(result/'metrics.json',selected)
    completed=records(run);leaderboard=[]
    for r in completed.values():
        if r.get('candidate'):leaderboard.append(dict(id=r['id'],group=r['group'],seed=r.get('seed'),image_miou=r['image_miou'],choices=r['context']['choices']))
    leaderboard.sort(key=lambda r:(-r['image_miou'],r['id']))
    write_json(result/'leaderboard.json',leaderboard)
    with (result/'leaderboard.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','group','seed','image_miou']);writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in leaderboard)
    baseline={r['image']:r['image_miou'] for r in read_json(BASE/'stage2/metrics.json')['images']}
    rows=['| Image | Baseline % | Selected % | Change pp |','|---|---:|---:|---:|']
    for r in selected['images']:
        rows.append(f'| {r["image"]} | {100*baseline[r["image"]]:.6f} | {100*r["image_miou"]:.6f} | {100*(r["image_miou"]-baseline[r["image"]]):+.6f} |')
    report=f'''# Model349 lite20 B4ws — 100개 학습 시드 탐색

최종 valid20 image mIoU: **{selected['image_miou']*100:.6f}%**  
기준: **{EXPECTED_SCORE*100:.6f}%**, 변화 **{(selected['image_miou']-EXPECTED_SCORE)*100:+.6f} %p**

시드 3407~3506, 학습 총 100회. GPU 4~7, conda komap. 최종 배포 모델은 20개(R11+E8+F1)입니다.
학습 데이터/fold/epoch/iteration/loss 및 lite20 route, TTA와 정밀도를 고정했습니다.
R/E 92개 학습 → 전체 R+E+F 평가 및 최대 두 차례 조합 선택 → 선택 HR의 train OOF 갱신 → F 8개 학습.
F OOF의 B4는 기존 pre-warmstart B4를 유지합니다. 최종 재확인에서는 선택된 E HR을 고정했습니다.

지표는 이미지별 4개 클래스 IoU 평균을 먼저 구한 뒤 20장 평균입니다. union=0이면 해당 클래스 IoU=0입니다.
global mIoU는 계산하지 않습니다. valid20으로 여러 후보를 선택했으므로 독립 검증 성능이 아닙니다.
test10은 최종 모델 선택 이후 추론만 수행했으며 점수나 모델 선택에 사용하지 않았습니다.
새 패키지의 이미지부터 시작한 valid20 추론이 탐색 예측과 픽셀 단위로 모두 일치함을 확인했습니다.

선택 조합: `{selected['context']['choices']}`

실행 경로: `{run}`

완료한 후보의 체크포인트, 명령, 로그, 평가와 해시는 실행 경로에 보존됩니다.
`selected_model/`에는 엄격한 해시 검증을 지원하는 최종 모델과 추론 코드가 있습니다.
`valid_mask_comparison/`은 원본/정답색 오류/예측/정답의 2×2 비교입니다.

'''+'\n'.join(rows)+'\n'
    (result/(result.name+'.md')).write_text(report)
    write_json(result/'complete.json',dict(ok=True,training_count=100,image_miou=selected['image_miou'],
        baseline_image_miou=EXPECTED_SCORE,run=str(run),fresh_valid_pixel_exact=True,test_images=10))
    write_json(run/'latest_result.json',dict(result=str(result)))
    return dict(result=str(result),image_miou=selected['image_miou'],checkpoint_count=20)
