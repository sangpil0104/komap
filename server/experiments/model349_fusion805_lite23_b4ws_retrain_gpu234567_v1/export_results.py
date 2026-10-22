"""Same result folder and panels as the preceding experiment, for both stages."""
from pathlib import Path
import csv
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from experiment import ROOT,REPO,read_json,write_json
from features import decode
from report_utils import allocate_result,class_ious,comparison


def score_stage(run,stage,rows):
    upstream=read_json(run/stage/'metrics.json')
    if upstream['count']!=20 or len(upstream['images'])!=20:raise ValueError('Expected valid20 metrics.')
    expected={r['image']:r for r in upstream['images']}
    if set(expected)!={r['name'] for r in rows}:raise ValueError('Metric image inventory differs.')
    scores=[]
    for row in rows:
        name=row['name']
        with Image.open(run/stage/f'predicted_labels/{name}.png') as im:pred=np.asarray(im)
        gt=decode(ROOT/f'data/valid/masks/{name}_mask.png')
        ious=class_ious(pred,gt)
        if not np.allclose(ious,expected[name]['class_iou'],atol=1e-12,rtol=0):raise ValueError('Independent class IoU mismatch.')
        scores.append(dict(image=name,class_iou=ious,image_miou=float(np.mean(ious))))
    mean=float(np.mean([r['image_miou'] for r in scores]))
    if abs(mean-upstream['image_miou'])>1e-12:raise ValueError('Independent image mIoU mismatch.')
    return dict(metric='mean of per-image 4-class IoU; empty union=0',count=20,image_miou=mean,images=scores)


def export_stage(run,stage,destination,rows,palette,metrics):
    destination.mkdir(parents=True,exist_ok=True)
    panels=destination/'valid_mask_comparison';panels.mkdir()
    for row,score in tqdm(list(zip(rows,metrics['images'])),desc=f'Export {stage} comparisons',unit='image'):
        name=row['name']
        with Image.open(run/stage/f'predicted_labels/{name}.png') as im:pred=np.asarray(im)
        gt=decode(ROOT/f'data/valid/masks/{name}_mask.png')
        with Image.open(row['path']) as im:
            panel=comparison(im,pred,gt,palette,name,score['image_miou'])
        panel.save(panels/f'{name}.png')
    for folder in ['predicted_labels','predicted_masks']:shutil.copytree(run/stage/folder,destination/folder)
    write_json(destination/'metrics.json',metrics)
    with (destination/'per_image.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['image','image_miou_percent','100_minus_image_miou_percent','Al3Ni','Primary Si','Al','Eutectic Si'])
        for r in metrics['images']:writer.writerow([r['image'],r['image_miou']*100,(1-r['image_miou'])*100,*[x*100 for x in r['class_iou']]])


def export(run,results_root=REPO/'results'):
    run=Path(run)
    manifest=read_json(run/'release_inference/input_manifest.json');rows=manifest['images']
    if len(rows)!=20 or len({r['name'] for r in rows})!=20:raise ValueError('Require twenty unique validation images.')
    stages={s:score_stage(run,s,rows) for s in ['stage1','stage2']}
    test=run/'test_inference/fusion'
    if len(list((test/'predicted_masks').glob('*.png')))!=10:raise ValueError('Require test10 masks before export.')
    destination=allocate_result(Path(results_root))
    export_stage(run,'stage2',destination,rows,manifest['config']['palette'],stages['stage2'])
    export_stage(run,'stage1',destination/'stage1',rows,manifest['config']['palette'],stages['stage1'])
    shutil.copytree(test,destination/'test_predictions')
    shutil.copytree(run/'retrained_model',destination/'selected_model')
    shutil.copytree(run/'implementation',destination/'source_snapshot',
                    ignore=shutil.ignore_patterns('data','folds','expert_splits'))
    for filename in ['plan.json','environment.freeze.txt','events.jsonl','preflight.json']:
        shutil.copy2(run/filename,destination/filename)
    shutil.copy2(run/'implementation/provenance.json',destination/'provenance.json')
    shutil.copy2(run/'expert/refiner/completion.json',destination/'refiner_training_receipt.json')
    release=read_json(run/'release_inference/metrics.json')['image_miou']
    summary=dict(release_image_miou=release,stage1_image_miou=stages['stage1']['image_miou'],
                 stage2_image_miou=stages['stage2']['image_miou'],training_count=37,deployed_weights=23,
                 primary_stage='stage2',selection_rule='Requested lite23 B4ws fixed recipe; no score-based reselection')
    write_json(destination/'fusion_comparison.json',summary)
    lines=['# Model349 fusion805 lite23 B4ws — 전체 재학습 결과','',
        f"대표 결과: **lite23 B4ws R+E+F image mIoU {stages['stage2']['image_miou']*100:.6f}%**.",'',
        '| 구성 | valid20 image mIoU |','|---|---:|',
        f'| 새로 학습한 R (exh14) | {release*100:.6f}% |',
        f"| 1단계 R+E (비교) | {stages['stage1']['image_miou']*100:.6f}% |",
        f"| 2단계 R+E+F (요청 모델) | {stages['stage2']['image_miou']*100:.6f}% |",'',
        'GitHub lite23 B4ws 고정 설정, S=0으로 총 37개 학습 작업을 완료했습니다.',
        '선행 Model6·62·97·98·194·195·239·242·257도 새로 학습했습니다. 기존 학습 가중치를 시작점으로 재사용하지 않았습니다.',
        '공개 ImageNet encoder 초기화에서 시작하며, R과 E의 동일 B4-329 fold 1·3·4는 한 번 학습한 가중치를 공유합니다.',
        'E는 원래 B4 4-fold와 HRNet 2 seeds × 4-fold를 학습하고, B4 4개를 같은 fold train으로 2,000회 추가 학습했습니다.',
        'B4 추가 학습은 seed 819349, lr 3e-5, BN running 통계 고정입니다. 각 fold에 통계 불변 검사 결과가 있습니다.',
        '최종 E는 B4ws 0.5 + HRNet seed1 fold2·4 평균 0.25 + seed2 fold1·4 평균 0.25입니다.',
        'F는 추가 학습 전 E 12개 모델의 train70 group OOF만으로 8 epochs/1,120 updates 학습했습니다.',
        '최종 저장 가중치는 R14 + B4ws4 + HR4 + F1 = 23개입니다. 나머지 중간 가중치는 실행 폴더에 보존됩니다.','',
        '지표는 각 이미지의 네 클래스 IoU 평균을 20장에 대해 평균하며 빈 클래스 IoU=0입니다. global mIoU는 계산하지 않습니다.',
        '원본 조합 개발과 R checkpoint 선택에는 valid20이 쓰였습니다. 이 점수는 독립 test 성능이 아닙니다.',
        '프로젝트 지침에 맞춰 과거 일부 R 학습기의 absent-class 제외 선택 지표도 image mIoU(empty=0)로 통일했습니다.',
        'GitHub의 공개 점수 80.5464316%는 이전 고정 가중치의 개발 결과입니다. 위 결과는 이번 새 가중치의 실측값입니다.',
        'test10은 정답이 없어 예측 마스크만 저장했으며 test mIoU를 계산하지 않았습니다.','',
        'GPU: 물리 2,3,4,5,6,7 / conda komap.',f'실행 원본·로그·모든 학습 결과: `{run}`','',
        '`valid_mask_comparison/`: 요청 모델의 20장. `stage1/valid_mask_comparison/`: F 적용 전 비교 20장.',
        '좌상 원본 / 우상 오류 / 좌하 예측 / 우하 정답. 맞은 픽셀은 흰색, 오류 픽셀은 정답 색.',
        '각 이미지 위에 image mIoU와 100%−image mIoU를 표시했습니다.',
        '`selected_model/`: 23개 새 가중치·추론 코드·SHA-256 manifest.',
        '`test_predictions/`: test10 예측. `source_snapshot/`: 실제 학습·실행 코드 사본.','',
        '| 이미지 | 1단계 image mIoU | 2단계 image mIoU |','|---|---:|---:|',
        *[f"| {a['image']} | {a['image_miou']*100:.6f}% | {b['image_miou']*100:.6f}% |" for a,b in zip(stages['stage1']['images'],stages['stage2']['images'])],'']
    (destination/f'{destination.name}.md').write_text('\n'.join(lines))
    (destination/'stage1/stage1_results.md').write_text(f"# F 적용 전 비교\n\nimage mIoU: **{stages['stage1']['image_miou']*100:.6f}%** (valid20, 빈 클래스=0).\n")
    write_json(destination/'complete.json',dict(ok=True,training_count=37,deployed_weights=23,primary_stage='stage2',stage1_panels=20,stage2_panels=20,test_images=10,source_run=str(run)))
    return destination
