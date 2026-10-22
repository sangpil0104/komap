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
    destination=allocate_result(Path(results_root))
    export_stage(run,'stage1',destination,rows,manifest['config']['palette'],stages['stage1'])
    export_stage(run,'stage2',destination/'stage2',rows,manifest['config']['palette'],stages['stage2'])
    shutil.copytree(run/'retrained_model',destination/'selected_model')
    shutil.copytree(run/'implementation',destination/'source_snapshot',
                    ignore=shutil.ignore_patterns('data','folds','initial_weights','expert_splits'))
    for filename in ['plan.json','environment.freeze.txt','events.jsonl','preflight.json']:
        shutil.copy2(run/filename,destination/filename)
    shutil.copy2(run/'implementation/provenance.json',destination/'provenance.json')
    shutil.copy2(run/'expert/refiner/completion.json',destination/'refiner_training_receipt.json')
    release=read_json(run/'release_inference/metrics.json')['image_miou']
    summary=dict(release_image_miou=release,stage1_image_miou=stages['stage1']['image_miou'],
                 stage2_image_miou=stages['stage2']['image_miou'],training_count=27,
                 primary_stage='stage1',selection_rule='Fixed by upstream documentation; not chosen using this run score')
    write_json(destination/'fusion_comparison.json',summary)
    lines=['# Model349 fusion805 — 구성 모델 재학습 결과','',
        f"대표 결과: **1단계 R+E image mIoU {stages['stage1']['image_miou']*100:.6f}%**.",'',
        '| 구성 | valid20 image mIoU |','|---|---:|',
        f'| 새로 학습한 R (exh14) | {release*100:.6f}% |',
        f"| 1단계 R+E (대표) | {stages['stage1']['image_miou']*100:.6f}% |",
        f"| 2단계 R+E+F (보존용 비교) | {stages['stage2']['image_miou']*100:.6f}% |",'',
        'R 14개 + E 12개 + F 1개 = 총 27개를 각각 새로 학습했습니다.',
        'R은 보존된 Model98·257에서 시작하는 미세조정 및 새 부모 연결을 이전 실험과 같이 유지했습니다.',
        'E는 ImageNet에서 B4 4-fold, HRNet 2 seeds × 4-fold를 학습했습니다.',
        'F는 새 E 모델의 train70 group OOF만으로 8 epochs/1,120 updates를 학습했습니다. valid/test 정답을 학습에 넣지 않았습니다.','',
        '지표는 각 이미지의 네 클래스 IoU 평균을 20장에 대해 평균하며 빈 클래스 IoU=0입니다.',
        '배포 문서에 따라 1단계를 대표로 기록합니다. 이번 점수로 단계를 다시 고르지 않았습니다.',
        '원본 fusion805 조합 개발과 R의 checkpoint 선택에 valid20이 쓰였으므로 독립 test 성능이 아닙니다.',
        'GitHub 공개 가중치·점수의 동일 재현을 주장하지 않습니다. 위 값은 이번 새 가중치의 실제 검증값입니다.','',
        'GPU: 물리 2,3,4,5,6,7 / conda komap.',f'실행 원본·로그·모든 학습 결과: `{run}`','',
        '`valid_mask_comparison/`: 1단계 20장. `stage2/valid_mask_comparison/`: 2단계 20장.',
        '좌상 원본 / 우상 오류 / 좌하 예측 / 우하 정답. 정답 위치는 흰색, 오류 위치는 정답 색.',
        '각 이미지 위에 image mIoU와 100%−image mIoU를 표시했습니다.',
        '`selected_model/`: 새 가중치 27개와 실행 소스, 새 SHA-256 manifest.',
        '`source_snapshot/`: 실제 학습·실행 코드 사본.','',
        '| 이미지 | 1단계 image mIoU | 2단계 image mIoU |','|---|---:|---:|',
        *[f"| {a['image']} | {a['image_miou']*100:.6f}% | {b['image_miou']*100:.6f}% |" for a,b in zip(stages['stage1']['images'],stages['stage2']['images'])],'']
    (destination/f'{destination.name}.md').write_text('\n'.join(lines))
    (destination/'stage2/stage2_results.md').write_text(f"# Fusion805 2단계 보존용 비교\n\nimage mIoU: **{stages['stage2']['image_miou']*100:.6f}%** (valid20, 빈 클래스=0).\n\n같은 실행의 R+E+F 결과이며 대표 후보는 상위 보고서의 1단계 R+E입니다.\n")
    write_json(destination/'complete.json',dict(ok=True,training_count=27,stage1_panels=20,stage2_panels=20,source_run=str(run)))
    return destination
