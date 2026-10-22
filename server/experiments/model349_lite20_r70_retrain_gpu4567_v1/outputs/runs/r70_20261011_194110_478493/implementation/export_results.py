"""Paired image-first metrics and project-standard result export; no test scoring."""
from pathlib import Path
import csv,shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from common import ROOT,REPO,BASE,read_json,write_json
from preflight import PALETTE,decode
from report_utils import class_ious,comparison,allocate_result

def names(split):return sorted(p.stem.removesuffix('_image') for p in (ROOT/f'data/{split}/images').glob('*_image.png'))

def labels(prob):return np.asarray([2,0,1,3],np.uint8)[prob[[2,0,1,3]].argmax(0)]

def score_variant(inference,output):
    rows={s:[] for s in ['R','stage1','stage2']};expected=names('valid')
    assert len(expected)==20
    for folder in [inference/'release/predicted_labels',inference/'fusion/predicted_labels']:
        assert {p.stem for p in folder.glob('*.png')}==set(expected)
    for name in tqdm(expected,desc='Score R / R+E / R+E+F',unit='image',mininterval=5):
        r=np.asarray(Image.open(inference/f'release/predicted_labels/{name}.png'))
        final=np.asarray(Image.open(inference/f'fusion/predicted_labels/{name}.png'))
        p=np.load(inference/f'expert/{name}.npy');assert p.dtype==np.float32 and np.isfinite(p).all()
        expert=labels(p);stage1=np.where(np.isin(expert,[0,2]),expert,r).astype(np.uint8)
        target=decode(ROOT/f'data/valid/masks/{name}_mask.png')
        for stage,pred in [('R',r),('stage1',stage1),('stage2',final)]:
            ious=class_ious(pred,target);score=float(np.mean(ious))
            rows[stage].append(dict(image=name,class_iou=ious,image_miou=score))
            destination=output/stage;destination.mkdir(parents=True,exist_ok=True)
            for d in ['predicted_labels','predicted_masks','valid_mask_comparison']:(destination/d).mkdir(exist_ok=True)
            Image.fromarray(pred).save(destination/f'predicted_labels/{name}.png')
            Image.fromarray(PALETTE[pred]).save(destination/f'predicted_masks/{name}_mask.png')
            with Image.open(ROOT/f'data/valid/images/{name}_image.png') as original:
                comparison(original,pred,target,PALETTE,name,score).save(destination/f'valid_mask_comparison/{name}.png')
    result={}
    for stage,images in rows.items():
        result[stage]=dict(metric='mean image-wise four-class IoU; empty union=0',count=20,
            image_miou=float(np.mean([r['image_miou'] for r in images])),images=images)
        write_json(output/stage/'metrics.json',result[stage])
        with (output/stage/'per_image.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['image','image_miou_percent','100_minus_image_miou_percent','Al3Ni','Primary Si','Al','Eutectic Si'])
            for r in images:writer.writerow([r['image'],r['image_miou']*100,(1-r['image_miou'])*100,*[v*100 for v in r['class_iou']]])
    return result

def paired_metrics(r70,control):
    a={r['image']:r['image_miou'] for r in r70['images']};b={r['image']:r['image_miou'] for r in control['images']}
    assert set(a)==set(b) and len(a)==20
    delta={k:100*(a[k]-b[k]) for k in sorted(a)};groups={}
    for name,value in delta.items():groups.setdefault(name.split('_')[0],[]).append(value)
    keys=list(groups);rng=np.random.default_rng(81795);boots=[]
    for _ in range(20000):
        picked=rng.integers(0,len(keys),len(keys));boots.append(float(np.mean([v for i in picked for v in groups[keys[i]]])))
    lo,hi=np.percentile(boots,[2.5,97.5]);worst=min(delta,key=delta.get)
    return dict(lite20_r70_image_miou=r70['image_miou'],lite20_image_miou=control['image_miou'],
        delta_pp=100*(r70['image_miou']-control['image_miou']),specimen_bootstrap95_pp=[float(lo),float(hi)],
        bootstrap_seed=81795,bootstrap_samples=20000,improved=sum(v>0 for v in delta.values()),
        worse=sum(v<0 for v in delta.values()),unchanged=sum(v==0 for v in delta.values()),
        worst_image=worst,worst_delta_pp=delta[worst],per_image_delta_pp=delta)

def test_difference(a,b):
    expected=set(names('test'));assert len(expected)==10
    assert {p.stem for p in a.glob('*.png')}=={p.stem for p in b.glob('*.png')}==expected
    count=total=0
    for name in sorted(expected):
        x=np.asarray(Image.open(a/f'{name}.png'));y=np.asarray(Image.open(b/f'{name}.png'))
        assert x.shape==y.shape and set(np.unique(x))<=set(range(4)) and set(np.unique(y))<=set(range(4))
        count+=int(np.count_nonzero(x!=y));total+=x.size
    return dict(changed_pixels=count,total_pixels=total,changed_percent=100*count/total,images=10,gt_read=False)

def export(run,attempt,done,results_root=REPO/'results'):
    metrics={};inferences={}
    for variant in ['lite20_r70','lite20']:
        inference=Path(done[f'{variant}_valid']['inference']);inferences[variant]=inference
        metrics[variant]=score_variant(inference,attempt/'scored'/variant)
    # The paired experiment changes R only; E/F assets and family predictions must remain identical.
    expert_differences={}
    for name in names('valid'):
        a=np.load(inferences['lite20_r70']/f'expert/{name}.npy');b=np.load(inferences['lite20']/f'expert/{name}.npy')
        assert a.shape==b.shape
        expert_differences[name]=float(np.max(np.abs(a-b)))
        if not np.array_equal(a,b):raise ValueError(f'Fixed E predictions changed between variants: {name}')
    pair={s:paired_metrics(metrics['lite20_r70'][s],metrics['lite20'][s]) for s in ['R','stage1','stage2']}
    tests={v:Path(done[f'{v}_test']['inference'])/'fusion' for v in ['lite20_r70','lite20']}
    changes=test_difference(tests['lite20_r70']/'predicted_labels',tests['lite20']/'predicted_labels')
    result=allocate_result(Path(results_root))
    for name in ['valid_mask_comparison','predicted_labels','predicted_masks']:
        shutil.copytree(attempt/'scored/lite20_r70/stage2'/name,result/name)
    for file in ['metrics.json','per_image.csv']:shutil.copy2(attempt/'scored/lite20_r70/stage2'/file,result/file)
    shutil.copytree(attempt/'scored/lite20_r70/R',result/'r_only')
    shutil.copytree(attempt/'scored/lite20_r70/stage1',result/'stage1')
    shutil.copytree(attempt/'scored/lite20',result/'baseline_lite20')
    shutil.copytree(tests['lite20_r70'],result/'test_predictions')
    shutil.copytree(tests['lite20'],result/'baseline_lite20/test_predictions')
    shutil.copytree(done['package']['models']['lite20_r70'],result/'selected_model')
    shutil.copytree(ROOT,result/'source_snapshot',ignore=shutil.ignore_patterns('data','train_data','checks','logs','outputs','upstream','downloads','__pycache__','*.pyc'))
    for name in ['plan.json','environment.freeze.txt','events.jsonl','preflight.json','implementation_manifest.json']:
        shutil.copy2(run/name,result/name)
    write_json(result/'paired_comparison.json',dict(metrics=pair,test_mask_difference=changes,
        fixed_expert_predictions_equal=True,baseline_previous_REF_image_miou=.8025584552762022,
        baseline_replay_delta_pp=100*(metrics['lite20']['stage2']['image_miou']-.8025584552762022)))
    write_json(result/'provenance.json',read_json(ROOT/'provenance.json'))
    summary=pair['stage2'];lo,hi=summary['specimen_bootstrap95_pp']
    lines=['# Model349 lite20_r70 — train70 B4 실험 결과','',
        f"대표 valid20 image mIoU: **{metrics['lite20_r70']['stage2']['image_miou']*100:.6f}%**",'',
        '| 구성 | lite20 기준 | lite20_r70 | 변화 (%p) |','|---|---:|---:|---:|',
        *[f"| {label} | {pair[s]['lite20_image_miou']*100:.6f}% | {pair[s]['lite20_r70_image_miou']*100:.6f}% | {pair[s]['delta_pp']:+.6f} |" for s,label in [('R','R 단독'),('stage1','R+E'),('stage2','R+E+F')]],'',
        '기준은 model_results_20261011_1의 lite20 S=0 가중치입니다. R의 B4-322/326/329만 ImageNet 초기화에서 새로 학습했습니다.',
        '세 모델 모두 train70 전체, seed0, 21,000 iterations, batch2, crop1536, lr3e-4, wd1e-2, CE+Dice, 마지막 checkpoint를 사용했습니다.',
        '322는 gray3/imagenet/scale3/area, 326은 zcdf3/image_global/scale3/area, 329는 zcdf3/image_global/target_width4000/uniform입니다.',
        'worker 및 Albumentations 시드를 고정했습니다. 새 B4는 valid/test를 읽거나 checkpoint 선택에 사용하지 않았습니다.',
        '그 외 R5+E8+F1=14개 가중치와 결합 규칙은 그대로 유지했습니다. r70은 R8+E8+F1=17개, 기준 lite20은 20개입니다.',
        '두 구성 모두 원본 이미지부터 새로 추론했습니다. E 확률의 픽셀 단위 일치 및 14개 공통 가중치의 SHA256 일치를 확인했습니다.',
        '시드 탐색 결과는 사용하지 않았습니다. 두 구성 중 점수가 높은 쪽으로 재선택하지 않고 요청한 r70 결과를 대표 결과로 저장했습니다.','',
        f"최종 차이의 시편 단위 bootstrap 95% 구간: [{lo:+.6f}, {hi:+.6f}] %p (20,000회, seed81795).",
        f"이미지 개선 {summary['improved']}장 / 악화 {summary['worse']}장 / 동일 {summary['unchanged']}장; 최악 {summary['worst_image']} ({summary['worst_delta_pp']:+.6f}%p).",
        '단일 시드의 개발용 비교이며, 이 구간이 독립 test 검증을 대신하지 않습니다.',
        f"test10 마스크 차이: {changes['changed_pixels']:,}/{changes['total_pixels']:,} pixels ({changes['changed_percent']:.6f}%). test 점수는 계산하지 않았습니다.",'',
        '지표는 이미지별 네 클래스 IoU의 평균을 valid20에 대해 평균합니다. union=0인 클래스는 IoU=0입니다. global mIoU는 계산하지 않습니다.',
        'GPU 물리4~7 / conda komap / nohup. 실제 학습은3개 작업이며 inference는4개 GPU 작업으로 분산했습니다.',
        f'실행 기록: `{run}`','',
        '`valid_mask_comparison/`: r70 최종 결과. `r_only/`: R 단독. `stage1/`: R+E. `baseline_lite20/`: 동일 가중치 기준 비교.',
        '2×2 배치는 좌상 원본/우상 정답색 오류(맞은 픽셀 흰색)/좌하 예측/우하 정답이며, image mIoU 및100%-image mIoU를 표시합니다.',
        '`selected_model/`: 17개 가중치와 엄격한 검증기 및 추론 코드. 후보·기준 모델과 학습 로그는 실행 경로에 보존됩니다.','',
        '| 이미지 | 기준 최종 (%) | r70 최종 (%) | 차이 (%p) |','|---|---:|---:|---:|']
    a={r['image']:r['image_miou'] for r in metrics['lite20_r70']['stage2']['images']}
    b={r['image']:r['image_miou'] for r in metrics['lite20']['stage2']['images']}
    lines += [f'| {n} | {100*b[n]:.6f} | {100*a[n]:.6f} | {100*(a[n]-b[n]):+.6f} |' for n in sorted(a)]
    (result/(result.name+'.md')).write_text('\n'.join(lines)+'\n')
    write_json(result/'complete.json',dict(ok=True,training_count=3,deployed_weights=17,baseline_weights=20,
        image_miou=metrics['lite20_r70']['stage2']['image_miou'],source_run=str(run),valid_images=20,test_images=10))
    write_json(run/'latest_result.json',dict(result=str(result)))
    return result
