"""Export measured image mIoU, inference timing, panels, and the exact 13 trained weights."""
from pathlib import Path
import argparse, csv, shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from common import ROOT, REPO, read, write, sha, now
from report_utils import allocate_result, class_ious, comparison

PALETTE=np.array([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],dtype=np.uint8)

def decode(path):
    with Image.open(path) as im:
        a=np.asarray(im)
        if a.ndim==2 and np.isin(a,range(4)).all():return a.copy()
        rgb=np.asarray(im.convert('RGB'))
    target=np.full(rgb.shape[:2],255,dtype=np.uint8)
    for c,color in enumerate(PALETTE):target[np.all(rgb==color,axis=-1)]=c
    if not np.isin(target,range(4)).all():raise ValueError(f'Invalid palette {path}')
    return target

def score_and_panels(data,labels,destination):
    panels=destination/'valid_mask_comparison';panels.mkdir(parents=True)
    files=sorted((data/'valid/images').glob('*_image.png'))
    names={p.stem.removesuffix('_image') for p in files}
    if len(names)!=20 or {p.stem for p in labels.glob('*.png')}!=names:raise ValueError('Expected exactly valid20 predictions')
    rows=[]
    for p in tqdm(files,desc='Valid comparison panels',unit='image',mininterval=10):
        name=p.stem.removesuffix('_image');gt=decode(data/f'valid/masks/{name}_mask.png')
        with Image.open(labels/f'{name}.png') as im:pred=np.asarray(im)
        vals=class_ious(pred,gt);score=float(np.mean(vals))
        rows.append(dict(image=name,class_iou=vals,image_miou=score))
        with Image.open(p) as im:panel=comparison(im,pred,gt,PALETTE,name,score)
        panel.save(panels/f'{name}.png')
    metrics=dict(metric='mean over 20 per-image four-class IoUs; empty union=0',count=20,
                 image_miou=float(np.mean([r['image_miou'] for r in rows])),images=rows)
    write(destination/'metrics.json',metrics)
    with (destination/'per_image.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image','image_miou_percent','100_minus_image_miou_percent','Al3Ni','Primary Si','Al','Eutectic Si'])
        for r in rows:w.writerow([r['image'],r['image_miou']*100,(1-r['image_miou'])*100,*[x*100 for x in r['class_iou']]])
    return metrics

def package(run,destination):
    model=destination/'selected_model';model.mkdir()
    shutil.copytree(run/'implementation/source',model/'code')
    shutil.copytree(run/'model/weights',model/'weights',symlinks=False)
    checkpoints=list((model/'weights').rglob('*.pt'))
    if len(checkpoints)!=13:raise ValueError('Package must contain exactly 13 weights')
    files={str(p.relative_to(model)):sha(p) for p in sorted(model.rglob('*')) if p.is_file()}
    write(model/'files_sha256.json',files)
    (model/'predict.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
pkg_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/sppark/miniconda3/etc/profile.d/conda.sh
conda activate komap
python - "$pkg_root" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1])
for name,digest in json.loads((r/'files_sha256.json').read_text()).items():
 h=hashlib.sha256()
 with (r/name).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 if h.hexdigest()!=digest:raise ValueError(name)
PY
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ALLOW_RETRAINED=1
exec bash "$pkg_root/code/run_predict.sh" "$pkg_root/weights" "$1" "$2" "${3:-3,4,5,6}"
''')
    (model/'predict.sh').chmod(0o755)
    (model/'README.md').write_text('# lite13 retuned 재학습 가중치\n\nR7 + E5 + F1 = 13개. `files_sha256.json`은 이번 가중치/소스의 실측 해시입니다.\n\n`bash predict.sh IMAGE_DIR NEW_OUTPUT_DIR 3,4,5,6`\n\n기존 공개 가중치의 해시 대신 이 패키지 manifest를 검증한 후 추론합니다.\n')
    return files

def export(run,results_root=REPO/'results'):
    run=Path(run);completion=read(run/'pipeline_completion.json')
    if not completion['ok']:raise ValueError('Pipeline incomplete')
    plan=read(run/'implementation/plan.json');receipts={p.stem:read(p) for p in (run/'stages').glob('*.json')}
    expected={r['name'] for r in plan['tasks']}|{'score_valid'}
    if set(receipts)!=expected or any(r['status']!='done' for r in receipts.values()):raise ValueError('Stage completion mismatch')
    timings={s:read(run/f'pred_{s}/inference_timing.json') for s in ['valid','test']}
    for s,n in [('valid',20),('test',10)]:
        t=timings[s]
        if t['images']!=n or t['wall_seconds']<=0 or abs(t['seconds_per_image']-t['wall_seconds']/n)>1e-9:raise ValueError('Invalid timing')
        names={p.stem.removesuffix('_image') for p in (ROOT/f'data/{s}/images').glob('*_image.png')}
        if {p.stem for p in (run/f'pred_{s}/fusion/predicted_masks').glob('*.png')}!=names:raise ValueError('Missing prediction')
    result=allocate_result(Path(results_root))
    metrics=score_and_panels(ROOT/'data',run/'pred_valid/fusion/predicted_labels',result)
    prior=read(run/'pred_valid/metrics.json')
    if abs(metrics['image_miou']-prior['image_miou'])>1e-12:raise ValueError('Independent score mismatch')
    originals={r['image']:r['class_iou'] for r in prior['images']}
    if any(not np.allclose(r['class_iou'],originals[r['image']],rtol=0,atol=1e-12) for r in metrics['images']):raise ValueError('Class score mismatch')
    for folder in ['predicted_labels','predicted_masks']:shutil.copytree(run/'pred_valid/fusion'/folder,result/folder)
    shutil.copytree(run/'pred_test/fusion',result/'test_predictions')
    write(result/'inference_timing.json',timings)
    with (result/'inference_timing.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['split','images','wall_seconds','seconds_per_image','images_per_second'])
        for s,t in timings.items():w.writerow([s,*[t[k] for k in ['images','wall_seconds','seconds_per_image','images_per_second']]])
    package(run,result)
    shutil.copytree(run/'implementation',result/'source_snapshot',ignore=shutil.ignore_patterns('data'))
    shutil.copytree(run/'stages',result/'stage_receipts')
    for name in ['preflight.json','environment.freeze.txt','pipeline_completion.json']:shutil.copy2(run/name,result/name)
    lines=['# Model349 lite13 retuned — 전체 재학습 결과','',
           f'valid20 image mIoU: **{metrics["image_miou"]*100:.6f}%**','',
           '최종 가중치 R7 + E5 + F1 = 13개. 선행/OOF를 포함한 학습 작업 33개를 완료했습니다.',
           'GPU 물리 3·4·5·6, conda komap. R 확률 결합과 E 비율은 GitHub retuned 설정을 고정했습니다.',
           '지표: 이미지별 네 클래스 IoU 평균을 valid20에 대해 평균합니다. 빈 클래스 IoU=0이며 global mIoU는 계산하지 않습니다.',
           '과거 R의 빈 클래스 제외 checkpoint 선택도 같은 지표로 통일했습니다. 원본과 다른 checkpoint가 선택될 수 있습니다.',
           '공개 80.779378%는 원래 가중치의 참조 점수이며 이번 재학습 결과가 아닙니다. valid는 일부 checkpoint 선택과 원본 확률 비율 조정에 사용되었습니다.',
           'test10은 정답 없이 추론만 수행했으므로 점수가 없습니다.','',
           '## 실측 추론시간','',
           '| 데이터 | 이미지 수 | 전체 시간(초) | 장당 평균(초) | 처리량(장/초) |',
           '|---|---:|---:|---:|---:|',
           *[f'| {s} | {t["images"]} | {t["wall_seconds"]:.3f} | {t["seconds_per_image"]:.3f} | {t["images_per_second"]:.5f} |' for s,t in timings.items()], '',
           '측정 범위: 준비, 프로세스 시작, 모델 로딩, 전처리, R(TTA/TENT)·E·F, 후처리, 확률/마스크 저장과 임시 폴더 정리까지입니다. 학습·정답 채점·보고서 그림 생성 시간은 제외했습니다.',
           '별도 warm-up이나 저장 예측 재사용 없이 각 split을 한 번 측정했습니다. 장당 시간은 전체 시간을 이미지 수로 나눈 처리시간이며 단일 이미지 지연시간은 아닙니다.',
           'R 구성요소는 GPU 3~6에 분배하며 E는 GPU 3, F는 CPU에서 실행합니다. R/E/F별 시간과 GPU UUID·이름은 inference_timing.json에 있습니다.',
           f'전체 학습·추론 파이프라인 경과시간(보고서 생성 제외): {completion["wall_seconds"]:.1f}초.','',
           '## 결과 파일','',
           '`valid_mask_comparison/`: 20장, 좌상 원본 / 우상 정답색 오류(맞은 픽셀 흰색) / 좌하 예측 / 우하 정답. image mIoU와 100%−image mIoU를 표시합니다.',
           '`selected_model/`: 새 13개 가중치, 코드, 설정, 해시 검증 추론 실행기.',
           '`inference_timing.json`, `.csv`: 실측 추론시간. `source_snapshot/`: 실제 실행 소스.',
           f'모든 중간 모델과 상세 로그: `{run}`','',
           '| 이미지 | image mIoU | 100%−image mIoU |','|---|---:|---:|',
           *[f'| {r["image"]} | {r["image_miou"]*100:.6f}% | {(1-r["image_miou"])*100:.6f}% |' for r in metrics['images']],'']
    (result/f'{result.name}.md').write_text('\n'.join(lines))
    write(result/'complete.json',dict(ok=True,source_run=str(run),weights=13,training_count=33,finished_at=now(),image_miou=metrics['image_miou']))
    write(run/'result.json',dict(result=str(result)));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();print(export(a.run))
