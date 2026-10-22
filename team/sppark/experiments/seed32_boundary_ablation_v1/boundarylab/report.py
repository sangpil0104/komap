import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io import read_json, sha256, write_json
from .metrics import CLASSES, PALETTE, summarize


def comparison_panel(original, target, prediction, score, image_id):
    original = np.asarray(original.convert('RGB')) if isinstance(original, Image.Image) else original
    h, w = target.shape
    if original.shape != (h,w,3):
        raise ValueError('Panel must preserve original image resolution')
    error = np.full((h,w,3), 255, np.uint8)
    wrong = prediction != target
    error[wrong] = PALETTE[target[wrong]]
    gap, header, caption = 24, 108, 44
    canvas = Image.new('RGB', (2*w+gap, 2*(h+caption)+gap+header), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 28)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    draw.text((10,4), image_id, font=small, fill='black')
    draw.text((10,36), f'image mIoU: {100*score:.4f}%', font=font, fill='black')
    draw.text((10,71), f'100% - image mIoU: {100*(1-score):.4f}%', font=font, fill='black')
    panels = [(original, 'Original image', 0, header),
              (error, 'Errors: ground-truth color', w+gap, header),
              (PALETTE[prediction], 'Predicted mask', 0, header+h+caption+gap),
              (PALETTE[target], 'Ground-truth mask', w+gap, header+h+caption+gap)]
    for array, title, x, y in panels:
        draw.text((x+8,y+5), title, font=small, fill='black')
        canvas.paste(Image.fromarray(array), (x,y+caption))
    return canvas


def finish_report(path, arm, rows, provenance):
    path = Path(path)
    summary = summarize(rows)
    summary.update(experiment=arm, provenance=provenance)
    write_json(path/'summary.json', summary, exclusive=True)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['image_id','image_miou','complement',*CLASSES,'boundary_error_1px','boundary_error_2px','boundary_error_3px'])
    for row in rows:
        writer.writerow([row['image_id'],row['image_miou'],1-row['image_miou'],*row['class_iou'],
                         *[row['boundary_error'][str(r)] for r in (1,2,3)]])
    (path/'per_image.csv').write_text(buf.getvalue())
    cfg = arm['config']
    lines = [f'# {arm["id"]} — {arm["description"]}', '',
             f'**image mIoU: {summary["image_miou"]*100:.6f}%** (valid {len(rows)} images, empty union=0).', '',
             '이미지별 4개 클래스 IoU 평균을 구한 뒤 이미지별 동일 가중치로 평균합니다. Global mIoU는 계산하지 않습니다.', '',
             f'- 비교 대조군: `{arm["contrast"] or "reference / control"}`',
             f'- 변경 요소: `{arm["change"]}`',
             f'- 초기화: `{arm["initialization"]}`',
             f'- 학습 방식: {"ImageNet 인코더 + seed 32로 새로 초기화한 디코더; 다른 분할 체크포인트를 이어 학습하지 않음" if arm["kind"] == "train" else "보존된 seed 32의 추론 전용 참고 결과; 학습 없음"}',
             f'- 체크포인트 선택: 고정 마지막 update; valid 성능에 따른 checkpoint 선택 없음',
             f'- crop={cfg["crop"]}, updates={cfg["steps"]}, global batch={cfg["global_batch"]}, D4={cfg["tta"]}',
             f'- provenance: `provenance.json` / weights SHA-256: `{provenance.get("weights_sha256")}`', '',
             '각 비교 사진은 좌상 원본, 우상 오류(정답 색), 좌하 예측, 우하 정답입니다. 맞은 픽셀은 흰색입니다.',
             '`100% − image mIoU`는 픽셀 오류율이 아닙니다. 경계 띠 오류율은 원본 픽셀 단위 진단값입니다.', '',
             '| 이미지 | image mIoU | 100%−image mIoU | 비교 |', '|---|---:|---:|---|']
    for row in rows:
        name = row['image_id']
        lines.append(f'| {name} | {row["image_miou"]*100:.4f}% | {(1-row["image_miou"])*100:.4f}% | [보기](valid_mask_comparison/{name}.png) |')
    report = path/(path.name+'.md')
    report.write_text('\n'.join(lines)+'\n')
    write_json(path/'provenance.json', provenance, exclusive=True)
    return summary


def result_manifest(path):
    path = Path(path)
    return {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob('*')) if p.is_file() and p.name != 'complete.json'}


def verify_result(path, expected_ids=None):
    path = Path(path)
    marker = read_json(path/'complete.json')
    for name, digest in marker['files'].items():
        if sha256(path/name) != digest:
            raise ValueError(f'Result changed: {path/name}')
    summary = read_json(path/'summary.json')
    ids = [r['image_id'] for r in summary['rows']]
    if len(ids) != 20 or len(set(ids)) != 20 or (expected_ids is not None and set(ids) != set(expected_ids)):
        raise ValueError('Result is not the official valid20')
    if summary['empty_union'] != 0:
        raise ValueError('Wrong empty-union policy')
    for name in ids:
        if not (path/'valid_mask_comparison'/f'{name}.png').is_file() or not (path/'predictions'/f'{name}.png').is_file():
            raise ValueError(f'Missing image artifact {name}')
    return summary


def suite_report(suite):
    suite = Path(suite)
    plan = read_json(suite/'plan.json')
    completed = {}
    for arm in plan['arms']:
        p = suite/'trials'/arm['id']/'complete.json'
        if p.is_file():
            marker = read_json(p)
            completed[arm['id']] = (marker, read_json(Path(marker['result'])/'summary.json'))
    lines = ['# Seed 32 구조 기반 단일 요소 제거·추가 실험', '',
             '학습 실험은 모두 ImageNet 인코더 + 새 디코더로 시작합니다. seed 32 분할 가중치는 R 실험의 추론에만 사용합니다.',
             '주 지표: valid20 image mIoU, 네 클래스 고정 평균, 빈 클래스 IoU=0.',
             'T 실험은 유효 배치 8 / SyncBatchNorm의 공통 조건입니다. 원본 seed32(배치2) 점수는 참고값이며 직접적인 단일 요소 대조군이 아닙니다.', '',
             '| 실험 | 변경 | 대조군 | image mIoU | 대조군 대비 %p | 결과 |', '|---|---|---|---:|---:|---|']
    out = io.StringIO()
    csvout = csv.writer(out)
    csvout.writerow(['id','kind','contrast','image_miou','delta_percentage_points','result'])
    for arm in plan['arms']:
        if arm['id'] not in completed:
            lines.append(f'| {arm["id"]} | {arm["description"]} | {arm["contrast"] or "—"} | 미완료 | — | — |')
            continue
        marker, summary = completed[arm['id']]
        score = summary['image_miou']
        control = completed.get(arm['contrast'])
        delta = 100*(score-control[1]['image_miou']) if control else None
        report = Path(marker['result'])/(Path(marker['result']).name+'.md')
        lines.append(f'| {arm["id"]} | {arm["description"]} | {arm["contrast"] or "—"} | {100*score:.6f}% | {f"{delta:+.6f}" if delta is not None else "—"} | [보고서]({report}) |')
        csvout.writerow([arm['id'],arm['kind'],arm['contrast'],score,delta,str(report)])
    (suite/'comparison.md').write_text('\n'.join(lines)+'\n')
    (suite/'comparison.csv').write_text(out.getvalue())
    return len(completed)
