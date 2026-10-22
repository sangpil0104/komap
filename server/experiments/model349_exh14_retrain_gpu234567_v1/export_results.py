"""Image-first four-class score and exact 2x2 validation comparison export."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import csv
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from experiment import ROOT, REPO, VALID, inference, read_json, write_json


def class_ious(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError('Prediction and ground truth must have equal H,W shapes.')
    if not (np.isin(prediction, range(4)).all() and np.isin(target, range(4)).all()):
        raise ValueError('Expected class labels 0..3.')
    values = []
    for class_id in range(4):
        intersection = np.count_nonzero((prediction == class_id) & (target == class_id))
        union = np.count_nonzero((prediction == class_id) | (target == class_id))
        values.append(float(intersection / union) if union else 0.0)
    return values


def comparison(original, prediction, target, palette, name, score):
    original = original.convert('RGB')
    width, height = original.size
    palette = np.asarray(palette, dtype=np.uint8)
    if target.shape != (height, width):
        raise ValueError('Original image and mask shapes differ.')
    errors = np.full((height, width, 3), 255, dtype=np.uint8)
    mismatch = prediction != target
    errors[mismatch] = palette[target[mismatch]]
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 22) if Path(font_path).is_file() else ImageFont.load_default()
    title_height, label_height = 80, 34
    panel = Image.new('RGB', (2 * width, title_height + 2 * (height + label_height)), 'white')
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), name, fill='black', font=font)
    draw.text((12, 40), f'image mIoU: {score * 100:.6f}%    100% - image mIoU: {(1 - score) * 100:.6f}%', fill='black', font=font)
    tiles = [(original, 'Original image', 0, 0),
             (Image.fromarray(errors), 'Errors (ground-truth color; correct = white)', 1, 0),
             (Image.fromarray(palette[prediction]), 'Predicted mask', 0, 1),
             (Image.fromarray(palette[target]), 'Ground-truth mask', 1, 1)]
    for tile, label, column, row in tiles:
        x, y = column * width, title_height + row * (height + label_height)
        draw.text((x + 10, y + 4), label, fill='black', font=font)
        panel.paste(tile, (x, y + label_height))
    return panel


def allocate_result(results_root):
    results_root.mkdir(parents=True, exist_ok=True)
    date = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d')
    for number in range(1, 1000000):
        destination = results_root / f'model_results_{date}_{number}'
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            continue
    raise RuntimeError('Could not allocate a new result directory.')


def export(run, results_root=REPO / 'results'):
    run = Path(run).resolve()
    work = run / 'inference'
    completion = read_json(work / 'completion.json')
    manifest = read_json(work / 'input_manifest.json')
    upstream_metrics = read_json(work / 'metrics.json')
    rows = manifest['images']
    if completion['images'] != 20 or len(rows) != 20 or upstream_metrics['count'] != 20:
        raise ValueError('Only complete valid20 runs can be exported.')
    if len({r['name'] for r in rows}) != 20:
        raise ValueError('Duplicate validation image names.')
    config = manifest['config']
    frozen = inference()
    scores, decoded = [], []
    for row in rows:
        name = row['name']
        with Image.open(work / 'predicted_labels' / f'{name}.png') as im:
            prediction = np.array(im)
        target = frozen.decode_gt(VALID / 'masks' / f'{name}_mask.png', config['palette'])
        ious = class_ious(prediction, target)
        scores.append(dict(image=name, class_iou=ious, image_miou=float(np.mean(ious))))
        decoded.append((prediction, target))
    mean = float(np.mean([r['image_miou'] for r in scores]))
    reported = {r['image']: r for r in upstream_metrics['images']}
    if set(reported) != {r['image'] for r in scores}:
        raise ValueError('Upstream metric image set differs.')
    for score in scores:
        if not np.allclose(score['class_iou'], reported[score['image']]['class_iou'], rtol=0, atol=1e-12):
            raise ValueError('Per-image scores differ from the original inference scorer.')
    if abs(mean - upstream_metrics['image_miou']) > 1e-12:
        raise ValueError('Mean image mIoU differs from the original inference scorer.')
    destination = allocate_result(Path(results_root))
    panels = destination / 'valid_mask_comparison'
    panels.mkdir()
    for row, score, (prediction, target) in tqdm(list(zip(rows, scores, decoded)), desc='Export valid comparison panels', unit='image'):
        with Image.open(row['path']) as original:
            panel = comparison(original, prediction, target, config['palette'], row['name'], score['image_miou'])
        panel.save(panels / f"{row['name']}.png")
    for folder in ['predicted_labels', 'predicted_masks']:
        shutil.copytree(work / folder, destination / folder)
    for name in ['completion.json', 'input_manifest.json']:
        shutil.copy2(work / name, destination / name)
    for name in ['provenance.json', 'input_manifest.json', 'plan.json']:
        shutil.copy2(run / 'implementation' / name, destination / name)
    shutil.copytree(run / 'implementation', destination / 'source_snapshot',
                    ignore=shutil.ignore_patterns('data', 'folds', 'initial_weights'))
    shutil.copytree(run / 'retrained_model', destination / 'selected_model')
    shutil.copy2(run / 'environment.freeze.txt', destination / 'environment.freeze.txt')
    shutil.copy2(run / 'events.jsonl', destination / 'training_events.jsonl')
    write_json(destination / 'metrics.json', dict(metric='mean of per-image 4-class IoU; empty union=0',
                count=20, image_miou=mean, class_order=config['class_order'], images=scores))
    with (destination / 'per_image.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', 'image_miou_percent', '100_minus_image_miou_percent', *config['class_order']])
        for row in scores:
            writer.writerow([row['image'], row['image_miou'] * 100, (1 - row['image_miou']) * 100,
                             *[x * 100 for x in row['class_iou']]])
    text = [
        '# Model349 exh14 — 14개 구성 재학습 결과', '',
        f"최종 image mIoU: **{mean * 100:.6f}%**. 검증 이미지: **20장**.", '',
        '이미지마다 4개 클래스 IoU를 평균한 뒤 20장을 동일 비중으로 평균했습니다. 빈 클래스 IoU는 0입니다.', '',
        'exh14 구성 가중치 14개를 각각 한 번 새로 학습한 후 FP32로 조합했습니다.',
        '결합 비율·fold·후처리는 원본 고정값이며, 고해상도 경로의 원래 TENT 5단계를 유지했습니다.',
        'Model98·257은 보존된 부모 가중치를 사용하고, Model109→restart51·59→restart61 및 HRNet336→347은 이번 실행의 새 부모를 사용했습니다.',
        'B4 7개와 HRNet336은 ImageNet 초기 가중치에서 학습했습니다. 최종 결합 비율·후처리는 다시 탐색하지 않았습니다.',
        'selected_model/에 새 체크포인트 14개와 추론 소스를, source_snapshot/에 학습 소스를 저장했습니다.', '',
        'exh14는 같은 valid20으로 선택한 조합이므로 이 점수는 독립 test 성능이 아닙니다.', '',
        f"GPU: 물리 인덱스 2,3,4,5,6,7. 추론 경과 시간: {completion['elapsed_seconds']:.1f}초.",
        f'실행 원본·로그·중간 확률: `{run}`', '',
        '`valid_mask_comparison/`: 좌상 원본 / 우상 오류 / 좌하 예측 / 우하 정답.',
        '맞은 픽셀은 흰색, 틀린 픽셀은 해당 위치의 정답 색입니다.',
        '상단에 image mIoU 및 100%−image mIoU를 표시했습니다.', '',
        '| 이미지 | image mIoU | 100%−image mIoU |', '|---|---:|---:|',
        *[f"| {r['image']} | {r['image_miou'] * 100:.6f}% | {(1-r['image_miou']) * 100:.6f}% |" for r in scores], '',
    ]
    (destination / f'{destination.name}.md').write_text('\n'.join(text))
    write_json(destination / 'complete.json', dict(ok=True, source_run=str(run), image_miou=mean, panels=20))
    return destination
