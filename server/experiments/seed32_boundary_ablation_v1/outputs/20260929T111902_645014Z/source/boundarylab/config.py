"""One-factor contrasts, trained afresh from ImageNet encoder + seed-32 decoder."""
from copy import deepcopy
from pathlib import Path

PROJECT = Path('/Data3/sppark/komap')
CHECKPOINT = PROJECT / 'experiments/seonghyeon_seed10_99_v1/outputs/20260926T082111_741695Z/seed_32/last.pt'
BASELINE_SOURCE = CHECKPOINT.parent.parent / 'source'
ENCODER_WEIGHTS = PROJECT / 'experiments/seonghyeon_seed10_v1/assets/efficientnet_b4_imagenet_encoder.pt'

COMMON = dict(
    seed=32, steps=16000, checkpoint_every=1000, crop=1536, target_width=4000,
    global_batch=8, workers_per_rank=1, learning_rate=3e-4,
    weight_decay=0.01, warmup=300, clip_grad=1.0,
    sync_bn=True, precision='bf16', geometry='legacy_scaled_crop',
    native_weight=0.0, boundary_sampling=0.0, boundary_ce_extra=0.0,
    boundary_radius=2, detail_branch=False, native_refiner=False,
    refiner_use_gray=True, gaussian_blur=True, decoder_interpolation='nearest',
    tta=True, restore='area',
)

# contrast is a comparison arm, NEVER the checkpoint used for initialization.
DEFINITIONS = [
    ('R00_seed32', 'eval', None, {}, 'Seed 32 원본 / D4'),
    ('R01_no_tta', 'eval', 'R00_seed32', {'tta': False}, '추론 D4만 제거'),
    ('R02_bilinear_restore', 'eval', 'R00_seed32', {'restore': 'bilinear'}, '최종 원본 복원만 bilinear'),
    ('T00_control', 'train', None, {}, 'B4 U-Net 신규 학습 대조군'),
    ('T01_native_geometry', 'train', 'T00_control', {'geometry': 'aligned_native_roi_v1'}, '원본·확대 좌표가 일치하는 crop 전처리'),
    ('T02_native_loss', 'train', 'T01_native_geometry', {'native_weight': 0.5}, '원본 해상도 NLL+Dice 손실 추가'),
    ('T03_boundary_sampling', 'train', 'T01_native_geometry', {'boundary_sampling': 0.35}, '35% 경계 중심 crop'),
    ('T04_boundary_weight', 'train', 'T01_native_geometry', {'boundary_ce_extra': 1.0}, '경계 띠 CE 가중치 1→2'),
    ('T05_detail_branch', 'train', 'T00_control', {'detail_branch': True}, '모델 입력 해상도 특징 경로 추가'),
    ('T06_no_gaussian', 'train', 'T00_control', {'gaussian_blur': False}, 'Gaussian blur만 제거; noise 유지'),
    ('T07_native_refiner', 'train', 'T02_native_loss', {'native_refiner': True}, '동일 native loss에 원본 영상 보정 모듈 추가'),
    ('T08_refiner_no_gray', 'train', 'T07_native_refiner', {'refiner_use_gray': False}, '보정 모듈의 원본 영상 입력만 0으로 차단'),
    ('T09_bilinear_decoder', 'train', 'T00_control', {'decoder_interpolation': 'bilinear'}, '디코더 보간만 nearest→bilinear'),
]


def make_plan(steps=16000, crop=1536):
    built = {}
    for name, kind, contrast, delta, description in DEFINITIONS:
        cfg = deepcopy(built[contrast]['config'] if contrast else COMMON)
        cfg.update(steps=steps, crop=crop)
        cfg.update(delta)
        built[name] = dict(id=name, kind=kind, contrast=contrast, change=delta,
                           description=description, config=cfg,
                           initialization=str(ENCODER_WEIGHTS if kind == 'train' else CHECKPOINT))
    arms = list(built.values())
    validate_plan(arms)
    return arms


def validate_plan(arms):
    by_id = {a['id']: a for a in arms}
    if len(by_id) != len(arms):
        raise ValueError('Duplicate experiment IDs')
    for arm in arms:
        cfg = arm['config']
        if cfg['crop'] < 64 or cfg['crop'] % 32 or cfg['steps'] < 1:
            raise ValueError('crop must be >=64 and divisible by 32; steps must be positive')
        if cfg['global_batch'] != 8:
            raise ValueError('This protocol fixes global batch size at eight')
        if cfg['native_refiner'] and cfg['native_weight'] <= 0:
            raise ValueError('Native refiner needs its matched native-supervision control')
        if (cfg['native_weight'] or cfg['boundary_sampling'] or cfg['boundary_ce_extra']) and cfg['geometry'] != 'aligned_native_roi_v1':
            raise ValueError('Native supervision/sampling needs aligned native geometry')
        if arm['contrast']:
            parent = by_id[arm['contrast']]
            changed = {k for k in cfg if cfg[k] != parent['config'][k]}
            if changed != set(arm['change']) or len(changed) != 1:
                raise ValueError(f'{arm["id"]} must differ by exactly ONE factor: {changed}')
        expected = ENCODER_WEIGHTS if arm['kind'] == 'train' else CHECKPOINT
        if arm['initialization'] != str(expected):
            raise ValueError('Training must load ONLY the ImageNet encoder; no seed-32 checkpoint fine-tuning')


def selected_arms(ids=None, **kwargs):
    arms = make_plan(**kwargs)
    if not ids:
        return arms
    lookup = {a['id']: a for a in arms}
    needed = {'R00_seed32'}
    for name in ids:
        if name not in lookup:
            raise ValueError(f'Unknown experiment {name}')
        while name:
            needed.add(name)
            name = lookup[name]['contrast']
    return [a for a in arms if a['id'] in needed]
