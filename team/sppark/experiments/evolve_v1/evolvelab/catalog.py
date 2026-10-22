"""100 configurations: 10 image-mIoU top-10 hundred_v1 parents x 10 technique variants each.
V1-V8 apply one technique alone at 'light' intensity (clean single-technique ablations); V9/V10
combine several techniques at 'heavy' intensity, targeting boundary quality (V9) and complex /
eutectic-texture regions (V10) respectively. See the approved plan for the full rationale:
/home/sppark/.claude/plans/fluttering-herding-meteor.md

Technique 2 (crop-mix) only has an effect during patch-based (`local_detail_global_context`)
training, since `sampling.mix` is only consulted for crop-center selection in that mode -- so the
three variants that turn technique 2 on (V2, V9, V10) get a short patch stage before the final
full-frame stage; every other variant trains full-frame only (like hundred_v1's local19 entries)."""
from copy import deepcopy

PARENTS = [
    'm011_seed2028_parent', 'm013_fullframe_parent', 'm063_aux_02', 'm085_local_contrast',
    'm009_head_only', 'm012_photo_parent', 'm008_decoder_only', 'm006_ema999',
    'm062_aux_01', 'm016_aspp_w64_staged',
]

CROP_MIX_LIGHT = dict(uniform=.7, complex_region=.15, boundary_rich=.1, hard_example=.05)
CROP_MIX_HEAVY = dict(uniform=.4, complex_region=.25, boundary_rich=.2, hard_example=.15)
CROP_MIX_OFF = dict(uniform=1., complex_region=0., boundary_rich=0., hard_example=0.)

BOUNDARY_LOSS_LIGHT = dict(boundary=.08, edge_weight=.3)
BOUNDARY_LOSS_HEAVY = dict(boundary=.15, edge_weight=.6)
COMPOSITION_LOSS_LIGHT = dict(composition=.08)
COMPOSITION_LOSS_HEAVY = dict(composition=.15)


def full_frame(updates=900, lr=2e-6, warmup=25):
    return dict(max_updates=updates, new_lr=lr, mode='full_frame', warmup_updates=warmup)


def patch_then_full(patch_updates, full_updates, core=512, patch_lr=1e-6, full_lr=2e-6):
    return [
        dict(max_updates=patch_updates, new_lr=patch_lr, mode='local_detail_global_context', core_sizes=[core], warmup_updates=25),
        dict(max_updates=full_updates, new_lr=full_lr, mode='full_frame', warmup_updates=25),
    ]


def technique(key, level='light'):
    return {key: dict(enabled=True, level=level)}


def merge_techniques(*items):
    out = {}
    for item in items:
        out.update(item)
    return out


VARIANTS = [
    dict(slug='v1_hires_decoder', group='light_single',
         description='기법1 High-res decoder 단독 (light)',
         model=dict(techniques=technique('hires_decoder', 'light')),
         stages=[full_frame(900)]),
    dict(slug='v2_crop_mix', group='light_single',
         description='기법2 크롭 믹스 단독 (light: uniform .7/복잡영역 .15/경계 .1/hard .05, patch->fullframe)',
         sampling=dict(mix=CROP_MIX_LIGHT),
         stages=patch_then_full(patch_updates=400, full_updates=500)),
    dict(slug='v3_hires_branch', group='light_single',
         description='기법3 고해상도 CNN 브랜치 단독 (light, DINOv3 제외)',
         model=dict(techniques=technique('hires_cnn_branch', 'light')),
         stages=[full_frame(900)]),
    dict(slug='v4_boundary_head', group='light_single',
         description='기법4 경계 보조 헤드 단독 (light: boundary=.08, edge_weight=.3)',
         model=dict(techniques=technique('boundary_aux_plus', 'light')),
         loss=deepcopy(BOUNDARY_LOSS_LIGHT),
         stages=[full_frame(900)]),
    dict(slug='v5_fullres_refine', group='light_single',
         description='기법5 전체 해상도 리파인먼트 단독 (light)',
         model=dict(techniques=technique('fullres_refine', 'light')),
         stages=[full_frame(900)]),
    dict(slug='v6_composition_head', group='light_single',
         description='기법6 국소 상 조성 헤드 단독 (light, decoder feedback 없음)',
         model=dict(techniques=technique('composition_head', 'light')),
         loss=deepcopy(COMPOSITION_LOSS_LIGHT),
         stages=[full_frame(900)]),
    dict(slug='v7_texture_branch', group='light_single',
         description='기법7 멀티스케일 텍스처 브랜치 단독 (light: ASPP류 컨텍스트만)',
         model=dict(techniques=technique('texture_branch', 'light')),
         stages=[full_frame(900)]),
    dict(slug='v8_complex_expert', group='light_single',
         description='기법8 Complex-region expert 단독 (light)',
         model=dict(techniques=technique('complex_expert', 'light')),
         stages=[full_frame(900)]),
    dict(slug='v9_boundary_combo', group='heavy_combo',
         description='경계 집중 heavy 조합: 기법1+4+5+6 heavy 동시 적용 + 기법2 light 믹스 (patch->fullframe)',
         model=dict(techniques=merge_techniques(
             technique('hires_decoder', 'heavy'), technique('boundary_aux_plus', 'heavy'),
             technique('fullres_refine', 'heavy'), technique('composition_head', 'heavy'))),
         loss=dict(**BOUNDARY_LOSS_HEAVY, **COMPOSITION_LOSS_HEAVY),
         sampling=dict(mix=CROP_MIX_LIGHT),
         stages=patch_then_full(patch_updates=500, full_updates=700)),
    dict(slug='v10_texture_combo', group='heavy_combo',
         description='복잡 텍스처 집중 heavy 조합: 기법2+3+7+8 heavy 동시 적용 (patch->fullframe)',
         model=dict(techniques=merge_techniques(
             technique('hires_cnn_branch', 'heavy'), technique('texture_branch', 'heavy'),
             technique('complex_expert', 'heavy'))),
         sampling=dict(mix=CROP_MIX_HEAVY),
         stages=patch_then_full(patch_updates=500, full_updates=700)),
]

assert len(VARIANTS) == 10
assert sum(v['group'] == 'light_single' for v in VARIANTS) == 8
assert sum(v['group'] == 'heavy_combo' for v in VARIANTS) == 2


def catalog():
    rows = {}
    n = 0
    for parent in PARENTS:
        for variant in VARIANTS:
            n += 1
            short = parent.split('_', 1)[0]
            name = f'e{n:03d}_{short}_{variant["slug"]}'
            rows[name] = dict(
                id=name, number=n, parent=parent, group=variant['group'],
                description=f'{parent} 기반, {variant["description"]}',
                model=deepcopy(variant.get('model', {})),
                augmentation=deepcopy(variant.get('augmentation', {})),
                loss=deepcopy(variant.get('loss', {})),
                sampling=deepcopy(variant.get('sampling', {})),
                train=deepcopy(variant.get('train', {})),
                stages=deepcopy(variant['stages']),
            )
    assert len(rows) == 100
    assert sum(r['group'] == 'light_single' for r in rows.values()) == 80
    assert sum(r['group'] == 'heavy_combo' for r in rows.values()) == 20
    return rows
