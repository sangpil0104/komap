"""100 substantive configurations: 1 recommendation, 19 local, 80 exploration."""
from copy import deepcopy

REFERENCES = {
 'aspp': 'https://arxiv.org/abs/1706.05587',
 'ppm': 'https://arxiv.org/abs/1612.01105',
 'se': 'https://arxiv.org/abs/1709.01507',
 'eca': 'https://arxiv.org/abs/1910.03151',
 'cbam': 'https://arxiv.org/abs/1807.06521',
 'attention': 'https://arxiv.org/abs/1804.03999',
 'convnext': 'https://arxiv.org/abs/2201.03545',
 'unet': 'https://arxiv.org/abs/1505.04597',
 'focal': 'https://arxiv.org/abs/1708.02002',
 'tversky': 'https://arxiv.org/abs/1706.05721',
 'gdice': 'https://arxiv.org/abs/1707.03237',
 'ohem': 'https://arxiv.org/abs/1604.03540',
 'torch': 'https://docs.pytorch.org/docs/stable/optim.html',
 'ce': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html',
}


def stage(updates=600, lr=2e-6, mode='full_frame', **extra):
    return dict(dict(max_updates=updates, new_lr=lr, mode=mode, warmup_updates=25), **extra)


def catalog():
    rows = []
    def add(slug, description, group, **changes):
        n = len(rows) + 1
        rows.append(dict(id=f'm{n:03d}_{slug}', number=n, group=group,
            description=description, **changes))
    def local(slug, description, **changes):
        add(slug, description, 'local19', **changes)
    add('recommended_fullframe', '추천: 부모 CNN에서 패치 없이 전체 영상 600 updates', 'recommended1')
    local('ff_short', '전체 영상 300 updates, LR 1e-6', stages=[stage(300, 1e-6)])
    local('ff_gentle_long', '전체 영상 1000 updates, LR 1e-6', stages=[stage(1000, 1e-6)])
    local('ff_lr5e6', '전체 영상 600 updates, LR 5e-6', stages=[stage(600, 5e-6)])
    local('ff_lr1e5', '전체 영상 1000 updates, LR 1e-5', stages=[stage(1000, 1e-5)])
    local('ema999', '추천 일정 + EMA 0.999', train=dict(ema_decay=.999))
    local('ema95', '추천 일정 + EMA 0.95', train=dict(ema_decay=.95))
    local('decoder_only', 'encoder 고정, decoder와 classifier 미세조정', stages=[stage(policy='decoder')])
    local('head_only', 'classifier만 미세조정', stages=[stage(policy='head')])
    local('encoder_slow', 'encoder LR은 decoder의 0.1배', stages=[stage(encoder_multiplier=.1)])
    local('seed2028_parent', '과거 seed2028 최고점에서 전체 영상 미세조정', parent='seed2028')
    local('photo_parent', '과거 photo_mild 모델에서 전체 영상 미세조정', parent='photo')
    local('fullframe_parent', '과거 fullframe 모델에서 전체 영상 미세조정', parent='fullframe')
    local('mild_photo', '전체 영상에서 약한 광도 증강 확률 0.3', augmentation=dict(photometric_probability=.3))
    for width, rates in [(32, [1, 2, 3, 6]), (64, [1, 3, 6, 9])]:
        local(f'aspp_w{width}_staged', f'ASPP width {width}: 새 블록 선행 학습 후 분리 LR',
            model=dict(adapter='aspp', adapter_width=width, rates=rates),
            stages=[stage(300, 1e-6, policy='new', adapter_lr=1e-4), stage(600, 1e-6, adapter_lr=2e-5)])
    local('light_boundary_ce', '경계 픽셀 CE 가중치 +0.2', loss=dict(edge_weight=.2))
    local('light_class_weights', '강한 가중치 대신 CE [1,1.05,1.1,1]', loss=dict(class_weights=[1.,1.05,1.1,1.]))
    local('short_hard_patch', '낮은 LR 768 hard patch 200 후 전체 영상 600', hard=True,
        sampling=dict(mix=dict(uniform=.65,eutectic_center=.25,error_or_boundary=.1)),
        stages=[stage(200, 1e-6, 'local_detail_global_context', core_sizes=[768]), stage()])
    local('aux_light', '낮은 비중 0.05의 다중 해상도 보조 학습', loss=dict(auxiliary=.05))
    assert len(rows) == 20
    # 24 architectural candidates: all use full-frame 1200 updates.
    adapters = [('aspp','aspp'),('ppm','ppm'),('large_kernel','convnext'),('axial','own'),
        ('bottleneck','own'),('se','se'),('eca','eca'),('cbam','cbam'),
        ('skip_gate','attention'),('decoder_refine','own'),('highres','own'),('channel_adapter','own')]
    for adapter, ref in adapters:
        for variant in (1,2):
            add(f'{adapter}_v{variant}', f'{adapter} 경량 변형 {variant}, 전체 영상 학습', 'explore80',
                family='architecture', reference=ref, model=dict(adapter=adapter,variant=variant),
                stages=[stage(1200, 2e-6, adapter_lr=2e-5)])
    # 20 loss candidates, differing in computation rather than names alone.
    losses = [
      ('ce_half', {'ce':.5}, 'own'), ('dice_one', {'dice':1.}, 'own'),
      ('lovasz_quarter', {'lovasz':.25}, 'own'),
      *[(f'focal_{g}', {'focal_gamma':g}, 'focal') for g in (.5,1.,2.)],
      *[(f'tversky_{a}', {'overlap_kind':'tversky','tversky_alpha':a}, 'tversky') for a in (.3,.7)],
      ('generalized_dice', {'overlap_kind':'generalized_dice'}, 'gdice'),
      *[(f'smooth_{s}', {'label_smoothing':s}, 'ce') for s in (.01,.03,.05)],
      *[(f'hard_pixels_{r}', {'ohem_fraction':r}, 'ohem') for r in (.75,.5,.25)],
      ('edge_ce_half', {'edge_weight':.5}, 'own'), ('edge_ce_one', {'edge_weight':1.}, 'own'),
      ('aux_01', {'auxiliary':.1}, 'own'), ('aux_02', {'auxiliary':.2}, 'own'),
      ('boundary_005', {'boundary':.05}, 'own')]
    assert len(losses) == 20
    for name, loss, ref in losses:
        add(name.replace('.','p'), f'손실 탐색: {name}', 'explore80', family='loss', reference=ref,
            loss=loss, stages=[stage(1200,2e-6)])
    optimizers = [
      ('wd_zero',dict(new_weight_decay=0.)),('wd_0001',dict(new_weight_decay=.0001)),
      ('wd_001',dict(new_weight_decay=.001)),('wd_05',dict(new_weight_decay=.05)),
      ('beta2_99',dict(betas=[.9,.99])),('beta1_95',dict(betas=[.95,.999])),
      ('sgd_001',dict(optimizer='sgd',rate=.001)),('sgd_003',dict(optimizer='sgd',rate=.003)),
      ('adam',dict(optimizer='adam')),('amsgrad',dict(amsgrad=True)),
      ('constant_lr',dict(schedule='constant')),('poly_lr',dict(schedule='poly'))]
    for name, options in optimizers:
        options=deepcopy(options); rate=options.pop('rate',2e-6)
        add(name,f'최적화 탐색: {name}','explore80',family='optimizer',reference='torch',
            train=options,stages=[stage(1200,rate)])
    augmentations = [
      ('brightness',dict(brightness_offset=[-.025,.025],contrast_gain=[1.,1.],gamma=[1.,1.])),
      ('contrast',dict(brightness_offset=[0.,0.],contrast_gain=[.85,1.15],gamma=[1.,1.])),
      ('gamma',dict(brightness_offset=[0.,0.],contrast_gain=[1.,1.],gamma=[.85,1.15])),
      ('noise',dict(noise_std=.005)),('blur',dict(blur_sigma=.6)),
      ('drop_zscore',dict(drop_channel=1)),('drop_cdf',dict(drop_channel=2)),
      ('sharpen',dict(sharpen=.2)),('local_contrast',dict(local_contrast=.2)),
      ('erase',dict(erase_fraction=.02)),
      ('sample_al3ni',dict(sample_class=1)),('sample_eutectic',dict(sample_class=2))]
    for name, options in augmentations:
        photo=.35 if name in ('brightness','contrast','gamma') else 0.
        add(name,f'데이터 탐색: {name}','explore80',family='data',reference='own',
            augmentation=dict(photometric_probability=photo, extra_probability=.25, **options),
            stages=[stage(1200,2e-6)])
    # 12 fully new networks, independent random initialization, no downloads.
    for block in ('residual','separable','convnext','attention'):
        for width in (16,24,32):
            add(f'scratch_{block}_w{width}',f'처음부터 학습: {block} U-Net, base width {width}',
                'explore80',family='scratch',reference='convnext' if block=='convnext' else 'unet', parent=None,
                model=dict(architecture='scratch',block=block,width=width),
                augmentation=dict(photometric_probability=.5),train=dict(ema_decay=.999),
                stages=[stage(1600,3e-4,'local_detail_global_context',core_sizes=[512],warmup_updates=100),
                        stage(1400,1e-4,'local_detail_global_context',core_sizes=[768],warmup_updates=100),
                        stage(1500,3e-5)])
    assert len(rows)==100
    assert [sum(r['group']==g for r in rows) for g in ('recommended1','local19','explore80')]==[1,19,80]
    return {r['id']:r for r in rows}
