"""8 configurations: a single parent (accuracy_v1's official baseline checkpoint, the project's
existing 78.4729% model) x a *cumulative* technique ladder built from evolve_v1's 8 light-level
techniques (evolve_v1/evolvelab/models.py::LEVELS), added one at a time in numeric plan order:

  1 hires_decoder -> 2 crop_mix -> 3 hires_cnn_branch -> 4 boundary_aux_plus -> 5 fullres_refine
  -> 6 composition_head -> 7 texture_branch -> 8 complex_expert

so c1 has only technique 1 enabled and c8 has all 8 enabled at once. Unlike evolve_v1's own catalog
(which ablates each technique alone, or builds two heavy combos aimed at specific classes),
this answers "how much does each additional technique buy, stacked in order, on top of the
project's existing best checkpoint" with one comparably-budgeted sequence. See
experiments/evolve_accuracy_v1/README.md for why accuracy_v1's baseline was chosen as the parent
and why the techniques are stacked in this order."""
from copy import deepcopy

PARENT = 'accuracy_v1_baseline'

CROP_MIX_LIGHT = dict(uniform=.7, complex_region=.15, boundary_rich=.1, hard_example=.05)
BOUNDARY_LOSS_LIGHT = dict(boundary=.08, edge_weight=.3)
COMPOSITION_LOSS_LIGHT = dict(composition=.08)


def full_frame(updates=900, lr=2e-6, warmup=25):
    return dict(max_updates=updates, new_lr=lr, mode='full_frame', warmup_updates=warmup)


def patch_then_full(patch_updates=400, full_updates=500, core=512, patch_lr=1e-6, full_lr=2e-6):
    return [
        dict(max_updates=patch_updates, new_lr=patch_lr, mode='local_detail_global_context', core_sizes=[core], warmup_updates=25),
        dict(max_updates=full_updates, new_lr=full_lr, mode='full_frame', warmup_updates=25),
    ]


# Each step lists the *cumulative* set of model techniques enabled by that point, plus whether
# crop_mix (technique 2, sampling-only) is on yet. c1 has crop_mix off (matching evolve_v1's v1,
# which trains full-frame only since technique 2 has no effect there); c2 onward keeps it on.
STEPS = [
    dict(slug='c1_hires_decoder', techniques=['hires_decoder'], crop_mix=False,
         description='기법1 hires_decoder 추가 (누적 1/8)'),
    dict(slug='c2_add_crop_mix', techniques=['hires_decoder'], crop_mix=True,
         description='+기법2 crop_mix 추가 (누적 2/8)'),
    dict(slug='c3_add_hires_cnn_branch', techniques=['hires_decoder', 'hires_cnn_branch'], crop_mix=True,
         description='+기법3 hires_cnn_branch 추가 (누적 3/8)'),
    dict(slug='c4_add_boundary_aux_plus',
         techniques=['hires_decoder', 'hires_cnn_branch', 'boundary_aux_plus'], crop_mix=True,
         description='+기법4 boundary_aux_plus 추가 (누적 4/8)'),
    dict(slug='c5_add_fullres_refine',
         techniques=['hires_decoder', 'hires_cnn_branch', 'boundary_aux_plus', 'fullres_refine'],
         crop_mix=True, description='+기법5 fullres_refine 추가 (누적 5/8)'),
    dict(slug='c6_add_composition_head',
         techniques=['hires_decoder', 'hires_cnn_branch', 'boundary_aux_plus', 'fullres_refine',
                     'composition_head'],
         crop_mix=True, description='+기법6 composition_head 추가 (누적 6/8)'),
    dict(slug='c7_add_texture_branch',
         techniques=['hires_decoder', 'hires_cnn_branch', 'boundary_aux_plus', 'fullres_refine',
                     'composition_head', 'texture_branch'],
         crop_mix=True, description='+기법7 texture_branch 추가 (누적 7/8)'),
    dict(slug='c8_all_techniques',
         techniques=['hires_decoder', 'hires_cnn_branch', 'boundary_aux_plus', 'fullres_refine',
                     'composition_head', 'texture_branch', 'complex_expert'],
         crop_mix=True, description='+기법8 complex_expert 추가 -> 8개 기법 전부 적용 (누적 8/8)'),
]


def catalog():
    rows = {}
    for i, step in enumerate(STEPS, 1):
        name = f'e{i:03d}_{step["slug"]}'
        techniques = {key: dict(enabled=True, level='light') for key in step['techniques']}
        loss = {}
        if 'boundary_aux_plus' in step['techniques']:
            loss.update(BOUNDARY_LOSS_LIGHT)
        if 'composition_head' in step['techniques']:
            loss.update(COMPOSITION_LOSS_LIGHT)
        sampling = dict(mix=CROP_MIX_LIGHT) if step['crop_mix'] else {}
        stages = patch_then_full() if step['crop_mix'] else [full_frame(900)]
        rows[name] = dict(
            id=name, number=i, parent=PARENT, group='cumulative_ladder',
            description=f'{PARENT} 기반, {step["description"]}',
            model=dict(techniques=techniques), augmentation={}, loss=loss, sampling=sampling,
            train={}, stages=deepcopy(stages),
        )
    assert len(rows) == len(STEPS) == 8
    return rows
