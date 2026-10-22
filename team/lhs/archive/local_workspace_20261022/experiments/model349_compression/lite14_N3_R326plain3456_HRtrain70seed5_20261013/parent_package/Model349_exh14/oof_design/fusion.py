"""exh14-family fusion as a function of a JSON-serializable configuration (GT-free).

A configuration fixes: which brightness bands keep their own high-res checkpoint (others use
restart51), whether 322 / 329 / 347 / 269 take part, the class-wise replacement gammas and the
post-processing parameters. DEFAULT reproduces the frozen Model349_exh14 settings.
"""
from __future__ import annotations

import json

import numpy as np

from common import add_inference_paths

add_inference_paths()
import model335_exhaustive_b4_route_search as route  # noqa: E402

DEFAULT = {
    'bands': ['model109', 'seed456', 'pairwise'],          # 'ema' band always falls back to restart51
    'use322': True, 'use329': True, 'use347': True, 'use269': True,
    'primary_gamma': 1.0, 'al3ni_gamma': 0.7, 'al_gamma': 0.6, 'eutectic_gamma': 0.7,
    'gamma329': [0.5, 1.0, 0.0, 0.0], 'gamma336': [0.0, 0.25, 0.75, 0.75], 'gamma347': [0.0, 0.0, 0.25, 0.0],
    'prior_scale': 1.25, 'beta_scale': 0.8, 'al_delta': 0.0, 'eut_delta': 0.15,
    'gaussian_blend': 0.5, 'eut_edge_blend': 0.25, 'cleanup_threshold': 128,
}


def key(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


def settings(cfg: dict) -> dict:
    s = dict(primary_source='326', al3ni_source='326', al_source='322', eutectic_source='269')
    for k in ('primary_gamma', 'al3ni_gamma', 'al_gamma', 'eutectic_gamma', 'prior_scale', 'beta_scale', 'al_delta',
              'eut_delta', 'gaussian_blend', 'eut_edge_blend', 'cleanup_threshold'):
        s[k] = cfg[k]
    if not cfg['use322']:
        s['al_gamma'] = 0.0
    if not cfg['use269']:
        s['eutectic_gamma'] = 0.0
    return s


def predict(arrays: dict, meta: dict, cfg: dict, weights=None) -> np.ndarray:
    """arrays: member probabilities (b4_322, b4_326, b4_329, hrnet336, hrnet347, n269, high_*), z, q."""
    routed = meta['routed']
    src = routed if routed in cfg['bands'] else 'restart51'
    base = arrays[f'high_{src}'].astype(np.float32)
    item = {'z': arrays['z'], 'q': arrays['q'], 'prior_z_stats': meta['prior_z_stats'],
            'prior_q_stats': meta['prior_q_stats'], 'params': meta['params']}
    zero = np.zeros_like(base)
    sources = {'322': arrays['b4_322'] if cfg['use322'] else zero, '326': arrays['b4_326'],
               '329': arrays['b4_329'], '336': arrays['hrnet336'], '347': arrays['hrnet347'],
               '269': arrays['n269'] if cfg['use269'] else zero}
    sources = {k: v.astype(np.float32) for k, v in sources.items()}
    s = settings(cfg)
    mixed = route.make_mixed({'item': item, 'base': base, 'sources': sources}, s)
    for comp, gam, on in (('329', cfg['gamma329'], cfg['use329']), ('336', cfg['gamma336'], True),
                          ('347', cfg['gamma347'], cfg['use347'])):
        if not on:
            continue
        for cid, g in enumerate(gam):
            if g:
                mixed = route.replace_classes(mixed, sources[comp], (cid,), g)
    if weights is None:
        weights = route.bilateral_weights(arrays['z'])
    return route.predict(item, mixed, s, weights, exact=True).astype(np.uint8)


# Fixed search space for OOF selection (documented in README; never tuned on test or held-out labels).
GRID = {
    'bands': [[b for i, b in enumerate(['model109', 'seed456', 'pairwise']) if m >> i & 1] for m in range(8)],
    'use322': [True, False], 'use329': [True, False], 'use347': [True, False], 'use269': [True, False],
    'primary_gamma': [0.0, 0.25, 0.5, 0.75, 1.0], 'al3ni_gamma': [0.0, 0.25, 0.5, 0.7, 1.0],
    'al_gamma': [0.0, 0.25, 0.4, 0.6, 0.8, 1.0], 'eutectic_gamma': [0.0, 0.25, 0.5, 0.7, 1.0],
    'gamma329': [0.0, 0.25, 0.5, 0.75, 1.0], 'gamma336': [0.0, 0.25, 0.5, 0.75, 1.0],
    'gamma347': [0.0, 0.25, 0.5, 0.75, 1.0],
    'prior_scale': [1.0, 1.25, 1.5], 'beta_scale': [0.6, 0.8, 1.0], 'al_delta': [-0.1, 0.0, 0.1],
    'eut_delta': [0.05, 0.15, 0.25], 'gaussian_blend': [0.25, 0.5, 0.75], 'eut_edge_blend': [0.0, 0.25, 0.5],
}
ORDER = (['bands', 'use322', 'use329', 'use347', 'use269', 'primary_gamma', 'al3ni_gamma', 'al_gamma', 'eutectic_gamma']
         + [f'gamma{c}[{i}]' for c in (329, 336, 347) for i in range(4)]
         + ['prior_scale', 'beta_scale', 'al_delta', 'eut_delta', 'gaussian_blend', 'eut_edge_blend'])


def neighbours(cfg: dict, coord: str):
    """All configurations that change only `coord` to another grid value (deterministic order)."""
    out = []
    if '[' in coord:
        name, idx = coord[:-1].split('[')
        idx = int(idx)
        for v in GRID[name]:
            if v != cfg[name][idx]:
                c = json.loads(key(cfg)); c[name][idx] = v; out.append(c)
    else:
        for v in GRID[coord]:
            if v != cfg[coord]:
                c = json.loads(key(cfg)); c[coord] = v; out.append(c)
    return out
