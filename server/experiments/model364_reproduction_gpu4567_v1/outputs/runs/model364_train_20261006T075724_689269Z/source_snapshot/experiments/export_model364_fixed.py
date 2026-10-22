"""Evaluate the recorded Model364 route on newly trained prerequisites."""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import model364_model361_entropy_into_360 as upstream
from local_image_metrics import class_ious
from local_result_export import export_final


def fixed_prediction(row, p347, p351, p361, route, gamma329, parameters):
    spatial = upstream.model360.spatial
    p336 = row['sources']['336']
    for probability in (p347, p351, p361):
        if probability.shape != p336.shape or not np.isfinite(probability).all():
            raise ValueError('Invalid expert probability shape or values')
    margin, strength = parameters['model360_model351_entropy_gate_margin_strength']
    gate = upstream.model360.entropy(p351) + margin < upstream.model360.entropy(p336)
    d4_base = p336 + strength * gate[None] * (p351 - p336)
    margin, strength = parameters['model364_model361_entropy_gate_margin_strength']
    gate = upstream.model360.entropy(p361) + margin < upstream.model360.entropy(d4_base)
    row['sources']['336'] = d4_base + strength * gate[None] * (p361 - d4_base)
    row['sources']['347'] = p347
    try:
        parts = spatial.spatial_parts(row, route, gamma329,
                                      parameters['model346_gamma336'],
                                      parameters['model349_gamma347'])
        g, e = parameters['model355_spatial_blend']
        return spatial.prediction(parts, dict(route, gaussian_blend=g, eut_edge_blend=e), g, e)
    finally:
        row['sources']['336'] = p336


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    source = Path(__file__).resolve().parents[1]
    recipe = json.loads((source / 'config/model364_recipe.json').read_text())
    params = recipe['fixed_route_parameters']
    spatial = upstream.model360.spatial
    outputs = root / 'outputs'
    rows, names, route, gamma329, _ = spatial.audit.model350.load_rows(
        root, outputs / 'model336_hrnet_best/valid_eval/valid_probabilities.npz')
    archives = {}
    for key, rel in [('347', 'model347_hrnet_inverse_ce_finetune/valid_eval'),
                     ('351', 'model351_model336_d4_tta/valid_eval_d4'),
                     ('361', 'model361_hrnet_scale3_finetune/valid_eval')]:
        _, archive = spatial.audit.model350.model341.route.load_archive(
            outputs / rel / 'valid_probabilities.npz', team_order=True)
        if set(archive) != set(names):
            raise ValueError(f'Model{key} validation image keys differ')
        archives[key] = archive
    predictions = {}
    scores = []
    for name, row in tqdm(list(zip(names, rows)), desc='Model364 fixed route', unit='image'):
        pred = fixed_prediction(row, archives['347'][name], archives['351'][name],
                                archives['361'][name], route, gamma329, params)
        predictions[id(row)] = pred
        scores.append(float(class_ious(pred, row['item']['target']).mean()))
    result = {'model': 364, 'fixed_parameters': params, 'base_route': route,
              'gamma329': gamma329, 'raw_best_valid_image_mean_miou_empty0': float(np.mean(scores)),
              'image_names': names, 'image_scores': scores,
              'route_selection': 'Fixed recorded parameters; no new final gate selection',
              'reference_only_recorded_score_percent': recipe['recorded_validation']['model364_raw_best_image_mean_miou_percent']}
    dest = outputs / 'model364_fixed_reproduction'
    dest.mkdir(exist_ok=False)
    (dest / 'results.json').write_text(json.dumps(result, indent=2))
    export_final(rows, names, lambda row: predictions[id(row)], root, result)


if __name__ == '__main__':
    main()
