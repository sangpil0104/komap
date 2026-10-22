"""Choose the stage-1 override rule on OOF and report dev / outer / per-seed scores.

    python select_rule.py --work WORK [--allow-partial]

Inputs: R labels written by oof_design/select_oof.py (work/select/labels/final, outer<j>,
final_seed<s>) and E OOF probabilities (work/oof_expert). Candidate rules: every subset of the four
classes (16, fixed order: size, then lexicographic; the empty rule = R alone). Selection maximizes the
mean image mIoU of the selection images; ties keep the earlier rule. For outer fold j the R labels
of the configuration chosen without fold j are used, and the rule is chosen without fold j too.
Writes work/select_rule/report.json and final_rule.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from expert_common import ORIGINAL_RULE, RULES, apply_rule, e_label
from common import decode_mask, image_miou, manifest, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--allow-partial', action='store_true')
    a = ap.parse_args()
    fold_of = {r['name']: r['fold'] for r in manifest(a.work)['labeled']}
    lab = a.work / 'select' / 'labels'
    seeds = {k: sorted(int(p.name[4:]) for p in (a.work / 'oof_expert' / f'fold{k}').glob('seed*')) for k in set(fold_of.values())}
    names = [n for n in sorted(fold_of) if (lab / 'final' / f'{n}.png').exists()
             and any((a.work / 'oof_expert' / f'fold{fold_of[n]}' / f'seed{s}' / f'{n}.npy').exists() for s in seeds[fold_of[n]])]
    if len(names) != len(fold_of) and not a.allow_partial:
        raise SystemExit(f'only {len(names)}/{len(fold_of)} images have R labels and E predictions')
    gt = {n: decode_mask(a.work / 'all' / 'train' / 'masks' / f'{n}_mask.png') for n in names}

    def e_probs(n, seed=None):
        k = fold_of[n]
        ss = [seed] if seed is not None else seeds[k]
        ps = [np.load(a.work / 'oof_expert' / f'fold{k}' / f'seed{s}' / f'{n}.npy').astype(np.float32) for s in ss]
        return np.mean(ps, 0)

    e_avg = {n: e_label(e_probs(n)) for n in names}

    def table(r_dir, e_lab, subset):
        return {rule: {n: image_miou(apply_rule(np.asarray(Image.open(r_dir / f'{n}.png')), e_lab[n], rule), gt[n])[0]
                       for n in subset} for rule in RULES}

    def choose(tab, subset):
        return max(RULES, key=lambda r: np.mean([tab[r][n] for n in subset]))     # first max wins ties

    dev_tab = table(lab / 'final', e_avg, names)
    rule = choose(dev_tab, names)
    held = {}
    outer = {}
    for j in sorted({fold_of[n] for n in names}):
        if not (lab / f'outer{j}').exists():
            continue
        inner = [n for n in names if fold_of[n] != j]
        test = [n for n in names if fold_of[n] == j]
        tab = table(lab / f'outer{j}', e_avg, names)
        rj = choose(tab, inner)
        held.update({n: tab[rj][n] for n in test})
        outer[j] = {'rule': list(rj), 'heldout_miou': float(np.mean([tab[rj][n] for n in test]) * 100)}
    per_seed = {}
    common_seeds = sorted(set.intersection(*[set(v) for v in seeds.values()])) if seeds else []
    for s in common_seeds:
        r_dir = lab / f'final_seed{s}'
        if not r_dir.exists():
            continue
        e_s = {n: e_label(e_probs(n, s)) for n in names}
        per_seed[s] = float(np.mean([image_miou(apply_rule(np.asarray(Image.open(r_dir / f'{n}.png')), e_s[n], rule), gt[n])[0]
                                     for n in names]) * 100)
    ps = np.array(list(per_seed.values()))
    mean = lambda r: float(np.mean([dev_tab[r][n] for n in names]) * 100)
    report = {'images': len(names), 'complete': len(names) == len(fold_of), 'rule': list(rule),
              'rule_classes': [['Al3Ni', 'Primary Si', 'Al', 'Eutectic Si'][c] for c in rule],
              'dev_oof': mean(rule), 'r_only_dev_oof': mean(()), 'original_fusion805_rule_dev_oof': mean(ORIGINAL_RULE),
              'outer_estimate': float(np.mean(list(held.values())) * 100) if held else None, 'outer_folds': outer,
              'per_seed': {'seeds': list(per_seed), 'scores': ps.round(4).tolist(),
                           'mean': float(ps.mean()) if len(ps) else None, 'sd': float(ps.std(ddof=1)) if len(ps) > 1 else None},
              'all_rules_dev': {str(list(r)): mean(r) for r in RULES}}
    write_json(a.work / 'select_rule' / 'report.json', report)
    write_json(a.work / 'select_rule' / 'final_rule.json', {'rule': list(rule)})
    print({k: report[k] for k in ('images', 'rule_classes', 'dev_oof', 'r_only_dev_oof', 'outer_estimate')}, report['per_seed'])


if __name__ == '__main__':
    main()
