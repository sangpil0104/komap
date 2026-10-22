#!/usr/bin/env python3
"""Combine exact per-image route grids into a conditional outer-group audit."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


GROUP_RE = re.compile(r"^(A\d+)_")
STEP4_RE = re.compile(r"_step4_")


def group_of(name: str) -> str:
    match = GROUP_RE.match(name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {name}")
    return match.group(1)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        scores = np.asarray(archive["image_scores"], dtype=np.float64)
        classes = np.asarray(archive["class_scores"], dtype=np.float64)
        settings = np.asarray(archive["gamma_settings"], dtype=np.float64)
        names = [str(value) for value in archive["image_names"].tolist()]
    if scores.ndim != 2 or classes.shape != (*scores.shape, 4):
        raise ValueError(f"bad score matrix shapes in {path}: {scores.shape}, {classes.shape}")
    if scores.shape[0] != len(names) or scores.shape[1] != len(settings):
        raise ValueError(f"matrix/name/settings count mismatch in {path}")
    if not np.isfinite(scores).all() or not np.isfinite(classes).all():
        raise ValueError(f"non-finite scores in {path}")
    return scores, classes, names, settings


def setting_index(settings: np.ndarray, value: list[float] | tuple[float, ...]) -> int:
    wanted = np.asarray(value, dtype=np.float64)
    matches = np.flatnonzero(np.all(np.isclose(settings, wanted, atol=1e-7, rtol=0), axis=1))
    if len(matches) != 1:
        raise ValueError(f"expected one route setting {wanted.tolist()}, found {len(matches)}")
    return int(matches[0])


def fold_choices(rows: list[dict], field: str) -> dict[str, tuple[float, ...]]:
    result = {}
    for row in rows:
        group = str(row["heldout_specimen"])
        if group in result:
            raise ValueError(f"duplicate held-out group: {group}")
        result[group] = tuple(float(value) for value in row[field])
    return result


def group_bootstrap(delta: np.ndarray, groups: list[str], draws: int,
                    seed: int) -> list[float]:
    unique = sorted(set(groups))
    by_group = {name: np.flatnonzero(np.asarray(groups) == name) for name in unique}
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sample = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([by_group[unique[item]] for item in sample])
        boot[index] = delta[rows].mean()
    return np.percentile(boot * 100.0, [2.5, 97.5]).tolist()


def summarize(reference: np.ndarray, candidate: np.ndarray,
              reference_class: np.ndarray, candidate_class: np.ndarray,
              names: list[str], draws: int, seed: int) -> dict:
    groups = [group_of(name) for name in names]
    delta = candidate - reference
    group_delta_fraction = {
        group: float(delta[np.asarray(groups) == group].mean())
        for group in sorted(set(groups))
    }
    group_delta_pp = {group: value * 100.0
                      for group, value in group_delta_fraction.items()}
    group_names = sorted(group_delta_fraction)
    step4 = np.asarray([STEP4_RE.search(name) is not None for name in names])
    step4_delta = float(delta[step4].mean() * 100.0) if step4.any() else None
    return {
        "images": len(names),
        "specimen_groups": len(set(groups)),
        "reference_image_mean_miou_empty0_pct": float(reference.mean() * 100.0),
        "candidate_image_mean_miou_empty0_pct": float(candidate.mean() * 100.0),
        "paired_image_weighted_delta_pp": float(delta.mean() * 100.0),
        "paired_group_bootstrap_95ci_delta_pp_image_weighted":
            group_bootstrap(delta, groups, draws, seed),
        "paired_group_equal_delta_pp": float(np.mean(list(group_delta_pp.values()))),
        "paired_group_bootstrap_95ci_delta_pp_group_equal": group_bootstrap(
            np.asarray([group_delta_fraction[group] for group in group_names]),
            group_names, draws, seed + 1),
        "positive_images": int((delta > 0).sum()),
        "positive_groups": int(sum(value > 0 for value in group_delta_fraction.values())),
        "step4_paired_delta_pp": step4_delta,
        "class_mean_iou_delta_pp": {
            name: float((candidate_class[:, index] - reference_class[:, index]).mean() * 100.0)
            for index, name in enumerate(("Al3Ni", "Primary_Si", "Al", "Eutectic_Si"))
        },
        "per_group_delta_pp": group_delta_pp,
        "per_image": [
            {"image": name, "group": groups[index],
             "reference_miou": float(reference[index] * 100.0),
             "candidate_miou": float(candidate[index] * 100.0),
             "delta_pp": float(delta[index] * 100.0)}
            for index, name in enumerate(names)
        ],
    }


def run(args: argparse.Namespace) -> dict:
    m346 = load_json(args.model346_json)
    m349 = load_json(args.model349_global_json)
    m349_alt = load_json(args.model349_alt_json)
    global_grid = load_json(args.global_grid_json)
    global_scores, global_classes, names, global_settings = load_matrix(args.global_matrix)
    alt_scores, alt_classes, alt_names, alt_settings = load_matrix(args.alt_matrix)
    if names != alt_names or len(names) != 20:
        raise ValueError("global/alternate matrices must contain identical ordered 20-image IDs")

    gamma336_full = tuple(float(value) for value in m346["raw_best_gamma336"])
    if tuple(float(value) for value in global_grid.get("fixed_gamma336", ())) != gamma336_full:
        raise ValueError("global score matrix was not generated from the recorded Model346 gamma")
    gamma336_by_group = fold_choices(
        m346["grouped_cv_vs_model341"]["folds"], "selected_gamma336")
    if set(gamma336_by_group) != {group_of(name) for name in names}:
        raise ValueError("Model346 outer-group gamma choices do not cover validation groups")
    non_global = {value for value in gamma336_by_group.values() if value != gamma336_full}
    if len(non_global) > 1:
        raise ValueError("more than one alternate gamma336 is present; extend preregistered audit")
    gamma336_alt = next(iter(non_global), gamma336_full)
    if tuple(float(value) for value in m349_alt.get("fixed_gamma336_audit_override", ())) != gamma336_alt:
        raise ValueError("alternate Model349 grid does not record the expected fixed gamma336")
    alt_groups = {group for group, value in gamma336_by_group.items()
                  if value == gamma336_alt and value != gamma336_full}
    if gamma336_alt != gamma336_full and alt_groups != {"A71"}:
        raise ValueError(f"unexpected alternate Model346 groups: {sorted(alt_groups)}")

    gamma347_global_by_group = fold_choices(
        m349["grouped_cv_vs_model346"]["folds"], "selected_gamma347")
    gamma347_alt_by_group = fold_choices(
        m349_alt["grouped_cv_vs_model346"]["folds"], "selected_gamma347")
    if set(gamma347_global_by_group) != set(gamma336_by_group):
        raise ValueError("global Model349 grouped route choices do not cover all groups")

    base_global_index = setting_index(global_settings, (0.0, 0.0, 0.0, 0.0))
    base_alt_index = setting_index(alt_settings, (0.0, 0.0, 0.0, 0.0))
    reference = np.empty(len(names), dtype=np.float64)
    candidate = np.empty(len(names), dtype=np.float64)
    reference_class = np.empty((len(names), 4), dtype=np.float64)
    candidate_class = np.empty((len(names), 4), dtype=np.float64)
    selected_gamma347: dict[str, tuple[float, ...]] = {}
    for index, name in enumerate(names):
        group = group_of(name)
        if gamma336_by_group[group] == gamma336_full:
            matrix_scores, matrix_classes, settings = global_scores, global_classes, global_settings
            base_index = base_global_index
            gamma347 = gamma347_global_by_group[group]
        else:
            matrix_scores, matrix_classes, settings = alt_scores, alt_classes, alt_settings
            base_index = base_alt_index
            gamma347 = gamma347_alt_by_group[group]
        route_index = setting_index(settings, gamma347)
        reference[index] = matrix_scores[index, base_index]
        reference_class[index] = matrix_classes[index, base_index]
        candidate[index] = matrix_scores[index, route_index]
        candidate_class[index] = matrix_classes[index, route_index]
        selected_gamma347[group] = gamma347

    group_names = sorted(gamma336_by_group)
    result = {
        "model": 380,
        "scope": "20 official validation images; leave-one-specimen-group-out route selection",
        "metric": "image-wise mean of four class IoUs; empty union=0",
        "conditional_anchor": "Model341/339 fixed historical route; its full-valid selection is not corrected",
        "gamma336_full_valid": list(gamma336_full),
        "gamma336_choices_by_heldout_group": {
            group: list(gamma336_by_group[group]) for group in group_names
        },
        "gamma347_choices_by_heldout_group": {
            group: list(selected_gamma347[group]) for group in group_names
        },
        "gamma336_choice_counts": {
            str(key): value for key, value in Counter(gamma336_by_group.values()).items()
        },
        "gamma347_choice_counts": {
            str(key): value for key, value in Counter(selected_gamma347.values()).items()
        },
        "paired_outer_group_comparison_349_vs_346": summarize(
            reference, candidate, reference_class, candidate_class,
            names, args.bootstrap, args.seed),
        "raw_full_valid_model346_pct":
            float(m346["raw_best_valid_image_mean_miou_empty0"] * 100.0),
        "raw_full_valid_model349_pct":
            float(m349["raw_best_valid_image_mean_miou_empty0"] * 100.0),
        "alternate_gamma336_group": sorted(alt_groups),
        "limitations": [
            "This cross-fit corrects gamma336/gamma347 selection conditional on Model341.",
            "Model341/339 route choices were selected using the full validation set and remain a source of selection bias.",
            "The 19-group interval is a small-sample uncertainty summary, not independent test performance.",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model346-json", type=Path, required=True)
    parser.add_argument("--model349-global-json", type=Path, required=True)
    parser.add_argument("--model349-alt-json", type=Path, required=True)
    parser.add_argument("--global-grid-json", type=Path, required=True)
    parser.add_argument("--global-matrix", type=Path, required=True)
    parser.add_argument("--alt-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=380)
    args = parser.parse_args()
    result = run(args)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["paired_outer_group_comparison_349_vs_346"], indent=2))


if __name__ == "__main__":
    main()
