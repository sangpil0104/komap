#!/usr/bin/env python3
"""Exhaustive exact validation routes for Model270 × Model322/326 B4 sources.

Per class, evaluate: keep the Model270 source, or blend a B4 candidate
(Model322, Model326, or their 50:50 probability mean) at gamma 0.5/1.0.
This is 7**4 = 2401 routes, including Eutectic Si. All scores use exact
spatial refinement; no model training or GPU inference is performed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import pickle
import re
import time
from pathlib import Path

import numpy as np

from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import (
    bilateral_weights,
    default_setting,
    predict,
    replace_classes,
)
from model323_model270_branch_route_probe import canonical_image_key, grouped_cv


TEAM_TO_MAIN = (1, 2, 0, 3)
CLASS_FIELDS = (
    (0, "Al3Ni", "al3ni_source", "al3ni_gamma"),
    (1, "Primary Si", "primary_source", "primary_gamma"),
    (2, "Al", "al_source", "al_gamma"),
    (3, "Eutectic Si", "eutectic_source", "eutectic_gamma"),
)
B4_SOURCES = ("322", "326", "b4mean")
GAMMAS = (0.50, 1.00)

_WORKER_ROWS = None
_WORKER_SETTINGS = None


def load_archive(path: Path, *, team_order: bool = False):
    with np.load(path) as archive:
        raw = {key: archive[key].astype(np.float32) for key in archive.files}
    values = {canonical_image_key(key): val for key, val in raw.items()}
    if len(values) != len(raw):
        raise ValueError(f"duplicate normalized image keys in {path}")
    if team_order:
        for key, val in values.items():
            if val.ndim != 3 or val.shape[0] != 4:
                raise ValueError(f"expected 4xHxW probability map for {key}: {val.shape}")
            values[key] = val[list(TEAM_TO_MAIN)]
    return list(values), values


def build_settings():
    reference = default_setting(
        stage="model270_reference",
        primary_source="195", primary_gamma=0.75,
        al3ni_source="217", al3ni_gamma=0.40,
        eutectic_source="269", eutectic_gamma=0.50,
        prior_scale=1.25, beta_scale=0.80,
        al_delta=0.0, eut_delta=0.15,
        gaussian_blend=0.50, eut_edge_blend=0.25,
        cleanup_threshold=128,
    )
    options = [(None, 0.0, "Model270")]
    for source in B4_SOURCES:
        for gamma in GAMMAS:
            options.append((source, gamma, f"{source}@{gamma:.2f}"))

    settings = []
    for index, choices in enumerate(itertools.product(options, repeat=4)):
        config = dict(reference)
        labels = []
        for (_, class_name, source_key, gamma_key), (source, gamma, label) in zip(
                CLASS_FIELDS, choices):
            labels.append(f"{class_name}={label}")
            if source is not None:
                config[source_key] = source
                config[gamma_key] = gamma
            elif source_key == "al_source":
                config.pop("al_source", None)
                config["al_gamma"] = 0.0
        config["stage"] = ("model270_reference" if index == 0
                           else f"exhaustive_{index:04d}")
        config["candidate"] = " | ".join(labels)
        settings.append(config)
    if len(settings) != 2401:
        raise AssertionError(f"expected 2401 routes, got {len(settings)}")
    return settings


def make_mixed(row: dict, setting: dict) -> np.ndarray:
    mixed = row["base"]
    # Preserve Model270's established replacement order and renormalization.
    mixed = replace_classes(
        mixed, row["sources"][setting["primary_source"]], (1,),
        setting["primary_gamma"])
    a_source, a_gamma = setting["al3ni_source"], setting["al3ni_gamma"]
    e_source, e_gamma = setting["eutectic_source"], setting["eutectic_gamma"]
    if a_source == e_source and a_gamma == e_gamma:
        mixed = replace_classes(mixed, row["sources"][a_source], (0, 3), a_gamma)
    else:
        mixed = replace_classes(mixed, row["sources"][a_source], (0,), a_gamma)
        mixed = replace_classes(mixed, row["sources"][e_source], (3,), e_gamma)
    al_gamma = float(setting.get("al_gamma", 0.0))
    if al_gamma:
        source = row["sources"][setting["al_source"]]
        mixed[2] = (1.0 - al_gamma) * mixed[2] + al_gamma * source[2]
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
    return mixed


def _init_worker(rows: list[dict], settings: list[dict]):
    global _WORKER_ROWS, _WORKER_SETTINGS
    _WORKER_ROWS = rows
    _WORKER_SETTINGS = settings


def _evaluate_image(image_index: int):
    row = _WORKER_ROWS[image_index]
    settings = _WORKER_SETTINGS
    route_count = len(settings)
    route_confusion = np.zeros((route_count, 4, 4), dtype=np.int64)
    route_scores = np.zeros(route_count, dtype=np.float64)
    route_scores_exclude = np.zeros(route_count, dtype=np.float64)
    weights = bilateral_weights(row["item"]["z"])
    target = row["item"]["target"]
    for route_index, setting in enumerate(settings):
        mixed = make_mixed(row, setting)
        prediction = predict(row["item"], mixed, setting, weights, exact=True)
        matrix = np.bincount(
            target.reshape(-1) * 4 + prediction.reshape(-1),
            minlength=16).reshape(4, 4)
        route_confusion[route_index] = matrix
        intersection = np.diag(matrix).astype(np.float64)
        union = matrix.sum(0) + matrix.sum(1) - intersection
        iou = intersection / np.maximum(union, 1.0)
        route_scores[route_index] = iou.mean()
        route_scores_exclude[route_index] = iou[union > 0].mean()
        if (route_index + 1) % 300 == 0:
            print(f"image {row['index'] + 1}: routes {route_index + 1}/{route_count}",
                  flush=True)
    return image_index, route_confusion, route_scores, route_scores_exclude


def evaluate(rows: list[dict], settings: list[dict], names: list[str], workers: int):
    count, n_images = len(settings), len(rows)
    confusion = np.zeros((count, 4, 4), dtype=np.int64)
    image_scores = np.zeros((count, n_images), dtype=np.float64)
    image_scores_exclude = np.zeros_like(image_scores)
    image_confusion = np.zeros((count, n_images, 4, 4), dtype=np.int64)
    start = time.monotonic()
    if workers > 1:
        context = mp.get_context("fork")
        with context.Pool(workers, initializer=_init_worker,
                          initargs=(rows, settings)) as pool:
            completed = pool.imap_unordered(_evaluate_image, range(n_images), chunksize=1)
            for done, (image_index, matrices, scores, scores_exclude) in enumerate(completed, 1):
                confusion += matrices
                image_confusion[:, image_index] = matrices
                image_scores[:, image_index] = scores
                image_scores_exclude[:, image_index] = scores_exclude
                elapsed = time.monotonic() - start
                print(f"completed image {done}/{n_images} (index {image_index + 1}); "
                      f"elapsed {elapsed / 60:.1f} min", flush=True)
    else:
        _init_worker(rows, settings)
        for image_index in range(n_images):
            result_index, matrices, scores, scores_exclude = _evaluate_image(image_index)
            confusion += matrices
            image_confusion[:, result_index] = matrices
            image_scores[:, result_index] = scores
            image_scores_exclude[:, result_index] = scores_exclude
            elapsed = time.monotonic() - start
            print(f"completed image {image_index + 1}/{n_images}; "
                  f"elapsed {elapsed / 60:.1f} min", flush=True)

    records = []
    for index, setting in enumerate(settings):
        matrix = confusion[index]
        intersection = np.diag(matrix).astype(np.float64)
        union = matrix.sum(0) + matrix.sum(1) - intersection
        global_iou = intersection / np.maximum(union, 1.0)
        records.append({
            **setting,
            "global_miou": float(global_iou.mean()),
            "global_iou": {
                label: float(value) for label, value in zip(
                    ("Al3Ni", "Primary Si", "Al", "Eutectic Si"), global_iou)},
            "image_mean_miou": float(image_scores[index].mean()),
            "image_mean_miou_exclude": float(image_scores_exclude[index].mean()),
            "image_scores": image_scores[index].tolist(),
            "image_scores_exclude": image_scores_exclude[index].tolist(),
            "confusion": matrix.tolist(),
        })
    return records, image_confusion


def grouped_cv_global(records: list[dict], image_confusion: np.ndarray,
                      names: list[str], seed: int = 335):
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        match = re.match(r"(A\d+)_", Path(name).name)
        if not match:
            raise ValueError(f"cannot parse specimen group from {name}")
        groups.setdefault(match.group(1), []).append(index)
    ref_index = next(i for i, r in enumerate(records)
                     if r["stage"] == "model270_reference")
    held_conf = np.zeros((len(names), 4, 4), dtype=np.int64)
    fold_choices = []
    for held_group, held_indices in groups.items():
        train_indices = [i for group, indices in groups.items() if group != held_group
                         for i in indices]
        best_index = ref_index
        best_score = -1.0
        for candidate_index in range(len(records)):
            cm = image_confusion[candidate_index, train_indices].sum(axis=0)
            inter = np.diag(cm).astype(np.float64)
            union = cm.sum(0) + cm.sum(1) - inter
            score = float((inter / np.maximum(union, 1.0)).mean())
            if score > best_score:
                best_index, best_score = candidate_index, score
        held_conf[held_indices] = image_confusion[best_index, held_indices]
        fold_choices.append({"heldout_specimen": held_group,
                             "selected": records[best_index]["candidate"]})

    ref_conf = image_confusion[ref_index]
    def score(conf):
        inter = np.diag(conf).astype(np.float64)
        union = conf.sum(0) + conf.sum(1) - inter
        return float((inter / np.maximum(union, 1.0)).mean())

    baseline_score = score(ref_conf.sum(axis=0))
    held_score = score(held_conf.sum(axis=0))
    rng = np.random.default_rng(seed)
    group_names = list(groups)
    bootstrap = np.empty(10000, dtype=np.float64)
    for draw in range(len(bootstrap)):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [i for group in sampled for i in groups[str(group)]]
        bootstrap[draw] = (score(held_conf[indices].sum(axis=0))
                           - score(ref_conf[indices].sum(axis=0))) * 100.0
    return {
        "image_mean_global_miou_delta_pp": (held_score - baseline_score) * 100.0,
        "group_bootstrap_95ci_delta_pp": [
            float(v) for v in np.quantile(bootstrap, [0.025, 0.975])],
        "baseline_global_miou": baseline_score,
        "heldout_route_global_miou": held_score,
        "specimen_group_count": len(groups),
        "fold_choices": fold_choices,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True, type=Path)
    parser.add_argument("--model195", required=True, type=Path)
    parser.add_argument("--tent20", required=True, type=Path)
    parser.add_argument("--model269", required=True, type=Path)
    parser.add_argument("--model322", required=True, type=Path)
    parser.add_argument("--model326", required=True, type=Path)
    parser.add_argument("--known-half", type=Path, default=None)
    parser.add_argument("--max-routes", type=int, default=0,
                        help="optional prefix limit for a smoke test; 0 evaluates all 2401")
    parser.add_argument("--workers", type=int, default=1,
                        help="CPU image-level workers; Linux fork is used for shared read-only inputs")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    entries = pickle.load(args.entries.open("rb"))
    paths = {
        "195": (args.model195, False),
        "217": (args.tent20, False),
        "269": (args.model269, False),
        "322": (args.model322, True),
        "326": (args.model326, True),
    }
    archives = {}
    names = None
    for source, (path, team_order) in paths.items():
        archive_names, values = load_archive(path, team_order=team_order)
        if names is None:
            names = archive_names
        elif set(archive_names) != set(names):
            raise ValueError(f"image-key mismatch for {source}")
        archives[source] = values
    assert names is not None
    if len(entries) != len(names):
        raise ValueError(f"entry count {len(entries)} != probability count {len(names)}")

    arrays322 = archives["322"]
    arrays326 = archives["326"]
    arrays_half = {}
    for name in names:
        a, b = arrays322[name], arrays326[name]
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for {name}: {a.shape} vs {b.shape}")
        half = 0.5 * (a + b)
        if not np.allclose(half.sum(axis=0), 1.0, atol=3e-3):
            raise ValueError(f"invalid probability sum for {name}")
        arrays_half[name] = half
    if args.known_half:
        known_names, known = load_archive(args.known_half, team_order=True)
        if set(known_names) != set(names):
            raise ValueError("known Model334 archive image-key mismatch")
        errors = [float(np.max(np.abs(arrays_half[n] - known[n]))) for n in names]
        if max(errors) > 3e-3:
            raise ValueError(f"equal blend mismatch against Model334: max error {max(errors)}")
        print(f"Model334 equal-blend verified; max abs diff={max(errors):.6g}", flush=True)
    archives["b4mean"] = arrays_half

    rows = []
    for index, name in enumerate(names):
        item = entries[index]
        rows.append({
            "index": index,
            "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {
                "base": None,
                "181": item["extra"].astype(np.float32),
                **{source: values[name] for source, values in archives.items()},
            },
        })

    settings = build_settings()
    full_route_count = len(settings)
    if args.max_routes:
        if args.max_routes < 1:
            raise ValueError("--max-routes must be positive")
        settings = settings[:args.max_routes]
    records, image_confusion = evaluate(rows, settings, names, args.workers)
    result = {
        "metric": "Exact image-wise 4-class IoU; empty=0 and NaN-skip, plus pooled global mIoU",
        "route_count": len(records),
        "full_route_count": full_route_count,
        "source_options": ["keep Model270", "Model322", "Model326", "50:50 Model322/326"],
        "gamma_options": list(GAMMAS),
        "class_order": ["Al3Ni", "Primary Si", "Al", "Eutectic Si"],
        "baseline": next(r for r in records if r["stage"] == "model270_reference"),
        "best_image_empty_zero": max(records, key=lambda r: r["image_mean_miou"]),
        "best_image_empty_skipped": max(records, key=lambda r: r["image_mean_miou_exclude"]),
        "best_global": max(records, key=lambda r: r["global_miou"]),
        "grouped_cv_image_empty_zero": grouped_cv(
            records, names, "image_scores", 336),
        "grouped_cv_image_empty_skipped": grouped_cv(
            records, names, "image_scores_exclude", 337),
        "grouped_cv_global": grouped_cv_global(records, image_confusion, names),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "route_count", "baseline", "best_image_empty_zero",
        "best_image_empty_skipped", "best_global",
        "grouped_cv_image_empty_zero", "grouped_cv_image_empty_skipped",
        "grouped_cv_global")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
