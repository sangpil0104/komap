#!/usr/bin/env python3
"""Train-OOF-fitted class-wise blend of Model322 and Model333 CE+Dice control.

The OOF mode selects gamma only from exactly-once specimen-group OOF
predictions and reports leave-one-specimen-group-out selection. The valid mode
applies the saved gamma once to independent valid probabilities; it never tunes
gamma on validation labels. Channel order is the team B4 order.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np


CLASS_ORDER = ("Al", "Al3Ni", "Primary Si", "Eutectic Si")


def image_group(name: str) -> str:
    match = re.match(r"(A\d+)_", Path(name).name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {name}")
    return match.group(1)


def class_iou(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    matrix = np.bincount(target.astype(np.int64).ravel() * 4 +
                         prediction.astype(np.int64).ravel(), minlength=16).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return np.divide(intersection, union, out=np.zeros(4, dtype=np.float64), where=union > 0)


def blend_prediction(parent: np.ndarray, candidate: np.ndarray,
                     gamma: tuple[float, ...]) -> np.ndarray:
    mixed = np.empty(parent.shape, dtype=np.float32)
    for class_id, weight in enumerate(gamma):
        mixed[class_id] = ((1.0 - weight) * parent[class_id]
                           + weight * candidate[class_id])
    return mixed.argmax(axis=0).astype(np.uint8)


def grouped_cv(scores: np.ndarray, names: list[str], bootstrap_draws: int,
               seed: int) -> dict:
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(image_group(name), []).append(index)
    selected = np.empty(len(names), dtype=np.int64)
    folds = []
    for held_group, held_indices in groups.items():
        train_indices = [index for group, indices in groups.items()
                         if group != held_group for index in indices]
        setting = int(np.argmax(scores[:, train_indices].mean(axis=1)))
        selected[held_indices] = setting
        folds.append({"heldout_specimen": held_group,
                      "selected_setting_index": setting})

    reference = scores[0]
    held = scores[selected, np.arange(len(names))]
    delta = held - reference
    rng = np.random.default_rng(seed)
    group_names = list(groups)
    bootstrap = np.empty(bootstrap_draws, dtype=np.float64)
    for draw in range(bootstrap_draws):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in sampled for index in groups[str(group)]]
        bootstrap[draw] = delta[indices].mean() * 100.0
    return {
        "reference": "Model322 parent (all gamma=0)",
        "image_mean_delta_pp": float(delta.mean() * 100.0),
        "positive_images": int((delta > 0).sum()),
        "image_count": len(names),
        "specimen_group_count": len(groups),
        "group_bootstrap_95ci_delta_pp": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "folds": folds,
    }


def run_oof(args: argparse.Namespace) -> dict:
    code = args.root / "model326_hybrid_fullstats/KoMaP"
    import sys
    sys.path.insert(0, str(code))
    from common import list_split, load_mask  # pylint: disable=import-outside-toplevel

    data = args.root / "data_model322_group_cv"
    rows = []
    for fold in range(1, 5):
        pairs = list_split(str(data), f"oof_fold{fold}")
        parent_path = args.root / "outputs/model322_team_b4_group_oof" / \
            f"fold{fold}_oof/probabilities.npz"
        candidate_path = args.root / "outputs/model333_b4_imagewise_lovasz_group_oof" / \
            "control" / f"fold{fold}_oof/probabilities.npz"
        for path in (parent_path, candidate_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with (np.load(parent_path) as parent_archive,
              np.load(candidate_path) as candidate_archive):
            for image_path, mask_path in pairs:
                name = Path(image_path).name.replace("_image.png", "")
                key_parent = name if name in parent_archive else f"{name}_image"
                key_candidate = name if name in candidate_archive else f"{name}_image"
                if key_parent not in parent_archive or key_candidate not in candidate_archive:
                    raise KeyError(f"missing paired OOF probability for {name}")
                parent = parent_archive[key_parent].astype(np.float32)
                candidate = candidate_archive[key_candidate].astype(np.float32)
                target = load_mask(mask_path)
                if parent.shape != candidate.shape or parent.shape != (4, *target.shape):
                    raise ValueError(f"probability shape mismatch for {name}: "
                                     f"{parent.shape}/{candidate.shape}/{target.shape}")
                rows.append((name, parent, candidate, target))

    names = [row[0] for row in rows]
    if len(names) != 70 or len(set(names)) != 70:
        raise ValueError(f"expected 70 unique exactly-once OOF images, found {len(names)}")
    settings = list(itertools.product(args.gammas, repeat=4))
    scores = np.empty((len(settings), len(rows)), dtype=np.float64)
    class_scores = np.empty((len(settings), len(rows), 4), dtype=np.float64)
    start = time.monotonic()
    for image_index, (name, parent, candidate, target) in enumerate(rows):
        for setting_index, gamma in enumerate(settings):
            iou = class_iou(blend_prediction(parent, candidate, gamma), target)
            scores[setting_index, image_index] = iou.mean()
            class_scores[setting_index, image_index] = iou
        print(f"scored OOF {image_index + 1}/{len(rows)}; "
              f"elapsed {(time.monotonic() - start) / 60.0:.1f} min", flush=True)

    best = int(np.argmax(scores.mean(axis=1)))
    cv = grouped_cv(scores, names, args.bootstrap, seed=344)
    return {
        "model": 344,
        "stage": "train grouped-OOF gamma selection",
        "metric": "image-wise four-class mIoU; empty union=0",
        "class_order": list(CLASS_ORDER),
        "candidate": "Model322 parent blended with Model333 CE+Dice control",
        "completed_folds": [1, 2, 3, 4],
        "image_count": len(names),
        "specimen_group_count": len(set(image_group(name) for name in names)),
        "gamma_grid": list(args.gammas),
        "candidate_count": len(settings),
        "selected_gamma": list(settings[best]),
        "selected_gamma_by_class": dict(zip(CLASS_ORDER, settings[best])),
        "model322_parent_train_oof_miou_empty0": float(scores[0].mean() * 100.0),
        "selected_train_oof_miou_empty0": float(scores[best].mean() * 100.0),
        "selected_train_oof_class_mean_iou_empty0":
            (class_scores[best].mean(axis=0) * 100.0).tolist(),
        "grouped_cv_vs_model322": cv,
        "settings_train_oof_mean_miou_empty0":
            (scores.mean(axis=1) * 100.0).tolist(),
        "settings_train_oof_class_mean_iou_empty0":
            (class_scores.mean(axis=1) * 100.0).tolist(),
    }


def run_valid(args: argparse.Namespace, oof_result: dict) -> dict:
    code = args.root / "model326_hybrid_fullstats/KoMaP"
    import sys
    sys.path.insert(0, str(code))
    from common import list_split, load_mask  # pylint: disable=import-outside-toplevel

    outputs = args.root / "outputs"
    parent_path = outputs / "model322_team_b4_group_oof/valid_ensemble/probabilities.npz"
    candidate_path = outputs / \
        "model333_b4_imagewise_lovasz_group_oof/control/valid_ensemble/probabilities.npz"
    for path in (parent_path, candidate_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    pairs = list_split(str(args.root / "data"), "valid")
    scores = []
    per_class = []
    images = []
    gamma = tuple(float(value) for value in oof_result["selected_gamma"])
    with (np.load(parent_path) as parent_archive,
          np.load(candidate_path) as candidate_archive):
        expected = {Path(image).name.replace("_image.png", "") for image, _ in pairs}
        if set(parent_archive.files) != expected or set(candidate_archive.files) != expected:
            raise ValueError("valid probability keys do not match the valid split")
        for image_path, mask_path in pairs:
            name = Path(image_path).name.replace("_image.png", "")
            parent = parent_archive[name].astype(np.float32)
            candidate = candidate_archive[name].astype(np.float32)
            target = load_mask(mask_path)
            iou = class_iou(blend_prediction(parent, candidate, gamma), target)
            score = float(iou.mean() * 100.0)
            scores.append(score)
            per_class.append(iou * 100.0)
            images.append({"image": name, "image_miou_empty0": score,
                           "class_iou_empty0": (iou * 100.0).tolist()})
    return {
        "model": 344,
        "stage": "fixed OOF-selected gamma, final validation scoring",
        "metric": "image-wise four-class mIoU; empty union=0",
        "class_order": list(CLASS_ORDER),
        "selected_gamma": list(gamma),
        "gamma_source": "Model344 exact train grouped-OOF results; valid labels not used for selection",
        "valid_image_count": len(scores),
        "valid_image_mean_miou_empty0": float(np.mean(scores)),
        "valid_class_mean_iou_empty0": np.mean(per_class, axis=0).tolist(),
        "valid_images": images,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--mode", choices=("oof", "valid", "both"), required=True)
    parser.add_argument("--oof-results", type=Path, default=None,
                        help="required for --mode valid; selected gamma JSON from OOF stage")
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value < 0 or value > 1 for value in args.gammas):
        raise ValueError("gamma candidates must lie in [0,1]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    if args.mode == "oof":
        result = run_oof(args)
    elif args.mode == "valid":
        if args.oof_results is None or not args.oof_results.is_file():
            raise FileNotFoundError("--mode valid requires --oof-results from Model344 OOF stage")
        result = run_valid(args, json.loads(args.oof_results.read_text(encoding="utf-8")))
    else:
        result = run_oof(args)
        result["validation"] = run_valid(args, result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("settings_train_oof_mean_miou_empty0",
                                     "settings_train_oof_class_mean_iou_empty0",
                                     "valid_images")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
