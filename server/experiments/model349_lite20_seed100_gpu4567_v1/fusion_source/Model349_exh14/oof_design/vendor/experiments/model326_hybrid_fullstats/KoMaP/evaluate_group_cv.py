#!/usr/bin/env python3
"""Score B4 grouped-OOF predictions and test a small preregistered fusion grid."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from common import list_split, load_mask

MAIN_TO_TEAM_ORDER = (2, 0, 1, 3)  # [Al3Ni, Primary, Al, Eutectic] -> [Al, Al3Ni, Primary, Eutectic]

def image_score(prob: np.ndarray, target: np.ndarray) -> tuple[float, float, np.ndarray]:
    prediction = prob.argmax(0)
    confusion = np.bincount(target.astype(np.int64).ravel() * 4 + prediction.ravel(),
                            minlength=16).reshape(4, 4)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.full(4, np.nan), where=union > 0)
    fixed = np.nan_to_num(iou, nan=0.0).mean()
    skipped = np.nanmean(iou)
    return float(fixed), float(skipped), iou


def group_of(name: str) -> str:
    match = re.match(r"(A\d+)_", name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {name}")
    return match.group(1)


def probability_for(archive, name: str) -> np.ndarray:
    for key in (name, f"{name}_image"):
        if key in archive:
            return archive[key].astype(np.float32)
    raise KeyError(f"{name} (or {name}_image) missing from probability archive")


def read_scores(prob_path: Path, pairs, convert_to_team_order: bool = False):
    scores, skipped_scores, ious, names, groups = [], [], [], [], []
    with np.load(prob_path) as archive:
        for image_path, mask_path in pairs:
            name = Path(image_path).name.replace("_image.png", "")
            try:
                probs = probability_for(archive, name)
            except KeyError as exc:
                raise KeyError(f"{exc} in {prob_path}") from exc
            if convert_to_team_order:
                probs = probs[list(MAIN_TO_TEAM_ORDER)]
            fixed, skipped, iou = image_score(probs, load_mask(mask_path))
            names.append(name)
            groups.append(group_of(name))
            scores.append(fixed)
            skipped_scores.append(skipped)
            ious.append(np.nan_to_num(iou, nan=0.0))
    return (names, groups, np.asarray(scores), np.asarray(skipped_scores),
            np.asarray(ious))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--parent-oof", required=True, type=Path)
    parser.add_argument("--candidate-oof", required=True, type=Path)
    parser.add_argument("--parent-valid", required=True, type=Path)
    parser.add_argument("--candidate-valid", required=True, type=Path)
    parser.add_argument("--parent-order", choices=("team", "main"), default="main",
                        help="parent archive channels; defaults to legacy main order")
    parser.add_argument("--candidate-order", choices=("team", "main"), default="team",
                        help="candidate archive channels")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    train_pairs = list_split(str(args.data), "train")
    valid_pairs = list_split(str(args.data), "valid")
    parent_is_main_order = args.parent_order == "main"
    names, groups, parent, parent_skip, parent_class = read_scores(
        args.parent_oof, train_pairs, convert_to_team_order=parent_is_main_order)
    candidate_names, candidate_groups, branch, branch_skip, branch_class = read_scores(
        args.candidate_oof, train_pairs, convert_to_team_order=(args.candidate_order == "main"))
    if names != candidate_names or groups != candidate_groups:
        raise ValueError("parent and candidate OOF key order mismatch")

    settings = [{"name": "parent", "gammas": [0.0, 0.0, 0.0, 0.0]}]
    class_ids = (0, 1, 2, 3)  # Include Al as a possible complementary B4 channel too.
    for subset in range(1, 1 << len(class_ids)):
        for gamma in (0.30, 0.50):
            weights = [0.0, 0.0, 0.0, 0.0]
            chosen = []
            for bit, class_id in enumerate(class_ids):
                if subset & (1 << bit):
                    weights[class_id] = gamma
                    chosen.append(str(class_id))
            settings.append({"name": f"b4_{','.join(chosen)}_{gamma:.2f}",
                             "gammas": weights})
    settings.append({"name": "b4_all", "gammas": [1.0, 1.0, 1.0, 1.0]})

    group_indices: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_indices.setdefault(group, []).append(index)
    score_matrix = np.empty((len(settings), len(names)), dtype=np.float64)
    skipped_matrix = np.empty_like(score_matrix)
    class_matrix = np.empty((len(settings), len(names), 4), dtype=np.float64)
    score_matrix[0], skipped_matrix[0], class_matrix[0] = parent, parent_skip, parent_class
    train_targets = [load_mask(mask_path) for _, mask_path in train_pairs]
    with np.load(args.parent_oof) as parent_archive, np.load(args.candidate_oof) as branch_archive:
        for index, name in enumerate(names):
            base = probability_for(parent_archive, name)
            if parent_is_main_order:
                base = base[list(MAIN_TO_TEAM_ORDER)]
            candidate = probability_for(branch_archive, name)
            if args.candidate_order == "main":
                candidate = candidate[list(MAIN_TO_TEAM_ORDER)]
            target = train_targets[index]
            for setting_index, setting in enumerate(settings[1:], start=1):
                p0 = base.copy()
                gamma = setting["gammas"]
                for class_id in class_ids:
                    alpha = gamma[class_id]
                    if alpha:
                        p0[class_id] = (1.0 - alpha) * p0[class_id] + alpha * candidate[class_id]
                denom = np.maximum(p0.sum(0, keepdims=True), 1e-8)
                fixed, skipped, iou = image_score(p0 / denom, target)
                score_matrix[setting_index, index] = fixed
                skipped_matrix[setting_index, index] = skipped
                class_matrix[setting_index, index] = np.nan_to_num(iou, nan=0.0)
    exact_means = score_matrix.mean(axis=1)
    parent_index = 0
    cv_delta = np.zeros(len(names), dtype=np.float64)
    cv_delta_skipped = np.zeros(len(names), dtype=np.float64)
    fold_routes = []
    for held_group, held_indices in group_indices.items():
        train_indices = [i for group, indices in group_indices.items() if group != held_group
                         for i in indices]
        selected = int(np.argmax(score_matrix[:, train_indices].mean(axis=1)))
        cv_delta[held_indices] = (score_matrix[selected, held_indices]
                                  - score_matrix[parent_index, held_indices])
        cv_delta_skipped[held_indices] = (skipped_matrix[selected, held_indices]
                                          - skipped_matrix[parent_index, held_indices])
        fold_routes.append({"heldout_specimen": held_group,
                            "selected": settings[selected]["name"]})

    rng = np.random.default_rng(322)
    group_names = list(group_indices)
    bootstrap = np.empty(10000)
    bootstrap_skipped = np.empty(10000)
    for draw in range(len(bootstrap)):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in sampled for index in group_indices[str(group)]]
        bootstrap[draw] = cv_delta[indices].mean() * 100
        bootstrap_skipped[draw] = cv_delta_skipped[indices].mean() * 100

    valid_names, valid_groups, valid_parent, valid_parent_skip, valid_parent_class = read_scores(
        args.parent_valid, valid_pairs, convert_to_team_order=parent_is_main_order)
    candidate_valid_names, _, valid_branch, valid_branch_skip, valid_branch_class = read_scores(
        args.candidate_valid, valid_pairs,
        convert_to_team_order=(args.candidate_order == "main"))
    if valid_names != candidate_valid_names:
        raise ValueError("parent and candidate validation key order mismatch")

    valid_setting_scores = np.empty((len(settings), len(valid_names)), dtype=np.float64)
    valid_setting_skipped = np.empty_like(valid_setting_scores)
    valid_setting_classes = np.empty((len(settings), len(valid_names), 4), dtype=np.float64)
    # Recompute probabilities once per image, then score the preregistered route grid.
    with np.load(args.parent_valid) as base_archive, np.load(args.candidate_valid) as branch_archive:
        valid_targets = [load_mask(mask_path) for _, mask_path in valid_pairs]
        for index, name in enumerate(valid_names):
            p_base = probability_for(base_archive, name)
            if parent_is_main_order:
                p_base = p_base[list(MAIN_TO_TEAM_ORDER)]
            p_branch = probability_for(branch_archive, name)
            if args.candidate_order == "main":
                p_branch = p_branch[list(MAIN_TO_TEAM_ORDER)]
            for setting_index, setting in enumerate(settings):
                p0 = p_base.copy()
                for class_id in class_ids:
                    alpha = setting["gammas"][class_id]
                    if alpha:
                        p0[class_id] = ((1.0 - alpha) * p0[class_id]
                                        + alpha * p_branch[class_id])
                p0 /= np.maximum(p0.sum(0, keepdims=True), 1e-8)
                fixed, skipped, iou = image_score(p0, valid_targets[index])
                valid_setting_scores[setting_index, index] = fixed
                valid_setting_skipped[setting_index, index] = skipped
                valid_setting_classes[setting_index, index] = np.nan_to_num(iou, nan=0.0)

    best_oof = int(np.argmax(exact_means))
    selected_cv_names = [row["selected"] for row in fold_routes]
    route_name = max((row["name"] for row in settings), key=selected_cv_names.count)
    # Use the most frequently selected route for a single deployable validation score.
    deployed_idx = next(i for i, row in enumerate(settings) if row["name"] == route_name)
    valid_matrix = valid_setting_scores
    valid_skip_matrix = valid_setting_skipped
    valid_best = int(np.argmax(valid_matrix.mean(axis=1)))

    def summary(idx: int, score_mat: np.ndarray, skip_mat: np.ndarray,
                class_mat: np.ndarray) -> dict:
        return {"setting": settings[idx], "image_mean_miou_empty_zero": float(score_mat[idx].mean()),
                "image_mean_miou_empty_skipped": float(skip_mat[idx].mean()),
                "per_class_image_mean_iou_empty_zero": class_mat[idx].mean(axis=0).tolist()}

    result = {
        "metric": "per-image mean of four class IoUs; empty union=0",
        "class_order": ["Al", "Al3Ni", "Primary Si", "Eutectic Si"],
        "train_count": len(names), "train_group_count": len(group_indices),
        "valid_count": len(valid_names), "candidate_count": len(settings),
        "parent_oof": {"image_mean_miou_empty_zero": float(parent.mean()),
                       "image_mean_miou_empty_skipped": float(parent_skip.mean()),
                       "per_class_image_mean_iou_empty_zero": parent_class.mean(axis=0).tolist()},
        "b4_oof_standalone": {"image_mean_miou_empty_zero": float(branch.mean()),
                              "image_mean_miou_empty_skipped": float(branch_skip.mean()),
                              "per_class_image_mean_iou_empty_zero": branch_class.mean(axis=0).tolist()},
        "best_exact_oof_setting": summary(best_oof, score_matrix, skipped_matrix,
                                           class_matrix),
        "grouped_cv_route": {
            "image_mean_delta_pp_empty_zero": float(cv_delta.mean() * 100),
            "image_mean_delta_pp_empty_skipped": float(cv_delta_skipped.mean() * 100),
            "positive_images": int((cv_delta > 0).sum()),
            "group_bootstrap_95ci_delta_pp_empty_zero": [
                float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
            "group_bootstrap_95ci_delta_pp_empty_skipped": [
                float(x) for x in np.quantile(bootstrap_skipped, [0.025, 0.975])],
            "fold_routes": fold_routes,
            "majority_route_for_valid": settings[deployed_idx],
        },
        "valid_parent": {"image_mean_miou_empty_zero": float(valid_parent.mean()),
                          "image_mean_miou_empty_skipped": float(valid_parent_skip.mean()),
                          "per_class_image_mean_iou_empty_zero": valid_parent_class.mean(axis=0).tolist()},
        "valid_b4_standalone": {"image_mean_miou_empty_zero": float(valid_branch.mean()),
                                "image_mean_miou_empty_skipped": float(valid_branch_skip.mean()),
                                "per_class_image_mean_iou_empty_zero": valid_branch_class.mean(axis=0).tolist()},
        "valid_group_cv_route": summary(deployed_idx, valid_matrix, valid_skip_matrix,
                                        valid_setting_classes),
        "valid_best_in_sample_not_for_selection": summary(valid_best, valid_matrix, valid_skip_matrix,
                                                            valid_setting_classes),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
