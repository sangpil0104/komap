#!/usr/bin/env python3
"""Locked CPU-only Model396 HRNet/B4 OOF blend audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/model396_fixed_architecture_oof_preregistration.md"
SCRIPT = "experiments/model396_fixed_architecture_oof.py"
LOCK = "experiments/results/model396_fixed_architecture_oof_lock.json"
RESULT = "experiments/results/model396_fixed_architecture_oof.json"
SEQUENCE_CLOSURE = "experiments/results/model392_sequence_closure_lock.json"
SEQUENCE_CLOSURE_SHA256 = "3df753de47ab636390ef806237018cd3c302636c0b59cc1dd2ea5236532a551e"
CV_ROOT = "data_model322_group_cv"
B4_ROOT = "outputs/model329_fullstats_width4000_uniform_group_oof"
HRNET_ROOT = "outputs/model382_seed_count_control"
PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
CLASS_NAMES = ("Al", "Al3Ni", "Primary Si", "Eutectic Si")
GROUP_RE = re.compile(r"^(A\d+)_")
STEP_RE = re.compile(r"_step(\d+)_")
MODEL382_EXPECTED_ARGS = {
    "epochs": 70,
    "architecture": "UnetPlusPlus",
    "encoder_name": "tu-hrnet_w48",
    "encoder_weights": "imagenet",
    "patch_size": 512,
    "input_scale": 2.0,
    "samples_per_image": 12,
    "batch_size": 4,
    "accumulation_steps": 2,
    "lr": 3e-4,
    "encoder_lr_mult": 1.0,
    "weight_decay": 1e-4,
    "boundary_head_weight": 1.0,
    "val_every": 0,
    "init_checkpoint": None,
}
B4_EXPECTED_ARGS = {
    "arch": "unet",
    "encoder": "efficientnet-b4",
    "norm": "image_global",
    "crop": 1536,
    "scale": 1.0,
    "target_width": 4000,
    "sample": "uniform",
    "bs": 2,
    "iters": 16000,
    "input": "zcdf3",
    "no_valid_eval": True,
}
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 396


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def mask_files(root: Path) -> dict[int, dict[str, Path]]:
    found: dict[int, dict[str, Path]] = {}
    for fold in range(1, 5):
        directory = root / CV_ROOT / f"oof_fold{fold}" / "masks"
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        items = {
            path.name.removesuffix("_mask.png"): path
            for path in sorted(directory.glob("*_mask.png"))
        }
        if not items:
            raise ValueError(f"no masks in fixed fold {fold}: {directory}")
        found[fold] = items
    names = [name for fold_items in found.values() for name in fold_items]
    groups: dict[str, int] = {}
    for fold, fold_items in found.items():
        for name in fold_items:
            match = GROUP_RE.match(name)
            if match is None or STEP_RE.search(name) is None:
                raise ValueError(f"cannot parse image/specimen/step id: {name}")
            group = match.group(1)
            if group in groups and groups[group] != fold:
                raise ValueError(f"specimen group {group} crosses fixed OOF folds")
            groups[group] = fold
    if len(names) != 70 or len(set(names)) != 70 or len(groups) != 59:
        raise ValueError(f"expected 70 unique OOF images and 59 groups; got {len(set(names))}/{len(groups)}")
    return found


def input_paths(root: Path) -> list[Path]:
    paths = [root / PROTOCOL, root / SCRIPT, root / SEQUENCE_CLOSURE]
    for fold, masks in mask_files(root).items():
        paths.extend(masks.values())
        paths.extend((
            root / B4_ROOT / f"fold{fold}" / "args.json",
            root / B4_ROOT / f"fold{fold}_oof" / "probabilities.npz",
            root / HRNET_ROOT / f"seed1/fold{fold}_oof/model382_manifest.json",
            root / HRNET_ROOT / f"seed1/fold{fold}_oof/valid_probabilities.npz",
            root / HRNET_ROOT / f"seed2/fold{fold}_oof/model382_manifest.json",
            root / HRNET_ROOT / f"seed2/fold{fold}_oof/valid_probabilities.npz",
        ))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted({path.resolve() for path in paths})


def create_lock(root: Path) -> dict:
    lock_path = root / LOCK
    if lock_path.exists():
        raise FileExistsError(f"refusing to overwrite lock: {lock_path}")
    closure_path = root / SEQUENCE_CLOSURE
    if sha256(closure_path) != SEQUENCE_CLOSURE_SHA256:
        raise ValueError("Model392 sequence-closure receipt hash differs from its frozen reference")
    masks = mask_files(root)
    b4_base = root / B4_ROOT
    for fold in range(1, 5):
        args = json.loads((b4_base / f"fold{fold}/args.json").read_text(encoding="utf-8"))
        if (any(args.get(key) != value for key, value in B4_EXPECTED_ARGS.items())
                or not str(args.get("data", "")).endswith("/data_model322_group_cv")
                or args.get("train_splits") != f"train_fold{fold}"):
            raise ValueError(f"Model329 recipe mismatch in fold{fold}/args.json")
    files = {
        relative(root, path): {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in input_paths(root)
    }
    receipt = {
        "experiment": "Model396 fixed architecture-diverse OOF blend",
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "decision": "fixed arms only; no validation/test; no Model392 artifacts or score accessed",
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "class_order": list(CLASS_NAMES),
        "folds": 4,
        "images": 70,
        "specimen_groups": 59,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "locked_files": files,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


def verify_lock(root: Path) -> dict:
    lock_path = root / LOCK
    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    if (receipt.get("experiment") != "Model396 fixed architecture-diverse OOF blend"
            or receipt.get("revision") != 1
            or receipt.get("metric") != "image-wise 4-class mIoU; empty-union=0"
            or receipt.get("bootstrap_draws") != BOOTSTRAP_DRAWS
            or receipt.get("bootstrap_seed") != BOOTSTRAP_SEED):
        raise ValueError("unexpected Model396 lock or protocol settings")
    expected_paths = {relative(root, path) for path in input_paths(root)}
    locked = receipt.get("locked_files", {})
    if set(locked) != expected_paths:
        raise ValueError("locked artifact inventory differs from current required inputs")
    for name, metadata in locked.items():
        path = root / name
        if path.stat().st_size != metadata["size_bytes"] or sha256(path) != metadata["sha256"]:
            raise ValueError(f"locked input changed: {name}")
    return receipt


def mask_to_ids(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        result[np.all(rgb == color, axis=2)] = class_id
    if np.any(result == 255):
        raise ValueError(f"unknown mask RGB values in {path}")
    return result


def class_ious(probabilities: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if probabilities.shape != (4, *truth.shape) or not np.isfinite(probabilities).all():
        raise ValueError(f"invalid probability shape/values {probabilities.shape} for {truth.shape}")
    if probabilities.min() < -1e-4 or probabilities.max() > 1.0001:
        raise ValueError("probability values outside [0,1]")
    if np.max(np.abs(probabilities.sum(axis=0) - 1.0)) > 0.03:
        raise ValueError("probability map is not normalized")
    prediction = probabilities.argmax(axis=0)
    matrix = np.bincount(
        truth.astype(np.int64).ravel() * 4 + prediction.astype(np.int64).ravel(),
        minlength=16,
    ).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros(4), where=union > 0)


def verify_model382_manifest(root: Path, closure: dict, seed: int, fold: int,
                              expected_ids: set[str]) -> Path:
    directory = root / HRNET_ROOT / f"seed{seed}/fold{fold}_oof"
    archive_path = directory / "valid_probabilities.npz"
    manifest_path = directory / "model382_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("model") != 382 or manifest.get("seed") != seed
            or manifest.get("fold") != fold or manifest.get("epochs") != 70
            or manifest.get("metric") != "image-wise 4-class mIoU; empty-union=0"
            or manifest.get("oof_probabilities_sha256") != sha256(archive_path)):
        raise ValueError(f"Model382 manifest provenance mismatch: seed{seed}/fold{fold}")
    heldout = Path(str(manifest.get("heldout_images", ""))).as_posix()
    if not heldout.endswith(f"/data_model322_group_cv/oof_fold{fold}"):
        raise ValueError(f"Model382 held-out split mismatch: seed{seed}/fold{fold}")
    args = manifest.get("training_args", {})
    if (any(args.get(key) != value for key, value in MODEL382_EXPECTED_ARGS.items())
            or args.get("seed") != seed):
        raise ValueError(f"Model382 recipe mismatch: seed{seed}/fold{fold}")
    closure_refs = closure["model382_same_seed_fold_references"][f"seed{seed}/fold{fold}"]
    if manifest.get("checkpoint_sha256") != closure_refs["checkpoint"]["sha256"]:
        raise ValueError(f"Model382 checkpoint binding mismatch: seed{seed}/fold{fold}")
    if closure_refs["oof_probabilities"]["sha256"] != sha256(archive_path):
        raise ValueError(f"Model382 OOF differs from sequence closure: seed{seed}/fold{fold}")
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_ids:
            raise ValueError(f"Model382 OOF IDs differ from fixed fold: seed{seed}/fold{fold}")
    return archive_path


def group_bootstrap_indices(rows: list[dict], draws: int, seed: int) -> list[np.ndarray]:
    by_fold_group: dict[int, dict[str, list[int]]] = {fold: {} for fold in range(1, 5)}
    for index, row in enumerate(rows):
        by_fold_group[row["fold"]].setdefault(row["group"], []).append(index)
    if any(not groups for groups in by_fold_group.values()):
        raise ValueError("empty fold in group bootstrap")
    rng = np.random.default_rng(seed)
    output = []
    for draw in range(draws):
        selected = []
        for fold in range(1, 5):
            groups = sorted(by_fold_group[fold])
            sample = rng.integers(0, len(groups), size=len(groups))
            for group_index in sample:
                selected.extend(by_fold_group[fold][groups[int(group_index)]])
        output.append(np.asarray(selected, dtype=np.int32))
    return output


def comparison(candidate: np.ndarray, reference: np.ndarray, rows: list[dict],
               bootstrap_indices: np.ndarray) -> dict:
    delta_class = (candidate - reference) * 100.0
    delta_image = delta_class.mean(axis=1)
    sampled = np.asarray([delta_image[indices].mean() for indices in bootstrap_indices])
    fold_delta = {
        str(fold): float(delta_image[np.asarray([r["fold"] == fold for r in rows])].mean())
        for fold in range(1, 5)
    }
    step4 = np.asarray([row["step"] == 4 for row in rows])
    group_sum: dict[str, float] = {}
    for row, value in zip(rows, delta_image):
        group_sum[row["group"]] = group_sum.get(row["group"], 0.0) + float(value)
    net_sum = float(delta_image.sum())
    largest_group_share = (
        max(0.0, max(group_sum.values())) / net_sum if net_sum > 0 else None
    )
    return {
        "delta_image_mean_pp": float(delta_image.mean()),
        "paired_group_bootstrap_95ci_pp": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
        "positive_images": int(np.count_nonzero(delta_image > 0)),
        "positive_folds": int(sum(value > 0 for value in fold_delta.values())),
        "fold_delta_pp": fold_delta,
        "class_delta_pp": {
            name: float(delta_class[:, index].mean()) for index, name in enumerate(CLASS_NAMES)
        },
        "step4_delta_pp": float(delta_image[step4].mean()),
        "largest_group_positive_contribution_share": largest_group_share,
        "group_delta_sum_pp": {key: group_sum[key] for key in sorted(group_sum)},
    }


def overall(class_scores: np.ndarray) -> dict:
    return {
        "image_mean_miou_percent": float(class_scores.mean(axis=1).mean() * 100.0),
        "class_image_mean_iou_percent": {
            name: float(class_scores[:, index].mean() * 100.0)
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def screen_gate(comparisons: dict[str, dict]) -> dict:
    checks = {}
    for name, single in (("D1", "S1"), ("D2", "S2")):
        against_b2 = comparisons[f"{name}_vs_B2"]
        against_single = comparisons[f"{name}_vs_{single}"]
        checks[name] = {
            "vs_B2_gain_at_least_0_30pp": against_b2["delta_image_mean_pp"] >= 0.30,
            "vs_B2_ci_lower_above_zero": against_b2["paired_group_bootstrap_95ci_pp"][0] > 0.0,
            "vs_B2_positive_in_at_least_3_folds": against_b2["positive_folds"] >= 3,
            "vs_single_gain_at_least_0_50pp": against_single["delta_image_mean_pp"] >= 0.50,
            "vs_single_ci_lower_above_zero": against_single["paired_group_bootstrap_95ci_pp"][0] > 0.0,
            "classes_and_step4_not_below_minus_0_50pp": (
                min(against_b2["class_delta_pp"].values()) >= -0.50
                and against_b2["step4_delta_pp"] >= -0.50
            ),
            "largest_group_share_at_most_25_percent": (
                against_b2["largest_group_positive_contribution_share"] is not None
                and against_b2["largest_group_positive_contribution_share"] <= 0.25
            ),
        }
    return {
        "checks": checks,
        "pass": all(all(values.values()) for values in checks.values()),
        "policy": "both architecture-diverse arms must pass every check; no weight search",
    }


def run(root: Path) -> dict:
    receipt = verify_lock(root)
    output_path = root / RESULT
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite result: {output_path}")
    closure_path = root / SEQUENCE_CLOSURE
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    if sha256(closure_path) != SEQUENCE_CLOSURE_SHA256:
        raise ValueError("Model382 sequence-closure lock changed")

    masks_by_fold = mask_files(root)
    rows: list[dict] = []
    scores = {arm: [] for arm in ("S1", "S2", "B2", "D1", "D2")}
    b4_base = root / B4_ROOT
    for fold in range(1, 5):
        mask_map = masks_by_fold[fold]
        expected_ids = set(mask_map)
        b4_args = json.loads((b4_base / f"fold{fold}/args.json").read_text(encoding="utf-8"))
        if (any(b4_args.get(key) != value for key, value in B4_EXPECTED_ARGS.items())
                or not str(b4_args.get("data", "")).endswith("/data_model322_group_cv")
                or b4_args.get("train_splits") != f"train_fold{fold}"):
            raise ValueError(f"Model329 recipe mismatch in fold{fold}")
        b4_path = b4_base / f"fold{fold}_oof/probabilities.npz"
        model382_paths = {
            seed: verify_model382_manifest(root, closure, seed, fold, expected_ids)
            for seed in (1, 2)
        }
        with np.load(b4_path, allow_pickle=False) as b4_archive, \
             np.load(model382_paths[1], allow_pickle=False) as s1_archive, \
             np.load(model382_paths[2], allow_pickle=False) as s2_archive:
            if any(set(archive.files) != expected_ids for archive in (b4_archive, s1_archive, s2_archive)):
                raise ValueError(f"OOF key mismatch in fold {fold}")
            for image_id in sorted(expected_ids):
                truth = mask_to_ids(mask_map[image_id])
                p_b4 = np.asarray(b4_archive[image_id], dtype=np.float32)
                p_s1 = np.asarray(s1_archive[image_id], dtype=np.float32)
                p_s2 = np.asarray(s2_archive[image_id], dtype=np.float32)
                class_scores = {
                    "S1": class_ious(p_s1, truth),
                    "S2": class_ious(p_s2, truth),
                    "B2": class_ious(0.5 * (p_s1 + p_s2), truth),
                    "D1": class_ious(0.5 * (p_s1 + p_b4), truth),
                    "D2": class_ious(0.5 * (p_s2 + p_b4), truth),
                }
                group = GROUP_RE.match(image_id).group(1)
                step = int(STEP_RE.search(image_id).group(1))
                rows.append({"image": image_id, "group": group, "step": step, "fold": fold})
                for arm in scores:
                    scores[arm].append(class_scores[arm])
        print(f"validated and scored fold {fold}/4 OOF IDs", flush=True)

    if len(rows) != 70 or len({row["image"] for row in rows}) != 70:
        raise ValueError("OOF evaluation did not cover exactly 70 unique images")
    if len({row["group"] for row in rows}) != 59:
        raise ValueError("OOF evaluation did not cover exactly 59 specimen groups")
    arrays = {arm: np.asarray(values, dtype=np.float64) for arm, values in scores.items()}
    bootstrap_indices = group_bootstrap_indices(rows, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    comparisons = {
        "D1_vs_B2": comparison(arrays["D1"], arrays["B2"], rows, bootstrap_indices),
        "D2_vs_B2": comparison(arrays["D2"], arrays["B2"], rows, bootstrap_indices),
        "D1_vs_S1": comparison(arrays["D1"], arrays["S1"], rows, bootstrap_indices),
        "D2_vs_S2": comparison(arrays["D2"], arrays["S2"], rows, bootstrap_indices),
    }
    gate = screen_gate(comparisons)
    result = {
        "experiment": "Model396 fixed architecture-diverse OOF blend",
        "status": "SCREEN_PASS" if gate["pass"] else "NO_GO",
        "metric": "image-wise 4-class mIoU; empty-union=0",
        "class_order": list(CLASS_NAMES),
        "scope": "train-only exactly-once grouped OOF; selection-conditional development evidence",
        "official_valid_used": False,
        "test_used": False,
        "gpu_used": False,
        "model392_outputs_read": False,
        "lock_sha256": sha256(root / LOCK),
        "arms": {arm: overall(arrays[arm]) for arm in scores},
        "comparisons": comparisons,
        "pre_registered_gate": gate,
        "decision": "A screen pass only permits considering a separately locked final candidate; it is not independent confirmation.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def self_test() -> None:
    truth = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    probabilities = np.zeros((4, 2, 2), dtype=np.float32)
    probabilities[0] = (truth == 0)
    probabilities[1] = (truth == 1)
    assert np.allclose(class_ious(probabilities, truth)[:2], 1.0)
    assert np.all(class_ious(probabilities, truth)[2:] == 0.0)
    rows = []
    for fold, groups in ((1, ((1, 1), (2, 2), (3, 3))),
                         (2, ((4, 1), (5, 2), (6, 3))),
                         (3, ((7, 1), (8, 2), (9, 3))),
                         (4, ((10, 1), (11, 2), (12, 3)))):
        for group, image_count in groups:
            for sample in range(image_count):
                rows.append({
                    "image": f"A{group}_step4_sample{sample + 1}",
                    "group": f"A{group}", "fold": fold, "step": 4,
                })
    indices_a = group_bootstrap_indices(rows, 100, BOOTSTRAP_SEED)
    indices_b = group_bootstrap_indices(rows, 100, BOOTSTRAP_SEED)
    assert len(indices_a) == 100 and all(
        np.array_equal(a, b) for a, b in zip(indices_a, indices_b)
    )
    assert len({len(indices) for indices in indices_a}) > 1
    print("Model396 smoke passed: official empty=0 metric and deterministic fold-group bootstrap.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--create-lock", action="store_true")
    mode.add_argument("--verify-lock", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.create_lock:
        receipt = create_lock(args.root.resolve())
        print(json.dumps({"locked_files": len(receipt["locked_files"]), "lock": str(args.root / LOCK)}, indent=2))
    elif args.verify_lock:
        receipt = verify_lock(args.root.resolve())
        print(json.dumps({"verified_files": len(receipt["locked_files"]), "lock_sha256": sha256(args.root.resolve() / LOCK)}, indent=2))
    else:
        print(json.dumps(run(args.root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
