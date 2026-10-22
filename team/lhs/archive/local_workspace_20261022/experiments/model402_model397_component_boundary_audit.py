#!/usr/bin/env python3
"""One-shot CPU audit of Model397 train-OOF errors by GT geometry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, label

import model396_fixed_architecture_oof as m396
import model397_fixed_b4_model392_blend as m397
import model397_train_oof_error_type_audit as m397audit


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/model402_model397_component_boundary_audit.md"
SCRIPT = "experiments/model402_model397_component_boundary_audit.py"
SMOKE = "experiments/model402_model397_component_boundary_audit_smoke.py"
LOCK = "experiments/results/model402_model397_component_boundary_audit_lock.json"
CLAIM = "experiments/results/model402_model397_component_boundary_audit.json.claim"
RESULT = "experiments/results/model402_model397_component_boundary_audit.json"
CV_ROOT = "data_model322_group_cv"
B4_ROOT = "outputs/model329_fullstats_width4000_uniform_group_oof"
MODEL392_ROOT = "outputs/model392_widthnorm"
MODEL397_LOCK = m397.LOCK
MODEL397_RESULT = m397.RESULT
PRIOR_AUDIT = m397audit.RESULT_REL
CLASSES = m396.CLASS_NAMES
N_CLASSES = len(CLASSES)
AREA_BINS = (
    (1, 64, "1_to_lt64"),
    (64, 256, "64_to_lt256"),
    (256, 1024, "256_to_lt1024"),
    (1024, 4096, "1024_to_lt4096"),
    (4096, None, "4096_or_more"),
)
DISTANCE_BINS = (
    (0.0, 2.0, "0_to_lt2"),
    (2.0, 4.0, "2_to_lt4"),
    (4.0, 8.0, "4_to_lt8"),
    (8.0, None, "8_or_more"),
)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 402_081
GROUP_RE = re.compile(r"^(A\d+)_")
STEP_RE = re.compile(r"_step(\d+)_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def exclusive_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                          indent=2) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def input_paths(root: Path) -> list[Path]:
    masks = m396.mask_files(root)
    paths = [
        root / PROTOCOL,
        root / SCRIPT,
        root / SMOKE,
        root / m396.SCRIPT,
        root / m397.SCRIPT,
        root / m397audit.__file__,
        root / MODEL397_LOCK,
        root / MODEL397_RESULT,
        root / PRIOR_AUDIT,
    ]
    for fold, fold_masks in masks.items():
        paths.extend(fold_masks.values())
        paths.extend((
            root / B4_ROOT / f"fold{fold}/args.json",
            root / B4_ROOT / f"fold{fold}_oof/probabilities.npz",
        ))
        for seed in (1, 2):
            oof = root / MODEL392_ROOT / f"seed{seed}/fold{fold}_oof"
            paths.extend((oof / "model392_manifest.json",
                          oof / "valid_probabilities.npz"))
    unique = sorted({path.resolve() for path in paths})
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(path)
    return unique


def verify_upstream(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    upstream_lock_path = root / MODEL397_LOCK
    upstream_lock = json.loads(upstream_lock_path.read_text(encoding="utf-8"))
    upstream_result = json.loads((root / MODEL397_RESULT).read_text(encoding="utf-8"))
    prior = json.loads((root / PRIOR_AUDIT).read_text(encoding="utf-8"))
    if (upstream_result.get("experiment") != "Model397 fixed B4/Model392 grouped-OOF blend"
            or upstream_result.get("lock_sha256") != sha256(upstream_lock_path)
            or upstream_result.get("official_valid_used") is not False
            or upstream_result.get("test_used") is not False
            or upstream_result.get("valid_stage_allowed") is not False
            or upstream_lock.get("valid_and_test_inputs_excluded") is not True):
        raise ValueError("upstream Model397 is not the expected train-OOF-only result")
    if (prior.get("analysis") != "exploratory class false-negative/false-positive audit on Model397 fixed train OOF"
            or prior.get("official_valid_or_test_read") is not False
            or prior.get("training_started") is not False):
        raise ValueError("existing Model397 diagnostic provenance mismatch")
    return upstream_lock, upstream_result, prior


def create_lock(root: Path) -> dict[str, Any]:
    root = root.resolve()
    lock_path = root / LOCK
    claim_path = root / CLAIM
    result_path = root / RESULT
    if any(path.exists() for path in (lock_path, claim_path, result_path)):
        raise FileExistsError("Model402 lock/claim/result exists; refusing overwrite")
    upstream_lock, upstream_result, prior = verify_upstream(root)
    masks = m396.mask_files(root)
    locked = upstream_lock.get("locked_files", {})
    audit_hashes = prior.get("input_sha256", {})
    files: dict[str, dict[str, Any]] = {}
    for path in input_paths(root):
        rel = relative(root, path)
        digest = sha256(path)
        size = path.stat().st_size
        if rel in locked:
            meta = locked[rel]
            if meta.get("sha256") != digest or meta.get("size_bytes") != size:
                raise ValueError(f"Model397 frozen input changed: {rel}")
        if rel in audit_hashes and audit_hashes[rel] != digest:
            raise ValueError(f"prior Model397 audit input hash changed: {rel}")
        files[rel] = {"size_bytes": size, "sha256": digest}
    expected_ids = {name for fold_masks in masks.values() for name in fold_masks}
    if len(expected_ids) != 70:
        raise ValueError("Model402 requires exactly 70 fixed grouped-OOF train images")
    body = {
        "experiment": "Model402 Model397 grouped train-OOF component/boundary audit",
        "revision": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "decision": "exploratory train-OOF geometry audit only; no training or official valid/test",
        "upstream_model397_lock_sha256": sha256(root / MODEL397_LOCK),
        "upstream_model397_result_sha256": sha256(root / MODEL397_RESULT),
        "metric": "image-wise four-class mIoU; empty-union=0",
        "class_order": list(CLASSES),
        "folds": [1, 2, 3, 4],
        "images": 70,
        "specimen_groups": 59,
        "component_connectivity": 8,
        "component_area_bins_native_pixels": [name for _, _, name in AREA_BINS],
        "boundary_distance_bins_native_pixels": [name for _, _, name in DISTANCE_BINS],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "locked_files": files,
    }
    receipt = {**body, "lock_sha256": canonical_sha256(body)}
    exclusive_json(lock_path, receipt)
    return receipt


def verify_lock(root: Path) -> dict[str, Any]:
    root = root.resolve()
    path = root / LOCK
    record = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in record.items() if key != "lock_sha256"}
    if (record.get("lock_sha256") != canonical_sha256(body)
            or record.get("experiment") != "Model402 Model397 grouped train-OOF component/boundary audit"
            or record.get("metric") != "image-wise four-class mIoU; empty-union=0"
            or record.get("bootstrap_draws") != BOOTSTRAP_DRAWS
            or record.get("bootstrap_seed") != BOOTSTRAP_SEED):
        raise ValueError("Model402 lock integrity or config mismatch")
    paths = input_paths(root)
    expected = {relative(root, item) for item in paths}
    files = record.get("locked_files", {})
    if set(files) != expected:
        raise ValueError("Model402 locked input inventory changed")
    for item in paths:
        rel = relative(root, item)
        metadata = files[rel]
        if item.stat().st_size != metadata["size_bytes"] or sha256(item) != metadata["sha256"]:
            raise ValueError(f"Model402 locked input changed: {rel}")
    upstream_lock, upstream_result, _ = verify_upstream(root)
    if (sha256(root / MODEL397_LOCK) != record["upstream_model397_lock_sha256"]
            or sha256(root / MODEL397_RESULT) != record["upstream_model397_result_sha256"]):
        raise ValueError("upstream Model397 lock/result changed")
    return record


def _iou_vector(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    confusion = np.bincount(
        truth.astype(np.int64).ravel() * N_CLASSES + prediction.astype(np.int64).ravel(),
        minlength=N_CLASSES * N_CLASSES,
    ).reshape(N_CLASSES, N_CLASSES)
    return _ious_from_confusion(confusion)


def _ious_from_confusion(confusion: np.ndarray) -> np.ndarray:
    diagonal = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - diagonal
    return np.divide(diagonal, union, out=np.zeros(N_CLASSES, dtype=np.float64), where=union > 0)


def _area_bin_index(area: int) -> int:
    for index, (low, high, _) in enumerate(AREA_BINS):
        if area >= low and (high is None or area < high):
            return index
    raise ValueError(f"invalid component area {area}")


def _cell_accumulator() -> dict[str, Any]:
    return {"support_pixels": 0, "false_negative_pixels": 0,
            "components": 0, "components_with_any_fn": 0,
            "component_fn_fraction_sum": 0.0, "groups": set(), "folds": {}}


def _fold_cell(row: dict[str, Any], fold: int) -> dict[str, Any]:
    return row["folds"].setdefault(str(fold), {
        "support_pixels": 0, "false_negative_pixels": 0,
        "components": 0, "components_with_any_fn": 0,
        "oracle_delta_pp_sum": 0.0, "images": 0,
    })


def _summarize_cell(acc: dict[str, Any], oracle_rows: list[dict[str, Any]],
                    seed: int) -> dict[str, Any]:
    support = acc["support_pixels"]
    components = acc["components"]
    result = {
        "support_pixels": int(support),
        "false_negative_pixels": int(acc["false_negative_pixels"]),
        "support_weighted_fn_rate_percent": (
            100.0 * acc["false_negative_pixels"] / support if support else None
        ),
        "components": int(components),
        "components_with_any_fn": int(acc["components_with_any_fn"]),
        "component_miss_rate_percent": (
            100.0 * acc["components_with_any_fn"] / components if components else None
        ),
        "mean_component_fn_fraction_percent": (
            100.0 * acc["component_fn_fraction_sum"] / components if components else None
        ),
        "specimen_groups": len(acc["groups"]),
        "oracle_miou_gain_pp": float(np.mean([row["delta_pp"] for row in oracle_rows])),
        "oracle_miou_gain_group_bootstrap_95ci_pp": _group_bootstrap_ci(oracle_rows, seed),
        "folds": {},
    }
    for fold, fold_acc in sorted(acc["folds"].items(), key=lambda pair: int(pair[0])):
        fold_support = fold_acc["support_pixels"]
        fold_components = fold_acc["components"]
        result["folds"][fold] = {
            "support_pixels": int(fold_support),
            "support_weighted_fn_rate_percent": (
                100.0 * fold_acc["false_negative_pixels"] / fold_support if fold_support else None
            ),
            "components": int(fold_components),
            "component_miss_rate_percent": (
                100.0 * fold_acc["components_with_any_fn"] / fold_components if fold_components else None
            ),
            "oracle_miou_gain_pp": fold_acc["oracle_delta_pp_sum"] / fold_acc["images"],
            "images": fold_acc["images"],
        }
    return result


def _group_bootstrap_ci(rows: list[dict[str, Any]], seed: int) -> list[float]:
    by_fold: dict[int, dict[str, list[float]]] = {fold: {} for fold in range(1, 5)}
    for row in rows:
        by_fold[row["fold"]].setdefault(row["group"], []).append(float(row["delta_pp"]))
    rng = np.random.default_rng(seed)
    numerator = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    denominator = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    for fold, groups in by_fold.items():
        names = sorted(groups)
        if not names:
            raise ValueError(f"bootstrap fold {fold} has no specimen groups")
        counts = rng.multinomial(len(names), np.full(len(names), 1.0 / len(names)),
                                 size=BOOTSTRAP_DRAWS)
        sums = np.asarray([sum(groups[name]) for name in names], dtype=np.float64)
        lengths = np.asarray([len(groups[name]) for name in names], dtype=np.float64)
        numerator += counts @ sums
        denominator += counts @ lengths
    values = numerator / denominator
    low, high = np.quantile(values, (0.025, 0.975))
    return [float(low), float(high)]


def _component_map(truth: np.ndarray, class_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    component_labels, count = label(truth == class_id, structure=np.ones((3, 3), dtype=np.uint8))
    area_by_id = np.bincount(component_labels.ravel(), minlength=count + 1).astype(np.int64)
    bin_by_id = np.full(count + 1, 255, dtype=np.uint8)
    for component_id in range(1, count + 1):
        bin_by_id[component_id] = _area_bin_index(int(area_by_id[component_id]))
    return component_labels.astype(np.int32, copy=False), area_by_id, bin_by_id[component_labels]


def _distance_bin_map(truth: np.ndarray, class_id: int) -> np.ndarray:
    mask = truth == class_id
    distance = distance_transform_edt(mask)
    bins = np.full(truth.shape, 255, dtype=np.uint8)
    for index, (low, high, _) in enumerate(DISTANCE_BINS):
        selected = mask & (distance >= low) & ((distance < high) if high is not None else True)
        bins[selected] = index
    if np.any(mask & (bins == 255)):
        raise ValueError("GT class pixel did not receive a boundary-distance bin")
    return bins


def _add_cell_stats(acc: dict[str, Any], *, selected: np.ndarray, errors: np.ndarray,
                    group: str, fold: int, labels: np.ndarray | None = None,
                    bin_ids: np.ndarray | None = None, bin_index: int | None = None) -> None:
    support = int(selected.sum())
    fn = int(np.count_nonzero(selected & errors))
    acc["support_pixels"] += support
    acc["false_negative_pixels"] += fn
    acc["groups"].add(group) if support else None
    fold_acc = _fold_cell(acc, fold)
    fold_acc["support_pixels"] += support
    fold_acc["false_negative_pixels"] += fn
    if labels is not None and bin_ids is not None and bin_index is not None:
        component_ids = np.unique(labels[selected])
        component_ids = component_ids[component_ids > 0]
        if component_ids.size:
            areas = np.bincount(labels[selected], minlength=int(labels.max()) + 1)
            misses = np.bincount(labels[selected & errors], minlength=int(labels.max()) + 1)
            component_areas = areas[component_ids]
            component_fn = misses[component_ids]
            missed = int(np.count_nonzero(component_fn))
            acc["components"] += int(component_ids.size)
            acc["components_with_any_fn"] += missed
            acc["component_fn_fraction_sum"] += float(np.sum(component_fn / component_areas))
            fold_acc["components"] += int(component_ids.size)
            fold_acc["components_with_any_fn"] += missed


def _new_arm_cells() -> dict[str, Any]:
    component = {(class_id, name): _cell_accumulator()
                 for class_id in range(N_CLASSES) for _, _, name in AREA_BINS}
    boundary = {(class_id, name): _cell_accumulator()
                for class_id in range(N_CLASSES) for _, _, name in DISTANCE_BINS}
    oracle_component = {(class_id, name): []
                        for class_id in range(N_CLASSES) for _, _, name in AREA_BINS}
    oracle_boundary = {(class_id, name): []
                       for class_id in range(N_CLASSES) for _, _, name in DISTANCE_BINS}
    return {"component": component, "boundary": boundary,
            "oracle_component": oracle_component, "oracle_boundary": oracle_boundary,
            "baseline_rows": []}


def _image_record(group: str, fold: int, delta: float) -> dict[str, Any]:
    return {"group": group, "fold": fold, "delta_pp": 100.0 * delta}


def _audit_arm_image(arm: dict[str, Any], truth: np.ndarray, prediction: np.ndarray,
                     image_id: str, group: str, fold: int) -> None:
    base_confusion = np.bincount(
        truth.astype(np.int64).ravel() * N_CLASSES + prediction.astype(np.int64).ravel(),
        minlength=N_CLASSES * N_CLASSES,
    ).reshape(N_CLASSES, N_CLASSES)
    baseline_class_iou = _ious_from_confusion(base_confusion)
    baseline = float(baseline_class_iou.mean())
    arm["baseline_rows"].append({"image_id": image_id, "group": group, "fold": fold,
                                 "image_miou": baseline,
                                 "class_iou": baseline_class_iou})
    errors_by_class = [prediction != class_id for class_id in range(N_CLASSES)]
    for class_id in range(N_CLASSES):
        component_labels, _, component_bins = _component_map(truth, class_id)
        for bin_index, (_, _, bin_name) in enumerate(AREA_BINS):
            selected = (truth == class_id) & (component_bins == bin_index)
            acc = arm["component"][(class_id, bin_name)]
            _add_cell_stats(acc, selected=selected, errors=errors_by_class[class_id],
                            group=group, fold=fold, labels=component_labels,
                            bin_ids=component_bins, bin_index=bin_index)
            selected_predictions = np.bincount(prediction[selected].astype(np.int64),
                                               minlength=N_CLASSES)
            corrected_confusion = base_confusion.copy()
            corrected_confusion[class_id, :] -= selected_predictions
            corrected_confusion[class_id, class_id] += int(selected.sum())
            delta = float(_ious_from_confusion(corrected_confusion).mean() - baseline)
            rows = arm["oracle_component"][(class_id, bin_name)]
            rows.append(_image_record(group, fold, delta))
            fold_acc = _fold_cell(acc, fold)
            fold_acc["oracle_delta_pp_sum"] += 100.0 * delta
            fold_acc["images"] += 1

        distance_bins = _distance_bin_map(truth, class_id)
        for bin_index, (_, _, bin_name) in enumerate(DISTANCE_BINS):
            selected = (truth == class_id) & (distance_bins == bin_index)
            acc = arm["boundary"][(class_id, bin_name)]
            _add_cell_stats(acc, selected=selected, errors=errors_by_class[class_id],
                            group=group, fold=fold)
            selected_predictions = np.bincount(prediction[selected].astype(np.int64),
                                               minlength=N_CLASSES)
            corrected_confusion = base_confusion.copy()
            corrected_confusion[class_id, :] -= selected_predictions
            corrected_confusion[class_id, class_id] += int(selected.sum())
            delta = float(_ious_from_confusion(corrected_confusion).mean() - baseline)
            rows = arm["oracle_boundary"][(class_id, bin_name)]
            rows.append(_image_record(group, fold, delta))
            fold_acc = _fold_cell(acc, fold)
            fold_acc["oracle_delta_pp_sum"] += 100.0 * delta
            fold_acc["images"] += 1


def run_audit(root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    masks_by_fold = m396.mask_files(root)
    arms = {name: _new_arm_cells() for name in ("B4", "H", "C")}
    used_inputs: dict[str, str] = {}
    for fold, fold_masks in masks_by_fold.items():
        ids = set(fold_masks)
        b4_path = root / B4_ROOT / f"fold{fold}_oof/probabilities.npz"
        args_path = root / B4_ROOT / f"fold{fold}/args.json"
        args = json.loads(args_path.read_text(encoding="utf-8"))
        if (any(args.get(key) != value for key, value in m396.B4_EXPECTED_ARGS.items())
                or args.get("train_splits") != f"train_fold{fold}"):
            raise ValueError(f"Model329 B4 OOF recipe mismatch in fold {fold}")
        input_hashes = [args_path, b4_path]
        seed_archives: dict[int, Path] = {}
        for seed in (1, 2):
            oof_dir = root / MODEL392_ROOT / f"seed{seed}/fold{fold}_oof"
            manifest_path = oof_dir / "model392_manifest.json"
            archive_path = oof_dir / "valid_probabilities.npz"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("model") != 392 or manifest.get("seed") != seed
                    or manifest.get("fold") != fold
                    or manifest.get("metric") != "image-wise 4-class mIoU; empty-union=0"
                    or manifest.get("oof_probabilities_sha256") != sha256(archive_path)):
                raise ValueError(f"Model392 seed{seed} fold{fold} OOF manifest mismatch")
            seed_archives[seed] = archive_path
            input_hashes.extend((manifest_path, archive_path))
        for path in input_hashes:
            used_inputs[relative(root, path)] = sha256(path)

        with np.load(b4_path, allow_pickle=False) as b4_archive, \
             np.load(seed_archives[1], allow_pickle=False) as seed1_archive, \
             np.load(seed_archives[2], allow_pickle=False) as seed2_archive:
            if any(set(archive.files) != ids for archive in (b4_archive, seed1_archive, seed2_archive)):
                raise ValueError(f"OOF probability archive IDs mismatch in fold {fold}")
            for image_id, mask_path in sorted(fold_masks.items()):
                truth = m396.mask_to_ids(mask_path)
                group_match = GROUP_RE.match(image_id)
                step_match = STEP_RE.search(image_id)
                if group_match is None or step_match is None:
                    raise ValueError(f"invalid fixed OOF image ID {image_id}")
                group = group_match.group(1)
                if group_match.group(1) != GROUP_RE.match(mask_path.name).group(1):
                    raise ValueError(f"group parse mismatch for {image_id}")
                p_b4 = np.asarray(b4_archive[image_id], dtype=np.float32)
                p_seed1 = np.asarray(seed1_archive[image_id], dtype=np.float32)
                p_seed2 = np.asarray(seed2_archive[image_id], dtype=np.float32)
                for probabilities in (p_b4, p_seed1, p_seed2):
                    m397audit._check_probabilities(probabilities, truth, image_id)
                p_b4, p_h, p_c = m397.blend_probabilities(p_b4, p_seed1, p_seed2)
                predictions = {
                    "B4": p_b4.argmax(axis=0).astype(np.uint8),
                    "H": p_h.argmax(axis=0).astype(np.uint8),
                    "C": p_c.argmax(axis=0).astype(np.uint8),
                }
                for arm_name, prediction in predictions.items():
                    _audit_arm_image(arms[arm_name], truth, prediction,
                                     image_id, group, fold)
        print(json.dumps({"stage": "model402_train_oof", "fold": fold,
                          "folds": 4, "images": sum(len(v) for v in masks_by_fold.values())},
                         sort_keys=True), flush=True)

    if any(len(arms[name]["baseline_rows"]) != 70 for name in arms):
        raise ValueError("expected exactly-once 70-image grouped OOF coverage")
    upstream_lock, upstream_result, _ = verify_upstream(root)
    candidate_miou = float(np.mean([row["image_miou"] for row in arms["C"]["baseline_rows"]]) * 100.0)
    recorded_miou = float(upstream_result["arms"]["C"]["image_mean_miou_percent"])
    if abs(candidate_miou - recorded_miou) > 1e-8:
        raise ValueError(f"Model397 OOF baseline does not reproduce: {candidate_miou} != {recorded_miou}")

    summary: dict[str, Any] = {}
    for arm_index, (arm_name, arm) in enumerate(arms.items()):
        rows = arm["baseline_rows"]
        class_means = np.mean(np.stack([row["class_iou"] for row in rows]), axis=0) * 100.0
        component_summary = {}
        boundary_summary = {}
        for class_id, class_name in enumerate(CLASSES):
            component_summary[class_name] = {}
            boundary_summary[class_name] = {}
            for bin_index, (_, _, bin_name) in enumerate(AREA_BINS):
                component_summary[class_name][bin_name] = _summarize_cell(
                    arm["component"][(class_id, bin_name)],
                    arm["oracle_component"][(class_id, bin_name)],
                    BOOTSTRAP_SEED + 1000 * arm_index + 10 * class_id + bin_index,
                )
            for bin_index, (_, _, bin_name) in enumerate(DISTANCE_BINS):
                boundary_summary[class_name][bin_name] = _summarize_cell(
                    arm["boundary"][(class_id, bin_name)],
                    arm["oracle_boundary"][(class_id, bin_name)],
                    BOOTSTRAP_SEED + 5000 + 1000 * arm_index + 10 * class_id + bin_index,
                )
        fold_means = {
            str(fold): float(np.mean([row["image_miou"] for row in rows if row["fold"] == fold]) * 100.0)
            for fold in range(1, 5)
        }
        summary[arm_name] = {
            "images": len(rows),
            "specimen_groups": len({row["group"] for row in rows}),
            "image_mean_miou_percent": float(np.mean([row["image_miou"] for row in rows]) * 100.0),
            "class_image_mean_iou_percent": {
                name: float(value) for name, value in zip(CLASSES, class_means)
            },
            "fold_image_mean_miou_percent": fold_means,
            "component_area_cells": component_summary,
            "boundary_distance_cells": boundary_summary,
        }
    return {
        "experiment": "Model402 Model397 grouped train-OOF component/boundary error audit",
        "status": "EXPLORATORY_DIAGNOSTIC_COMPLETE",
        "scope": "70 exactly-once held-out train images; 59 specimen groups; fixed four folds",
        "metric": "image-wise four-class mIoU; empty-union=0",
        "class_order": list(CLASSES),
        "component_connectivity": 8,
        "component_area_bins_native_pixels": [name for _, _, name in AREA_BINS],
        "boundary_distance_bins_native_pixels": [name for _, _, name in DISTANCE_BINS],
        "oracle_definition": "set only GT pixels in the named class/geometry cell to that GT class; recompute all four class IoUs per image with empty union=0",
        "bootstrap": {
            "method": "fold-stratified specimen-group bootstrap; resample groups and retain all their image rows",
            "draws": BOOTSTRAP_DRAWS,
            "seed_base": BOOTSTRAP_SEED,
        },
        "arms": summary,
        "used_input_sha256": used_inputs,
        "upstream_model397_lock_sha256": upstream_lock.get("lock_sha256", sha256(root / MODEL397_LOCK)),
        "upstream_model397_result_sha256": sha256(root / MODEL397_RESULT),
        "official_valid_or_test_read": False,
        "test_used": False,
        "training_started": False,
        "training_authorized": False,
        "diagnostic_only": True,
        "post_selection_exploratory": True,
        "interpretation_guard": "oracle gains are unattainable upper bounds; fixed-OOF bootstrap is conditional, not independent confirmation; no automatic training or model-selection decision follows",
        "lock_sha256": lock["lock_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test", action="store_true")
    actions.add_argument("--create-lock", action="store_true")
    actions.add_argument("--verify-lock", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.self_test:
        from model402_model397_component_boundary_audit_smoke import main as smoke
        smoke()
    elif args.create_lock:
        result = create_lock(root)
        print(json.dumps({"status": "LOCKED", "lock_sha256": result["lock_sha256"]}, sort_keys=True))
    elif args.verify_lock:
        result = verify_lock(root)
        print(json.dumps({"status": "LOCK_VERIFIED", "lock_sha256": result["lock_sha256"]}, sort_keys=True))
    elif args.run:
        lock = verify_lock(root)
        if (root / CLAIM).exists() or (root / RESULT).exists():
            raise FileExistsError("Model402 claim/result exists; refusing rerun")
        claim = {
            "experiment": "Model402 Model397 grouped train-OOF component/boundary audit",
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "lock_sha256": lock["lock_sha256"],
            "stage": "train_oof_diagnostic",
        }
        claim_hash = exclusive_json(root / CLAIM, claim)
        result = run_audit(root, lock)
        result["claim_sha256"] = claim_hash
        exclusive_json(root / RESULT, result)
        print(json.dumps({"status": result["status"], "image_mean_miou_percent": {
            key: value["image_mean_miou_percent"] for key, value in result["arms"].items()
        }, "result": RESULT}, sort_keys=True))


if __name__ == "__main__":
    main()
