#!/usr/bin/env python3
"""Paired, group-aware audit of completed Model374 and Model329 train-OOF maps.

Uses only grouped train OOF probabilities and their masks. Official validation
labels are never opened. The fixed 50:50 blend is diagnostic, not a tuned
weight. All reported mIoU values are the official image-wise four-class metric
with empty union assigned IoU 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

MODEL329_SHA256 = "539bc1983bffb3ff55303f6865fc9a2094edc5dea9910551a481005fc1a01837"
CLASS_ORDER = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(root.rglob("*.py")):
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def image_id(path: Path) -> str:
    return path.name.removesuffix("_image.png")


def group_id(name: str) -> str:
    match = re.match(r"(A\d+)_", name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {name}")
    return match.group(1)


def probability(archive, name: str) -> np.ndarray:
    for key in (name, f"{name}_image"):
        if key in archive:
            value = np.asarray(archive[key], dtype=np.float32)
            if (value.ndim != 3 or value.shape[0] != 4
                    or not np.isfinite(value).all()
                    or np.any(value < -1e-5)
                    or not np.allclose(value.sum(axis=0), 1.0, atol=3e-2)):
                raise ValueError(f"invalid 4-class probability map for {name}: {value.shape}")
            return value
    raise KeyError(f"missing probability map: {name}")


def class_iou(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    confusion = np.bincount(target.astype(np.int64).ravel() * 4
                            + prediction.astype(np.int64).ravel(), minlength=16).reshape(4, 4)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros(4), where=union > 0), union


def absent_class_audit(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    """Count false positives for GT-absent classes and compute a GT oracle ceiling."""
    target_count = np.bincount(target.ravel(), minlength=4)
    prediction_count = np.bincount(prediction.ravel(), minlength=4)
    absent = target_count == 0
    fp_by_class = np.where(absent, prediction_count, 0)
    oracle = prediction.copy()
    for class_id in np.flatnonzero(absent):
        selected = prediction == class_id
        # Every selected pixel is false positive for this GT-absent class.
        # Reassigning it to GT is label-aware and diagnostic only.
        oracle[selected] = target[selected]
    oracle_iou, _ = class_iou(oracle, target)
    return {
        "gt_absent_classes": [int(class_id) for class_id in np.flatnonzero(absent)],
        "gt_absent_class_fp_pixels": fp_by_class.astype(int).tolist(),
        "gt_absent_class_fp_image_class_pairs": int(np.count_nonzero(fp_by_class)),
        "gt_absent_class_fp_pixel_count": int(fp_by_class.sum()),
        "gt_absent_class_fp_fraction_of_image": float(fp_by_class.sum() / target.size),
        "gt_oracle_miou_after_relabeling_absent_class_fp_empty0": float(oracle_iou.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-data", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True,
                        help="Model374 output root containing foldN_oof/valid_probabilities.npz")
    parser.add_argument("--reference", type=Path, required=True,
                        help="Pinned Model329 exactly-once grouped train-OOF archive")
    parser.add_argument("--reference-fold-root", type=Path, required=True,
                        help="Model329 output root containing foldN_oof/probabilities.npz")
    parser.add_argument("--common-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1374)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.out}")
    if args.bootstrap < 100:
        raise ValueError("use at least 100 group-bootstrap draws")
    if len(set(args.folds)) != len(args.folds) or any(fold not in (1, 2, 3, 4) for fold in args.folds):
        raise ValueError(f"invalid or duplicate folds: {args.folds}")

    reference_sha = sha256(args.reference)
    if reference_sha != MODEL329_SHA256:
        raise ValueError(f"pinned Model329 reference hash mismatch: {reference_sha}")
    sys.path.insert(0, str(args.common_dir))
    from common import list_split, load_mask  # pylint: disable=import-outside-toplevel

    reference_train_source_path = args.common_dir / "train.py"
    reference_common_source_path = args.common_dir / "common.py"
    reference_train_source = reference_train_source_path.read_text(encoding="utf-8")
    reference_common_source = reference_common_source_path.read_text(encoding="utf-8")
    if ("model = build_model(args.arch, args.encoder).cuda()" not in reference_train_source
            or "load_state_dict(" in reference_train_source or "torch.load(" in reference_train_source):
        raise ValueError("Model329 trainer no longer matches the audited fresh-initialization path")
    if ("def build_model(arch, encoder, pretrained=True):" not in reference_common_source
            or 'encoder_weights="imagenet" if pretrained else None' not in reference_common_source):
        raise ValueError("Model329 model builder no longer matches the audited ImageNet initialization")

    rows: list[dict[str, object]] = []
    group_owner: dict[str, int] = {}
    reference_fold_maps_compared = 0
    reference_fold_provenance: dict[str, dict[str, object]] = {}
    candidate_fold_provenance: dict[str, dict[str, object]] = {}
    provenance_warnings: list[str] = [
        "Model329 trainer/model-builder source hashes were measured during audit, not at training time"]
    with np.load(args.reference) as reference_archive:
        reference_keys = {key.removesuffix("_image") for key in reference_archive.files}
        for fold in args.folds:
            pairs = list_split(str(args.cv_data), f"oof_fold{fold}")
            expected = {image_id(Path(image_path)) for image_path, _ in pairs}
            train_pairs = list_split(str(args.cv_data), f"train_fold{fold}")
            train_groups = {group_id(image_id(Path(image_path))) for image_path, _ in train_pairs}
            held_groups = {group_id(name) for name in expected}
            overlap = train_groups & held_groups
            if overlap:
                raise ValueError(f"fold {fold} train/OOF specimen leakage: {sorted(overlap)}")
            reference_args_path = args.reference_fold_root / f"fold{fold}/args.json"
            reference_args = json.loads(reference_args_path.read_text(encoding="utf-8"))
            if Path(reference_args.get("data", "")).resolve() != args.cv_data.resolve():
                raise ValueError(f"Model329 fold {fold} used an unexpected data root")
            if reference_args.get("train_splits") != f"train_fold{fold}":
                raise ValueError(
                    f"Model329 fold {fold} was not trained on train_fold{fold}: "
                    f"{reference_args.get('train_splits')!r}")
            reference_run = args.reference_fold_root / f"fold{fold}"
            reference_log_path = reference_run / "train.log"
            reference_log = reference_log_path.read_text(encoding="utf-8", errors="replace")
            if reference_args.get("no_valid_eval") is not True:
                raise ValueError(f"Model329 fold {fold} did not disable validation evaluation")
            if reference_args.get("iters") != 16000 or reference_args.get("seed") != 0:
                raise ValueError(f"Model329 fold {fold} does not match the pinned training recipe")
            if Path(reference_args.get("out", "")).resolve() != reference_run.resolve():
                raise ValueError(f"Model329 fold {fold} args point to a different output directory")
            if not reference_log.splitlines() or (
                    f"train {len(train_pairs)} valid 0" not in reference_log.splitlines()[0]):
                raise ValueError(f"Model329 fold {fold} log does not confirm an empty validation loader")
            if "[valid" in reference_log:
                raise ValueError(f"Model329 fold {fold} log contains validation-based evaluation")
            reference_checkpoint = reference_run / "last.pt"
            if not reference_checkpoint.is_file():
                raise FileNotFoundError(reference_checkpoint)
            reference_oof_manifest = args.reference_fold_root / f"fold{fold}_oof/inference_manifest.json"
            if not reference_oof_manifest.is_file():
                provenance_warnings.append(
                    f"fold {fold}: Model329 has no inference manifest binding last.pt to its OOF map")
            reference_fold_provenance[str(fold)] = {
                "training_images": len(train_pairs),
                "training_specimen_groups": len(train_groups),
                "heldout_images": len(pairs),
                "heldout_specimen_groups": len(held_groups),
                "train_heldout_group_overlap": [],
                "model329_training_split": reference_args["train_splits"],
                "no_valid_eval": True,
                "validation_loader_empty_confirmed_by_log": True,
                "validation_evaluation_messages": False,
                "fixed_training_iterations": reference_args["iters"],
                "training_args_sha256": sha256(reference_args_path),
                "training_log_sha256": sha256(reference_log_path),
                "checkpoint_sha256": sha256(reference_checkpoint),
                "external_init_checkpoint": reference_args.get("init_checkpoint"),
                "fresh_init_path_verified_in_current_source": True,
                "source_hashes_measured_at_audit_time": True,
                "trainer_source_sha256": sha256(reference_train_source_path),
                "model_builder_source_sha256": sha256(reference_common_source_path),
                "checkpoint_to_oof_archive_binding_verified": reference_oof_manifest.is_file(),
            }
            candidate_path = args.candidate_root / f"fold{fold}_oof/valid_probabilities.npz"
            candidate_manifest_path = candidate_path.parent / "inference_manifest.json"
            candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
            candidate_data_root = Path(candidate_manifest["inference_data_root"])
            if candidate_manifest.get("fold") != fold:
                raise ValueError(f"Model374 fold-{fold} inference manifest has wrong fold")
            if candidate_manifest.get("train_state_epoch") != 70:
                raise ValueError(f"Model374 fold {fold} was not evaluated from epoch 70")
            if candidate_manifest.get("inference_split") != "valid":
                raise ValueError(f"Model374 fold {fold} used an unexpected inference split")
            if Path(candidate_manifest["heldout_data_root"]).resolve() != (
                    args.cv_data / f"oof_fold{fold}").resolve():
                raise ValueError(f"Model374 fold {fold} inference used an unexpected held-out split")
            if ((candidate_data_root / "train").resolve() != (
                    args.cv_data / f"train_fold{fold}").resolve()
                    or (candidate_data_root / "valid").resolve() != (
                        args.cv_data / f"oof_fold{fold}").resolve()):
                raise ValueError(f"Model374 fold {fold} training/inference links do not match grouped split")
            candidate_args_path = Path(candidate_manifest["training_args"])
            if sha256(candidate_args_path) != candidate_manifest.get("training_args_sha256"):
                raise ValueError(f"Model374 fold {fold} training args changed since OOF inference")
            candidate_args = json.loads(candidate_args_path.read_text(encoding="utf-8"))
            expected_candidate_args = {
                "architecture": "UnetPlusPlus", "encoder_name": "tu-hrnet_w48",
                "encoder_weights": "imagenet", "patch_size": 512, "input_scale": 2.0,
                "samples_per_image": 12, "batch_size": 4, "accumulation_steps": 2,
                "num_workers": 4, "epochs": 70, "lr": 3e-4, "encoder_lr_mult": 1.0,
                "weight_decay": 1e-4, "ce_class_weights": None, "init_checkpoint": None,
                "no_boundary_head": False, "boundary_head_weight": 1.0, "seed": 0,
                "val_every": 0, "no_amp": False, "resume": True,
            }
            expected_arg_keys = set(expected_candidate_args) | {"data_root", "output_dir", "device"}
            if set(candidate_args) != expected_arg_keys or any(
                    candidate_args.get(key) != value
                    for key, value in expected_candidate_args.items()
                    if key != "init_checkpoint"):
                raise ValueError(f"Model374 fold {fold} args do not match the complete fixed recipe")
            if ("init_checkpoint" not in candidate_args
                    or candidate_args["init_checkpoint"] is not None
                    or candidate_args.get("device") != "cuda:0"
                    or Path(candidate_args.get("data_root", "")).resolve() != candidate_data_root.resolve()):
                raise ValueError(f"Model374 fold {fold} did not follow the fixed training protocol")
            candidate_checkpoint = Path(candidate_manifest["checkpoint"])
            if candidate_checkpoint.resolve() != (
                    Path(candidate_args["output_dir"]).resolve() / "last.pt"):
                raise ValueError(f"Model374 fold {fold} checkpoint is outside its own training run")
            if sha256(candidate_checkpoint) != candidate_manifest.get("checkpoint_sha256"):
                raise ValueError(f"Model374 fold {fold} checkpoint changed since OOF inference")
            candidate_project_root = candidate_data_root.parent
            candidate_code_root = candidate_project_root / "model336_hrnet_best"
            candidate_eval_script = candidate_code_root / "evaluate_image_mean.py"
            candidate_runner = candidate_project_root / "experiments/runners/run_model374_hrnet_group_oof.sh"
            if sha256(candidate_eval_script) != candidate_manifest.get("evaluation_script_sha256"):
                raise ValueError(f"Model374 fold {fold} evaluation code changed since OOF inference")
            if source_tree_sha256(candidate_code_root) != candidate_manifest.get("model_code_tree_sha256"):
                raise ValueError(f"Model374 fold {fold} model code changed since OOF inference")
            current_runner_sha = sha256(candidate_runner)
            runner_sha_matches = current_runner_sha == candidate_manifest.get("runner_sha256")
            if not runner_sha_matches:
                provenance_warnings.append(
                    f"fold {fold}: runner SHA differs from inference manifest; "
                    "training args, checkpoint, model code and evaluation code hashes match")
            candidate_probability_sha = sha256(candidate_path)
            if candidate_manifest.get("probability_archive_sha256") is None:
                provenance_warnings.append(
                    f"fold {fold}: candidate probability archive SHA is recorded now but was not bound "
                    "to the inference-time manifest")
            elif candidate_manifest["probability_archive_sha256"] != candidate_probability_sha:
                raise ValueError(f"Model374 fold {fold} probability archive hash differs from manifest")
            candidate_fold_provenance[str(fold)] = {
                "heldout_images": len(pairs),
                "train_state_epoch": candidate_manifest["train_state_epoch"],
                "inference_split": candidate_manifest["inference_split"],
                "validation_checkpoint_selection": False,
                "external_initial_checkpoint": None,
                "training_args_sha256_verified": True,
                "checkpoint_sha256_verified": True,
                "model_code_tree_sha256_verified": True,
                "evaluation_script_sha256_verified": True,
                "runner_sha256_verified": runner_sha_matches,
                "runner_sha256_in_manifest": candidate_manifest.get("runner_sha256"),
                "runner_sha256_current": current_runner_sha,
                "probability_archive_sha256_recorded": candidate_probability_sha,
                "probability_archive_sha256_bound_to_manifest": (
                    candidate_manifest.get("probability_archive_sha256") == candidate_probability_sha),
                "inference_manifest_sha256": sha256(candidate_manifest_path),
            }
            reference_fold_path = args.reference_fold_root / f"fold{fold}_oof/probabilities.npz"
            reference_fold_provenance[str(fold)]["fold_oof_probability_archive_sha256"] = sha256(
                reference_fold_path)
            with np.load(candidate_path) as candidate_archive, np.load(reference_fold_path) as fold_archive:
                candidate_keys = {key.removesuffix("_image") for key in candidate_archive.files}
                reference_fold_keys = {key.removesuffix("_image") for key in fold_archive.files}
                if candidate_keys != expected or reference_fold_keys != expected:
                    raise ValueError(
                        f"fold {fold} key mismatch: expected={len(expected)}, "
                        f"candidate={len(candidate_keys)}, reference_fold={len(reference_fold_keys)}")
                if not expected <= reference_keys:
                    raise ValueError(f"Model329 merged OOF archive missing fold-{fold} keys")
                for name in sorted(expected):
                    group = group_id(name)
                    if group in group_owner and group_owner[group] != fold:
                        raise ValueError(f"specimen group {group} occurs in multiple audited folds")
                    group_owner[group] = fold
                    target_path = next(Path(mask_path) for image_path, mask_path in pairs
                                       if image_id(Path(image_path)) == name)
                    target = load_mask(str(target_path))
                    reference_prob = probability(fold_archive, name)
                    merged_reference_prob = probability(reference_archive, name)
                    if (reference_prob.shape != merged_reference_prob.shape
                            or not np.array_equal(reference_prob, merged_reference_prob)):
                        raise ValueError(
                            f"Model329 fold-{fold} map differs from merged OOF archive for {name}")
                    reference_fold_maps_compared += 1
                    candidate_prob = probability(candidate_archive, name)
                    if reference_prob.shape != candidate_prob.shape or reference_prob.shape[1:] != target.shape:
                        raise ValueError(f"probability/target shape mismatch for {name}")
                    reference_pred = reference_prob.argmax(axis=0)
                    candidate_pred = candidate_prob.argmax(axis=0)
                    blend_pred = (0.5 * reference_prob + 0.5 * candidate_prob).argmax(axis=0)
                    reference_iou, reference_union = class_iou(reference_pred, target)
                    candidate_iou, candidate_union = class_iou(candidate_pred, target)
                    blend_iou, blend_union = class_iou(blend_pred, target)
                    candidate_confidence = candidate_prob.max(axis=0)
                    reference_confidence = reference_prob.max(axis=0)
                    choose_candidate = candidate_confidence > reference_confidence
                    confidence_gate_pred = np.where(choose_candidate, candidate_pred, reference_pred)
                    confidence_gate_iou, _ = class_iou(confidence_gate_pred, target)
                    absent_audits = {
                        "Model329": absent_class_audit(reference_pred, target),
                        "Model374": absent_class_audit(candidate_pred, target),
                        "fixed_half_blend": absent_class_audit(blend_pred, target),
                        "fixed_max_confidence_gate": absent_class_audit(confidence_gate_pred, target),
                    }
                    ref_correct = reference_pred == target
                    cand_correct = candidate_pred == target
                    image_oracle = max(float(reference_iou.mean()), float(candidate_iou.mean()))
                    pixel_oracle_pred = reference_pred.copy()
                    pixel_oracle_pred[(~ref_correct) & cand_correct] = target[(~ref_correct) & cand_correct]
                    pixel_oracle_iou, _ = class_iou(pixel_oracle_pred, target)
                    rows.append({
                        "image": name,
                        "fold": fold,
                        "specimen_group": group,
                        "pixels": int(target.size),
                        "model329_miou_empty0": float(reference_iou.mean()),
                        "model374_miou_empty0": float(candidate_iou.mean()),
                        "fixed_half_blend_miou_empty0": float(blend_iou.mean()),
                        "fixed_max_confidence_gate_miou_empty0": float(confidence_gate_iou.mean()),
                        "model374_delta_vs_model329_pp": float(
                            (candidate_iou.mean() - reference_iou.mean()) * 100.0),
                        "blend_delta_vs_model329_pp": float(
                            (blend_iou.mean() - reference_iou.mean()) * 100.0),
                        "confidence_gate_delta_vs_model329_pp": float(
                            (confidence_gate_iou.mean() - reference_iou.mean()) * 100.0),
                        "confidence_gate_delta_vs_blend_pp": float(
                            (confidence_gate_iou.mean() - blend_iou.mean()) * 100.0),
                        "model329_class_iou": reference_iou.tolist(),
                        "model374_class_iou": candidate_iou.tolist(),
                        "blend_class_iou": blend_iou.tolist(),
                        "confidence_gate_class_iou": confidence_gate_iou.tolist(),
                        "class_delta_blend_vs_model329_pp": (
                            (blend_iou - reference_iou) * 100.0).tolist(),
                        "class_delta_confidence_gate_vs_model329_pp": (
                            (confidence_gate_iou - reference_iou) * 100.0).tolist(),
                        "class_delta_confidence_gate_vs_blend_pp": (
                            (confidence_gate_iou - blend_iou) * 100.0).tolist(),
                        "confidence_gate_candidate_pixel_fraction": float(choose_candidate.mean()),
                        "target_absent_class_count": int((np.bincount(target.ravel(), minlength=4) == 0).sum()),
                        "absent_target_predicted_by_model329": int(
                            ((np.bincount(target.ravel(), minlength=4) == 0)
                             & (np.bincount(reference_pred.ravel(), minlength=4) > 0)).sum()),
                        "absent_target_predicted_by_model374": int(
                            ((np.bincount(target.ravel(), minlength=4) == 0)
                             & (np.bincount(candidate_pred.ravel(), minlength=4) > 0)).sum()),
                        "absent_target_predicted_by_blend": int(
                            ((np.bincount(target.ravel(), minlength=4) == 0)
                             & (np.bincount(blend_pred.ravel(), minlength=4) > 0)).sum()),
                        "absent_target_predicted_by_confidence_gate": int(
                            ((np.bincount(target.ravel(), minlength=4) == 0)
                             & (np.bincount(confidence_gate_pred.ravel(), minlength=4) > 0)).sum()),
                        "absent_class_audit": absent_audits,
                        "prediction_disagreement_fraction": float((reference_pred != candidate_pred).mean()),
                        "model374_correct_only_pixels": int(((~ref_correct) & cand_correct).sum()),
                        "model329_correct_only_pixels": int((ref_correct & (~cand_correct)).sum()),
                        "image_oracle_best_model_miou_empty0": image_oracle,
                        "pixel_oracle_miou_empty0_diagnostic_only": float(pixel_oracle_iou.mean()),
                    })

    if len(rows) != sum(len(list_split(str(args.cv_data), f"oof_fold{fold}")) for fold in args.folds):
        raise AssertionError("row count does not match requested held-out folds")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["specimen_group"]), []).append(index)
    ref_scores = np.asarray([row["model329_miou_empty0"] for row in rows])
    candidate_scores = np.asarray([row["model374_miou_empty0"] for row in rows])
    blend_scores = np.asarray([row["fixed_half_blend_miou_empty0"] for row in rows])
    confidence_gate_scores = np.asarray([
        row["fixed_max_confidence_gate_miou_empty0"] for row in rows])
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty((args.bootstrap, 4), dtype=np.float64)
    names = list(groups)
    for draw in range(args.bootstrap):
        sampled_groups = rng.choice(names, size=len(names), replace=True)
        indices = [index for group in sampled_groups for index in groups[str(group)]]
        bootstrap[draw] = (
            (candidate_scores[indices] - ref_scores[indices]).mean() * 100.0,
            (blend_scores[indices] - ref_scores[indices]).mean() * 100.0,
            (confidence_gate_scores[indices] - ref_scores[indices]).mean() * 100.0,
            (confidence_gate_scores[indices] - blend_scores[indices]).mean() * 100.0,
        )
    candidate_delta = (candidate_scores - ref_scores) * 100.0
    blend_delta = (blend_scores - ref_scores) * 100.0
    confidence_gate_delta = (confidence_gate_scores - ref_scores) * 100.0
    confidence_gate_delta_vs_blend = (confidence_gate_scores - blend_scores) * 100.0
    fold_summary = {}
    for fold in args.folds:
        fold_rows = [row for row in rows if row["fold"] == fold]
        fold_reference = np.asarray([row["model329_miou_empty0"] for row in fold_rows])
        fold_candidate = np.asarray([row["model374_miou_empty0"] for row in fold_rows])
        fold_blend = np.asarray([row["fixed_half_blend_miou_empty0"] for row in fold_rows])
        fold_gate = np.asarray([
            row["fixed_max_confidence_gate_miou_empty0"] for row in fold_rows])
        fold_class_delta = np.asarray([
            row["class_delta_blend_vs_model329_pp"] for row in fold_rows])
        fold_gate_class_delta = np.asarray([
            row["class_delta_confidence_gate_vs_model329_pp"] for row in fold_rows])
        fold_summary[str(fold)] = {
            "image_count": len(fold_rows),
            "specimen_group_count": len({row["specimen_group"] for row in fold_rows}),
            "model329_image_mean_miou_empty0": float(fold_reference.mean()),
            "model374_image_mean_miou_empty0": float(fold_candidate.mean()),
            "fixed_half_blend_image_mean_miou_empty0": float(fold_blend.mean()),
            "fixed_max_confidence_gate_image_mean_miou_empty0": float(fold_gate.mean()),
            "model374_delta_vs_model329_pp": float((fold_candidate - fold_reference).mean() * 100.0),
            "fixed_half_blend_delta_vs_model329_pp": float((fold_blend - fold_reference).mean() * 100.0),
            "confidence_gate_delta_vs_model329_pp": float((fold_gate - fold_reference).mean() * 100.0),
            "confidence_gate_delta_vs_blend_pp": float((fold_gate - fold_blend).mean() * 100.0),
            "fixed_half_blend_positive_images": int((fold_blend > fold_reference).sum()),
            "confidence_gate_positive_images_vs_model329": int((fold_gate > fold_reference).sum()),
            "confidence_gate_positive_images_vs_blend": int((fold_gate > fold_blend).sum()),
            "confidence_gate_candidate_pixel_fraction": float(np.mean([
                row["confidence_gate_candidate_pixel_fraction"] for row in fold_rows])),
            "class_delta_blend_vs_model329_pp": {
                name: float(fold_class_delta[:, index].mean())
                for index, name in enumerate(CLASS_ORDER)
            },
            "class_delta_confidence_gate_vs_model329_pp": {
                name: float(fold_gate_class_delta[:, index].mean())
                for index, name in enumerate(CLASS_ORDER)
            },
        }
    class_names = list(CLASS_ORDER)
    blend_class_delta = np.asarray([row["class_delta_blend_vs_model329_pp"] for row in rows])
    confidence_gate_class_delta = np.asarray([
        row["class_delta_confidence_gate_vs_model329_pp"] for row in rows])
    absent_summary = {}
    for model_name in ("Model329", "Model374", "fixed_half_blend", "fixed_max_confidence_gate"):
        audit_rows = [row["absent_class_audit"][model_name] for row in rows]
        absent_pairs = sum(len(row["gt_absent_classes"]) for row in audit_rows)
        absent_summary[model_name] = {
            "gt_absent_image_class_pairs": int(absent_pairs),
            "false_positive_absent_image_class_pairs": int(sum(
                row["gt_absent_class_fp_image_class_pairs"] for row in audit_rows)),
            "false_positive_pixels_on_gt_absent_classes": int(sum(
                row["gt_absent_class_fp_pixel_count"] for row in audit_rows)),
            "mean_image_fraction_predicted_as_gt_absent_classes": float(np.mean([
                row["gt_absent_class_fp_fraction_of_image"] for row in audit_rows])),
            "image_mean_miou_after_gt_oracle_relabels_absent_class_fp": float(np.mean([
                row["gt_oracle_miou_after_relabeling_absent_class_fp_empty0"]
                for row in audit_rows])),
            "oracle_gain_pp_diagnostic_only": float(np.mean([
                row["gt_oracle_miou_after_relabeling_absent_class_fp_empty0"]
                - rows[index][{"Model329": "model329_miou_empty0",
                               "Model374": "model374_miou_empty0",
                               "fixed_half_blend": "fixed_half_blend_miou_empty0",
                               "fixed_max_confidence_gate": "fixed_max_confidence_gate_miou_empty0"}[model_name]]
                for index, row in enumerate(audit_rows)]) * 100.0),
        }
        class_summary = {}
        for class_id, class_name in enumerate(class_names):
            absent_cases = [row for row in audit_rows if class_id in row["gt_absent_classes"]]
            predicted_cases = [row for row in absent_cases
                               if row["gt_absent_class_fp_pixels"][class_id] > 0]
            class_summary[class_name] = {
                "gt_absent_images": len(absent_cases),
                "images_with_false_positive": len(predicted_cases),
                "total_false_positive_pixels": int(sum(
                    row["gt_absent_class_fp_pixels"][class_id] for row in absent_cases)),
                "mean_false_positive_fraction_when_gt_absent": float(np.mean([
                    row["gt_absent_class_fp_pixels"][class_id] / rows[index]["pixels"]
                    for index, row in enumerate(audit_rows)
                    if class_id in row["gt_absent_classes"]])) if absent_cases else 0.0,
            }
        absent_summary[model_name]["by_class"] = class_summary
    result = {
        "audit": "paired grouped train-OOF metrics recomputed from saved maps; no official-valid/test labels",
        "score_source": "saved per-fold probability archives and train-OOF masks; checkpoint inference was not rerun by this audit",
        "metric": "per-image mean of four class IoUs; empty union=0",
        "class_order": class_names,
        "channel_alignment": "identity; Model329 and Model374 both [Al, Al3Ni, Primary_Si, Eutectic_Si]",
        "reference_model": 329,
        "candidate_model": 374,
        "reference_sha256": reference_sha,
        "reference_fold_maps_compared_to_merged_archive": reference_fold_maps_compared,
        "reference_fold_map_parity_exact": True,
        "reference_fold_training_provenance_verified": reference_fold_provenance,
        "candidate_fold_training_provenance_verified": candidate_fold_provenance,
        "provenance_status": "complete" if not provenance_warnings else "partial_with_warnings",
        "provenance_warnings": provenance_warnings,
        "folds": list(args.folds),
        "fold_summary": fold_summary,
        "image_count": len(rows),
        "specimen_group_count": len(groups),
        "group_ownership_exactly_once": True,
        "model329_image_mean_miou_empty0": float(ref_scores.mean()),
        "model374_image_mean_miou_empty0": float(candidate_scores.mean()),
        "fixed_half_blend_image_mean_miou_empty0": float(blend_scores.mean()),
        "fixed_max_confidence_gate_image_mean_miou_empty0": float(confidence_gate_scores.mean()),
        "fixed_max_confidence_gate_rule": {
            "choose_model374_when": "its max class probability is strictly greater than Model329's",
            "ties": "choose Model329",
            "fitted_parameters": 0,
        },
        "model374_delta_vs_model329_pp": float(candidate_delta.mean()),
        "fixed_half_blend_delta_vs_model329_pp": float(blend_delta.mean()),
        "confidence_gate_delta_vs_model329_pp": float(confidence_gate_delta.mean()),
        "confidence_gate_delta_vs_blend_pp": float(confidence_gate_delta_vs_blend.mean()),
        "model374_positive_images": int((candidate_delta > 0).sum()),
        "fixed_half_blend_positive_images": int((blend_delta > 0).sum()),
        "confidence_gate_positive_images_vs_model329": int((confidence_gate_delta > 0).sum()),
        "confidence_gate_positive_images_vs_blend": int((confidence_gate_delta_vs_blend > 0).sum()),
        "model374_group_bootstrap_95ci_delta_pp": [float(x) for x in
            np.quantile(bootstrap[:, 0], [0.025, 0.975])],
        "fixed_half_blend_group_bootstrap_95ci_delta_pp": [float(x) for x in
            np.quantile(bootstrap[:, 1], [0.025, 0.975])],
        "confidence_gate_group_bootstrap_95ci_vs_model329_pp": [float(x) for x in
            np.quantile(bootstrap[:, 2], [0.025, 0.975])],
        "confidence_gate_group_bootstrap_95ci_vs_blend_pp": [float(x) for x in
            np.quantile(bootstrap[:, 3], [0.025, 0.975])],
        "confidence_gate_candidate_pixel_fraction_image_mean": float(np.mean([
            row["confidence_gate_candidate_pixel_fraction"] for row in rows])),
        "class_delta_blend_vs_model329_pp": {
            name: float(blend_class_delta[:, index].mean())
            for index, name in enumerate(class_names)
        },
        "class_delta_confidence_gate_vs_model329_pp": {
            name: float(confidence_gate_class_delta[:, index].mean())
            for index, name in enumerate(class_names)
        },
        "mean_image_oracle_best_model_miou_empty0_diagnostic_only": float(np.mean([
            row["image_oracle_best_model_miou_empty0"] for row in rows])),
        "mean_pixel_oracle_miou_empty0_diagnostic_only": float(np.mean([
            row["pixel_oracle_miou_empty0_diagnostic_only"] for row in rows])),
        "positive_pixel_complementarity": {
            "model374_correct_only_pixels": int(sum(row["model374_correct_only_pixels"] for row in rows)),
            "model329_correct_only_pixels": int(sum(row["model329_correct_only_pixels"] for row in rows)),
            "prediction_disagreement_fraction_image_mean": float(np.mean([
                row["prediction_disagreement_fraction"] for row in rows])),
        },
        "absent_class_false_positive_audit": absent_summary,
        "interpretation": "single-fold or partial-fold OOF is exploratory; fixed 0.5 blend and max-confidence gate have no fitted weights or threshold",
        "per_image": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "folds", "image_count", "specimen_group_count", "model329_image_mean_miou_empty0",
        "model374_image_mean_miou_empty0", "fixed_half_blend_image_mean_miou_empty0",
        "fixed_max_confidence_gate_image_mean_miou_empty0",
        "model374_delta_vs_model329_pp", "fixed_half_blend_delta_vs_model329_pp",
        "confidence_gate_delta_vs_model329_pp", "confidence_gate_delta_vs_blend_pp",
        "model374_group_bootstrap_95ci_delta_pp", "fixed_half_blend_group_bootstrap_95ci_delta_pp",
        "confidence_gate_group_bootstrap_95ci_vs_blend_pp",
        "class_delta_blend_vs_model329_pp", "mean_image_oracle_best_model_miou_empty0_diagnostic_only",
        "mean_pixel_oracle_miou_empty0_diagnostic_only", "positive_pixel_complementarity",
        "absent_class_false_positive_audit")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
