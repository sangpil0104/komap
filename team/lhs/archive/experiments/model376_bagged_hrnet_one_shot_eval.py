#!/usr/bin/env python3
"""One prelocked Model360 evaluation using the label-free Model374 fold mean.

This script performs no search. It averages the four Model374 fold maps 50:50
with Model336 flip4 and Model351 D4 maps, keeps Model360's entropy gate and
all spatial/class routes fixed, then computes exactly one candidate score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

import model355_model349_spatial_audit as spatial
import model360_entropy_gated_d4_into_355 as model360


def group_for(name: str) -> str:
    match = re.match(r"(A\d+)_", name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {name}")
    return match.group(1)


def group_bootstrap_ci(delta: np.ndarray, groups: list[str], draws: int,
                       seed: int) -> list[float]:
    unique = sorted(set(groups))
    by_group = {group: np.flatnonzero(np.asarray(groups) == group) for group in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        estimates[draw] = delta[indices].mean() * 100.0
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def imported_local_source_hashes(source_root: Path) -> dict[str, str]:
    """Bind the route result to every imported Python source under experiments/."""
    hashes = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename or Path(filename).suffix != ".py":
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(source_root)
        except ValueError:
            continue
        hashes[relative.as_posix()] = sha256(path)
    return dict(sorted(hashes.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--fold-predictions", type=Path,
                        default=Path("/workspace/komap_model1/outputs/model376_bagged_hrnet_valid_preds"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    outputs = args.root / "outputs"
    code_dir = args.root / "model336_hrnet_best"
    sys.path.insert(0, str(code_dir))
    from komap_seg.palette import CLASS_NAMES as RAW_CLASS_NAMES

    manifest_path = args.fold_predictions / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_raw_order = ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"]
    expected_route_order = ["Al3Ni", "Primary Si", "Al", "Eutectic Si"]
    expected_inference = {"patch_size": 512, "overlap": 170, "input_scale": 2.0,
                          "batch_size": 2, "tta_modes": ["flip4", "d4"],
                          "downsample_antialias": False}
    if (manifest.get("model") != 376 or manifest.get("class_order") != expected_raw_order
            or list(RAW_CLASS_NAMES) != expected_raw_order
            or manifest.get("image_count") != 20
            or manifest.get("probability_averaging")
            != "arithmetic mean of four fold probability maps"
            or manifest.get("inference") != expected_inference
            or Path(manifest.get("inference_code_dir", "")).resolve()
            != code_dir.resolve()
            or manifest.get("class_palette_sha256")
            != sha256(code_dir / "komap_seg" / "palette.py")):
        raise ValueError("fold inference manifest/classes/settings differ from the prelocked protocol")
    expected_code_files = {"train.py", "evaluate_image_mean.py", *(
        str(path.relative_to(code_dir)) for path in (code_dir / "komap_seg").rglob("*.py"))}
    code_hashes = manifest.get("inference_code_sha256", {})
    if set(code_hashes) != expected_code_files or any(
            sha256(code_dir / relative) != code_hashes[relative]
            for relative in expected_code_files):
        raise ValueError("Model376 inference code differs from its manifest")
    predictor_path = Path(__file__).resolve().with_name("model376_bagged_hrnet_predict_only.py")
    if sha256(predictor_path) != manifest.get("predictor_script_sha256"):
        raise ValueError("label-free predictor code differs from the inference manifest")
    if tuple(spatial.audit.model350.CLASS_ORDER) != tuple(expected_route_order):
        raise AssertionError("Model360 route class order changed")

    valid_image_dir = (args.root / "data" / "valid" / "images").resolve()
    expected_image_hashes = manifest.get("valid_image_sha256", {})
    current_images = sorted(valid_image_dir.glob("*_image.png"))
    current_image_hashes = {
        path.name[:-len("_image.png")]: sha256(path) for path in current_images
    }
    if (Path(manifest.get("valid_image_dir", "")).resolve() != valid_image_dir
            or len(current_images) != 20 or current_image_hashes != expected_image_hashes):
        raise ValueError("fold predictions were not generated from the current valid images")

    fold_checkpoint_root = outputs / "model374_hrnet_group_oof"
    fold_records = manifest.get("folds", [])
    if [record.get("fold") for record in fold_records] != [1, 2, 3, 4]:
        raise ValueError("fold inference manifest does not list exactly folds 1–4")
    for fold, record in enumerate(fold_records, start=1):
        expected_checkpoint = fold_checkpoint_root / f"fold{fold}" / "last.pt"
        expected_args = expected_checkpoint.parent / "args.json"
        expected_source_manifest = fold_checkpoint_root / f"fold{fold}_oof" / "inference_manifest.json"
        expected_hashes = (
            ("checkpoint", expected_checkpoint, "checkpoint_sha256"),
            ("training_args", expected_args, "training_args_sha256"),
            ("source_manifest", expected_source_manifest, "source_manifest_sha256"),
        )
        for label, expected_path, hash_field in expected_hashes:
            if Path(record.get(label, "")).resolve() != expected_path.resolve():
                raise ValueError(f"fold {fold} {label} path differs from Model374 output")
            if sha256(expected_path) != record.get(hash_field):
                raise ValueError(f"fold {fold} {label} hash differs from inference manifest")
        training_args = json.loads(expected_args.read_text(encoding="utf-8"))
        source_manifest = json.loads(expected_source_manifest.read_text(encoding="utf-8"))
        expected_training = {
            "architecture": "UnetPlusPlus", "encoder_name": "tu-hrnet_w48",
            "encoder_weights": "imagenet", "patch_size": 512, "input_scale": 2.0,
            "samples_per_image": 12, "batch_size": 4, "accumulation_steps": 2,
            "epochs": 70, "lr": 3e-4, "encoder_lr_mult": 1.0,
            "weight_decay": 1e-4, "ce_class_weights": None,
            "init_checkpoint": None, "no_boundary_head": False,
            "boundary_head_weight": 1.0, "seed": 0, "val_every": 0,
            "no_amp": False, "resume": True,
        }
        if any(training_args.get(key) != value for key, value in expected_training.items()):
            raise ValueError(f"fold {fold} training args differ from the fixed Model374 recipe")
        expected_data_root = (args.root / f"data_model374_fold{fold}").resolve()
        if (Path(training_args.get("data_root", "")).resolve() != expected_data_root
                or Path(training_args.get("output_dir", "")).resolve()
                != expected_checkpoint.parent.resolve()
                or "done. final model:" not in (expected_checkpoint.parent / "train.log")
                    .read_text(encoding="utf-8", errors="replace")
                or source_manifest.get("fold") != fold
                or source_manifest.get("checkpoint_sha256")
                != record.get("checkpoint_sha256")
                or Path(source_manifest.get("checkpoint", "")).resolve()
                != expected_checkpoint.resolve()
                or Path(source_manifest.get("training_args", "")).resolve()
                != expected_args.resolve()
                or Path(source_manifest.get("training_data_root", "")).resolve()
                != expected_data_root
                or Path(source_manifest.get("heldout_data_root", "")).resolve()
                != (args.root / "data_model322_group_cv" / f"oof_fold{fold}").resolve()):
            raise ValueError(f"fold {fold} checkpoint training config is not Model374")

    gate_path = fold_checkpoint_root / "fold1_kill_switch_official.json"
    gate_record = manifest.get("fold1_official_kill_gate", {})
    if (Path(gate_record.get("path", "")).resolve() != gate_path.resolve()
            or sha256(gate_path) != gate_record.get("sha256")):
        raise ValueError("official Model374 fold-1 OOF gate is missing or differs from manifest")
    fold1_summary_path = fold_checkpoint_root / "fold1_oof" / "valid_summary.json"
    summary_record = manifest.get("fold1_oof_summary", {})
    if (Path(summary_record.get("path", "")).resolve() != fold1_summary_path.resolve()
            or sha256(fold1_summary_path) != summary_record.get("sha256")):
        raise ValueError("Model374 fold-1 OOF summary is missing or differs from manifest")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_delta = (float(gate.get("candidate_oof_image_mean_miou_empty0", float("nan")))
                  - float(gate.get("model329_same_fold_oof_image_mean_miou_empty0",
                                   float("nan")))) * 100.0
    fold1_oof_summary = json.loads(fold1_summary_path.read_text(encoding="utf-8"))
    if (gate.get("candidate_model") != 374 or gate.get("fold") != 1
            or gate.get("continue") is not True
            or gate.get("model329_recomputed_from_grouped_train_oof_probabilities") is not True
            or gate.get("kill_switch_threshold_pp") != -2.0
            or not np.isfinite(gate_delta) or gate_delta <= -2.0
            or not np.isclose(gate_delta, float(gate.get("delta_pp", float("nan"))),
                              atol=1e-9, rtol=0)
            or not np.isclose(gate_delta, float(gate_record.get("delta_pp", float("nan"))),
                              atol=1e-9, rtol=0)
            or not np.isclose(
                float(fold1_oof_summary.get("image_miou_empty_zero", float("nan"))),
                float(gate.get("candidate_oof_image_mean_miou_empty0", float("nan"))),
                atol=1e-10, rtol=0)):
        raise ValueError("official Model374 fold-1 OOF gate is failed or internally inconsistent")

    fold_archive_paths = {}
    average_records = manifest.get("averages", {})
    for mode in ("flip4", "d4"):
        records = manifest.get("fold_archives", {}).get(mode, [])
        if [record.get("fold") for record in records] != [1, 2, 3, 4]:
            raise ValueError(f"manifest {mode} does not list fold archives 1–4")
        paths = []
        for fold, record in enumerate(records, start=1):
            expected_path = args.fold_predictions / f"fold{fold}_{mode}.npz"
            if (Path(record.get("path", "")).resolve() != expected_path.resolve()
                    or sha256(expected_path) != record.get("sha256")):
                raise ValueError(f"fold {fold} {mode} archive differs from its manifest")
            paths.append(expected_path)
        average = average_records.get(mode, {})
        mean_path = args.fold_predictions / f"fold_mean_{mode}.npz"
        if (average.get("folds") != 4 or average.get("images") != 20
                or Path(average.get("probabilities", "")).resolve() != mean_path.resolve()
                or sha256(mean_path) != average.get("sha256")):
            raise ValueError(f"fold-mean {mode} archive does not match manifest")
        fold_archive_paths[mode] = paths

    load_team_archive = spatial.audit.model350.model341.route.load_archive
    _, full_flip4_reference = load_team_archive(
        outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz", team_order=True)
    _, full_d4_reference = load_team_archive(
        outputs / "model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz",
        team_order=True)
    fold_archives = {}
    for mode in ("flip4", "d4"):
        fold_path = args.fold_predictions / f"fold_mean_{mode}.npz"
        expected_sha = manifest.get("averages", {}).get(mode, {}).get("sha256")
        if not expected_sha or sha256(fold_path) != expected_sha:
            raise ValueError(f"fold-mean {mode} archive does not match inference manifest")
        _, route_ordered = spatial.audit.model350.model341.route.load_archive(
            fold_path, team_order=True)
        fold_archives[mode] = route_ordered
    fold_flip4, fold_d4 = fold_archives["flip4"], fold_archives["d4"]
    expected_names = set(full_flip4_reference)
    if len(expected_names) != 20:
        raise ValueError(f"expected 20 Model336 valid maps, got {len(expected_names)}")
    names = sorted(expected_names)
    for mode, mean_archive in (("flip4", fold_flip4), ("d4", fold_d4)):
        paths = fold_archive_paths[mode]
        for path in paths:
            with np.load(path) as archive:
                if set(archive.files) != expected_names:
                    raise ValueError(f"fold archive keys do not match valid images: {path}")
        mean_path = args.fold_predictions / f"fold_mean_{mode}.npz"
        with np.load(mean_path) as mean_archive:
            for name in names:
                total = None
                for path in paths:
                    with np.load(path) as fold_archive:
                        raw = np.asarray(fold_archive[name], dtype=np.float32)
                    if (raw.ndim != 3 or raw.shape[0] != 4 or not np.isfinite(raw).all()
                            or (total is not None and raw.shape != total.shape)
                            or not np.allclose(raw.sum(0), 1.0, atol=3e-3)):
                        raise ValueError(f"invalid raw fold map in {path} for {name}")
                    total = raw.copy() if total is None else total + raw
                expected_mean = (total / len(paths)).astype(np.float16)
                if not np.array_equal(expected_mean, mean_archive[name]):
                    raise AssertionError(f"fold mean {mode} is not the exact four-fold average for {name}")
    for label, source in (("Model351 D4", full_d4_reference),
                          ("fold mean flip4", fold_flip4), ("fold mean D4", fold_d4)):
        if set(source) != expected_names:
            raise ValueError(f"{label} valid keys differ from the fixed image list")
    agreement = {mode: np.zeros((4, 4), dtype=np.int64) for mode in ("flip4", "d4")}
    for name in names:
        references = {"flip4": full_flip4_reference[name],
                      "d4": full_d4_reference[name]}
        predictions = {"flip4": fold_flip4[name], "d4": fold_d4[name]}
        for mode in ("flip4", "d4"):
            full, bag = references[mode], predictions[mode]
            if full.shape != bag.shape or full.ndim != 3 or full.shape[0] != 4:
                raise ValueError(f"{mode} reference/fold-mean shape mismatch for {name}")
        for probabilities in (*references.values(), *predictions.values()):
            if (probabilities.shape != full.shape or not np.isfinite(probabilities).all()
                    or not np.allclose(probabilities.sum(0), 1.0, atol=3e-3)):
                raise ValueError(f"invalid or non-normalized probabilities for {name}")
        for mode in ("flip4", "d4"):
            full, bag = references[mode], predictions[mode]
            encoded = full.argmax(axis=0).astype(np.int64) * 4 + bag.argmax(axis=0).astype(np.int64)
            agreement[mode] += np.bincount(encoded.ravel(), minlength=16).reshape(4, 4)
    agreement_rate = {mode: float(np.trace(matrix) / matrix.sum())
                      for mode, matrix in agreement.items()}
    for mode, matrix in agreement.items():
        diagonal = np.diag(matrix)
        row_totals = matrix.sum(axis=1)
        column_totals = matrix.sum(axis=0)
        active_rows = row_totals > 0
        active_columns = column_totals > 0
        if (np.any(diagonal[active_rows] <= row_totals[active_rows] - diagonal[active_rows])
                or np.any(diagonal[active_columns]
                          <= column_totals[active_columns] - diagonal[active_columns])):
            raise AssertionError(f"{mode} argmax agreement is not classwise diagonally dominant")
    _, prob347 = load_team_archive(
        outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz",
        team_order=True)
    _, prob351 = load_team_archive(
        outputs / "model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz",
        team_order=True)
    for label, source in (("Model347", prob347), ("Model351", prob351)):
        if set(source) != expected_names:
            raise ValueError(f"{label} valid keys differ from the fixed image list")

    flip4_inference = {"patch_size": 512, "input_scale": 2.0, "overlap": 170,
                       "tta": "4-way horizontal/vertical flips"}
    d4_inference = {"patch_size": 512, "input_scale": 2.0, "overlap": 170,
                    "tta": "8-way dihedral rotations and flips"}
    source_summaries = {
        "Model336": outputs / "model336_hrnet_best/valid_eval/valid_summary.json",
        "Model347": outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_summary.json",
        "Model351": outputs / "model351_model336_d4_tta/valid_eval_d4/valid_summary.json",
    }
    for source, path in source_summaries.items():
        summary = json.loads(path.read_text(encoding="utf-8"))
        expected = d4_inference if source == "Model351" else flip4_inference
        if summary.get("inference") != expected:
            raise ValueError(f"{source} source inference settings differ from the prelocked route")

    model_paths = {
        "model346": outputs / "model346_model336_into_341_route/results.json",
        "model349": outputs / "model349_model347_into_346_route/results.json",
        "model355": outputs / "model355_model349_spatial_audit/results.json",
        "model360": outputs / "model360_entropy_gated_d4_into_355/results.json",
    }
    model346, model349, model355, fixed360 = (
        json.loads(model_paths[key].read_text(encoding="utf-8"))
        for key in ("model346", "model349", "model355", "model360"))
    if tuple(fixed360.get("image_names", ())) != tuple(names):
        raise ValueError("fixed Model360 route image keys/order changed")
    fixed360_scores = np.asarray(fixed360["raw_best_image_scores"], dtype=np.float64)
    if (fixed360_scores.shape != (len(names),)
            or not np.isclose(fixed360_scores.mean(),
                              float(fixed360["raw_best_valid_image_mean_miou_empty0"]),
                              atol=1e-10, rtol=0)):
        raise ValueError("stored Model360 baseline scores are internally inconsistent")

    # All artifact/class-axis checks above use probabilities only. Open valid
    # masks only after they pass, for the single predeclared official score.
    rows, names_from_labels, base_route, gamma329, _ = spatial.audit.model350.load_rows(
        args.root, outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz")
    if tuple(names_from_labels) != tuple(names):
        raise ValueError("valid mask image ordering differs from archived probabilities")
    names = names_from_labels

    gamma336 = tuple(model346["raw_best_gamma336"])
    gamma347 = tuple(model349["raw_best_gamma347"])
    blend_primary, blend_eutectic = model355["raw_best_spatial_blend"]
    margin, strength = fixed360["raw_best_entropy_gate"]
    fixed_route = dict(base_route, gaussian_blend=blend_primary,
                       eut_edge_blend=blend_eutectic)
    baseline_scores = np.asarray(fixed360["raw_best_image_scores"], dtype=np.float64)
    if baseline_scores.shape != (len(rows),):
        raise ValueError("Model360 baseline per-image score shape mismatch")

    if tuple(names) != tuple(fixed360["image_names"]):
        raise AssertionError("Model360 image order changed")
    if not np.isclose(baseline_scores.mean(),
                      float(fixed360["raw_best_valid_image_mean_miou_empty0"]),
                      atol=1e-10, rtol=0):
        raise AssertionError("stored Model360 score does not match its per-image scores")

    # Validate every baseline image before calculating any candidate score.
    for index, (name, row) in enumerate(zip(names, rows)):
        full_flip4 = row["sources"]["336"].astype(np.float32)
        full_d4 = prob351[name]
        target = row["item"]["target"]
        base_gate = (model360.entropy(full_d4) + margin
                     < model360.entropy(full_flip4)).astype(np.float32)
        row["sources"]["336"] = full_flip4 + strength * base_gate[None] * (
            full_d4 - full_flip4)
        row["sources"]["347"] = prob347[name]
        base_parts = spatial.spatial_parts(row, base_route, gamma329, gamma336, gamma347)
        base_pred = spatial.prediction(base_parts, fixed_route,
                                       blend_primary, blend_eutectic)
        base_ious = spatial.audit.class_iou(base_pred, target)
        if not np.isclose(base_ious.mean(), baseline_scores[index], atol=1e-9, rtol=0):
            raise AssertionError(
                f"Model360 parity failed for {name}: {base_ious.mean()} vs "
                f"{baseline_scores[index]}")
        row["sources"]["336"] = full_flip4

    candidate_scores = np.empty(len(rows), dtype=np.float64)
    class_scores = np.empty((len(rows), 4), dtype=np.float64)
    image_records = []
    for index, (name, row) in enumerate(zip(names, rows)):
        full_flip4 = row["sources"]["336"].astype(np.float32)
        full_d4 = prob351[name]
        bag_flip4 = fold_flip4[name]
        bag_d4 = fold_d4[name]
        shapes = {value.shape for value in (full_flip4, full_d4, bag_flip4, bag_d4)}
        if len(shapes) != 1 or len(next(iter(shapes))) != 3:
            raise ValueError(f"probability shape mismatch for {name}: {shapes}")
        mixed_flip4 = 0.5 * (full_flip4 + bag_flip4)
        mixed_d4 = 0.5 * (full_d4 + bag_d4)
        gate = (model360.entropy(mixed_d4) + margin
                < model360.entropy(mixed_flip4)).astype(np.float32)
        row["sources"]["336"] = mixed_flip4 + strength * gate[None] * (
            mixed_d4 - mixed_flip4)
        row["sources"]["347"] = prob347[name]
        parts = spatial.spatial_parts(row, base_route, gamma329, gamma336, gamma347)
        pred = spatial.prediction(parts, fixed_route, blend_primary, blend_eutectic)
        ious = spatial.audit.class_iou(pred, target)
        class_scores[index] = ious
        candidate_scores[index] = float(ious.mean())
        image_records.append({"image": name, "specimen_group": group_for(name),
                              "image_miou_empty0": candidate_scores[index],
                              "class_iou_empty0": ious.tolist()})
        print(f"scored fixed candidate {index + 1}/{len(rows)}: {name}", flush=True)

    baseline_mean = float(fixed360["raw_best_valid_image_mean_miou_empty0"])
    delta = candidate_scores - baseline_scores
    groups = [record["specimen_group"] for record in image_records]
    input_paths = {
        "Model336_valid_probabilities": outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz",
        "Model347_valid_probabilities": outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz",
        "Model351_valid_probabilities": outputs / "model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz",
        **{f"{name}_summary": path for name, path in source_summaries.items()},
        **{f"{name}_route_config": path for name, path in model_paths.items()},
    }
    result = {
        "model": 376,
        "status": "single prelocked validation evaluation; no parameter search",
        "metric": "20-image mean of four class IoUs; empty union=0",
        "candidate": {
            "fold_ensemble": "arithmetic mean of four Model374 checkpoint probability maps",
            "combine_with_full_hrnet": "0.50 Model336 flip4 + 0.50 fold-mean flip4; "
                                       "0.50 Model351 D4 + 0.50 fold-mean D4",
            "entropy_gate": {"margin": margin, "strength": strength},
            "Model360_class_route_and_spatial_settings": "unchanged",
        },
        "route_class_order": expected_route_order,
        "fold_inference_manifest": str(manifest_path),
        "fold_inference_manifest_sha256": sha256(manifest_path),
        "fold_mean_archive_sha256": {
            mode: manifest["averages"][mode]["sha256"] for mode in ("flip4", "d4")
        },
        "fold_archive_sha256": {
            mode: [record["sha256"] for record in manifest["fold_archives"][mode]]
            for mode in ("flip4", "d4")
        },
        "valid_image_sha256": manifest["valid_image_sha256"],
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "inference_code_sha256": manifest["inference_code_sha256"],
        "predictor_script_sha256": manifest["predictor_script_sha256"],
        "evaluator_script_sha256": sha256(Path(__file__).resolve()),
        "route_support_source_sha256": imported_local_source_hashes(
            Path(__file__).resolve().parent),
        "label_free_channel_axis_sanity": {
            "argmax_agreement_with_full_model": agreement_rate,
            "class_order": expected_route_order,
            "confusion_matrices_rows_full_cols_fold_mean": {
                mode: matrix.tolist() for mode, matrix in agreement.items()
            },
        },
        "baseline_model360_image_mean_miou_empty0": baseline_mean,
        "candidate_image_mean_miou_empty0": float(candidate_scores.mean()),
        "delta_pp_vs_model360": float(delta.mean() * 100.0),
        "paired_specimen_group_bootstrap_95ci_delta_pp": group_bootstrap_ci(
            delta, groups, args.bootstrap, seed=376),
        "class_image_mean_iou_empty0": class_scores.mean(axis=0).tolist(),
        "positive_images": int((delta > 0).sum()),
        "positive_specimen_groups": int(sum(
            np.mean(delta[np.asarray(groups) == group]) > 0 for group in set(groups))),
        "valid_was_previously_used_for_model_selection": True,
        "per_image": image_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "baseline_model360_image_mean_miou_empty0", "candidate_image_mean_miou_empty0",
        "delta_pp_vs_model360", "paired_specimen_group_bootstrap_95ci_delta_pp",
        "class_image_mean_iou_empty0", "positive_images")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
