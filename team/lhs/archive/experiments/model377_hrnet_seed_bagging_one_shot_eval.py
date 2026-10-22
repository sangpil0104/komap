#!/usr/bin/env python3
"""One fixed Model377 full-train seed ensemble evaluation on valid.

The two predeclared full-train HRNet seeds are averaged per TTA mode, blended
50:50 with Model336 flip4 and Model351 D4 maps, and passed through the frozen
Model360 route. No parameters are searched.
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("/workspace/komap_model1/outputs/model377_hrnet_seed_valid_preds"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model376-result", type=Path,
                        default=Path("/workspace/komap_model1/outputs/model376_bagged_hrnet_valid_eval/results.json"))
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    root = args.root.resolve()
    predictions = args.predictions.resolve()
    outputs = root / "outputs"
    model376_result_path = args.model376_result.resolve()
    model376_result = json.loads(model376_result_path.read_text(encoding="utf-8"))
    if (model376_result.get("model") != 376
            or model376_result.get("metric")
            != "20-image mean of four class IoUs; empty union=0"):
        raise ValueError("Model376 result does not use the fixed official metric")
    model376_score = float(model376_result["candidate_image_mean_miou_empty0"])
    if not np.isfinite(model376_score) or model376_score >= 0.80:
        raise ValueError(f"Model377 fallback is allowed only if Model376 < 0.80; got {model376_score}")
    code_dir = root / "model336_hrnet_best"
    sys.path.insert(0, str(code_dir))
    from komap_seg.palette import CLASS_NAMES as RAW_CLASS_NAMES

    manifest_path = predictions / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_raw_order = ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"]
    expected_route_order = ["Al3Ni", "Primary Si", "Al", "Eutectic Si"]
    expected_inference = {"patch_size": 512, "overlap": 170, "input_scale": 2.0,
                          "batch_size": 2, "tta_modes": ["flip4", "d4"],
                          "downsample_antialias": False}
    if (manifest.get("model") != 377 or manifest.get("class_order") != expected_raw_order
            or list(RAW_CLASS_NAMES) != expected_raw_order
            or tuple(spatial.audit.model350.CLASS_ORDER) != tuple(expected_route_order)
            or manifest.get("image_count") != 20
            or manifest.get("probability_averaging")
            != "arithmetic mean of seed 1 and seed 2 probability maps"
            or manifest.get("inference") != expected_inference
            or Path(manifest.get("inference_code_dir", "")).resolve() != code_dir.resolve()):
        raise ValueError("Model377 inference manifest/classes/settings differ from fixed protocol")
    if (Path(manifest.get("fallback_trigger_model376_result", "")).resolve()
            != model376_result_path.resolve()
            or manifest.get("fallback_trigger_model376_result_sha256")
            != sha256(model376_result_path)
            or not np.isclose(float(manifest.get("fallback_trigger_model376_score", float("nan"))),
                              model376_score, atol=0.0, rtol=0)):
        raise ValueError("Model377 fallback trigger differs from the fixed Model376 result")
    expected_code = {"train.py", "evaluate_image_mean.py", *(
        str(path.relative_to(code_dir)) for path in (code_dir / "komap_seg").rglob("*.py"))}
    code_hashes = manifest.get("inference_code_sha256", {})
    if set(code_hashes) != expected_code or any(
            sha256(code_dir / name) != code_hashes[name] for name in expected_code):
        raise ValueError("Model377 inference code differs from manifest")
    predictor_path = Path(__file__).resolve().with_name(
        "model377_hrnet_seed_bagging_predict_only.py")
    if sha256(predictor_path) != manifest.get("predictor_script_sha256"):
        raise ValueError("Model377 label-free predictor differs from manifest")
    runner_path = (Path(__file__).resolve().parent / "runners"
                   / "run_model377_hrnet_full_train_seeds.sh")
    if sha256(runner_path) != manifest.get("training_runner_sha256"):
        raise ValueError("Model377 training runner differs from inference manifest")

    train_image_paths = sorted((root / "data/train/images").glob("*_image.png"))
    train_mask_paths = sorted((root / "data/train/masks").glob("*_mask.png"))
    valid_image_paths = sorted((root / "data/valid/images").glob("*_image.png"))
    train_image_hashes = {path.name: sha256(path) for path in train_image_paths}
    train_mask_hashes = {path.name: sha256(path) for path in train_mask_paths}
    valid_image_hashes = {
        path.name[:-len("_image.png")]: sha256(path) for path in valid_image_paths
    }
    if (Path(manifest.get("valid_image_dir", "")).resolve()
            != (root / "data/valid/images").resolve()
            or len(train_image_paths) != 70 or len(train_mask_paths) != 70
            or len(valid_image_paths) != 20
            or train_image_hashes != manifest.get("training_image_sha256")
            or train_mask_hashes != manifest.get("training_mask_sha256")
            or valid_image_hashes != manifest.get("valid_image_sha256")):
        raise ValueError("Model377 training or valid image set differs from inference manifest")

    checkpoint_root = outputs / "model377_hrnet_full_train_seeds"
    checkpoints = manifest.get("checkpoints", [])
    if [record.get("seed") for record in checkpoints] != [1, 2]:
        raise ValueError("Model377 manifest must contain seeds 1 and 2 in fixed order")
    expected_training = {
        "architecture": "UnetPlusPlus", "encoder_name": "tu-hrnet_w48",
        "encoder_weights": "imagenet", "patch_size": 512, "input_scale": 2.0,
        "samples_per_image": 12, "batch_size": 4, "accumulation_steps": 2,
        "num_workers": 4,
        "epochs": 70, "lr": 3e-4, "encoder_lr_mult": 1.0,
        "weight_decay": 1e-4, "ce_class_weights": None,
        "init_checkpoint": None, "no_boundary_head": False,
        "boundary_head_weight": 1.0, "val_every": 0,
        "no_amp": False, "resume": True,
    }
    for seed, record in enumerate(checkpoints, start=1):
        checkpoint = checkpoint_root / f"seed{seed}/last.pt"
        args_path = checkpoint.parent / "args.json"
        log_path = checkpoint.parent / "train.log"
        for label, expected_path, hash_field in (
                ("checkpoint", checkpoint, "checkpoint_sha256"),
                ("training_args", args_path, "training_args_sha256"),
                ("training_log", log_path, "training_log_sha256")):
            if (Path(record.get(label, "")).resolve() != expected_path.resolve()
                    or sha256(expected_path) != record.get(hash_field)):
                raise ValueError(f"seed {seed} {label} differs from manifest")
        train_args = json.loads(args_path.read_text(encoding="utf-8"))
        if any(train_args.get(key) != value for key, value in expected_training.items()):
            raise ValueError(f"seed {seed} training args differ from fixed recipe")
        if (train_args.get("seed") != seed
                or Path(train_args.get("data_root", "")).resolve() != (root / "data").resolve()
                or Path(train_args.get("output_dir", "")).resolve()
                != checkpoint.parent.resolve()
                or "done. final model:" not in log_path.read_text(
                    encoding="utf-8", errors="replace")):
            raise ValueError(f"seed {seed} is not a completed full-train run")

    archives_by_mode = {}
    for mode in ("flip4", "d4"):
        records = manifest.get("seed_archives", {}).get(mode, [])
        if [record.get("seed") for record in records] != [1, 2]:
            raise ValueError(f"{mode} manifest must list seed archives 1 and 2")
        paths = []
        for seed, record in enumerate(records, start=1):
            path = predictions / f"seed{seed}_{mode}.npz"
            if (Path(record.get("path", "")).resolve() != path.resolve()
                    or sha256(path) != record.get("sha256")):
                raise ValueError(f"seed {seed} {mode} archive differs from manifest")
            paths.append(path)
        average = manifest.get("averages", {}).get(mode, {})
        mean_path = predictions / f"seed_mean_{mode}.npz"
        if (average.get("seeds") != [1, 2] or average.get("images") != 20
                or Path(average.get("probabilities", "")).resolve() != mean_path.resolve()
                or sha256(mean_path) != average.get("sha256")):
            raise ValueError(f"seed-mean {mode} archive differs from manifest")
        with np.load(mean_path) as mean_archive:
            for name in sorted(valid_image_hashes):
                total = None
                for path in paths:
                    with np.load(path) as archive:
                        probability = np.asarray(archive[name], dtype=np.float32)
                    if (probability.shape[0] != 4 or not np.isfinite(probability).all()
                            or not np.allclose(probability.sum(0), 1.0, atol=3e-3)):
                        raise ValueError(f"invalid probabilities in {path} for {name}")
                    total = probability if total is None else total + probability
                if not np.array_equal((total / 2).astype(np.float16), mean_archive[name]):
                    raise AssertionError(f"seed mean {mode} is not exact for {name}")
        archives_by_mode[mode] = mean_path

    route = spatial.audit.model350.model341.route
    _, seed_flip4 = route.load_archive(archives_by_mode["flip4"], team_order=True)
    _, seed_d4 = route.load_archive(archives_by_mode["d4"], team_order=True)
    _, reference_flip4 = route.load_archive(
        outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz", team_order=True)
    _, reference_d4 = route.load_archive(
        outputs / "model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz",
        team_order=True)
    expected_names = set(valid_image_hashes)
    for label, archive in (("seed flip4", seed_flip4), ("seed d4", seed_d4),
                           ("Model336 flip4", reference_flip4),
                           ("Model351 D4", reference_d4)):
        if set(archive) != expected_names:
            raise ValueError(f"{label} probability keys differ from fixed valid images")
        for name, probability in archive.items():
            if (probability.ndim != 3 or probability.shape[0] != 4
                    or not np.isfinite(probability).all()
                    or not np.allclose(probability.sum(0), 1.0, atol=3e-3)):
                raise ValueError(f"invalid {label} probabilities for {name}")
    _, reference_347 = route.load_archive(
        outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz",
        team_order=True)
    for name in sorted(expected_names):
        shapes = {archive[name].shape for archive in (
            seed_flip4, seed_d4, reference_flip4, reference_d4, reference_347)}
        if len(shapes) != 1:
            raise ValueError(f"probability source shapes differ for {name}: {shapes}")
    for source, summary_path, expected in (
            ("Model336", outputs / "model336_hrnet_best/valid_eval/valid_summary.json",
             {"patch_size": 512, "input_scale": 2.0, "overlap": 170,
              "tta": "4-way horizontal/vertical flips"}),
            ("Model347", outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_summary.json",
             {"patch_size": 512, "input_scale": 2.0, "overlap": 170,
              "tta": "4-way horizontal/vertical flips"}),
            ("Model351", outputs / "model351_model336_d4_tta/valid_eval_d4/valid_summary.json",
             {"patch_size": 512, "input_scale": 2.0, "overlap": 170,
              "tta": "8-way dihedral rotations and flips"})):
        if json.loads(summary_path.read_text(encoding="utf-8")).get("inference") != expected:
            raise ValueError(f"{source} inference settings differ from frozen protocol")

    pre_mask_inputs = {
        "Model195_valid_probabilities": outputs / "model195_native_resume/valid_probs.npz",
        "Model217_valid_probabilities": outputs / "model220_native_tent_extended/valid_probs_06.npz",
        "Model269_valid_probabilities": outputs / "model269_fullstats_1x_extended/valid_probs.npz",
        "Model322_valid_probabilities": outputs / "model322_team_b4_group_oof/valid_ensemble/probabilities.npz",
        "Model326_valid_probabilities": outputs / "model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz",
        "Model329_valid_probabilities": outputs / "model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz",
        "Model336_valid_probabilities": outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz",
        "Model347_valid_probabilities": outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz",
        "Model351_valid_probabilities": outputs / "model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz",
        "Model336_valid_summary": outputs / "model336_hrnet_best/valid_eval/valid_summary.json",
        "Model347_valid_summary": outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_summary.json",
        "Model351_valid_summary": outputs / "model351_model336_d4_tta/valid_eval_d4/valid_summary.json",
        "Model339_route": outputs / "model339_joint_gamma_route_search/results.json",
        "Model341_route": outputs / "model341_model329_joint_route_search/results.json",
        "Model346_route": outputs / "model346_model336_into_341_route/results.json",
        "Model349_route": outputs / "model349_model347_into_346_route/results.json",
        "Model355_route": outputs / "model355_model349_spatial_audit/results.json",
        "Model360_route": outputs / "model360_entropy_gated_d4_into_355/results.json",
    }
    pre_mask_input_hashes = {name: sha256(path) for name, path in pre_mask_inputs.items()}

    # Fixed route settings and all probability-source inputs are checked before
    # load_rows opens validation masks.
    prob347 = reference_347
    if set(prob347) != expected_names:
        raise ValueError("Model347 probability keys differ from fixed valid images")
    for name, probability in prob347.items():
        if (probability.shape[0] != 4 or not np.isfinite(probability).all()
                or not np.allclose(probability.sum(0), 1.0, atol=3e-3)):
            raise ValueError(f"invalid Model347 probabilities for {name}")
    model346 = json.loads((outputs / "model346_model336_into_341_route/results.json")
                           .read_text(encoding="utf-8"))
    model349 = json.loads((outputs / "model349_model347_into_346_route/results.json")
                           .read_text(encoding="utf-8"))
    model355 = json.loads((outputs / "model355_model349_spatial_audit/results.json")
                           .read_text(encoding="utf-8"))
    fixed360 = json.loads((outputs / "model360_entropy_gated_d4_into_355/results.json")
                           .read_text(encoding="utf-8"))
    if tuple(fixed360.get("image_names", ())) != tuple(sorted(expected_names)):
        raise ValueError("Model360 image order differs from fixed valid filenames")
    baseline_scores = np.asarray(fixed360["raw_best_image_scores"], dtype=np.float64)
    if (baseline_scores.shape != (20,)
            or not np.isclose(baseline_scores.mean(),
                              float(fixed360["raw_best_valid_image_mean_miou_empty0"]),
                              atol=1e-10, rtol=0)):
        raise ValueError("stored Model360 image scores are internally inconsistent")
    gamma336 = tuple(model346["raw_best_gamma336"])
    gamma347 = tuple(model349["raw_best_gamma347"])
    blend_primary, blend_eutectic = model355["raw_best_spatial_blend"]
    margin, strength = fixed360["raw_best_entropy_gate"]

    # All provenance, input hashes and class-axis checks finish before this call
    # opens any valid masks.
    rows, names, base_route, gamma329, _ = spatial.audit.model350.load_rows(
        root, outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz")
    names = list(names)
    if set(names) != expected_names:
        raise ValueError("valid mask names differ from prediction archive keys")
    valid_mask_paths = sorted((root / "data/valid/masks").glob("*_mask.png"))
    valid_mask_hashes = {
        path.name[:-len("_mask.png")]: sha256(path) for path in valid_mask_paths
    }
    if set(valid_mask_hashes) != expected_names:
        raise ValueError("valid mask hash keys differ from fixed valid images")
    fixed_route = dict(base_route, gaussian_blend=blend_primary,
                       eut_edge_blend=blend_eutectic)
    pre_mask_input_hashes["Model183_route_entries"] = sha256(
        outputs / "model183_metric_calibration/entries.pkl")
    for index, (name, row) in enumerate(zip(names, rows)):
        original_flip4 = row["sources"]["336"].astype(np.float32)
        original_d4 = reference_d4[name]
        row["sources"]["336"] = original_flip4 + strength * (
            model360.entropy(original_d4) + margin
            < model360.entropy(original_flip4))[None].astype(np.float32) * (
                original_d4 - original_flip4)
        row["sources"]["347"] = prob347[name]
        parts = spatial.spatial_parts(row, base_route, gamma329, gamma336, gamma347)
        prediction = spatial.prediction(parts, fixed_route, blend_primary, blend_eutectic)
        baseline_ious = spatial.audit.class_iou(prediction, row["item"]["target"])
        if not np.isclose(baseline_ious.mean(), baseline_scores[index], atol=1e-9, rtol=0):
            raise AssertionError(f"Model360 parity failed for {name}")

    candidate = np.empty(20, dtype=np.float64)
    class_scores = np.empty((20, 4), dtype=np.float64)
    per_image = []
    for index, (name, row) in enumerate(zip(names, rows)):
        full_flip4 = reference_flip4[name].astype(np.float32)
        full_d4 = reference_d4[name].astype(np.float32)
        mixed_flip4 = 0.5 * (full_flip4 + seed_flip4[name])
        mixed_d4 = 0.5 * (full_d4 + seed_d4[name])
        gate = (model360.entropy(mixed_d4) + margin
                < model360.entropy(mixed_flip4)).astype(np.float32)
        row["sources"]["336"] = mixed_flip4 + strength * gate[None] * (
            mixed_d4 - mixed_flip4)
        row["sources"]["347"] = prob347[name]
        parts = spatial.spatial_parts(row, base_route, gamma329, gamma336, gamma347)
        prediction = spatial.prediction(parts, fixed_route, blend_primary, blend_eutectic)
        ious = spatial.audit.class_iou(prediction, row["item"]["target"])
        class_scores[index] = ious
        candidate[index] = float(ious.mean())
        per_image.append({"image": name, "specimen_group": group_for(name),
                          "image_miou_empty0": candidate[index],
                          "class_iou_empty0": ious.tolist()})
    groups = [record["specimen_group"] for record in per_image]
    delta = candidate - baseline_scores
    input_paths = {
        "Model336_valid_probabilities": outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz",
        "Model347_valid_probabilities": outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz",
        "Model351_valid_probabilities": outputs / "model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz",
        "Model346_route": outputs / "model346_model336_into_341_route/results.json",
        "Model349_route": outputs / "model349_model347_into_346_route/results.json",
        "Model355_route": outputs / "model355_model349_spatial_audit/results.json",
        "Model360_route": outputs / "model360_entropy_gated_d4_into_355/results.json",
    }
    result = {
        "model": 377,
        "status": "single prelocked validation evaluation; no parameter search",
        "metric": "20-image mean of four class IoUs; empty union=0",
        "candidate": "Model336/351 plus 50:50 two-seed full-train HRNet means, Model360 route frozen",
        "fallback_trigger": "predeclared; run only if Model376 official score is below 0.80",
        "fallback_trigger_model376_score": model376_score,
        "fallback_trigger_model376_result": str(model376_result_path),
        "fallback_trigger_model376_result_sha256": sha256(model376_result_path),
        "route_class_order": expected_route_order,
        "inference_manifest": str(manifest_path),
        "inference_manifest_sha256": sha256(manifest_path),
        "seed_mean_archive_sha256": {
            mode: manifest["averages"][mode]["sha256"] for mode in ("flip4", "d4")},
        "valid_image_sha256": manifest["valid_image_sha256"],
        "valid_mask_sha256": valid_mask_hashes,
        "training_image_sha256": manifest["training_image_sha256"],
        "training_mask_sha256": manifest["training_mask_sha256"],
        "input_sha256": {**pre_mask_input_hashes,
                          **{name: sha256(path) for name, path in input_paths.items()}},
        "inference_code_sha256": manifest["inference_code_sha256"],
        "predictor_script_sha256": manifest["predictor_script_sha256"],
        "evaluator_script_sha256": sha256(Path(__file__).resolve()),
        "baseline_model360_image_mean_miou_empty0": float(baseline_scores.mean()),
        "candidate_image_mean_miou_empty0": float(candidate.mean()),
        "delta_pp_vs_model360": float(delta.mean() * 100.0),
        "paired_specimen_group_bootstrap_95ci_delta_pp": group_bootstrap_ci(
            delta, groups, args.bootstrap, seed=377),
        "class_image_mean_iou_empty0": class_scores.mean(axis=0).tolist(),
        "positive_images": int((delta > 0).sum()),
        "valid_was_previously_used_for_model_selection": True,
        "per_image": per_image,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "baseline_model360_image_mean_miou_empty0", "candidate_image_mean_miou_empty0",
        "delta_pp_vs_model360", "paired_specimen_group_bootstrap_95ci_delta_pp",
        "class_image_mean_iou_empty0")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
