"""Strict YAML inheritance and effective settings, with no ignored feature flags."""

from copy import deepcopy
from pathlib import Path
import yaml
from .io import ROOT, REPO, digest, local_path, jsonl_read
from .contracts import CLASS_NAMES, PALETTE

DEFAULT = {
    "schema_version": 1,
    "run": {"name": "legacy_reproduction", "seed": 32, "output_root": "outputs"},
    "paths": {
        "dataset_root": str(REPO / "dataset"),
        "dataset_manifest": "manifests/dataset.jsonl",
        "checkpoint_manifest": "manifests/checkpoints.jsonl",
        "legacy_source": str(
            REPO
            / "experiments/seonghyeon_seed10_99_v1/outputs/20260926T082111_741695Z/source"
        ),
        "legacy_checkpoint": None,
        "encoder_weights": str(
            REPO
            / "experiments/seonghyeon_seed10_v1/assets/efficientnet_b4_imagenet_encoder.pt"
        ),
    },
    "classes": {
        "names": list(CLASS_NAMES),
        "rgb": PALETTE.tolist(),
        "ignore_index": 255,
    },
    "model": {
        "family": "legacy_b4_unet",
        "encoder_name": "efficientnet-b4",
        "new_branches": {"texture": False, "context": False, "residual_head": False},
        "residual_last_layer_zero_init": True,
        "texture_channels": [16, 32, 64],
        "context_channels": [32, 64, 128, 256],
        "fusion_channels": [64, 32, 16],
        "new_module_norm": "groupnorm",
        "new_module_gn_groups": 8,
    },
    "geometry": {
        "profile": "legacy_scaled_crop",
        "target_width": 4000,
        "detail_input_size": 1536,
        "context_input_size": 768,
        "context_factor": 2.0,
        "align_corners": False,
        "antialias": False,
        "padding": "reflect",
        "mask_interpolation": "nearest-exact",
        "native_side_rounding": "half_up",
        "scale_jitter_enabled": False,
        "legacy_resize_policy": "opencv_linear_fx_fy; pad32_reflect; bf16; flip_then_rot90; softmax_fp32; mean_D4; native_area_if_upscaled_else_bilinear",
    },
    "relative_features": {
        "sigma_native_px": [4.0, 16.0],
        "eps_std": 0.01,
        "clip_z": 5.0,
        "gaussian_truncate": 3.0,
        "image_photometric_before_feature": True,
    },
    "sampling": {
        "image_selection": "area",
        "random": 1.0,
        "mixed": 0.0,
        "hard": 0.0,
        "local_window_native_px": 41,
        "hard_cache": None,
        "allow_train_in_sample_mining": True,
        "allow_validation_mining": False,
        "missing_hard_policy": "random_same_image",
        "hard_warmup_updates": 500,
        "hard_refresh_updates": 1000,
        "hard_mining_d4": False,
    },
    "training": {
        "recipe": "scratch_comparable",
        "initialize_from": None,
        "batch_size": 2,
        "accumulation": 1,
        "total_optimizer_updates": 16000,
        "warmup_updates": 300,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "optimizer": "AdamW",
        "stage_a_updates": 500,
        "stage_b_updates": 3500,
        "stage_a_new_lr": 0.0003,
        "stage_b_encoder_lr": 0.00001,
        "stage_b_decoder_lr": 0.00003,
        "stage_b_new_lr": 0.0001,
        "stage_b_warmup_updates": 100,
        "freeze_bn": False,
        "validation_every_updates": 1000,
        "checkpoint_every_updates": 1000,
        "ema_enabled": False,
        "ema_decay": 0.999,
        "precision": "bf16",
        "grad_clip": 1.0,
        "workers": 0,
        "sample_rng": "indexed_seed_step_micro_sample",
        "train_ids": None,
        "valid_ids": None,
        "validation_enabled": True,
        "augmentation": "legacy_exact_operators",
        "native_blur_noise": False,
        "save_results": True,
    },
    "loss": {
        "native_enabled": False,
        "native_weight": 0.5,
        "native_lovasz_weight": 0.5,
        "native_lovasz_per_image": True,
        "native_lovasz_classes": "present",
        "probability_eps": 1e-6,
        "boundary_enabled": False,
    },
    "evaluation": {
        "primary_metric": "image_miou",
        "empty_union_policy": "one",
        "class_reduction": "macro_four",
        "image_reduction": "equal_mean",
        "native_resolution": True,
        "d4": True,
    },
    "inference": {
        "mode": "legacy_full_frame",
        "d4": True,
        "probability_cache_dtype": "float32",
        "ensemble_accumulation_dtype": "float64",
        "temperature": 1.0,
        "class_bias": [0.0] * 4,
        "native_overlap_fraction": 0.5,
        "blend": "gaussian",
        "blend_weight_floor": 0.001,
    },
    "ensemble": {
        "fixed_member_ids": [f"seed_{i:02d}" for i in range(100)],
        "weights": None,
        "all_members_required": True,
        "cache_root": "outputs/probabilities",
        "output": "outputs/ensemble_uniform",
        "selection": "predeclared_equal_weight",
        "k_curve": [1, 3, 5, 10, 20, 50, 100],
    },
    "oof": {
        "source_split": "train",
        "n_folds": 5,
        "fold_seed": 2026,
        "grouping_policy": "image",
        "teacher_config": "configs/experiments/E02_scratch.yaml",
        "teacher_seeds": [0],
        "teacher_fixed_updates": 16000,
        "strict_lineage": True,
        "root": "outputs/oof",
        "fold_manifest": "manifests/oof_folds.jsonl",
        "deployment_teacher_manifest": None,
    },
    "refiner": {
        "input_channels": 8,
        "channel_names": [
            "gray",
            "P_Al",
            "P_Al3Ni",
            "P_PrimarySi",
            "P_EutecticSi",
            "relative4",
            "relative16",
            "entropy",
        ],
        "channels": [32, 64, 128, 256],
        "norm": "groupnorm",
        "zero_init_last": True,
        "probability_eps": 1e-6,
        "native_crop_size": 512,
        "batch_size": 2,
        "learning_rate": 0.0001,
        "optimizer_updates": 4000,
        "training_cache": "outputs/oof/train/manifest.jsonl",
        "validation_cache": None,
        "checkpoint": None,
    },
    "submission": {
        "checkpoint": None,
        "output": "outputs/submission",
        "filename_suffix": "_mask.png",
    },
    "execution": {
        "dry_run": True,
        "overwrite_existing_outputs": False,
        "gpus": [0, 1, 2, 3, 4, 5],
        "device": "cuda:0",
        "cpu_threads": 1,
    },
}


def merge(base, patch, trail=""):
    result = deepcopy(base)
    for k, v in patch.items():
        if k not in base:
            raise ValueError(f"Unknown config key: {trail}{k}")
        old = base[k]
        if isinstance(old, dict):
            if not isinstance(v, dict):
                raise TypeError(f"{trail}{k} must be a mapping")
            result[k] = merge(old, v, trail + k + ".")
        else:
            if old is not None:
                ok = isinstance(v, type(old)) or (
                    isinstance(old, float) and type(v) in (float, int)
                )
                if isinstance(old, bool):
                    ok = type(v) is bool
                if not ok:
                    raise TypeError(
                        f"{trail}{k}: expected {type(old).__name__}, got {type(v).__name__}"
                    )
            result[k] = v
    return result


def _read(path, stack):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Config inherit cycle: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    parent = data.pop("inherit", None)
    return merge(
        _read(path.parent / parent, stack + [path]) if parent else DEFAULT, data
    )


def load_config(path, overrides=()):
    cfg = _read(local_path(path), [])
    for override in overrides:
        key, raw = override.split("=", 1)
        patch = value = yaml.safe_load(raw)
        for part in reversed(key.split(".")):
            patch = {part: patch}
        cfg = merge(cfg, patch)
    validate(cfg)
    return cfg


def validate(c):
    def require(ok, msg):
        if not ok:
            raise ValueError(msg)

    require(c["schema_version"] == 1, "Unknown schema_version")
    require(c["classes"] == DEFAULT["classes"], "Class order/palette/ignore mismatch")
    for section in ["relative_features"]:
        require(c[section] == DEFAULT[section], f"Unsupported {section} contract")
    for k, value in DEFAULT["evaluation"].items():
        if k != "d4":
            require(c["evaluation"][k] == value, f"Unsupported evaluation.{k}")
    require(
        c["execution"]["overwrite_existing_outputs"] is False,
        "Overwriting existing runs is prohibited",
    )
    require(
        c["execution"]["gpus"] and set(c["execution"]["gpus"]) <= set(range(6)),
        "Only GPUs 0–5 are authorized",
    )
    require(c["execution"]["cpu_threads"] > 0, "cpu_threads must be positive")
    b = c["model"]["new_branches"]
    g = c["geometry"]
    t = c["training"]
    s = c["sampling"]
    l = c["loss"]
    inf = c["inference"]
    require(
        c["model"]["encoder_name"] == "efficientnet-b4", "Only audited B4 supported"
    )
    require(
        c["model"]["family"] in ["legacy_b4_unet", "enhanced_b4_unet"],
        "Unknown model family",
    )
    require(
        not (b["context"] or b["texture"]) or b["residual_head"],
        "Texture/context require residual head",
    )
    require(
        c["model"]["family"] != "legacy_b4_unet" or not any(b.values()),
        "Legacy model cannot enable new branches",
    )
    for k in [
        "residual_last_layer_zero_init",
        "texture_channels",
        "context_channels",
        "fusion_channels",
        "new_module_norm",
        "new_module_gn_groups",
    ]:
        require(c["model"][k] == DEFAULT["model"][k], f"Unsupported model.{k}")
    require(g["profile"] in ["legacy_scaled_crop", "native_roi_v1"], "Unknown geometry")
    require(
        g["detail_input_size"] > 0 and g["detail_input_size"] % 32 == 0,
        "Detail input must be a positive multiple of32",
    )
    require(
        g["context_input_size"] > 0 and g["context_input_size"] % 16 == 0,
        "Context input must be divisible by16",
    )
    require(g["target_width"] > 0 and g["context_factor"] >= 1, "Invalid field of view")
    for k in [
        "align_corners",
        "antialias",
        "padding",
        "mask_interpolation",
        "native_side_rounding",
        "scale_jitter_enabled",
        "legacy_resize_policy",
    ]:
        require(g[k] == DEFAULT["geometry"][k], f"Unsupported geometry.{k}")
    require(
        not any(b.values()) or g["profile"] == "native_roi_v1",
        "New branches require native ROI",
    )
    require(
        (inf["mode"] == "legacy_full_frame") == (g["profile"] == "legacy_scaled_crop"),
        "Inference/geometry mismatch",
    )
    require(
        inf["mode"] in ["legacy_full_frame", "dual_fov_native_tiles"],
        "Unknown inference",
    )
    require(inf["d4"] == c["evaluation"]["d4"], "Evaluation/inference D4 mismatch")
    for k in [
        "probability_cache_dtype",
        "ensemble_accumulation_dtype",
        "temperature",
        "class_bias",
        "blend",
    ]:
        require(inf[k] == DEFAULT["inference"][k], f"Unsupported inference.{k}")
    require(
        0 <= inf["native_overlap_fraction"] < 1 and 0 < inf["blend_weight_floor"] <= 1,
        "Invalid blending",
    )
    require(
        t["recipe"] in ["scratch_comparable", "finetune_existing"], "Unknown recipe"
    )
    require(
        t["optimizer"] == "AdamW" and t["precision"] == "bf16",
        "Only audited AdamW/BF16 supported",
    )
    require(
        t["sample_rng"] == "indexed_seed_step_micro_sample" and t["workers"] == 0,
        "Indexed reproducible sampler uses workers=0",
    )
    require(
        t["batch_size"] > 0
        and t["accumulation"] > 0
        and t["total_optimizer_updates"] > 0,
        "Invalid training count",
    )
    require(
        t["validation_every_updates"] > 0 and t["checkpoint_every_updates"] > 0,
        "Invalid checkpoint/validation interval",
    )
    require(
        t["learning_rate"] > 0 and t["weight_decay"] >= 0 and 0 < t["ema_decay"] < 1,
        "Invalid optimizer or EMA",
    )
    require(
        t["recipe"] != "finetune_existing" or t["initialize_from"] is not None,
        "training.initialize_from is required",
    )
    if t["recipe"] == "finetune_existing" and b["residual_head"]:
        require(
            t["stage_a_updates"] + t["stage_b_updates"] == t["total_optimizer_updates"],
            "A+B must equal total updates",
        )
    require(s["image_selection"] in ["area", "uniform"], "Unknown image selection")
    require(
        abs(sum(s[x] for x in ["random", "mixed", "hard"]) - 1) < 1e-8
        and min(s[x] for x in ["random", "mixed", "hard"]) >= 0,
        "Sampler weights must sum to1",
    )
    require(not s["allow_validation_mining"], "Validation mining is forbidden")
    require(s["missing_hard_policy"] == "random_same_image", "Unknown hard fallback")
    require(
        s["hard_warmup_updates"] >= 0 and s["hard_refresh_updates"] > 0,
        "Invalid hard mining schedule",
    )
    require(
        s["local_window_native_px"] > 0 and s["local_window_native_px"] % 2 == 1,
        "Mixed window must be odd",
    )
    require(not l["boundary_enabled"], "Boundary loss is not supported in this version")
    require(
        l["native_lovasz_per_image"] and l["native_lovasz_classes"] == "present",
        "Native Lovasz contract mismatch",
    )
    require(
        not l["native_enabled"] or g["profile"] == "native_roi_v1",
        "Native loss requires native ROI",
    )
    if g["profile"] == "legacy_scaled_crop":
        require(
            s["image_selection"] == "area" and s["random"] == 1,
            "Legacy sampler must be area/random",
        )
        require(
            t["augmentation"] == "legacy_exact_operators",
            "Legacy augmentation mismatch",
        )
    else:
        require(
            s["image_selection"] == "uniform"
            and t["augmentation"] == "native_shared_d4_photo",
            "Native geometry needs uniform/shared-D4 augmentation",
        )
    require(
        c["oof"]["strict_lineage"] and c["oof"]["source_split"] == "train",
        "Strict train-only OOF required",
    )
    require(
        c["oof"]["grouping_policy"] == "image",
        "Only image grouping implemented; group hints reported separately",
    )
    require(
        c["refiner"]["input_channels"] == 8
        and c["refiner"]["channel_names"] == DEFAULT["refiner"]["channel_names"],
        "Refiner input order mismatch",
    )
    for k in ["channels", "norm", "zero_init_last"]:
        require(c["refiner"][k] == DEFAULT["refiner"][k], f"Unsupported refiner.{k}")
    require(c["ensemble"]["all_members_required"], "Partial ensembles are prohibited")
    require(
        c["ensemble"]["selection"] == "predeclared_equal_weight"
        and c["ensemble"]["weights"] is None,
        "Only fixed equal-weight ensemble implemented",
    )
    require(
        len(set(c["ensemble"]["fixed_member_ids"]))
        == len(c["ensemble"]["fixed_member_ids"])
        > 0,
        "Duplicate/empty ensemble IDs",
    )
    require(
        c["oof"]["n_folds"] >= 2 and c["oof"]["teacher_fixed_updates"] > 0,
        "Invalid OOF run length/folds",
    )
    require(
        c["oof"]["teacher_seeds"]
        and len(set(c["oof"]["teacher_seeds"])) == len(c["oof"]["teacher_seeds"]),
        "Invalid teacher seed list",
    )
    require(
        c["refiner"]["native_crop_size"] >= 8
        and c["refiner"]["batch_size"] > 0
        and c["refiner"]["optimizer_updates"] > 0
        and c["refiner"]["learning_rate"] > 0,
        "Invalid refiner recipe",
    )
    require(
        c["submission"]["filename_suffix"] == "_mask.png",
        "Only the documented _mask.png convention is supported",
    )
    require(
        c["oof"]["deployment_teacher_manifest"] is None,
        "External deployment teacher manifests are not supported; build matching deployment teachers with build-oof",
    )
    return c


def preflight(c, training=False):
    needed = [
        "dataset_root",
        "dataset_manifest",
        "checkpoint_manifest",
        "legacy_source",
        "encoder_weights",
    ]
    for k in needed:
        p = c["paths"][k]
        if p is None or not local_path(p).exists():
            raise FileNotFoundError(f"paths.{k}: {p}")
    if training and c["training"]["recipe"] == "finetune_existing":
        p = c["training"]["initialize_from"]
        if not local_path(p).is_file():
            raise FileNotFoundError(f"initialize_from: {p}")
    if (
        c["sampling"]["hard_cache"] is not None
        and not local_path(c["sampling"]["hard_cache"]).is_file()
    ):
        raise FileNotFoundError("hard_cache manifest does not exist")


def save_config(path, c):
    from .io import atomic

    atomic(path, lambda p: p.write_text(yaml.safe_dump(c, sort_keys=False)))


def seed_config(c, seed):
    c = deepcopy(c)
    c["run"]["seed"] = seed
    if c["training"]["recipe"] == "finetune_existing":
        rows = jsonl_read(local_path(c["paths"]["checkpoint_manifest"]))
        c["training"]["initialize_from"] = next(
            r["checkpoint"] for r in rows if r["seed"] == seed
        )
    return c


def inference_hash(c):
    return digest(
        {
            k: c[k]
            for k in ["classes", "model", "geometry", "relative_features", "inference"]
        }
    )
