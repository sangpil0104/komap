"""Strict train70 OOF teachers. Existing all-train seeds can never initialize folds."""

from copy import deepcopy
from pathlib import Path
import numpy as np
from .io import (
    ROOT,
    local_path,
    jsonl_read,
    jsonl_write,
    json_write,
    sha256,
    digest,
    ids_hash,
)
from .config import load_config, save_config, inference_hash
from .data import dataset_rows, verify_rows
from .jobs import run_jobs
from .cache import cache_model, ensemble, load_probability

LINEAGE_FIELDS = [
    "teacher_weight_training_ids",
    "teacher_ancestor_checkpoint_training_ids",
    "teacher_checkpoint_selection_ids",
    "teacher_bn_calibration_ids",
    "teacher_postprocessor_training_ids",
]


def validate_lineage(lineage, train_rows=None):
    image_id = lineage.get("predicted_image_id")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("Missing predicted image ID")
    if lineage.get("fold_id") is None:
        raise ValueError("Missing fold ID")
    required = [
        "teacher_config_hash",
        "teacher_checkpoint_hashes",
        "teacher_ensemble_definition",
    ]
    for key in required:
        if not lineage.get(key) or lineage[key] == "UNKNOWN":
            raise ValueError(f"Unknown lineage {key}")
    all_ids = set()
    for field in LINEAGE_FIELDS:
        ids = lineage.get(field)
        if not isinstance(ids, list) or any(
            not isinstance(i, str) or i == "UNKNOWN" for i in ids
        ):
            raise ValueError(f"Unknown lineage {field}")
        if image_id in ids:
            raise ValueError(f"OOF leakage: {image_id} in {field}")
        all_ids.update(ids)
    if not lineage["teacher_weight_training_ids"]:
        raise ValueError("Empty teacher training lineage")
    if train_rows is not None:
        known = {r["image_id"] for r in train_rows if r["split"] == "train"}
        if image_id not in known or not all_ids <= known:
            raise ValueError("OOF contains valid/test/unknown IDs")
    return True


def make_folds(cfg):
    rows = dataset_rows(cfg, "train")
    verify_rows(rows, "train")
    ids = sorted(r["image_id"] for r in rows)
    o = cfg["oof"]
    n = o["n_folds"]
    if n < 2 or n > len(ids):
        raise ValueError("Invalid fold count")
    rng = np.random.default_rng(o["fold_seed"])
    shuffled = np.array(ids)[rng.permutation(len(ids))]
    result = [
        {"image_id": str(image_id), "fold_id": i, "split": "train"}
        for i, part in enumerate(np.array_split(shuffled, n))
        for image_id in part
    ]
    path = local_path(o["fold_manifest"])
    if path.exists() and jsonl_read(path) != result:
        raise ValueError("Existing fold manifest differs")
    if not path.exists():
        jsonl_write(path, result)
    return result


def build_oof(cfg, execute=False):
    folds = make_folds(cfg)
    rows = dataset_rows(cfg, "train")
    valid = dataset_rows(cfg, "valid")
    root = local_path(cfg["oof"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    teacher = load_config(cfg["oof"]["teacher_config"])
    if (
        teacher["training"]["recipe"] != "scratch_comparable"
        or teacher["training"]["initialize_from"] is not None
    ):
        raise ValueError(
            "Strict fold teachers must start from audited ImageNet, never all-train checkpoints"
        )
    from .io import REPO

    expected_encoder = (
        REPO
        / "experiments/seonghyeon_seed10_v1/assets/efficientnet_b4_imagenet_encoder.pt"
    )
    if sha256(local_path(teacher["paths"]["encoder_weights"])) != sha256(
        expected_encoder
    ):
        raise ValueError("Unknown ancestor encoder weights")
    jobs = []
    configs = []
    for fold_id in list(range(cfg["oof"]["n_folds"])) + ["deployment"]:
        held = (
            [r["image_id"] for r in folds if r["fold_id"] == fold_id]
            if fold_id != "deployment"
            else []
        )
        training_ids = [r["image_id"] for r in rows if r["image_id"] not in held]
        for seed in cfg["oof"]["teacher_seeds"]:
            c = deepcopy(teacher)
            name = f"teacher_fold{fold_id}"
            c["run"].update(name=name, seed=seed, output_root=str(root / "teachers"))
            c["training"].update(
                train_ids=training_ids,
                valid_ids=[],
                validation_enabled=False,
                save_results=False,
                total_optimizer_updates=cfg["oof"]["teacher_fixed_updates"],
            )
            cp = root / "configs" / f"{name}_seed{seed}.yaml"
            cp.parent.mkdir(parents=True, exist_ok=True)
            if cp.exists() and load_config(cp) != c:
                raise ValueError("OOF teacher config changed")
            if not cp.exists():
                save_config(cp, c)
            run = root / "teachers" / f"{name}_seed{seed}"
            checkpoint = run / "last.pt"
            args = ["train", "--config", str(cp)]
            if run.exists() and not (run / "completed.json").exists():
                if checkpoint.exists():
                    args += ["--resume", str(checkpoint)]
                else:
                    raise RuntimeError(f"Incomplete teacher without checkpoint: {run}")
            if not (run / "completed.json").exists():
                jobs.append(
                    {
                        "name": f"{name}_seed{seed}",
                        "args": args,
                        "log": str(root / "logs" / f"{name}_seed{seed}.log"),
                    }
                )
            configs.append((fold_id, seed, c, checkpoint, held))
    json_write(
        root / "plan.json",
        {
            "folds": folds,
            "teacher_seed_count": len(cfg["oof"]["teacher_seeds"]),
            "jobs": jobs,
            "training_counts": {
                "fold": len(rows)
                * (cfg["oof"]["n_folds"] - 1)
                // cfg["oof"]["n_folds"],
                "deployment": len(rows),
            },
            "profile_hash": inference_hash(teacher),
            "initialization": "audited_ImageNet_only",
        },
    )
    run_jobs(jobs, cfg["execution"]["gpus"], execute)
    if not execute:
        return root / "plan.json"
    registry = []
    for fold_id, seed, c, checkpoint, held in configs:
        effective = deepcopy(c)
        effective["execution"]["dry_run"] = False
        registry.append(
            {
                "fold_id": fold_id,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "config_hash": digest(effective),
                "profile_hash": inference_hash(c),
            }
        )
    jsonl_write(root / "teacher_registry.jsonl", registry)
    # Cache each completed teacher, then same-seed-count probability averages.
    combined = []
    for fold_id in list(range(cfg["oof"]["n_folds"])) + ["deployment"]:
        manifests = []
        fold_configs = [x for x in configs if x[0] == fold_id]
        teacher_hashes = []
        for _, seed, c, checkpoint, held in fold_configs:
            import torch

            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            effective = deepcopy(c)
            effective["execution"]["dry_run"] = False
            if (
                state["config"] != effective
                or state["training_ids"] != c["training"]["train_ids"]
                or state["checkpoint_selection_ids"]
                or state["parent_checkpoint_sha256"] is not None
            ):
                raise ValueError("Teacher checkpoint lineage/config mismatch")
            images = (
                valid
                if fold_id == "deployment"
                else [r for r in rows if r["image_id"] in held]
            )
            h = sha256(checkpoint)
            teacher_hashes.append(h)

            def lineage(image):
                return {
                    "predicted_image_id": image["image_id"],
                    "fold_id": fold_id,
                    "teacher_weight_training_ids": c["training"]["train_ids"],
                    "teacher_ancestor_checkpoint_training_ids": [],
                    "teacher_checkpoint_selection_ids": [],
                    "teacher_bn_calibration_ids": [],
                    "teacher_postprocessor_training_ids": [],
                    "teacher_config_hash": inference_hash(c),
                    "teacher_checkpoint_hashes": [h],
                    "teacher_ensemble_definition": {
                        "seeds": [seed],
                        "aggregation": "native_probability_mean",
                        "dtype": "float32",
                    },
                }

            manifests.append(
                cache_model(
                    c,
                    checkpoint,
                    images,
                    root / "members" / f"fold{fold_id}_seed{seed}",
                    cfg["execution"]["device"],
                    c["training"]["train_ids"],
                    [],
                    lineage_factory=lineage if fold_id != "deployment" else None,
                )
            )
        out = root / ("valid" if fold_id == "deployment" else f"fold{fold_id}")
        merged = ensemble(cfg, manifests, out, with_curve=False)
        merged_rows = jsonl_read(merged)
        if fold_id != "deployment":
            for r in merged_rows:
                r["lineage"] = {
                    "predicted_image_id": r["image_id"],
                    "fold_id": fold_id,
                    "teacher_weight_training_ids": fold_configs[0][2]["training"][
                        "train_ids"
                    ],
                    "teacher_ancestor_checkpoint_training_ids": [],
                    "teacher_checkpoint_selection_ids": [],
                    "teacher_bn_calibration_ids": [],
                    "teacher_postprocessor_training_ids": [],
                    "teacher_config_hash": inference_hash(fold_configs[0][2]),
                    "teacher_checkpoint_hashes": teacher_hashes,
                    "teacher_ensemble_definition": {
                        "seeds": cfg["oof"]["teacher_seeds"],
                        "aggregation": "native_probability_mean",
                        "dtype": "float32",
                    },
                }
                validate_lineage(r["lineage"], rows)
                # One aggregate manifest with relative links, arrays remain in fold ownership.
                r["probability_path"] = str(
                    Path("..") / f"fold{fold_id}" / r["probability_path"]
                )
                combined.append(r)
            jsonl_write(root / "train/manifest.jsonl", combined)
    json_write(
        root / "teacher_distribution.json",
        {
            "profile_hash": inference_hash(teacher),
            "teacher_seeds": cfg["oof"]["teacher_seeds"],
            "fold_training_count": len(rows)
            * (cfg["oof"]["n_folds"] - 1)
            // cfg["oof"]["n_folds"],
            "deployment_training_count": len(rows),
            "same_ensemble_size": True,
        },
    )
    return validate_oof(cfg)


def validate_oof(cfg):
    rows = dataset_rows(cfg, "train")
    path = local_path(cfg["refiner"]["training_cache"])
    caches = jsonl_read(path)
    if len(caches) != len(rows) or {r["image_id"] for r in caches} != {
        r["image_id"] for r in rows
    }:
        raise ValueError("OOF must cover every train image exactly once")
    folds = {r["image_id"]: r["fold_id"] for r in make_folds(cfg)}
    images = {r["image_id"]: r for r in rows}
    profiles = set()
    definitions = set()
    overlap = []
    registry_path = local_path(cfg["oof"]["root"]) / "teacher_registry.jsonl"
    if not registry_path.is_file():
        raise ValueError("Strict OOF requires verifiable teacher checkpoint registry")
    registry = {r["checkpoint_sha256"]: r for r in jsonl_read(registry_path)}
    verified = {}
    for row in caches:
        load_probability(row, path.parent)
        line = row.get("lineage", {})
        validate_lineage(line, rows)
        if (
            line["predicted_image_id"] != row["image_id"]
            or line["fold_id"] != folds[row["image_id"]]
        ):
            raise ValueError("OOF ID/fold mismatch")
        expected = sorted(i for i in images if folds[i] != line["fold_id"])
        if sorted(line["teacher_weight_training_ids"]) != expected:
            raise ValueError("Teacher train IDs do not equal fold complement")
        for ckhash in line["teacher_checkpoint_hashes"]:
            if ckhash not in registry:
                raise ValueError("Unknown teacher checkpoint lineage")
            entry = registry[ckhash]
            if ckhash not in verified:
                import torch

                cp = local_path(entry["checkpoint"])
                if sha256(cp) != ckhash:
                    raise ValueError("Teacher checkpoint hash mismatch")
                state = torch.load(cp, map_location="cpu", weights_only=True)
                if (
                    state.get("kind") != "alphase_training_v1"
                    or state.get("parent_checkpoint_sha256") is not None
                ):
                    raise ValueError("Unknown/all-train warm-start teacher ancestry")
                teacher_cfg = state["config"]
                if (
                    digest(teacher_cfg) != entry["config_hash"]
                    or teacher_cfg["training"]["recipe"] != "scratch_comparable"
                    or teacher_cfg["training"]["initialize_from"] is not None
                ):
                    raise ValueError("Teacher initialization/config mismatch")
                if sha256(
                    local_path(teacher_cfg["paths"]["encoder_weights"])
                ) != sha256(local_path(cfg["paths"]["encoder_weights"])):
                    raise ValueError("Unknown ImageNet ancestor")
                if (
                    state["checkpoint_selection_ids"]
                    or state["step"] != cfg["oof"]["teacher_fixed_updates"]
                ):
                    raise ValueError(
                        "Teacher selection must be fixed-update without holdout use"
                    )
                verified[ckhash] = {
                    "training_ids": state["training_ids"],
                    "profile_hash": inference_hash(teacher_cfg),
                }
            if (
                entry["fold_id"] != line["fold_id"]
                or sorted(verified[ckhash]["training_ids"]) != expected
                or verified[ckhash]["profile_hash"] != line["teacher_config_hash"]
            ):
                raise ValueError(
                    "Teacher actual checkpoint does not match claimed OOF lineage"
                )
        if (
            row["source_split"] != "train"
            or row["image_sha256"] != images[row["image_id"]]["image_sha256"]
        ):
            raise ValueError("OOF image identity mismatch")
        profiles.add(line["teacher_config_hash"])
        definitions.add(digest(line["teacher_ensemble_definition"]))
        group = images[row["image_id"]]["group_hint"]
        if any(
            images[i]["group_hint"] == group
            for i in line["teacher_weight_training_ids"]
        ):
            overlap.append(row["image_id"])
    if len(profiles) != 1 or len(definitions) != 1:
        raise ValueError("OOF teachers have inconsistent profiles/ensemble definitions")
    result = {
        "status": "validated",
        "images": len(caches),
        "teacher_config_hash": next(iter(profiles)),
        "teacher_ensemble_hash": next(iter(definitions)),
        "unconfirmed_group_hint_overlap_ids": overlap,
        "grouping_policy": "image; A-prefix is not confirmed specimen identity",
    }
    json_write(local_path(cfg["oof"]["root"]) / "validation.json", result)
    return result
