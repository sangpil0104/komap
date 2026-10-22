"""CLI entry points. Training and long sweeps require explicit --execute."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from .io import ROOT, local_path, jsonl_read, jsonl_write, json_write, sha256, digest
from .config import load_config, save_config, preflight, seed_config, inference_hash


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=[
            "audit",
            "validate-config",
            "reproduce",
            "smoke",
            "cache-probs",
            "cache-member",
            "ensemble",
            "diagnose",
            "train",
            "launch",
            "evaluate",
            "build-oof",
            "validate-oof",
            "train-refiner",
            "predict",
            "validate-submission",
            "mine-hard",
        ],
    )
    p.add_argument("--config", default="configs/resolved_legacy.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--execute", action="store_true")
    p.add_argument("--device")
    p.add_argument("--seed", type=int)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/experiments/E03a.yaml",
            "configs/experiments/E03b.yaml",
            "configs/experiments/E04.yaml",
            "configs/experiments/E05.yaml",
        ],
    )
    p.add_argument("--gpus", default="0,1,2,3,4,5")
    p.add_argument("--split", choices=["train", "valid", "test"])
    p.add_argument("--members", default="all")
    p.add_argument("--member")
    p.add_argument("--image-id")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--full-size", action="store_true")
    p.add_argument("--real-train-patches", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--run-dir")
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--prediction-manifest")
    p.add_argument(
        "--prediction-manifests",
        nargs="+",
        help="Explicit equal-weight member cache manifests for a heterogeneous ensemble",
    )
    p.add_argument("--output")
    return p


def reproduce(cfg, image_id=None):
    from .models import load_model
    from .legacy import native_probs, original_module
    from .contracts import (
        rgb_to_ids,
        confusion_matrix,
        metrics_from_cm,
        summarize_images,
    )
    from .data import dataset_rows

    device = cfg["execution"]["device"]
    model = load_model(cfg, cfg["paths"]["legacy_checkpoint"], device).eval()
    original = original_module(cfg)
    rows = dataset_rows(cfg, "valid")
    if image_id:
        rows = [r for r in rows if r["image_id"] == image_id]
        if not rows:
            raise ValueError(f"Unknown image ID: {image_id}")
    results = []
    cms = []
    for row in tqdm(rows, desc="legacy regression"):
        gray = np.array(Image.open(row["image_path"]).convert("L"))
        prob = native_probs(model, gray, cfg, device)
        ref = original.predict_probs(
            [
                original.make_member(
                    model.base, "imagenet", target_width=cfg["geometry"]["target_width"]
                )
            ],
            gray,
            tta=cfg["inference"]["d4"],
            device=device,
        )
        target = rgb_to_ids(np.array(Image.open(row["mask_path"])))
        cm = confusion_matrix(target, prob.argmax(0).cpu().numpy())
        cms.append(cm)
        results.append(
            {
                "image_id": row["image_id"],
                "max_abs_probability_difference": float((ref - prob).abs().max()),
                "argmax_mismatch_pixels": int((ref.argmax(0) != prob.argmax(0)).sum()),
                **metrics_from_cm(cm),
            }
        )
    report = {**summarize_images(cms), "per_image": results}
    json_write(ROOT / "reports" / f'reproduction_{image_id or "valid20"}.json', report)
    if any(
        r["max_abs_probability_difference"] != 0 or r["argmax_mismatch_pixels"] != 0
        for r in results
    ):
        raise RuntimeError("Legacy regression mismatch; see report")
    return report


def cache_member(cfg, member, split, device):
    from .cache import cache_model
    from .data import dataset_rows

    rows = jsonl_read(local_path(cfg["paths"]["checkpoint_manifest"]))
    selected = [r for r in rows if r["member_id"] == member]
    if len(selected) != 1:
        raise ValueError(f"Unknown member: {member}")
    r = selected[0]
    if sha256(r["checkpoint"]) != r["checkpoint_sha256"]:
        raise ValueError("Historical checkpoint hash changed")
    root = local_path(cfg["ensemble"]["cache_root"]) / member
    if split != "valid":
        root = root / split
    return cache_model(
        cfg,
        r["checkpoint"],
        dataset_rows(cfg, split),
        root,
        device,
        r["training_ids"],
        r["checkpoint_selection_ids"],
    )


def launch(args, cfg):
    from .jobs import run_jobs

    seeds = [int(x) for x in args.seeds.split(",")]
    jobs = []
    gpus = [int(x) for x in args.gpus.split(",")]
    folder = ROOT / "outputs/launch_configs"
    folder.mkdir(parents=True, exist_ok=True)
    for path in args.configs:
        original = load_config(path)
        for seed in seeds:
            c = seed_config(original, seed)
            preflight(c, training=True)
            name = f'{c["run"]["name"]}_seed{seed}'
            p = folder / (name + ".yaml")
            if p.exists() and load_config(p) != c:
                raise ValueError(f"Launch config collision: {p}")
            if not p.exists():
                save_config(p, c)
            run = local_path(c["run"]["output_root"]) / name
            command = ["train", "--config", str(p)]
            if run.exists():
                if (run / "completed.json").exists():
                    continue
                if (run / "last.pt").exists():
                    command += ["--resume", str(run / "last.pt")]
                else:
                    raise RuntimeError(
                        f"Existing incomplete run without checkpoint: {run}"
                    )
            jobs.append(
                {
                    "name": name,
                    "args": command,
                    "log": str(ROOT / "outputs/logs" / (name + ".log")),
                }
            )
    json_write(ROOT / "outputs/launch_plan.json", jobs)
    return run_jobs(jobs, gpus, args.execute)


def predict(cfg, split, execute):
    from .data import dataset_rows, verify_rows
    from .models import load_model, NativeRefiner
    from .inference import predict_probs
    from .contracts import ids_to_rgb

    checkpoint = cfg["submission"]["checkpoint"]
    if checkpoint is None or not local_path(checkpoint).is_file():
        raise FileNotFoundError("submission.checkpoint is required")
    rows = dataset_rows(cfg, split)
    verify_rows(rows, split)
    out = local_path(cfg["submission"]["output"])
    if not execute:
        return {
            "dry_run": True,
            "checkpoint": checkpoint,
            "images": len(rows),
            "output": str(out),
        }
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    model = load_model(cfg, checkpoint, cfg["execution"]["device"]).eval()
    refiner = None
    if cfg["refiner"]["checkpoint"]:
        state = torch.load(
            local_path(cfg["refiner"]["checkpoint"]),
            map_location="cpu",
            weights_only=True,
        )
        # A single base teacher must match the OOF ensemble's profile exactly.
        expected = digest([inference_hash(cfg)])
        if state["lineage"]["teacher_inference_hash"] != expected:
            raise ValueError("Refiner teacher distribution mismatch")
        if state["channel_names"] != cfg["refiner"]["channel_names"]:
            raise ValueError("Refiner input metadata mismatch")
        refiner = (
            NativeRefiner(cfg["refiner"]["probability_eps"])
            .to(cfg["execution"]["device"])
            .eval()
        )
        refiner.load_state_dict(state["model"], strict=True)
    manifests = []
    for row in tqdm(rows, desc=f"predict {split}"):
        gray = np.array(Image.open(row["image_path"]).convert("L"))
        prob = predict_probs(model, gray, cfg, cfg["execution"]["device"])
        if refiner:
            with torch.inference_mode():
                x = (
                    torch.from_numpy(gray.copy()).float()[None, None].to(prob.device)
                    / 255
                )
                prob = refiner(x, prob[None]).softmax(1)[0]
        path = out / (row["image_id"] + cfg["submission"]["filename_suffix"])
        Image.fromarray(ids_to_rgb(prob.argmax(0).cpu().numpy())).save(path)
        manifests.append(
            {
                "image_id": row["image_id"],
                "source_split": split,
                "native_hw": row["native_hw"],
                "image_sha256": row["image_sha256"],
                "prediction_path": path.name,
                "prediction_sha256": sha256(path),
                "checkpoint_sha256": sha256(local_path(checkpoint)),
                "inference_config_hash": inference_hash(cfg),
            }
        )
    jsonl_write(out / "manifest.jsonl", manifests)
    return validate_submission(cfg, split)


def validate_submission(cfg, split="test"):
    from .contracts import rgb_to_ids
    from .data import dataset_rows

    out = local_path(cfg["submission"]["output"])
    expected = {
        r["image_id"] + cfg["submission"]["filename_suffix"]: r
        for r in dataset_rows(cfg, split)
    }
    files = {p.name: p for p in out.glob("*.png")}
    if set(files) != set(expected):
        raise ValueError("Submission filename/count mismatch")
    for name, path in files.items():
        im = Image.open(path)
        if (
            im.format != "PNG"
            or list(np.array(im).shape[:2]) != expected[name]["native_hw"]
        ):
            raise ValueError("Submission format/shape mismatch")
        rgb_to_ids(np.array(im))
    result = {
        "validated_images": len(files),
        "native_size": True,
        "palette": True,
        "archive_rules": "not supplied; PNG validation only",
    }
    json_write(out / "validation.json", result)
    return result


def main():
    args = parser().parse_args()
    cfg = load_config(args.config, args.set)
    if args.split is None:
        args.split = (
            "test" if args.command in ["predict", "validate-submission"] else "valid"
        )
    if args.seed is not None:
        cfg = seed_config(cfg, args.seed)
    if args.device:
        cfg["execution"]["device"] = args.device
    selected_device = cfg["execution"]["device"]
    if selected_device.startswith("cuda"):
        logical = int(selected_device.split(":")[-1]) if ":" in selected_device else 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical = int(visible.split(",")[logical]) if visible else logical
        if physical not in range(6):
            raise ValueError("Only physical GPUs0–5 are authorized")
    torch.set_num_threads(cfg["execution"]["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cmd = args.command
    result = None
    if cmd == "audit":
        import subprocess

        subprocess.run(
            [sys.executable, str(ROOT / "scripts/audit_repository.py")], check=True
        )
    elif cmd == "validate-config":
        preflight(cfg, training=cfg["training"]["recipe"] == "finetune_existing")
        result = {
            "valid": True,
            "config_hash": digest(cfg),
            "hard_cache_policy": (
                "random_same_image_until_cache_is_supplied"
                if cfg["sampling"]["hard_cache"] is None
                else "validated_at_training_preflight"
            ),
            "unused_legacy_warmup": cfg["training"]["recipe"] == "finetune_existing",
            "resolved": cfg,
        }
    elif cmd == "reproduce":
        preflight(cfg)
        result = reproduce(cfg, args.image_id)
    elif cmd == "smoke":
        from .smoke import smoke

        result = smoke(
            cfg,
            args.steps,
            cfg["execution"]["device"],
            args.full_size,
            args.real_train_patches,
            args.execute,
        )
    elif cmd == "launch":
        result = launch(args, cfg)
    elif cmd in ["cache-probs", "cache-member"]:
        preflight(cfg)
        if cmd == "cache-member":
            if not args.execute:
                result = {"dry_run": True, "member": args.member, "split": args.split}
            else:
                result = str(
                    cache_member(
                        cfg, args.member, args.split, cfg["execution"]["device"]
                    )
                )
        else:
            from .jobs import run_jobs

            members = (
                [
                    r["member_id"]
                    for r in jsonl_read(local_path(cfg["paths"]["checkpoint_manifest"]))
                ]
                if args.members == "all"
                else args.members.split(",")
            )
            valid_ids = {
                r["member_id"]
                for r in jsonl_read(local_path(cfg["paths"]["checkpoint_manifest"]))
            }
            if not set(members) <= valid_ids:
                raise ValueError("Unknown ensemble members")
            resolved = ROOT / "outputs/cache_configs" / f"{digest(cfg)}.yaml"
            save_config(resolved, cfg)
            jobs = [
                {
                    "name": m,
                    "args": [
                        "cache-member",
                        "--config",
                        str(resolved),
                        "--member",
                        m,
                        "--split",
                        args.split,
                    ],
                    "log": str(ROOT / "outputs/logs" / f"cache_{m}_{args.split}.log"),
                }
                for m in members
            ]
            result = run_jobs(jobs, cfg["execution"]["gpus"], args.execute)
    elif cmd == "ensemble":
        from .cache import ensemble
        from .evaluation import report_predictions, write_csv

        if not args.execute:
            result = {
                "dry_run": True,
                "members": cfg["ensemble"]["fixed_member_ids"],
                "output": cfg["ensemble"]["output"],
            }
        else:
            member_paths = (
                [local_path(p) for p in args.prediction_manifests]
                if args.prediction_manifests
                else None
            )
            manifest = ensemble(cfg, member_paths)
            result = report_predictions(manifest, cfg["paths"]["dataset_manifest"])
            curves = []
            for p in sorted(manifest.parent.glob("prefix_*/manifest.jsonl")):
                s = report_predictions(
                    p, cfg["paths"]["dataset_manifest"], panels=False
                )
                curves.append(
                    {
                        "K": int(p.parent.name.split("_")[-1]),
                        "val_image_miou": s["val_image_miou"],
                        "selection": "predeclared_seed_prefix",
                    }
                )
            write_csv(manifest.parent / "k_curve.csv", curves)
    elif cmd == "diagnose":
        from .evaluation import report_predictions

        if not args.prediction_manifest:
            raise ValueError("--prediction-manifest required")
        result = report_predictions(
            args.prediction_manifest, cfg["paths"]["dataset_manifest"], args.output
        )
    elif cmd == "train":
        from .training import train

        preflight(cfg, training=True)
        if not args.execute:
            result = {
                "dry_run": True,
                "run": cfg["run"],
                "updates": cfg["training"]["total_optimizer_updates"],
                "parent": cfg["training"]["initialize_from"],
                "device": cfg["execution"]["device"],
            }
        else:
            result = str(train(cfg, args.resume))
    elif cmd == "evaluate":
        from .training import export_evaluation

        if not args.run_dir:
            raise ValueError("--run-dir required")
        run = local_path(args.run_dir)
        cfg = load_config(run / "resolved.yaml")
        key = {
            "best": "best_image_miou.pt",
            "last": "last.pt",
            "best_ema": "best_ema_image_miou.pt",
        }.get(args.checkpoint, args.checkpoint)
        result = export_evaluation(
            cfg,
            run,
            run / key,
            args.device or cfg["execution"]["device"],
            use_ema=key.startswith("best_ema"),
        )
    elif cmd == "build-oof":
        from .oof import build_oof

        result = build_oof(cfg, args.execute)
    elif cmd == "validate-oof":
        from .oof import validate_oof

        result = validate_oof(cfg)
    elif cmd == "train-refiner":
        from .refiner import train_refiner, load_refiner_data

        if not args.execute:
            result = {
                "dry_run": True,
                "requires": {
                    "strict_oof": cfg["refiner"]["training_cache"],
                    "deployment_valid": cfg["refiner"]["validation_cache"],
                },
                "updates": cfg["refiner"]["optimizer_updates"],
            }
        else:
            result = str(train_refiner(cfg, args.resume))
    elif cmd == "mine-hard":
        from .cache import cache_model
        from .data import dataset_rows

        if not args.run_dir:
            raise ValueError("--run-dir required")
        run = local_path(args.run_dir)
        c = load_config(run / "resolved.yaml")
        cp = run / "last.pt"
        if not cp.is_file():
            raise FileNotFoundError(cp)
        if not args.execute:
            result = {
                "dry_run": True,
                "policy": "in_sample_train_mining; sampler only, NEVER OOF",
                "checkpoint": str(cp),
            }
        else:
            state = torch.load(cp, map_location="cpu", weights_only=True)
            result = str(
                cache_model(
                    c,
                    cp,
                    dataset_rows(c, "train"),
                    run / "hard_train_cache",
                    args.device or c["execution"]["device"],
                    state["training_ids"],
                    state["checkpoint_selection_ids"],
                )
            )
    elif cmd == "predict":
        preflight(cfg)
        result = predict(cfg, args.split, args.execute)
    elif cmd == "validate-submission":
        result = validate_submission(cfg, args.split)
    if result is not None:
        print(json.dumps(result, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
