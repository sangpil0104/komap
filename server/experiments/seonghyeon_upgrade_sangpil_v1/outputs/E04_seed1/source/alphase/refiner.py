"""Four-class native refiner, trained exclusively on validated strict OOF caches."""

from pathlib import Path
from copy import deepcopy
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from segmentation_models_pytorch.losses.lovasz import _lovasz_softmax
from .io import (
    ROOT,
    local_path,
    jsonl_read,
    jsonl_write,
    json_write,
    torch_save,
    sha256,
    digest,
    ids_hash,
)
from .config import save_config, preflight
from .oof import validate_oof
from .data import dataset_rows, verify_rows, sample_seed
from .cache import load_probability, write_probability
from .contracts import (
    rgb_to_ids,
    relative_features,
    d4_forward,
    confusion_matrix,
    summarize_images,
)
from .geometry import crop_resize, native_target
from .models import NativeRefiner
from .training import seed_all, rng_state, restore_rng, EMA


def load_refiner_data(cfg):
    validation = validate_oof(cfg)
    train = dataset_rows(cfg, "train")
    valid = dataset_rows(cfg, "valid")
    verify_rows(train, "train")
    verify_rows(valid, "valid")
    if cfg["refiner"]["validation_cache"] is None:
        raise ValueError("refiner.validation_cache is required")
    paths = [
        local_path(cfg["refiner"][key])
        for key in ["training_cache", "validation_cache"]
    ]
    results = []
    profiles = []
    for images, path in zip([train, valid], paths):
        rows = jsonl_read(path)
        index = {r["image_id"]: r for r in rows}
        if len(index) != len(rows) or set(index) != {r["image_id"] for r in images}:
            raise ValueError("Refiner cache coverage mismatch")
        data = []
        for image in images:
            row = index[image["image_id"]]
            if (
                row["image_sha256"] != image["image_sha256"]
                or row["source_split"] != image["split"]
            ):
                raise ValueError("Refiner cache identity mismatch")
            if image["split"] == "valid" and row["training_ids_hash"] != ids_hash(
                [r["image_id"] for r in train]
            ):
                raise ValueError("Deployment teacher train IDs differ")
            profiles.append(row["inference_config_hash"])
            data.append(
                (
                    image,
                    np.array(Image.open(image["image_path"]).convert("L")),
                    rgb_to_ids(np.array(Image.open(image["mask_path"]))),
                    load_probability(row, path.parent),
                )
            )
        results.append(data)
    if len(set(profiles)) != 1:
        raise ValueError(
            "OOF and deployment teacher preprocessing/ensemble distribution differ"
        )
    return *results, {
        "oof_validation": validation,
        "cache_hashes": [sha256(p) for p in paths],
        "teacher_inference_hash": profiles[0],
    }


def make_refiner_batch(data, cfg, step, device):
    r = cfg["refiner"]
    size = r["native_crop_size"]
    xs = []
    ps = []
    rs = []
    ys = []
    for j in range(r["batch_size"]):
        rng = np.random.default_rng(sample_seed(cfg["run"]["seed"], step, 0, j))
        _, gray, target, prob = data[int(rng.integers(len(data)))]
        h, w = gray.shape
        x = int(rng.integers(0, max(w - size, 0) + 1))
        y = int(rng.integers(0, max(h - size, 0) + 1))
        d = (x, y, size, size)
        image = torch.from_numpy(gray.copy()).float()[None, None].to(device) / 255
        probability = torch.from_numpy(prob).to(device)[None]
        rel = relative_features(image)[:, 1:]
        label, valid = native_target(
            torch.from_numpy(target.astype(np.int64)).to(device), d
        )
        k = int(rng.integers(4))
        flip = bool(rng.integers(2))
        xs.append(d4_forward(crop_resize(image, d, (size, size)), k, flip))
        ps.append(d4_forward(crop_resize(probability, d, (size, size)), k, flip))
        rs.append(d4_forward(crop_resize(rel, d, (size, size)), k, flip))
        ys.append(d4_forward(label, k, flip))
    return torch.cat(xs), torch.cat(ps), torch.cat(rs), torch.stack(ys)


@torch.inference_mode()
def evaluate_refiner(model, data, device, output=None, cfg=None):
    model.eval()
    cms = []
    rows = []
    for image, gray, target, prob in tqdm(
        data, desc="valid native refiner", leave=False
    ):
        x = torch.from_numpy(gray.copy()).float()[None, None].to(device) / 255
        p = torch.from_numpy(prob).to(device)[None]
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"
        ):
            logits = model(x, p)
        refined = logits.float().softmax(1)[0].cpu().numpy()
        cms.append(confusion_matrix(target, refined.argmax(0)))
        if output:
            from .cache import cache_metadata

            meta = cache_metadata(
                image,
                cfg,
                sha256(Path(output).parent / "selected_weights.pt"),
                [r["image_id"] for r in dataset_rows(cfg, "train")],
                [r["image_id"] for r in dataset_rows(cfg, "valid")],
            )
            rows.append(write_probability(output, meta, refined))
    if output:
        from .io import jsonl_write

        jsonl_write(Path(output) / "manifest.jsonl", rows)
    return summarize_images(cms)


def train_refiner(cfg, resume=None, device=None):
    cfg = deepcopy(cfg)
    cfg["execution"]["dry_run"] = False
    preflight(cfg)
    device = device or cfg["execution"]["device"]
    train, valid, lineage = load_refiner_data(cfg)
    r = cfg["refiner"]
    t = cfg["training"]
    run = (
        local_path(cfg["run"]["output_root"])
        / f'{cfg["run"]["name"]}_seed{cfg["run"]["seed"]}'
    )
    state = (
        torch.load(local_path(resume), map_location="cpu", weights_only=True)
        if resume
        else None
    )
    if state:
        if (
            state.get("kind") != "alphase_refiner_v1"
            or state["config"] != cfg
            or state["lineage"] != lineage
            or state["run_dir"] != str(run)
        ):
            raise ValueError("Refiner resume config/cache/lineage mismatch")
    else:
        run.mkdir(parents=True, exist_ok=False)
        save_config(run / "resolved.yaml", cfg)
        import shutil

        shutil.copytree(
            ROOT / "alphase",
            run / "source/alphase",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    seed_all(cfg["run"]["seed"])
    torch.set_num_threads(cfg["execution"]["cpu_threads"])
    model = NativeRefiner(r["probability_eps"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=r["learning_rate"], weight_decay=t["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: 0.5
        * (
            1
            + math.cos(
                math.pi * min(step, r["optimizer_updates"]) / r["optimizer_updates"]
            )
        ),
    )
    ema = EMA(model, t["ema_decay"]) if t["ema_enabled"] else None
    step = 0
    best = {"raw": -1.0, "ema": -1.0}
    elapsed = 0.0
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        step = state["step"]
        best = state["best"]
        elapsed = state["wall_seconds"]
        if ema:
            ema.model.load_state_dict(state["ema"])
            ema.updates = state["ema_updates"]
        restore_rng(state["rng"])
    else:
        baseline = summarize_images(
            [confusion_matrix(y, p.argmax(0)) for _, _, y, p in valid]
        )
        json_write(run / "teacher_baseline.json", baseline)
    started = time.monotonic()
    if state and (run / "history.jsonl").exists():
        jsonl_write(
            run / "history.jsonl",
            [row for row in jsonl_read(run / "history.jsonl") if row["step"] <= step],
        )

    def snapshot():
        return {
            "kind": "alphase_refiner_v1",
            "run_dir": str(run),
            "config": cfg,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "amp_scaler": None,
            "ema": ema.model.state_dict() if ema else None,
            "ema_updates": ema.updates if ema else 0,
            "step": step,
            "rng": rng_state(),
            "sampler": {"next_step": step, "policy": "indexed_seed_step_sample"},
            "lineage": lineage,
            "training_ids": [v[0]["image_id"] for v in train],
            "checkpoint_selection_ids": [v[0]["image_id"] for v in valid],
            "best": best.copy(),
            "wall_seconds": elapsed + time.monotonic() - started,
            "channel_names": r["channel_names"],
        }

    with tqdm(
        total=r["optimizer_updates"],
        initial=step,
        desc="strict OOF refiner",
        unit="update",
    ) as bar:
        while step < r["optimizer_updates"]:
            model.train()
            opt.zero_grad(set_to_none=True)
            x, p, rel, y = make_refiner_batch(train, cfg, step, device)
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=torch.device(device).type == "cuda",
            ):
                logits = model(x, p, rel)
            probability = logits.float().softmax(1)
            loss = F.cross_entropy(logits.float(), y, ignore_index=255) + cfg["loss"][
                "native_lovasz_weight"
            ] * _lovasz_softmax(
                probability, y, classes="present", per_image=True, ignore=255
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Refiner nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), t["grad_clip"], error_if_nonfinite=True
            )
            opt.step()
            sched.step()
            step += 1
            if ema:
                ema.update(model)
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "lr": sched.get_last_lr(),
                "samples_seen": step * r["batch_size"],
            }
            del x, p, rel, y, logits, probability, loss
            if (
                step % t["validation_every_updates"] == 0
                or step == r["optimizer_updates"]
            ):
                for name, net in [("raw", model)] + (
                    [("ema", ema.model)] if ema else []
                ):
                    score = evaluate_refiner(net, valid, device)["val_image_miou"]
                    record["val_image_miou_" + name] = score
                    if score > best[name]:
                        best[name] = score
                        torch_save(
                            run
                            / (
                                "best_image_miou.pt"
                                if name == "raw"
                                else "best_ema_image_miou.pt"
                            ),
                            snapshot(),
                        )
            import json

            with (run / "history.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            if (
                step % t["checkpoint_every_updates"] == 0
                or step == r["optimizer_updates"]
            ):
                torch_save(run / "last.pt", snapshot())
            bar.update()
            bar.set_postfix(
                loss=f'{record["loss"]:.4f}', miou=f"{max(best.values())*100:.3f}%"
            )
    selected = run / (
        "best_ema_image_miou.pt" if best["ema"] > best["raw"] else "best_image_miou.pt"
    )
    state = torch.load(selected, map_location="cpu", weights_only=True)
    weights = state["ema"] if best["ema"] > best["raw"] else state["model"]
    model.load_state_dict(weights)
    torch_save(
        run / "selected_weights.pt",
        {"model": weights, "lineage": lineage, "channel_names": r["channel_names"]},
    )
    json_write(
        run / "completed.json",
        {"step": step, "selected_checkpoint": str(selected), "best": best},
    )
    if t["save_results"]:
        from .evaluation import report_predictions, reserve_results

        evaluate_refiner(model, valid, device, run / "evaluation", cfg)
        result = reserve_results()
        summary = report_predictions(
            run / "evaluation/manifest.jsonl", cfg["paths"]["dataset_manifest"], result
        )
        json_write(
            run / "result.json",
            {"result_dir": str(result), "val_image_miou": summary["val_image_miou"]},
        )
    return run
