from __future__ import annotations

import gc
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from alphase.data.dataset import load_image, load_target
from alphase.data.palette import PALETTE, save_mask
from alphase.evaluation.metrics import image_miou
from alphase.inference.predictor import predict_probabilities
from alphase.models.factory import build_model
from alphase.utils import code_hash, json_hash, sha256, write_json
from lab.workspace import (BASELINE, SOURCE, OUTPUTS, atomic_npy, checkpoint_for, output_path,
                           read_json, require_manifest, safe_name)

CLASS_NAMES = ["primary_si", "al3ni", "eutectic_si", "al"]


def select_images(state, split):
    table = pd.read_csv(BASELINE/"data/manifest.csv").fillna("")
    if split == "valid":
        ids = state["validation_ids"]
        if not ids:
            raise ValueError("This checkpoint has no held-out validation set (for example all90)")
        rows = table[table.image_id.isin(ids)]
        if set(rows.image_id) != set(ids):
            raise ValueError("Validation IDs are missing from the frozen manifest")
        if set(rows.image_id) & set(state["train_ids"]):
            raise ValueError("Validation images overlap training images")
        if any(not p for p in rows.mask_path):
            raise ValueError("Validation labels are missing")
    elif split == "test":
        rows = table[table.source_split == "test"]
        if set(rows.image_id) & set(state["train_ids"]):
            raise ValueError("Test IDs appear in the training set")
    else:
        raise ValueError("Prediction split must be valid or test")
    return rows.sort_values("image_id").to_dict("records")


def validate_probability(array, height, width):
    if array.shape != (4, height, width) or array.dtype != np.float32:
        raise ValueError(f"Expected float32 [4,{height},{width}], got {array.dtype} {array.shape}")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Invalid prediction probability")
    if not np.allclose(array.sum(0), 1., atol=2e-5):
        raise ValueError("Per-pixel probabilities must sum to one")
    return array


def weighted_average(arrays, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if len(arrays) != len(weights) or not len(arrays):
        raise ValueError("A positive weight is required for every member")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Ensemble weights must be finite and positive")
    # Scale before summing to avoid overflow for otherwise finite weights.
    weights = weights/weights.max()
    weights = weights/weights.sum()
    result = np.zeros_like(arrays[0], dtype=np.float32)
    for array, weight in zip(arrays, weights):
        if array.shape != result.shape:
            raise ValueError("Ensemble members have different image dimensions")
        validate_probability(array, *array.shape[-2:])
        result += array*float(weight)
    return result/result.sum(0, keepdims=True)


def cache_probabilities(checkpoint, split="valid", d4=False, scales=(1.,), selected="selected"):
    """Read one model at a time; cache probabilities with data/model/source hashes."""
    if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("Positive finite inference scales are required")
    dataset_hash = require_manifest()
    checkpoint = Path(checkpoint)
    checkpoint_hash = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["palette"] != PALETTE.tolist():
        raise ValueError("Checkpoint class palette/order differs from this experiment")
    if state["dataset_hash"] != dataset_hash:
        raise ValueError("Checkpoint was trained with a different dataset")
    rows = select_images(state, split)
    cfg = deepcopy(state["resolved_config"])
    cfg["inference"].update(d4=bool(d4), scales=list(scales))
    if cfg["model"]["name"] == "alphase_hybrid_v1":
        cfg["model"]["dino_repo"] = str(SOURCE/"third_party/dinov3")
    choice = state["selected_weights"] if selected == "selected" else selected
    if choice not in ("raw", "ema"):
        raise ValueError("Weight selection must be raw or ema")
    signature = dict(checkpoint_sha256=checkpoint_hash, selected_weights=choice,
                     dataset_hash=dataset_hash, class_names=CLASS_NAMES, palette=PALETTE.tolist(),
                     inference=cfg["inference"], model_config=cfg["model"], precision=cfg["train"]["precision"],
                     baseline_source_hash=code_hash(), torch_version=str(torch.__version__), split=split,
                     train_ids=sorted(state["train_ids"]), validation_ids=sorted(state["validation_ids"]),
                     images=[dict(image_id=r["image_id"], width=r["width"], height=r["height"],
                                  image_sha256=r["image_sha256"]) for r in rows])
    cache = output_path("cache", json_hash(signature))
    manifest = cache/"provenance.json"
    if manifest.exists():
        old = read_json(manifest)
        if old["signature"] != signature:
            raise ValueError("Cache provenance mismatch")
        for record in old["probabilities"]:
            path = cache/record["file"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise ValueError(f"Cached prediction was modified or is missing: {path}")
        return cache, rows, old
    cache.mkdir(parents=True, exist_ok=True)
    # A provenance file is written only once every image has completed successfully.
    torch.set_num_threads(cfg["train"].get("cpu_threads", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, load_pretrained=False)
    model.load_state_dict(state["ema_model"] if choice == "ema" else state["model"])
    model.to(device).eval()
    del state
    records = []
    for row in tqdm(rows, desc=f"{checkpoint.parent.name} {split} D4={d4}"):
        gray = load_image(row["image_path"]).astype(np.float32)/255
        probability = predict_probabilities(model, gray, cfg, device).astype(np.float32)
        validate_probability(probability, row["height"], row["width"])
        path = cache/f"{row['image_id']}.npy"
        atomic_npy(path, probability)
        records.append(dict(image_id=row["image_id"], file=path.name, sha256=sha256(path)))
    metadata = dict(signature=signature, probabilities=records, checkpoint=str(checkpoint))
    write_json(manifest, metadata)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache, rows, metadata


def write_results(rows, probabilities, folder, provenance):
    """The iterator yields original-resolution probabilities in the manifest order."""
    folder.mkdir(parents=True, exist_ok=True)
    is_validation = bool(rows and rows[0]["mask_path"])
    records, files = [], []
    for index, (row, probability) in enumerate(zip(rows, probabilities, strict=True)):
        validate_probability(probability, row["height"], row["width"])
        labels = probability.argmax(0)
        image_id = row["image_id"]
        output = output_path(*folder.relative_to(OUTPUTS).parts, "predictions", f"{image_id}_mask.png")
        save_mask(labels, output)
        files.append(dict(image_id=image_id, file=str(output.relative_to(folder)), sha256=sha256(output),
                          width=row["width"], height=row["height"]))
        if is_validation:
            target = load_target(row["mask_path"])
            record = dict(image_id=image_id, H=row["height"], W=row["width"])
            for policy in ("exclude", "one", "zero"):
                score, iou, cm = image_miou(target, labels, policy)
                record[f"mIoU_{policy}"] = 100*score
                if policy == "exclude":
                    record.update({f"IoU_{c}": float(iou[c]*100) if np.isfinite(iou[c]) else None for c in range(4)})
                    record["confusion_matrix"] = json.dumps(cm.tolist())
            record["GT_phase_fractions"] = json.dumps((np.bincount(target.ravel(), minlength=4)/target.size).tolist())
            record["predicted_phase_fractions"] = json.dumps((np.bincount(labels.ravel(), minlength=4)/labels.size).tolist())
            records.append(record)
            if index < 3:
                gray = load_image(row["image_path"])
                rgb = np.repeat(gray[..., None], 3, -1)
                preview = Image.fromarray(np.concatenate([rgb, PALETTE[target], PALETTE[labels]], 1))
                preview.thumbnail((1440, 480))
                preview.save(folder/f"preview_{index+1}.png")
    summary = dict(num_images=len(rows), evaluation_kind="validation" if is_validation else "unlabeled_test")
    if is_validation:
        frame = pd.DataFrame(records)
        frame.to_csv(folder/"per_image_metrics.csv", index=False)
        summary.update({f"mIoU_{policy}": float(frame[f"mIoU_{policy}"].mean()) for policy in ("exclude", "one", "zero")})
        summary["per_class_iou"] = [None if frame[f"IoU_{c}"].isna().all() else float(frame[f"IoU_{c}"].mean()) for c in range(4)]
    else:
        from alphase.cli.check_submission import check
        check(Path(rows[0]["image_path"]).parent, folder/"predictions", expected_count=len(rows))
    write_json(folder/"metrics_summary.json", summary)
    write_json(folder/"inference_manifest.json", dict(class_names=CLASS_NAMES, palette=PALETTE.tolist(), images=files))
    # This marker is written last. A partial directory can be safely reconstructed.
    write_json(folder/"provenance.json", provenance)
    return summary


def inference_tag(d4, scales):
    return ("d4" if d4 else "plain")+"_"+"-".join(f"{s:g}".replace(".", "p") for s in scales)


def complete_result(folder, provenance):
    marker = folder/"provenance.json"
    if not marker.exists():
        return None
    old = read_json(marker)
    if old != provenance:
        raise FileExistsError(f"Existing result uses different settings; choose another --tag: {folder}")
    exported = read_json(folder/"inference_manifest.json")
    for entry in exported["images"]:
        path = folder/entry["file"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Existing prediction has changed: {path}")
    return read_json(folder/"metrics_summary.json")


def evaluate_one(name, model_split="official", split="valid", d4=False, scales=(1.,), smoke=False, tag=None):
    checkpoint = checkpoint_for(name, model_split, smoke)
    cache, rows, metadata = cache_probabilities(checkpoint, split, d4, scales)
    tag = safe_name(tag or inference_tag(d4, scales))
    folder = output_path("smoke_evaluations" if smoke else "evaluations", safe_name(name), model_split, split, tag)
    provenance = dict(kind="single", experiment=name, model_split=model_split, split=split,
                      cache=str(cache.relative_to(OUTPUTS)), member=metadata["signature"], smoke_test=smoke)
    existing = complete_result(folder, provenance)
    if existing is not None:
        return folder, existing
    probabilities = (np.load(cache/f"{r['image_id']}.npy", allow_pickle=False) for r in rows)
    return folder, write_results(rows, probabilities, folder, provenance)


def ensemble(config, model_split="official", split="valid", smoke=False):
    import yaml
    config = yaml.safe_load(Path(config).read_text())
    name = safe_name(config["name"])
    members = config["members"]
    if not members:
        raise ValueError("Ensemble has no members")
    names = [m["experiment"] for m in members]
    if len(set(names)) != len(names):
        raise ValueError("Each experiment may appear only once in an ensemble")
    weights = [float(m.get("weight", 1)) for m in members]
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Ensemble weights must be finite and positive")
    # Check all files before spending GPU time generating any predictions.
    checkpoints = [checkpoint_for(n, model_split, smoke) for n in names]
    if len(set(sha256(p) for p in checkpoints)) != len(checkpoints):
        raise ValueError("Duplicate checkpoint bytes: an ensemble needs distinct models")
    caches, signatures, rows = [], [], None
    for checkpoint in checkpoints:
        cache, member_rows, metadata = cache_probabilities(checkpoint, split, config.get("d4", False), config.get("scales", [1.]))
        signature = metadata["signature"]
        if signatures:
            for field in ("dataset_hash", "class_names", "palette", "images", "train_ids", "validation_ids", "inference"):
                if signature[field] != signatures[0][field]:
                    raise ValueError(f"Ensemble provenance mismatch: {field}. Use the same split and inference settings.")
        rows = member_rows
        caches.append(cache)
        signatures.append(signature)
    folder = output_path("smoke_ensembles" if smoke else "ensembles", name, model_split, split)
    provenance = dict(kind="probability_ensemble", name=name, model_split=model_split, split=split,
                      members=signatures, weights=((np.asarray(weights)/max(weights))/(np.asarray(weights)/max(weights)).sum()).tolist(), smoke_test=smoke)
    existing = complete_result(folder, provenance)
    if existing is not None:
        return folder, existing
    def predictions():
        for row in rows:
            arrays = [np.load(cache/f"{row['image_id']}.npy", allow_pickle=False) for cache in caches]
            yield weighted_average(arrays, weights)
    return folder, write_results(rows, predictions(), folder, provenance)
