"""Ported from experiments/hundred_v1/hundredlab/evaluation.py (import paths only)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from alphase.data.audit import verify_manifest
from alphase.data.dataset import load_image
from alphase.data.palette import save_mask
from alphase.data.splits import select_rows
from alphase.evaluation.evaluate import evaluate
from alphase.inference.predictor import predict_probabilities
from alphase.utils import sha256, write_json
from evolvelab.models import build_model
from evolvelab.workspace import PROJECT, output_path, read_json, run_dir, source_hash


def load_run(name, smoke=False, d4=False, checkpoint_kind="best_trained"):
    checkpoint = run_dir(name, smoke) / (checkpoint_kind + ".pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["resolved_config"]
    if cfg["hundred"]["source_sha256"] != source_hash():
        raise ValueError("evolve_v1 source changed since this model was trained")
    if verify_manifest(cfg["data"]["manifest"]) != state["dataset_hash"]:
        raise ValueError("Dataset changed since training")
    train, valid = select_rows(cfg, "official")
    if set(state["train_ids"]) != {r["image_id"] for r in train} or set(state["validation_ids"]) != {r["image_id"] for r in valid}:
        raise ValueError("Checkpoint split mismatch")
    if set(state["train_ids"]) & set(state["validation_ids"]):
        raise ValueError("Training and validation IDs overlap")
    cfg["inference"]["d4"] = d4
    torch.set_num_threads(cfg["train"].get("cpu_threads", 4))
    model = build_model(cfg, load_pretrained=False)
    selected = state["selected_weights"]
    model.load_state_dict(state["ema_model"] if selected == "ema" else state["model"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    provenance = dict(experiment=name, smoke_test=smoke, checkpoint_kind=checkpoint_kind, optimizer_step=state["optimizer_step"], checkpoint_sha256=sha256(checkpoint),
                      source_sha256=source_hash(), selected_weights=selected, dataset_hash=state["dataset_hash"],
                      train_ids=state["train_ids"], validation_ids=state["validation_ids"],
                      inference=cfg["inference"], architecture=cfg["model"]["name"], torch_version=str(torch.__version__))
    return model.to(device).eval(), cfg, device, valid, provenance


def begin_output(folder, expected):
    marker, request = folder / "provenance.json", folder / "request.json"
    if marker.is_file():
        saved = read_json(marker)
        if saved["request"] != expected:
            raise ValueError(f"Completed result belongs to different settings: {folder}")
        for filename, digest in saved["artifacts"].items():
            if sha256(folder / filename) != digest:
                raise ValueError(f"Completed output changed: {filename}")
        return False
    if folder.exists() and any(folder.iterdir()):
        if not request.is_file() or read_json(request) != expected:
            raise ValueError(f"Unknown incomplete output: {folder}")
        backup = folder.with_name(folder.name + f".interrupted-{time.time_ns()}")
        folder.rename(backup)
        print(f"Interrupted output preserved: {backup}", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(request, expected)
    return True


def finish_output(folder, expected):
    artifacts = {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob("*"))
                 if p.is_file() and p.name not in ("request.json", "provenance.json")}
    write_json(folder / "provenance.json", dict(request=expected, artifacts=artifacts))


def metric_summary(frame):
    cm = sum(np.array(json.loads(v), dtype=np.int64) for v in frame.confusion_matrix)
    diagonal = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - diagonal
    global_iou = np.divide(diagonal, union, out=np.full(4, np.nan), where=union > 0) * 100
    return dict(global_mIoU=float(np.nanmean(global_iou)), global_class_iou=global_iou.tolist(),
                image_mIoU=float(frame.mIoU_exclude.mean()), num_images=len(frame))


def evaluate_run(name, smoke=False, d4=False):
    model, cfg, device, rows, provenance = load_run(name, smoke, d4)
    out = output_path("smoke_evaluations" if smoke else "evaluations", name, "d4" if d4 else "plain")
    if not begin_output(out, provenance):
        print(f"평가 완료 결과 재사용: {name} {'D4' if d4 else 'plain'}", flush=True)
        return read_json(out / "metrics_summary.json")
    summary = evaluate(model, rows, cfg, device, out, label=f"{name} {'D4' if d4 else 'plain'}")
    frame = pd.read_csv(out / "per_image_metrics.csv")
    summary.update(metric_summary(frame))
    write_json(out / "metrics_summary.json", summary)
    finish_output(out, provenance)
    print(f"{name}: image mIoU={summary['image_mIoU']:.4f}% / global mIoU={summary['global_mIoU']:.4f}%", flush=True)
    return summary
