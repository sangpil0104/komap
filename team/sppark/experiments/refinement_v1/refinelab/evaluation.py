from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from alphase.cli.check_submission import check
from alphase.data.audit import verify_manifest
from alphase.data.dataset import load_image
from alphase.data.palette import save_mask
from alphase.data.splits import select_rows
from alphase.evaluation.evaluate import evaluate
from alphase.inference.predictor import predict_probabilities
from alphase.utils import sha256, write_json
from refinelab.models import build_model
from refinelab.workspace import BASELINE, PROJECT, output_path, read_json, run_dir, source_hash


def load_run(name, seed=2026, smoke=False, d4=False):
    checkpoint = run_dir(name, seed, smoke) / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["resolved_config"]
    if cfg["refinement"]["source_sha256"] != source_hash():
        raise ValueError("Refinement source changed since this model was trained")
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
    provenance = dict(experiment=name, seed=seed, smoke_test=smoke, checkpoint_sha256=sha256(checkpoint),
                      source_sha256=source_hash(), selected_weights=selected, dataset_hash=state["dataset_hash"],
                      train_ids=state["train_ids"], validation_ids=state["validation_ids"],
                      inference=cfg["inference"], architecture=cfg["model"]["name"], torch_version=str(torch.__version__))
    return model.to(device).eval(), cfg, device, valid, provenance


def begin_output(folder, expected):
    """Resume an interrupted evaluation by keeping its partial output in a backup."""
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


def evaluate_run(name, seed=2026, smoke=False, d4=False):
    model, cfg, device, rows, provenance = load_run(name, seed, smoke, d4)
    out = output_path("smoke_evaluations" if smoke else "evaluations", name, f"seed{seed}", "d4" if d4 else "plain")
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


def predict_test(name, seed=2026, smoke=False, d4=False):
    model, cfg, device, _, provenance = load_run(name, seed, smoke, d4)
    provenance = {**provenance, "evaluation_kind": "unlabeled_test"}
    out = output_path("smoke_predictions" if smoke else "predictions", name, f"seed{seed}", "d4" if d4 else "plain")
    if not begin_output(out, provenance):
        return out
    frame = pd.read_csv(cfg["data"]["manifest"]).fillna("")
    rows = frame[frame.source_split == "test"].to_dict("records")
    for row in tqdm(rows, desc=f"Test masks {name}", unit="image", dynamic_ncols=True):
        gray = load_image(row["image_path"]).astype(np.float32) / 255
        pred = predict_probabilities(model, gray, cfg, device).argmax(0)
        save_mask(pred, out / "masks" / f"{row['image_id']}_mask.png")
    count = check(PROJECT / "dataset/test/images", out / "masks")
    write_json(out / "check.json", dict(images=count, color_and_dimensions_verified=True, has_test_ground_truth=False))
    finish_output(out, provenance)
    print(f"Test masks: {out / 'masks'}", flush=True)
    return out


def compare():
    candidates = [("baseline", 2026, "plain", BASELINE / "run/evaluation/per_image_metrics.csv"),
                  ("baseline", 2026, "d4", BASELINE / "run/evaluation_d4_review/per_image_metrics.csv")]
    for p in sorted(output_path("evaluations").glob("*/seed*/*/per_image_metrics.csv")):
        if p.parent.name not in ("plain", "d4") or not (p.parent / "provenance.json").is_file():
            continue
        prov = read_json(p.parent / "provenance.json")["request"]
        if not prov["smoke_test"]:
            candidates.append((prov["experiment"], prov["seed"], p.parent.name, p))
    frames = {(n, seed, mode): pd.read_csv(p).set_index("image_id") for n, seed, mode, p in candidates}
    rows = []
    for name, seed, mode, p in candidates:
        ref = frames["baseline", 2026, mode]
        frame = frames[name, seed, mode]
        if set(frame.index) != set(ref.index) or len(frame) != 20:
            raise ValueError("Comparison requires the same 20 official validation images")
        frame = frame.loc[ref.index]
        delta = frame.mIoU_exclude - ref.mIoU_exclude
        cms = np.stack([json.loads(v) for v in frame.confusion_matrix])
        common = (cms.sum(2) > 0).all(1)
        row = dict(experiment=name, seed=seed, inference=mode, **metric_summary(frame),
                   delta_baseline_pp=float(delta.mean()), improved_images=int((delta > 1e-8).sum()),
                   delta_common19_pp=float(delta.to_numpy()[common].mean()),
                   miou_one=float(frame.mIoU_one.mean()), miou_zero=float(frame.mIoU_zero.mean()), source=str(p))
        control = frames.get(("e00_ft_control", seed, mode))
        row["delta_finetune_control_pp"] = None if control is None else float(frame.mIoU_exclude.mean()-control.mIoU_exclude.mean())
        for c in range(4):
            row[f"iou_{c}"] = float(frame[f"IoU_{c}"].mean())
            row[f"iou_{c}_image_count"] = int(frame[f"IoU_{c}"].notna().sum())
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("image_mIoU", ascending=False)
    folder = output_path("reports")
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "comparison.csv", index=False)
    cols = ["experiment", "seed", "inference", "image_mIoU", "global_mIoU", "delta_baseline_pp", "delta_finetune_control_pp", "delta_common19_pp"]
    text = table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    print(text)
    (folder / "comparison.txt").write_text("주 지표: 이미지별 mIoU. Global mIoU는 보조 지표. 같은 공식 검증 20장의 개발 결과.\n" + text)
    write_json(folder / "selection.json", dict(best_observed=table.iloc[0][["experiment", "seed", "inference", "image_mIoU", "global_mIoU"]].to_dict(),
        automatic_model_replacement=False, note="Validation selection only; inspect class counts/common19 and confirm independently."))
    return table
