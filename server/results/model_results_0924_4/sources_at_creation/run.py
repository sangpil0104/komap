#!/usr/bin/env python3
"""Entry point: preflight, fit, evaluate, suite, predict. No GPU dependencies."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

import numpy as np
from PIL import Image
from tqdm import tqdm

from core import (CLASS_NAMES, PALETTE, confusion, fit, image_metrics, load_model,
                  predict, read_rgb, read_target, sha256, summarize)
from reporting import comparison, histogram, write_report

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, data):
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    with open(path, "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Progress:
    def __init__(self, label, total, overall_start=None, final_stage=False):
        self.label, self.total = label, total
        self.started = time.monotonic()
        self.overall_start = overall_start if overall_start is not None else self.started
        self.final_stage = final_stage
        self.bar = tqdm(total=total, desc=label, unit="image", disable=not sys.stderr.isatty())
        print(f"[{now()}] {label}: {total} images; ETA calculated after first image", flush=True)

    def __call__(self, completed, image_id):
        elapsed = time.monotonic() - self.started
        overall = time.monotonic() - self.overall_start
        remaining = elapsed / completed * (self.total - completed)
        finish = datetime.now().astimezone() + timedelta(seconds=remaining)
        text = (f"{completed}/{self.total} {image_id} | elapsed={elapsed:.1f}s "
                f"remaining~{remaining:.1f}s | finish~{finish.isoformat(timespec='seconds')} "
                f"| overall_elapsed={overall:.1f}s")
        if self.final_stage:
            text += f" | total_estimate~{overall + remaining:.1f}s (plus final report write)"
        if self.bar.disable:
            print(f"[{self.label}] {text}", flush=True)
        else:
            self.bar.update(completed - self.bar.n)
            self.bar.set_postfix_str(text)
        if completed == self.total:
            self.bar.close()


def manifest_rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "source_split", "image_path", "mask_path", "width", "height",
                "image_sha256", "mask_sha256"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest missing required fields: {sorted(required)}")
    ids = [row["image_id"] for row in rows]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[\w.-]+", x) or x in (".", "..") for x in ids):
        raise ValueError("Image IDs must be unique and safe filenames")
    counts = {split: sum(row["source_split"] == split for row in rows)
              for split in ("train", "valid", "test")}
    if counts != {"train": 70, "valid": 20, "test": 10} or len(rows) != 100:
        raise ValueError(f"Expected fixed train70/valid20/test10 split, received {counts}")
    for row in rows:
        if row["source_split"] != "test" and (not row["mask_path"] or not row["mask_sha256"]):
            raise ValueError(f"Train/valid mask is required: {row['image_id']}")
        for key in ("image_path", "mask_path"):
            if not row[key]:
                continue
            item = Path(row[key])
            row[key] = str((path.parent / item).resolve() if not item.is_absolute() else item)
    return rows


def verify_row(row):
    rgb = read_rgb(row["image_path"])
    target = read_target(row["mask_path"]) if row["mask_path"] else None
    expected = (int(row["height"]), int(row["width"]))
    if rgb.shape[:2] != expected or (target is not None and target.shape != expected):
        raise ValueError(f"Dimensions differ from manifest: {row['image_id']}")
    for key in ("image", "mask"):
        if key == "mask" and target is None:
            continue
        if sha256(row[f"{key}_path"]) != row[f"{key}_sha256"]:
            raise ValueError(f"File hash differs from manifest: {row['image_id']} {key}")
    return rgb, target


def preflight(args):
    rows = manifest_rows(args.manifest)
    progress = Progress("preflight (read-only)", len(rows), final_stage=True)
    for index, row in enumerate(rows):
        verify_row(row)
        progress(index + 1, row["image_id"])
    result = {"checked_at": now(), "manifest": str(args.manifest),
              "manifest_sha256": sha256(args.manifest), "checked_images": len(rows),
              "checked_masks": sum(bool(r["mask_path"]) for r in rows),
              "split_counts": {"train": 70, "valid": 20, "test": 10},
              "checks": ["unique IDs", "opaque images", "known palette in available masks", "native dimensions", "file SHA-256"],
              "training_started": False}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def source_snapshot(destination):
    destination.mkdir()
    hashes = {}
    for path in sorted(HERE.rglob("*")):
        relative = path.relative_to(HERE)
        if path.is_file() and len(relative.parts) <= 2 and path.suffix in (".py", ".md", ".sh", ".txt"):
            if relative.parts[0] in ("runs", "logs", "predictions") or "__pycache__" in relative.parts:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(relative)] = sha256(target)
    return hashes


def fresh_run(root):
    directory = root / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def fresh_result(root):
    root.mkdir(parents=True, exist_ok=True)
    prefix = "model_results_" + datetime.now().strftime("%m%d") + "_"
    numbers = [int(path.name[len(prefix):]) for path in root.iterdir()
               if path.name.startswith(prefix) and path.name[len(prefix):].isdigit()]
    number = max(numbers, default=0) + 1
    while True:
        directory = root / f"{prefix}{number}"
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            number += 1


def train(args, overall_start=None):
    rows = [r for r in manifest_rows(args.manifest) if r["source_split"] == "train"]
    checks = Progress("train data check", len(rows), overall_start)
    for index, row in enumerate(rows):
        verify_row(row)
        checks(index + 1, row["image_id"])
    directory = fresh_run(args.runs_root)
    source_hashes = source_snapshot(directory / "sources_at_creation")
    write_csv(directory / "train_manifest.csv", rows)
    progress = Progress("fit train-only phase mapping", len(rows), overall_start)
    model, candidates = fit(rows, progress)
    model.update({"created_at": now(), "manifest_sha256": sha256(args.manifest),
                  "source_sha256": source_hashes})
    model_path = directory / "model.json"
    write_json(model_path, model)
    write_json(directory / "mapping_candidates.json", candidates)
    print(f"MODEL={model_path}\nBin to phase: " + " -> ".join(CLASS_NAMES[c] for c in model["bin_to_class"]), flush=True)
    print(f"Train image mIoU={model['train_image_miou_one']:.4f}% (calibration score, not validation)", flush=True)
    return model_path


def evaluate(args, model_path, overall_start=None):
    started = time.monotonic()
    model = load_model(model_path)
    rows = [r for r in manifest_rows(args.manifest) if r["source_split"] == "valid"]
    # Prevent accidental use of calibration images as validation, even under renamed IDs.
    train_ids = {row["image_id"] for row in model["train_images"]}
    train_hashes = {row["image_sha256"] for row in model["train_images"]}
    if any(row["image_id"] in train_ids or row["image_sha256"] in train_hashes for row in rows):
        raise ValueError("Validation overlaps model calibration images")
    directory = fresh_result(args.results_root)
    source_hashes = source_snapshot(directory / "sources_at_creation")
    for name in ("masks", "valid_mask_comparison", "histograms"):
        (directory / name).mkdir()
    shutil.copy2(model_path, directory / "model.json")
    records, csv_rows, checks = [], [], []
    progress = Progress("valid prediction + reports", len(rows), overall_start, final_stage=True)
    for index, row in enumerate(rows):
        original, target = verify_row(row)
        prediction, info = predict(original, model)
        matrix = confusion(target, prediction)
        metrics = image_metrics(matrix)
        image_id = row["image_id"]
        flat = {"image_id": image_id, "threshold_1": info["thresholds"][0],
                "threshold_2": info["thresholds"][1], "threshold_3": info["thresholds"][2],
                "threshold_source": info["threshold_source"], **metrics}
        csv_rows.append(flat)
        records.append({**flat, **info, "confusion_matrix": matrix.tolist()})
        Image.fromarray(PALETTE[prediction]).convert("RGBA").save(directory / "masks" / f"{image_id}.png")
        checks.append(comparison(directory / "valid_mask_comparison" / f"{image_id}.png",
                                 original, target, prediction, image_id, metrics["image_miou_one"]))
        histogram(directory / "histograms" / f"{image_id}.png", original,
                  info["thresholds"], model["bin_to_class"])
        progress(index + 1, image_id)
    summary = summarize(csv_rows)
    write_json(directory / "summary.json", summary)
    write_csv(directory / "per_image.csv", csv_rows)
    write_json(directory / "per_image.json", records)
    write_json(directory / "verification.json", {"saved_panels_verified": True, "images": checks})
    write_json(directory / "provenance.json", {"created_at": now(), "model_path": str(model_path),
               "model_sha256": sha256(model_path), "manifest_path": str(args.manifest),
               "manifest_sha256": sha256(args.manifest), "evaluation_source_sha256": source_hashes,
               "validation_ids": [r["image_id"] for r in rows], "data": rows,
               "selection_split": "train", "evaluation_split": "valid", "global_miou_used": False})
    write_report(directory, model, summary, csv_rows, model_path, time.monotonic() - started)
    print(f"RESULTS={directory}\nValidation image mIoU={summary['image_miou_one']:.4f}% (empty=1)", flush=True)
    return directory


def predict_images(args):
    model = load_model(args.model)
    paths = []
    for item in args.input:
        if item.is_dir():
            paths.extend(p for p in sorted(item.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
        else:
            paths.append(item)
    if not paths:
        raise ValueError("No input images found")
    directory = fresh_run(args.output_root)
    shutil.copy2(args.model, directory / "model.json")
    records = []
    progress = Progress("predict", len(paths), final_stage=True)
    for index, path in enumerate(paths):
        original = read_rgb(path)
        prediction, info = predict(original, model)
        name = f"{index + 1:04d}_{path.stem}"
        Image.fromarray(PALETTE[prediction]).convert("RGBA").save(directory / f"{name}_mask.png")
        histogram(directory / f"{name}_histogram.png", original, info["thresholds"], model["bin_to_class"])
        records.append({"input": str(path), "image_sha256": sha256(path), "output_stem": name, **info})
        progress(index + 1, path.name)
    write_json(directory / "predictions.json", records)
    print(f"PREDICTIONS={directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "fit", "evaluate", "suite", "predict"):
        sub = commands.add_parser(command)
        if command != "predict":
            sub.add_argument("--manifest", type=Path, default=ROOT / "data/processed/manifest.csv")
        if command == "preflight":
            sub.add_argument("--output", type=Path, help="Optional new JSON path (never overwrite)")
        if command in ("fit", "suite"):
            sub.add_argument("--runs-root", type=Path, default=HERE / "runs")
        if command in ("evaluate", "suite"):
            sub.add_argument("--results-root", type=Path, default=ROOT / "results")
        if command in ("evaluate", "predict"):
            sub.add_argument("--model", required=True, type=Path)
        if command == "predict":
            sub.add_argument("--input", required=True, nargs="+", type=Path, help="Images or directories; no masks needed")
            sub.add_argument("--output-root", type=Path, default=HERE / "predictions")
    args = parser.parse_args()
    started = time.monotonic()
    print(f"[{now()}] command={args.command}; CPU only; PID={os.getpid()}", flush=True)
    if args.command == "preflight":
        preflight(args)
    elif args.command == "fit":
        train(args, started)
    elif args.command == "evaluate":
        evaluate(args, args.model, started)
    elif args.command == "suite":
        print("Stages: train data check -> train calibration -> valid20 evaluation/figures. "
              "Whole-run finish estimate begins after the first validation image; earlier ETAs are stage-specific.", flush=True)
        model_path = train(args, started)
        evaluate(args, model_path, started)
    else:
        predict_images(args)
    print(f"[{now()}] DONE | total_elapsed={time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
