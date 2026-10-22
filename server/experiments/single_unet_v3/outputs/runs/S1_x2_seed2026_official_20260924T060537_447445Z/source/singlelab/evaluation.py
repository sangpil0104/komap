from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from .common import CLASSES, PALETTE, dataset_signature, digest, read_json, records, sources, stamp, write_json
from .data import gray, labels
from .inference import predict
from .metrics import measure, summarize
from .model import EfficientUNet


def checkpoint_path(run):
    return run / read_json(run / "latest.json")["checkpoint"]


def load_run(run, device):
    run = Path(run).resolve()
    cfg = read_json(run / "config.json")
    if read_json(run / "source_hashes.json") != sources():
        raise ValueError("Experiment source changed; use the preserved source snapshot with this run")
    if read_json(run / "dataset.json") != dataset_signature(cfg, verify=True):
        raise ValueError("Dataset/split changed")
    path = checkpoint_path(run)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = EfficientUNet(cfg, pretrained=False)
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), cfg, path, saved["update"]


def comparison(path, g, target, pred, score):
    h, w = target.shape
    width = min(w, 800)
    height = round(h * width / w)
    canvas = Image.new("RGB", (2 * width, 2 * (height + 30) + 40), "white")
    original = Image.fromarray(np.clip(g * 255, 0, 255).astype(np.uint8)).convert("RGB")
    error = np.repeat(np.clip(g * 255, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    error[target != pred] = [255, 50, 50]
    panels = [(original, "Original"), (Image.fromarray(error), "Errors (red)"),
              (Image.fromarray(PALETTE[pred]), f"Prediction / image mIoU {score:.3f}%"),
              (Image.fromarray(PALETTE[target]), "Ground truth")]
    draw = ImageDraw.Draw(canvas)
    for i, (panel, title) in enumerate(panels):
        x, y = (i % 2) * width, (i // 2) * (height + 30)
        draw.text((x + 8, y + 8), title, fill="black")
        canvas.paste(panel.resize((width, height), Image.Resampling.NEAREST), (x, y + 30))
    draw.text((8, canvas.height - 27), "Green: Primary Si | Blue: Al3Ni | Gray: Eutectic Si | Brown: Al", fill="black")
    canvas.save(path)


def evaluate(run, device, overrides=None, split="valid", model_bundle=None):
    run = Path(run).resolve()
    model, cfg, checkpoint, update = model_bundle or load_run(run, device)
    cfg = copy.deepcopy(cfg)
    if overrides:
        cfg["inference"].update(overrides)
    settings = cfg["inference"]
    tag = f"{split}_{settings['mode']}_pad{settings['pad']}_halo{settings['halo']}_off{settings['offset']}_d4{int(settings['d4'])}"
    folder = run / "evaluations" / f"{tag}_{stamp()}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "masks").mkdir()
    if split == "valid":
        (folder / "comparisons").mkdir()
    request = dict(run=str(run), checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                   update=update, config=cfg, split=split, completed=False)
    write_json(folder / "request.json", request)
    rows, started = [], time.monotonic()
    for row in tqdm(records(cfg)[split], desc=f"{cfg['experiment']} {tag}", unit="image", dynamic_ncols=True):
        g = gray(row["image_path"])
        probs = predict(model, g, cfg, device, show_tiles=True)
        pred = probs.argmax(0).astype(np.uint8)
        rgb = PALETTE[pred]
        rgba = np.concatenate((rgb, np.full((*pred.shape, 1), 255, np.uint8)), axis=2)
        Image.fromarray(rgba).save(folder / "masks" / f"{row['image_id']}_mask.png")
        if split == "valid":
            target = labels(row["mask_path"])
            result = dict(image_id=row["image_id"], **measure(target, pred, cfg["evaluation"]))
            rows.append(result)
            comparison(folder / "comparisons" / f"{row['image_id']}.png", g, target, pred, result["image_miou"])
    if rows:
        with (folder / "per_image.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
        summary = summarize(rows, cfg["evaluation"]["empty_union"])
    else:
        summary = dict(num_images=len(records(cfg)[split]), test_has_no_labels=True)
    summary.update(elapsed_seconds=time.monotonic() - started)
    write_json(folder / "summary.json", summary)
    request["completed"] = True
    write_json(folder / "request.json", request)
    return folder


def border_comparison(run, device):
    bundle = load_run(run, device)
    # Identical checkpoint/scale, no TTA; only context/padding/tile grid change.
    variants = [dict(mode="full", pad=0, d4=False), dict(mode="full", pad=64, d4=False),
                dict(mode="tiled", pad=0, halo=64, offset=0, d4=False),
                dict(mode="tiled", pad=0, halo=128, offset=0, d4=False),
                dict(mode="tiled", pad=0, halo=64, offset=128, d4=False)]
    folders = []
    for variant in tqdm(variants, desc="Boundary inference comparison", unit="setting"):
        folders.append(evaluate(run, device, variant, model_bundle=bundle))
    return folders
