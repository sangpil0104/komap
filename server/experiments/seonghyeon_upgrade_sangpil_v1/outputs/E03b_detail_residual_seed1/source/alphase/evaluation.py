import csv
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation
from tqdm import tqdm
from .contracts import (
    CLASS_NAMES,
    PALETTE,
    rgb_to_ids,
    ids_to_rgb,
    confusion_matrix,
    metrics_from_cm,
    summarize_images,
)
from .data import boundary_mask, dataset_rows
from .io import ROOT, REPO, json_write, jsonl_read, local_path, atomic
from .cache import load_probability


def unique_records(rows):
    """Remove DistributedSampler padding duplicates; reject conflicting records."""
    out = {}
    for r in rows:
        key = r["image_id"]
        if key in out and out[key] != r:
            raise ValueError(f"Conflicting duplicate validation ID {key}")
        out[key] = r
    return list(out.values())


def diagnostics(target, pred):
    cm = confusion_matrix(target, pred)
    result = metrics_from_cm(cm)
    valid = target != 255
    wrong = (target != pred) & valid
    band = binary_dilation(boundary_mask(target), iterations=2) & valid
    reasons = {}
    for name, mask in [
        ("boundary_2px_error_rate", band),
        ("interior_error_rate", valid & ~band),
    ]:
        result[name] = float(wrong[mask].mean()) if mask.any() else None
        if not mask.any():
            reasons[name] = "empty_pixel_region"
    for a, b, name in [(1, 3, "Al3Ni_to_EutecticSi"), (3, 1, "EutecticSi_to_Al3Ni")]:
        result[name + "_count"] = int(cm[a, b])
        total = int(cm[a].sum())
        result[name + "_gt_rate"] = float(cm[a, b] / total) if total else None
        if not total:
            reasons[name + "_gt_rate"] = "no_gt_pixels"
        result[name + "_fraction_of_all_errors"] = (
            float(cm[a, b] / wrong.sum()) if wrong.any() else None
        )
        if not wrong.any():
            reasons[name + "_fraction_of_all_errors"] = "no_errors"
    for i, c in enumerate(CLASS_NAMES):
        if result["precision"][i] is None:
            reasons["precision_" + c] = "no_predicted_pixels"
        if result["recall"][i] is None:
            reasons["recall_" + c] = "no_gt_pixels"
    result["undefined_reasons"] = reasons
    result["confusion_matrix"] = cm.tolist()
    return result


def comparison(path, gray, pred, target, score):
    h, w = gray.shape
    margin = 16
    gap = 24
    header = 76
    canvas = Image.new(
        "RGB",
        (2 * w + 2 * margin + gap, 2 * h + 2 * header + 2 * margin + gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = (
        ImageFont.truetype(font_path, 24)
        if Path(font_path).exists()
        else ImageFont.load_default()
    )
    x1 = margin
    x2 = margin + w + gap
    y1 = margin + header
    y2 = y1 + h + gap + header
    error = np.full((h, w, 3), 255, np.uint8)
    wrong = pred != target
    error[wrong] = PALETTE[target[wrong]]
    for x, y, arr in [
        (x1, y1, np.repeat(gray[:, :, None], 3, 2)),
        (x2, y1, error),
        (x1, y2, ids_to_rgb(pred)),
        (x2, y2, ids_to_rgb(target)),
    ]:
        canvas.paste(Image.fromarray(arr), (x, y))
    draw.text((x1, margin + 20), "Original image", fill="black", font=font)
    draw.text(
        (x2, margin + 4), f"image mIoU: {score*100:.4f}%", fill="black", font=font
    )
    draw.text(
        (x2, margin + 38),
        f"100% - image mIoU: {(1-score)*100:.4f}%",
        fill="black",
        font=font,
    )
    draw.text((x1, y2 - header + 20), "Predicted mask", fill="black", font=font)
    draw.text((x2, y2 - header + 20), "Ground-truth mask", fill="black", font=font)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))

    def write(p):
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: (
                            json.dumps(v, allow_nan=False)
                            if isinstance(v, (dict, list))
                            else v
                        )
                        for k, v in row.items()
                    }
                )

    atomic(path, write)


def report_predictions(manifest, dataset_manifest=None, out=None, panels=True):
    manifest = local_path(manifest)
    out = Path(out) if out else manifest.parent / "report"
    out.mkdir(parents=True, exist_ok=True)
    images = {
        r["image_id"]: r
        for r in jsonl_read(local_path(dataset_manifest or "manifests/dataset.jsonl"))
    }
    rows = []
    cms = []
    for row in tqdm(unique_records(jsonl_read(manifest)), desc="diagnostics"):
        image = images[row["image_id"]]
        if (
            image["image_sha256"] != row["image_sha256"]
            or image["split"] != row["source_split"]
        ):
            raise ValueError("Prediction dataset identity mismatch")
        prob = load_probability(row, manifest.parent)
        if not image.get("mask_path"):
            continue
        target = rgb_to_ids(np.array(Image.open(image["mask_path"])))
        pred = prob.argmax(0)
        metric = diagnostics(target, pred)
        ni = target == 1
        metric["gt_Al3Ni_mean_P_EutecticSi"] = (
            float(prob[3, ni].mean()) if ni.any() else None
        )
        votes = manifest.parent / "diagnostics" / f'{row["image_id"]}_vote_fraction.npy'
        if votes.exists():
            v = np.load(votes, allow_pickle=False)
            metric["gt_Al3Ni_mean_EutecticSi_vote_fraction"] = (
                float(v[3, ni].mean()) if ni.any() else None
            )
        rows.append(
            {
                "image_id": row["image_id"],
                "split": image["split"],
                "height": image["native_hw"][0],
                "width": image["native_hw"][1],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "inference_config_hash": row["inference_config_hash"],
                **metric,
            }
        )
        cms.append(np.array(metric["confusion_matrix"], np.int64))
        if panels:
            gray = np.array(Image.open(image["image_path"]).convert("L"))
            comparison(
                out / "valid_mask_comparison" / f'{row["image_id"]}.png',
                gray,
                pred,
                target,
                metric["image_miou"],
            )
    if not rows:
        raise ValueError("No labeled images to evaluate")
    summary = {
        **summarize_images(cms),
        "per_image": rows,
        "mean_class_iou": np.mean([r["iou"] for r in rows], 0).tolist(),
        "score_role": (
            "development_validation"
            if rows[0]["split"] == "valid"
            else "training_diagnostic"
        ),
    }
    json_write(out / "summary.json", summary)
    write_csv(out / "per_image.csv", rows)
    write_csv(
        out / "per_class.csv",
        [
            {"class": c, "mean_image_iou": summary["mean_class_iou"][i]}
            for i, c in enumerate(CLASS_NAMES)
        ],
    )
    json_write(
        out / "confusion_matrices.json",
        {r["image_id"]: r["confusion_matrix"] for r in rows},
    )
    (out / (out.name + ".md")).write_text(
        f'# {out.name}\n\nImage mIoU: **{summary["val_image_miou"]*100:.6f}%**, {len(rows)} images, four-class mean, empty union=1.\n\n'
        + "\n".join(
            ["| Image | image mIoU |", "|---|---:|"]
            + [f'| {r["image_id"]} | {r["image_miou"]*100:.4f}% |' for r in rows]
        )
        + "\n\nUpper right: correct pixels white, errors in GT color; labels show image mIoU and its complement.\n"
    )
    return summary


def reserve_results():
    date = datetime.now(timezone.utc).strftime("%m%d")
    i = 1
    while True:
        p = REPO / "results" / f"model_results_{date}_{i}"
        try:
            p.mkdir()
            return p
        except FileExistsError:
            i += 1


def paired_bootstrap(a, b, n=10000, seed=2026):
    if set(a) != set(b):
        raise ValueError("Paired image IDs differ")
    delta = np.array([b[k] - a[k] for k in sorted(a)])
    rng = np.random.default_rng(seed)
    values = delta[rng.integers(len(delta), size=(n, len(delta)))].mean(1)
    return {
        "mean_difference": float(delta.mean()),
        "image_bootstrap_95pct_ci": np.quantile(values, [0.025, 0.975]).tolist(),
        "replicates": n,
        "seed": seed,
        "grouping": "image; specimen identities unavailable",
    }
