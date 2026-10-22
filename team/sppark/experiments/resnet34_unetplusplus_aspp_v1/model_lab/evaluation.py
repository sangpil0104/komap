from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_lab.dataset import KomapEvalDataset, PALETTE
from model_lab.workspace import get_dataset_samples, write_json, OUTPUTS_DIR


def predict_full_frame(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    original_h: int,
    original_w: int,
    d4: bool = False,
    device: torch.device = torch.device("cuda:0"),
) -> np.ndarray:
    """Predicts full-frame class probabilities or logits, supporting D4 TTA.
    
    image_tensor: [3, pad_h, pad_w]
    returns: [original_h, original_w] predicted integer labels in [0, 3].
    """
    model.eval()
    x = image_tensor.unsqueeze(0).to(device)  # [1, 3, pad_h, pad_w]

    with torch.no_grad():
        if not d4:
            out = model(x)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = F.softmax(logits, dim=1)[0]
        else:
            accum_probs = None
            # 8 dihedral transformations
            for k in range(8):
                rot_k = k % 4
                flip = (k // 4) == 1
                inp = torch.rot90(x, rot_k, dims=(-2, -1))
                if flip:
                    inp = torch.flip(inp, dims=[-1])

                out = model(inp)
                logits = out["logits"] if isinstance(out, dict) else out
                cur_probs = F.softmax(logits, dim=1)

                # Invert transformations
                if flip:
                    cur_probs = torch.flip(cur_probs, dims=[-1])
                if rot_k > 0:
                    cur_probs = torch.rot90(cur_probs, 4 - rot_k, dims=(-2, -1))

                if accum_probs is None:
                    accum_probs = cur_probs
                else:
                    accum_probs += cur_probs

            probs = (accum_probs / 8.0)[0]

        # Crop back to original dimensions
        cropped_probs = probs[:, :original_h, :original_w]
        pred_labels = torch.argmax(cropped_probs, dim=0).cpu().numpy().astype(np.int64)

    return pred_labels


def evaluate_dataset(
    model: torch.nn.Module,
    samples: list[dict[str, Any]] | None = None,
    d4: bool = False,
    device: torch.device = torch.device("cuda:0"),
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Runs evaluation on 20 validation images, computing per-image and global mIoU and per-class IoU."""
    if samples is None:
        samples = get_dataset_samples("valid")

    eval_ds = KomapEvalDataset(samples, has_masks=True)
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

    num_classes = len(PALETTE)
    per_image_records = []
    global_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    pbar = tqdm(loader, desc=f"Evaluating ({'D4' if d4 else 'Plain'})", unit="image")
    for batch in pbar:
        img = batch["image"][0]
        h = int(batch["original_h"][0])
        w = int(batch["original_w"][0])
        image_id = batch["image_id"][0]
        target = batch["mask"][0].numpy()

        pred = predict_full_frame(model, img, h, w, d4=d4, device=device)

        # Compute confusion matrix for this image
        valid_mask = target < num_classes
        flat_target = target[valid_mask]
        flat_pred = pred[valid_mask]

        hist = np.bincount(
            flat_target * num_classes + flat_pred,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        global_confusion += hist

        # Class IoUs for this image
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp
        denom = tp + fp + fn
        present = hist.sum(axis=1) > 0

        ious = np.full(num_classes, np.nan, dtype=np.float64)
        ious[denom > 0] = tp[denom > 0] / denom[denom > 0]
        image_miou = float(np.nanmean(ious[present])) if present.any() else 0.0

        rec: dict[str, Any] = {
            "image_id": image_id,
            "mIoU": image_miou,
            "IoU_Primary_Si": float(ious[0]) if not np.isnan(ious[0]) else None,
            "IoU_Al3Ni": float(ious[1]) if not np.isnan(ious[1]) else None,
            "IoU_Eutectic_Si": float(ious[2]) if not np.isnan(ious[2]) else None,
            "IoU_Al": float(ious[3]) if not np.isnan(ious[3]) else None,
            "confusion_matrix": hist.tolist(),
        }
        per_image_records.append(rec)

    # Compute aggregate metrics
    df = pd.DataFrame(per_image_records)
    image_mean_miou = float(df["mIoU"].mean())

    gtp = np.diag(global_confusion)
    gfp = global_confusion.sum(axis=0) - gtp
    gfn = global_confusion.sum(axis=1) - gtp
    gdenom = gtp + gfp + gfn
    global_ious = np.where(gdenom > 0, gtp / gdenom, 0.0)
    global_miou = float(np.mean(global_ious))

    summary = {
        "mode": "d4" if d4 else "plain",
        "num_images": len(samples),
        "image_mean_miou": round(image_mean_miou * 100.0, 4),
        "global_miou": round(global_miou * 100.0, 4),
        "class_ious": {
            "Primary_Si": round(float(global_ious[0]) * 100.0, 4),
            "Al3Ni": round(float(global_ious[1]) * 100.0, 4),
            "Eutectic_Si": round(float(global_ious[2]) * 100.0, 4),
            "Al": round(float(global_ious[3]) * 100.0, 4),
        },
    }

    if out_dir is not None:
        p_out = Path(out_dir)
        p_out.mkdir(parents=True, exist_ok=True)
        df.to_csv(p_out / "per_image_metrics.csv", index=False)
        write_json(p_out / "summary.json", summary)

    return summary

