import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from alphase.data.dataset import load_image, load_target
from alphase.data.palette import PALETTE
from alphase.evaluation.metrics import image_miou
from alphase.inference.predictor import predict_probabilities
from alphase.utils import json_hash, write_json


def evaluate(model, rows, cfg, device, output=None, save_probabilities=False, distributed=False, label="Validation"):
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    results = []
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
    for row in tqdm(rows[rank::world], desc=label, disable=rank != 0, leave=False):
        gray = load_image(row["image_path"]).astype(np.float32)/255
        target = load_target(row["mask_path"])
        prob = predict_probabilities(model, gray, cfg, device)
        pred = prob.argmax(0)
        record = dict(image_id=row["image_id"], H=target.shape[0], W=target.shape[1], inference_config_hash=json_hash(cfg["inference"]))
        for policy in ("exclude", "one", "zero"):
            score, iou, cm = image_miou(target, pred, policy)
            record[f"mIoU_{policy}"] = 100*score
            if policy == "exclude":
                for c in range(4):
                    record[f"IoU_{c}"] = 100*float(iou[c]) if np.isfinite(iou[c]) else None
                record["confusion_matrix"] = json.dumps(cm.tolist())
        record["GT_phase_fractions"] = json.dumps((np.bincount(target.ravel(), minlength=4)/target.size).tolist())
        record["predicted_phase_fractions"] = json.dumps((np.bincount(pred.ravel(), minlength=4)/pred.size).tolist())
        results.append(record)
        if output and save_probabilities:
            folder = Path(output)/"probabilities"
            folder.mkdir(exist_ok=True)
            np.save(folder/f"{row['image_id']}.npy", prob.astype(np.float32))
        if output and rank == 0 and len(results) <= 3:
            rgb = np.repeat((gray*255).astype(np.uint8)[..., None], 3, -1)
            preview = Image.fromarray(np.concatenate([rgb, PALETTE[target], PALETTE[pred]], 1))
            preview.thumbnail((1440, 480))
            preview.save(Path(output)/f"preview_{len(results)}.png")
    if distributed:
        gathered = [None] * world
        dist.all_gather_object(gathered, results)
        results = [item for part in gathered for item in part]
    if not results:
        raise ValueError("No validation images")
    frame = pd.DataFrame(results).sort_values("image_id")
    summary = {f"mIoU_{p}": float(frame[f"mIoU_{p}"].mean()) for p in ("exclude", "one", "zero")}
    summary["per_class_iou"] = [None if frame[f"IoU_{c}"].isna().all() else float(frame[f"IoU_{c}"].mean()) for c in range(4)]
    summary["num_images"] = len(rows)
    summary["inference_config_hash"] = json_hash(cfg["inference"])
    if rank == 0 and output:
        frame.to_csv(Path(output)/"per_image_metrics.csv", index=False)
        write_json(Path(output)/"metrics_summary.json", summary)
    return summary
