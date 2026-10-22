from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from tqdm import tqdm

from alphase.cli.common import load_trained
from alphase.data.audit import verify_manifest
from alphase.data.dataset import load_image, load_target
from alphase.data.splits import select_rows
from alphase.inference.predictor import predict_probabilities
from alphase.utils import sha256, write_json
from refinelab.models import validate_parent
from refinelab.workspace import PARENT, config, output_path, read_json


def ensure_mining():
    cfg = config("e01_hard")
    train, valid = select_rows(cfg, "official")
    assert not {r["image_id"] for r in train} & {r["image_id"] for r in valid}
    dataset_hash = verify_manifest(cfg["data"]["manifest"])
    expected = dict(parent_sha256=sha256(PARENT), dataset_hash=dataset_hash,
                    train_ids=[r["image_id"] for r in train], validation_images_used=[],
                    policy="wrong_predictions_on_gt_classes_1_2; fallback_all_errors", version=1,
                    code_sha256=sha256(Path(__file__)))
    if expected["parent_sha256"] != cfg["refinement"]["parent_sha256"]:
        raise ValueError("Parent checkpoint changed")
    folder = output_path("mining")
    folder.mkdir(parents=True, exist_ok=True)
    request, marker = folder / "request.json", folder / "manifest.json"
    if request.exists() and read_json(request) != expected:
        raise ValueError("Mining cache belongs to different data/code/parent; keep it in a separate folder")
    write_json(request, expected)
    records = read_json(marker) if marker.exists() else {}
    pending = []
    for row in train:
        record = records.get(row["image_id"])
        file = folder / f"{row['image_id']}.npz"
        if record:
            if not file.is_file() or sha256(file) != record["sha256"]:
                raise ValueError(f"Mining cache damaged: {file}")
        else:
            pending.append(row)
    if not pending:
        print("학습 70장의 오분류 위치 캐시 재사용", flush=True)
        return folder
    model, parent_cfg, device, state, _ = load_trained(PARENT)
    validate_parent(state, cfg)
    if state["dataset_hash"] != dataset_hash:
        raise ValueError("Dataset changed since parent training")
    for row in tqdm(pending, desc="Mining TRAIN images only", unit="image", dynamic_ncols=True):
        gray = load_image(row["image_path"]).astype(np.float32) / 255
        target = load_target(row["mask_path"])
        pred = predict_probabilities(model, gray, parent_cfg, device).argmax(0)
        wrong = pred != target
        hard = wrong & np.isin(target, [1, 2])
        fallback = not hard.any()
        if fallback:
            hard = wrong
        file = folder / f"{row['image_id']}.npz"
        temp = file.with_suffix(".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, hard=hard.astype(bool))
        temp.replace(file)
        records[row["image_id"]] = dict(file=file.name, sha256=sha256(file), shape=list(hard.shape),
                                        hard_pixels=int(hard.sum()), fallback_all_errors=fallback)
        write_json(marker, records)
    write_json(folder / "completed.json", dict(**expected, num_images=len(records)))
    return folder


def verify_mining(cfg):
    folder = Path(cfg["refinement"]["mining_folder"])
    train, _ = select_rows(cfg, "official")
    ids = {r["image_id"] for r in train}
    request = read_json(folder / "request.json")
    records = read_json(folder / "manifest.json")
    if set(request["train_ids"]) != ids or set(records) != ids or request["validation_images_used"]:
        raise ValueError("Mining must contain exactly the official training images")
    if request["parent_sha256"] != cfg["refinement"]["parent_sha256"] or request["code_sha256"] != sha256(Path(__file__)):
        raise ValueError("Stale mining cache")
    for record in records.values():
        file = folder / record["file"]
        if file.parent != folder or sha256(file) != record["sha256"]:
            raise ValueError("Invalid mining record")
    return records


@lru_cache(maxsize=70)
def load_hard(folder, image_id):
    with np.load(Path(folder) / f"{image_id}.npz", allow_pickle=False) as data:
        return data["hard"].copy()
