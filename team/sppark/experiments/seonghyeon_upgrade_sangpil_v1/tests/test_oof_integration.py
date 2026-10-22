"""Synthetic teacher workflow test; no real dataset training/checkpoint optimization."""

from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from alphase.config import load_config, save_config
from alphase.io import sha256, jsonl_write, jsonl_read, json_write, torch_save
from alphase.contracts import ids_to_rgb
from alphase.cache import cache_metadata, write_probability
import alphase.oof as oof
from alphase.refiner import load_refiner_data


def test_oof_registry_deployment_and_refiner_data(tmp_path, monkeypatch):
    cfg = load_config("configs/oof_train70.yaml")
    data = []
    for i in range(6):
        image = tmp_path / f"{i}.png"
        mask = tmp_path / f"{i}_mask.png"
        Image.fromarray(np.full((9, 13), 100, np.uint8)).save(image)
        Image.fromarray(ids_to_rgb(np.full((9, 13), i % 4, np.uint8))).save(mask)
        data.append(
            {
                "image_id": f"A{i}",
                "group_hint": f"A{i}",
                "split": "train" if i < 4 else "valid",
                "image_path": str(image),
                "mask_path": str(mask),
                "native_hw": [9, 13],
                "image_sha256": sha256(image),
                "mask_sha256": sha256(mask),
            }
        )
    manifest = tmp_path / "dataset.jsonl"
    jsonl_write(manifest, data)
    cfg["paths"]["dataset_manifest"] = str(manifest)
    cfg["oof"].update(
        root=str(tmp_path / "oof"),
        fold_manifest=str(tmp_path / "folds.jsonl"),
        n_folds=2,
        teacher_fixed_updates=2,
    )
    cfg["refiner"].update(
        training_cache=str(tmp_path / "oof/train/manifest.jsonl"),
        validation_cache=str(tmp_path / "oof/valid/manifest.jsonl"),
    )
    teacher = load_config("configs/experiments/E02_scratch.yaml")
    teacher["paths"]["dataset_manifest"] = str(manifest)
    teacher_config = tmp_path / "teacher.yaml"
    save_config(teacher_config, teacher)
    cfg["oof"]["teacher_config"] = str(teacher_config)

    def fake_jobs(jobs, gpus, execute):
        assert execute
        for job in jobs:
            c = load_config(job["args"][2])
            run = (
                Path(c["run"]["output_root"])
                / f'{c["run"]["name"]}_seed{c["run"]["seed"]}'
            )
            c["execution"]["dry_run"] = False
            run.mkdir(parents=True)
            torch_save(
                run / "last.pt",
                {
                    "kind": "alphase_training_v1",
                    "config": c,
                    "training_ids": c["training"]["train_ids"],
                    "checkpoint_selection_ids": [],
                    "parent_checkpoint_sha256": None,
                    "step": 2,
                    "model": {"synthetic_only": torch.zeros(1)},
                },
            )
            json_write(run / "completed.json", {"step": 2})

    def fake_cache(
        c,
        checkpoint,
        images,
        out,
        device,
        train_ids,
        selection_ids,
        lineage_factory=None,
    ):
        out = Path(out)
        records = []
        for image in images:
            meta = cache_metadata(
                image, c, sha256(checkpoint), train_ids, selection_ids
            )
            if lineage_factory:
                meta["lineage"] = lineage_factory(image)
            records.append(
                write_probability(out, meta, np.full((4, 9, 13), 0.25, np.float32))
            )
        jsonl_write(out / "manifest.jsonl", records)
        return out / "manifest.jsonl"

    monkeypatch.setattr(oof, "run_jobs", fake_jobs)
    monkeypatch.setattr(oof, "cache_model", fake_cache)
    result = oof.build_oof(cfg, execute=True)
    assert result["images"] == 4
    train, valid, lineage = load_refiner_data(cfg)
    assert len(train) == 4 and len(valid) == 2
    caches = jsonl_read(cfg["refiner"]["training_cache"])
    caches[0]["lineage"]["teacher_checkpoint_hashes"] = ["unknown_checkpoint"]
    jsonl_write(cfg["refiner"]["training_cache"], caches)
    with pytest.raises(ValueError, match="Unknown teacher"):
        oof.validate_oof(cfg)
