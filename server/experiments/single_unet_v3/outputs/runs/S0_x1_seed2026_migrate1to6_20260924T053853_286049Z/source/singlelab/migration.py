"""Import a saved run into v2 without changing its checkpoints or source."""
from pathlib import Path
import shutil

import torch

from .common import OUTPUTS, dataset_signature, digest, read_json, snapshot, stamp, write_json
from .model import EfficientUNet


def import_run(source, target_world=6):
    source = Path(source).resolve()
    cfg = read_json(source / "config.json")
    latest = read_json(source / "latest.json")
    checkpoint = source / latest["checkpoint"]
    hashes = read_json(source / "source_hashes.json")
    for relative, expected in hashes.items():
        if digest(source / "source" / relative) != expected:
            raise ValueError(f"Original source snapshot changed: {relative}")
    signature = dataset_signature(cfg, verify=True)
    if signature != read_json(source / "dataset.json"):
        raise ValueError("Original dataset or split changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["update"] != latest["update"] or state["config"] != cfg:
        raise ValueError("Checkpoint does not match run metadata")
    if state["update"] >= cfg["train"]["updates"]:
        raise ValueError("Run is already trained to its planned budget")
    # Validate architecture/state before allocating a continuation run.
    model = EfficientUNet(cfg, pretrained=False)
    model.load_state_dict(state["model"], strict=True)
    source_world, update = state["world_size"], state["update"]
    del model, state
    run = OUTPUTS / "runs" / f"{cfg['experiment']}_x{cfg['scale']}_seed{cfg['seed']}_migrate{source_world}to{target_world}_{stamp()}"
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", cfg)
    write_json(run / "dataset.json", signature)
    snapshot(run)
    if (source / "environment.json").exists():
        shutil.copy2(source / "environment.json", run / "source_environment.json")
    relative = f"checkpoints/imported_update_{update:06d}.pt"
    target = run / relative
    target.parent.mkdir()
    shutil.copy2(checkpoint, target)
    checksum = digest(checkpoint)
    if digest(target) != checksum:
        raise ValueError("Checkpoint copy hash mismatch")
    write_json(run / "latest.json", dict(checkpoint=relative, update=update))
    details = dict(source_run=str(source), source_checkpoint=str(checkpoint), checkpoint_sha256=checksum,
                   source_world_size=source_world, target_world_size=target_world, imported_update=update,
                   effective_batch=cfg["train"]["effective_batch"], remaining_updates=cfg["train"]["updates"] - update,
                   source_hashes=hashes, source_elapsed_time_in_checkpoint=True,
                   stochastic_note="Weights, optimizer, update, sample index and LR preserved; changing GPU count changes RNG streams and BN execution, so not bitwise identical")
    write_json(run / "migration.json", details)
    if (source / "training.jsonl").exists():
        shutil.copy2(source / "training.jsonl", run / "training_before_migration.jsonl")
    write_json(OUTPUTS / "migration" / "ready.json", dict(run=str(run), **details))
    return run
