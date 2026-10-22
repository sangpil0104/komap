"""Conda komap launcher. No training unless train --execute is explicit."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "adapters"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def verify_source():
    for row in json.loads((ROOT / "source_manifest.json").read_text()):
        if hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Source changed since setup: {row['path']}; preserve this version and make a new experiment")


def prepare_data():
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    from local_result_export import PALETTE
    records = []
    for split, expected in (("train", 70), ("valid", 20)):
        images = sorted((REPO / "dataset" / split / "images").glob("*_image.png"))
        masks = sorted((REPO / "dataset" / split / "masks").glob("*_mask.png"))
        if len(images) != expected or len(masks) != expected:
            raise ValueError(f"{split}: expected {expected} pairs; got {len(images)}/{len(masks)}")
        for image in tqdm(images, desc=f"Check {split}", unit="image"):
            mask = image.parent.parent / "masks" / image.name.replace("_image.png", "_mask.png")
            with Image.open(image) as im, Image.open(mask) as gt:
                if im.size != gt.size:
                    raise ValueError(f"Size mismatch: {image}")
                im.load()
                rgb = np.array(gt.convert("RGB"))
                valid = np.zeros(rgb.shape[:2], dtype=bool)
                for color in PALETTE:
                    valid |= np.all(rgb == color, axis=-1)
                if not valid.all():
                    raise ValueError(f"Unknown mask colors: {mask}")
                if "A" in gt.getbands() and np.any(np.array(gt.getchannel("A")) != 255):
                    raise ValueError(f"Transparent mask: {mask}")
            for p in (image, mask):
                records.append({"path": str(p.relative_to(REPO / "dataset")),
                                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    data_root = ROOT / "datasets"
    data_root.mkdir(exist_ok=True)
    manifest = data_root / "manifest.json"
    if manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored["files"] != records:
            raise ValueError("Dataset changed; create a new experiment rather than replacing prepared data")
        for split, checksum in stored["archives"].items():
            if hashlib.sha256((data_root / f"{split}.zip").read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Prepared archive changed: {split}.zip")
    else:
        checksums = {}
        for split in ("train", "valid"):
            archive = data_root / f"{split}.zip"
            with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as z:
                selected = [r for r in records if r["path"].startswith(split + "/")]
                for row in tqdm(selected, desc=f"Prepare {split}.zip", unit="file"):
                    z.write(REPO / "dataset" / row["path"], row["path"])
            checksums[split] = hashlib.sha256(archive.read_bytes()).hexdigest()
        write_json(manifest, {"files": records, "archives": checksums})
    # Independently check exactly-once OOF group coverage, beyond upstream preflight.
    import re
    all_groups = {re.match(r"(A\d+)_", Path(r["path"]).name).group(1)
                  for r in records if r["path"].startswith("train/images/")}
    folds = json.loads((ROOT / "source/bundle_assets/training/model303_progress.json").read_text())["completed_folds"]
    assert sorted(int(f["fold"]) for f in folds) == [1, 2, 3, 4]
    held_seen = set()
    for fold in folds:
        train, held = set(fold["train_groups"]), set(fold["heldout_groups"])
        if train & held or train | held != all_groups or held & held_seen:
            raise ValueError(f"Invalid group fold: {fold['fold']}")
        held_seen |= held
    if held_seen != all_groups:
        raise ValueError("OOF folds do not cover every specimen")
    return {"train_images": 70, "valid_images": 20, "group_folds": 4, "data": str(data_root)}


def environment():
    import torch
    if Path(sys.prefix).name != "komap" or os.environ.get("CONDA_DEFAULT_ENV") != "komap":
        raise RuntimeError("Use ./run.sh to run in conda komap")
    packages = ["torch", "torchvision", "numpy", "Pillow", "scipy", "tqdm",
                "segmentation-models-pytorch", "timm", "albumentations", "opencv-python-headless"]
    versions = {name: importlib.metadata.version(name) for name in packages}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in conda komap")
    return {"python": sys.executable, "packages": versions, "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def launch(mode, gpu, workers):
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    lock = (outputs / "pipeline.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This experiment already has an active pipeline")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = outputs / "runs" / f"model349_{mode}_{tag}"
    env = os.environ.copy()
    env.update(KOMAP_PYTHON=sys.executable, KOMAP_DATA_DIR=str(ROOT / "datasets"),
               KOMAP_RUN_ROOT=str(run), KOMAP_SETUP_ROOT=str(ROOT),
               KOMAP_RESULTS_ROOT=str(REPO / "results"), KOMAP_WORKERS=str(workers),
               CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = ["bash", str(ROOT / "source/experiments/run_model349_pipeline.sh"), mode]
    print(f"RUN_DIRECTORY={run}\nGPU={gpu}\nCONDA_PYTHON={sys.executable}", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, start_new_session=True, pass_fds=(lock.fileno(),))
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    try:
        code = proc.wait()
        if code:
            raise RuntimeError(f"Pipeline failed ({code}); logs/checkpoints preserved in {run}")
        print(f"COMPLETE: {run}; elapsed={time.monotonic() - started:.1f}s", flush=True)
        return run
    except BaseException:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        raise
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", nargs="?", choices=("check", "train"), default="check")
    p.add_argument("--execute", action="store_true", help="Start actual training (user only)")
    p.add_argument("--gpu", type=int, default=int(os.environ.get("MODEL349_GPU", "0")))
    p.add_argument("--workers", type=int, default=int(os.environ.get("KOMAP_WORKERS", "4")))
    args = p.parse_args()
    if args.execute and args.command != "train":
        p.error("--execute is only valid with train")
    if args.workers < 1:
        p.error("--workers must be positive")
    verify_source()
    info = environment()
    if args.gpu not in range(len(info["gpus"])):
        p.error("--gpu must be a physical GPU index present on this machine")
    info["dataset"] = prepare_data()
    info["training_started"] = args.command == "train" and args.execute
    print(json.dumps(info, indent=2), flush=True)
    if args.command == "train" and not args.execute:
        print("DRY RUN: no training. Start explicitly with ./run.sh train --execute", flush=True)
        return
    run = launch("train" if args.execute else "preflight", args.gpu, args.workers)
    if args.command == "check":
        write_json(ROOT / "reports" / f"preflight_{run.name}.json", info)


if __name__ == "__main__":
    main()
