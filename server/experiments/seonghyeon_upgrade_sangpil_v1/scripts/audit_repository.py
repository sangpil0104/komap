"""Read-only source/data/checkpoint audit; writes only in this experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.metadata as md
import inspect
import json
import platform
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp
from alphase.io import ROOT, REPO, sha256, digest, ids_hash, jsonl_write, json_write
from alphase.contracts import CLASS_NAMES, rgb_to_ids

rows = []
for split, count in [("train", 70), ("valid", 20), ("test", 10)]:
    paths = sorted((REPO / "dataset" / split / "images").glob("*_image.png"))
    assert len(paths) == count, (split, len(paths))
    for p in tqdm(paths, desc=f"audit {split}"):
        image = np.array(Image.open(p))
        assert image.ndim == 3 and image.shape[2] in [3, 4]
        assert np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(
            image[:, :, 0], image[:, :, 2]
        )
        if image.shape[2] == 4:
            assert (image[:, :, 3] == 255).all()
        mask = p.parent.parent / "masks" / p.name.replace("_image.png", "_mask.png")
        r = dict(
            image_id=p.stem.removesuffix("_image"),
            split=split,
            image_path=str(p),
            mask_path=None,
            native_hw=list(image.shape[:2]),
            image_sha256=sha256(p),
            pixel_sha256=digest(image[:, :, 0].tolist()),
            group_hint=p.name.split("_")[0],
            group_hint_is_confirmed_specimen=False,
        )
        if split != "test":
            labels = rgb_to_ids(np.array(Image.open(mask)))
            assert labels.shape == image.shape[:2]
            r.update(
                mask_path=str(mask),
                mask_sha256=sha256(mask),
                class_counts=np.bincount(labels.ravel(), minlength=4).tolist(),
            )
        rows.append(r)
    if split != "test":
        assert (
            len(list((REPO / "dataset" / split / "masks").glob("*_mask.png"))) == count
        )
assert len({r["image_id"] for r in rows}) == 100
assert len({r["pixel_sha256"] for r in rows}) == 100
jsonl_write(ROOT / "manifests/dataset.jsonl", rows)
train_ids = [r["image_id"] for r in rows if r["split"] == "train"]
valid_ids = [r["image_id"] for r in rows if r["split"] == "valid"]
checkpoints = []
for seed in tqdm(range(100), desc="hash 100 checkpoints"):
    result = json.loads(
        (
            REPO / f"results/model_results_0926_2/metrics/seed_{seed:02d}.json"
        ).read_text()
    )
    p = Path(result["checkpoint"])
    args = json.loads((p.parent / "args.json").read_text())
    assert (
        args["seed"] == seed
        and args["arch"] == "unet"
        and args["encoder"] == "efficientnet-b4"
    )
    checkpoints.append(
        dict(
            member_id=f"seed_{seed:02d}",
            seed=seed,
            checkpoint=str(p),
            checkpoint_sha256=sha256(p),
            bytes=p.stat().st_size,
            class_names=list(CLASS_NAMES),
            training_ids=train_ids,
            training_ids_hash=ids_hash(train_ids),
            checkpoint_selection_ids=[],
            checkpoint_selection_ids_hash=ids_hash([]),
            selection_policy="fixed_final_16000_updates; valid monitored, not selected",
            ancestor_training_ids=[],
            ancestor_initialization="ImageNet",
            historical_image_miou=result["image_miou"],
            args=args,
        )
    )
jsonl_write(ROOT / "manifests/checkpoints.jsonl", checkpoints)
model = smp.Unet(
    encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=4
)
state = torch.load(checkpoints[32]["checkpoint"], map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
with torch.no_grad():
    features = model.encoder(torch.zeros(1, 3, 64, 96))
    decoded = model.decoder(features)
source = (
    REPO / "experiments/seonghyeon_seed10_99_v1/outputs/20260926T082111_741695Z/source"
)
packages = {
    name: md.version(name)
    for name in [
        "torch",
        "torchvision",
        "segmentation-models-pytorch",
        "timm",
        "albumentations",
        "opencv-python-headless",
        "numpy",
        "Pillow",
        "tqdm",
        "PyYAML",
        "pytest",
    ]
}
audit = dict(
    python=sys.executable,
    python_version=platform.python_version(),
    packages=packages,
    source=str(source),
    source_hashes={str(p.relative_to(source)): sha256(p) for p in source.rglob("*.py")},
    input_hashes={
        str(p.relative_to(ROOT)): sha256(p)
        for p in (ROOT / "input").rglob("*")
        if p.is_file()
    },
    encoder_channels=model.encoder.out_channels,
    feature_shapes=[list(f.shape) for f in features],
    decoder_output_shape=list(decoded.shape),
    decoder_repr=str(model.decoder),
    head_repr=str(model.segmentation_head),
    dataset_counts={"train": 70, "valid": 20, "test": 10},
    dataset_hash=digest(rows),
    training_ids_hash=ids_hash(train_ids),
    valid_ids_hash=ids_hash(valid_ids),
    shared_group_hints=sorted(
        {r["group_hint"] for r in rows if r["split"] == "train"}
        & {r["group_hint"] for r in rows if r["split"] == "valid"}
    ),
    gpus=[
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(torch.cuda.device_count())
    ],
)
json_write(ROOT / "reports/audit.json", audit)
(ROOT / "reports/repo_audit.md").write_text(
    f"""# Repository audit\n\n- Python: `{sys.executable}`; exact historical environment reused, no package upgrades.\n- Data: train70 / valid20 / test10, exact image/mask pairs, all opaque RGBA grayscale; strict palette validated; no identical grayscale arrays.\n- Checkpoints: 100 existing last.pt files SHA-256 hashed; seed32 strict load passed.\n- Source: `{source}`. source/train.py is the actual entry, NOT vendor/train.py.\n- B4 encoder channels: {model.encoder.out_channels}; decoder channels [256,128,64,32,16], BN, no attention; final 3x3 head16→4.\n- Input: gray/255, scalar mean0.449/std0.226, copied3. OpenCV linear resize fx=fy=4000/W (OpenCV rounding), reflect right/bottom padding to32.\n- Inference: BF16 forward, FP32 softmax, flip-before-rotation D4 in k0..7 order; average8 then restore native with area when enlarged, bilinear otherwise.\n- Historical training: area-weighted sampling, batch2, 1536 crops, 16000 updates; AdamW lr3e-4 wd.01, 300 warmup*cosine, gradient clip1; CE+batch/class Dice(smooth1). BN unchanged, no EMA.\n- Source augmentation: native pre-crop int(1536*2.1/scale), Affine(.7–1.4,±180°,p.7), reflect pad, random crop, D4, brightness/contrast(.3/.4,p.8), gamma60–160(p.5), blur/noise(p.3).\n- Seed32 historical image mIoU: {checkpoints[32]['historical_image_miou']:.15f}. Reproduction still to run.\n- No checkpoint selection on valid: fixed last at16000. Validation observed during development, so scores are development results.\n- Root README describes an older DINO design; current input spec and verified B4 code govern this implementation.\n- GPUs0–5: six RTX4090 24GiB; independent one-GPU jobs preserve batch2. GPUs6/7 excluded.\n- A-prefix overlap across train/valid: {audit['shared_group_hints']}; these are unconfirmed specimen hints, not proof of independent physical groups.\n- Submission PNG naming will follow original stems with _mask.png; official archive requirements are not present, so no claimed competition-ready zip.\n- input/README mentions SHA256SUMS and VALIDATION_REPORT files that are absent; all available input files are hashed in audit.json.\n"""
)
(ROOT / "requirements.lock.txt").write_text(
    "\n".join(f"{k}=={v}" for k, v in packages.items()) + "\n"
)
print("Audit complete:", ROOT / "reports/audit.json")
