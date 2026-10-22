from __future__ import annotations

from pathlib import Path
import json
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
CONFIG_PATH = EXPERIMENT_DIR / "configs/default.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_config(override_path: str | Path | None = None) -> dict:
    path = Path(override_path) if override_path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataset_samples(split: str = "train") -> list[dict[str, Any]]:
    """Returns sample records for train (70), valid (20), or test (10)."""
    split_dir = DATASET_DIR / split
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    images = sorted(list(img_dir.glob("*_image.png")))
    samples = []
    for img_path in images:
        token = img_path.name.replace("_image.png", "")
        record: dict[str, Any] = {
            "image_id": token,
            "image_path": str(img_path),
            "split": split,
        }
        if split in ("train", "valid"):
            mask_path = mask_dir / f"{token}_mask.png"
            if not mask_path.is_file():
                # Fallback: check if mask without _mask exists or similar
                mask_cand = list(mask_dir.glob(f"{token}*mask*.png"))
                if mask_cand:
                    mask_path = mask_cand[0]
            record["mask_path"] = str(mask_path)
        else:
            record["mask_path"] = None
        samples.append(record)

    return samples

