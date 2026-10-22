#!/usr/bin/env python3
"""Post-hoc group-level association of mask width proxies with raster diagonal."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments/results/model393_synthetic_mask_feasibility.json"
LOCK = ROOT / "experiments/results/model393_synthetic_mask_feasibility_lock.json"
TRAIN_ZIP = ROOT / "train.zip"
OUTPUT = ROOT / "experiments/results/model393_scale_width_resolution_followup.json"
BOOTSTRAPS = 5000
SEED = 393
TARGETS = ("Eutectic_Si", "Al3Ni")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate() -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    if source.get("lock_sha256") != sha256(LOCK):
        raise ValueError("Model393 mask audit does not match its immutable lock")
    if source.get("source_train_zip_sha256") != sha256(TRAIN_ZIP):
        raise ValueError("train.zip differs from the Model393 locked archive")
    if lock.get("train_zip_sha256") != sha256(TRAIN_ZIP):
        raise ValueError("Model393 lock refers to a different train.zip")
    if source.get("images") != 70 or source.get("specimen_groups") != 59:
        raise ValueError("unexpected Model393 train-mask inventory")

    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in source["per_image"]:
        diagonal = float(np.hypot(row["width"], row["height"]))
        eutectic = float(
            row["distance_width_proxy"]["Eutectic_Si"]["width_proxy_px_q"][0]
        )
        al3ni = float(
            row["distance_width_proxy"]["Al3Ni"]["width_proxy_px_q"][0]
        )
        grouped[row["specimen_group"]].append((diagonal, eutectic, al3ni))
    groups = np.asarray(
        [
            [float(np.mean([image[i] for image in values])) for i in range(3)]
            for _, values in sorted(grouped.items())
        ],
        dtype=np.float64,
    )
    if len(groups) != 59:
        raise ValueError(f"expected 59 groups, found {len(groups)}")

    rng = np.random.default_rng(SEED)
    associations = {}
    for column, target in ((1, TARGETS[0]), (2, TARGETS[1])):
        rho = float(spearmanr(groups[:, 0], groups[:, column]).statistic)
        bootstrap = []
        for _ in range(BOOTSTRAPS):
            sample = groups[rng.integers(0, len(groups), len(groups))]
            if np.ptp(sample[:, 0]) and np.ptp(sample[:, column]):
                bootstrap.append(
                    float(spearmanr(sample[:, 0], sample[:, column]).statistic)
                )
        interval = np.quantile(bootstrap, (0.025, 0.975))
        associations[target] = {
            "spearman_raster_diagonal_vs_group_mean_width_q50": rho,
            "group_bootstrap_percentile_95pct": [float(x) for x in interval],
            "groups": int(len(groups)),
            "valid_bootstrap_replicates": len(bootstrap),
        }

    return {
        "model": 393,
        "stage": "posthoc_resolution_width_association",
        "status": "exploratory; not preregistered before mask morphology was inspected",
        "data_scope": "locked train.zip mask-derived geometry only; no images, valid/test, OOF, checkpoints, or scores read",
        "group_aggregation": "average image-level raster diagonal and width proxy within each specimen group; correlate 59 group means",
        "width_measure": "per-image doubled-distance-transform pixel-width q50 proxy, then group mean",
        "bootstrap": {
            "unit": "specimen group",
            "replicates": BOOTSTRAPS,
            "seed": SEED,
            "interval": "percentile 95%",
        },
        "associations": associations,
        "interpretation_guard": "Association with raster diagonal cannot distinguish acquisition magnification from real morphology or group composition; it does not establish that test-time scale normalization will improve segmentation.",
        "model393_lock_sha256": sha256(LOCK),
        "source_train_zip_sha256": sha256(TRAIN_ZIP),
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite exploratory result: {OUTPUT}")
    result = calculate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
    temporary.replace(OUTPUT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
