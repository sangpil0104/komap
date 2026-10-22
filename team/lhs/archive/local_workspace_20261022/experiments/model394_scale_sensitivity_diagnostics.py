#!/usr/bin/env python3
"""Post-hoc, result-only robustness checks for the Model394 scale feature."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments/results/model394_image_scale_mask_association.json"
OUTPUT = ROOT / "experiments/results/model394_scale_sensitivity_diagnostics.json"
BOOTSTRAPS = 5_000
SEED = 39_430
CLASSES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
TARGETS = ("Eutectic_Si", "Al3Ni")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def partial_rank_correlation(rows: list[dict], target: str) -> float:
    bins = np.asarray([row["raster_bin"] for row in rows])
    all_bins = sorted(set(bins.tolist()))
    feature = rankdata([row["mean_apparent_wavelength_px"] for row in rows]).astype(float)
    width = rankdata([row[f"{target}_mean_mask_width_q50_px"] for row in rows]).astype(float)
    fraction_index = CLASSES.index(target)
    fraction = rankdata([row["mean_phase_fraction"][fraction_index] for row in rows]).astype(float)
    feature = (feature - feature.mean()) / feature.std()
    width = (width - width.mean()) / width.std()
    fraction = (fraction - fraction.mean()) / fraction.std()
    nuisance = np.column_stack(
        [np.ones(len(rows)), fraction, *[(bins == value).astype(float) for value in all_bins[1:]]]
    )
    feature_residual = feature - nuisance @ np.linalg.lstsq(nuisance, feature, rcond=None)[0]
    width_residual = width - nuisance @ np.linalg.lstsq(nuisance, width, rcond=None)[0]
    return float(
        feature_residual @ width_residual
        / (np.linalg.norm(feature_residual) * np.linalg.norm(width_residual))
    )


def calculate() -> dict:
    input_hash = sha256(INPUT)
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = source["per_group"]
    raster_bins = sorted({row["raster_bin"] for row in rows})
    scale = np.asarray([row["mean_apparent_wavelength_px"] for row in rows], dtype=float)
    near_high = int(np.count_nonzero(scale >= 60.0))
    near_low = int(np.count_nonzero(scale <= 2.75))
    seed_rng = np.random.default_rng(SEED)
    output = {}

    for target in TARGETS:
        width = np.asarray([row[f"{target}_mean_mask_width_q50_px"] for row in rows], dtype=float)
        full_rho = partial_rank_correlation(rows, target)
        strata = {
            value: np.asarray([i for i, row in enumerate(rows) if row["raster_bin"] == value])
            for value in raster_bins
        }
        bootstrap = []
        for _ in range(BOOTSTRAPS):
            indices = np.concatenate(
                [seed_rng.choice(group_indices, len(group_indices), replace=True)
                 for group_indices in strata.values()]
            )
            sample = [rows[int(i)] for i in indices]
            bootstrap.append(partial_rank_correlation(sample, target))
        interval = np.quantile(bootstrap, (0.025, 0.975))
        within = {}
        for value, indices in strata.items():
            within[value] = float(spearmanr(scale[indices], width[indices]).statistic)
        leave_one_bin_out = {}
        for omitted in raster_bins:
            sample = [row for row in rows if row["raster_bin"] != omitted]
            leave_one_bin_out[omitted] = partial_rank_correlation(sample, target)
        output[target] = {
            "full_partial_rank_rho": full_rho,
            "raster_stratified_group_bootstrap_95pct": [float(x) for x in interval],
            "bootstrap_replicates": len(bootstrap),
            "within_bin_unadjusted_spearman": within,
            "leave_one_bin_out_partial_rank_rho": leave_one_bin_out,
        }

    return {
        "model": 394,
        "stage": "posthoc_scale_feature_robustness_diagnostics",
        "status": "exploratory; diagnostics selected after reviewing Model394 results",
        "input_scope": "Model394 group summary JSON only; no new train image, valid/test, OOF, checkpoint, or score read",
        "groups": len(rows),
        "raster_bin_group_counts": {
            value: sum(row["raster_bin"] == value for row in rows) for value in raster_bins
        },
        "feature_boundary_saturation_proxy": {
            "group_mean_scale_q10_q50_q90_px": [float(x) for x in np.quantile(scale, (0.1, 0.5, 0.9))],
            "group_means_ge_60px_of_64px_upper_bound": near_high,
            "group_means_le_2_75px_of_2_5px_lower_bound": near_low,
            "note": "group-mean proximity only; crop-level peak boundary rate was not stored",
        },
        "targets": output,
        "bootstrap": {
            "unit": "specimen group, resampled within fixed raster-size strata",
            "replicates": BOOTSTRAPS,
            "seed": SEED,
            "interval": "percentile 95%",
        },
        "interpretation_guard": "These post-hoc robustness checks do not validate physical scale, correct the exploratory p-value, establish error causation, or authorize model training.",
        "source_model394_result_sha256": input_hash,
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite Model394 sensitivity output: {OUTPUT}")
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
