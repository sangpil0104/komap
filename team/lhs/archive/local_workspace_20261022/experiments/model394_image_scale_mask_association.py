#!/usr/bin/env python3
"""CPU-only, post-hoc train-image apparent-scale association screen."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "train.zip"
MODEL393_RESULT = ROOT / "experiments/results/model393_synthetic_mask_feasibility.json"
MODEL393_LOCK = ROOT / "experiments/results/model393_synthetic_mask_feasibility_lock.json"
LOCK_PATH = ROOT / "experiments/results/model394_image_scale_mask_association_lock.json"
OUTPUT = ROOT / "experiments/results/model394_image_scale_mask_association.json"
IMAGE_PREFIX = "train/images/"
IMAGE_SUFFIX = "_image.png"
PATCH_SIZE = 320
GAUSSIAN_SIGMA = 16.0
WAVELENGTH_MIN = 2.5
WAVELENGTH_MAX = 64.0
PERMUTATIONS = 10_000
BOOTSTRAPS = 5_000
PERMUTATION_SEED = 39_410
BOOTSTRAP_SEED = 39_420
CLASSES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apparent_wavelength(rgb: np.ndarray) -> float:
    """Median dominant radial wavelength from five fixed native-resolution crops."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB image")
    height, width = rgb.shape[:2]
    if height < PATCH_SIZE or width < PATCH_SIZE:
        raise ValueError(f"image smaller than fixed {PATCH_SIZE}px crop")
    gray = (
        0.2126 * rgb[:, :, 0].astype(np.float32)
        + 0.7152 * rgb[:, :, 1].astype(np.float32)
        + 0.0722 * rgb[:, :, 2].astype(np.float32)
    )
    y_starts = (0, height - PATCH_SIZE, (height - PATCH_SIZE) // 2)
    x_starts = (0, width - PATCH_SIZE, (width - PATCH_SIZE) // 2)
    locations = (
        (y_starts[0], x_starts[0]),
        (y_starts[0], x_starts[1]),
        (y_starts[1], x_starts[0]),
        (y_starts[1], x_starts[1]),
        (y_starts[2], x_starts[2]),
    )
    window = np.outer(np.hanning(PATCH_SIZE), np.hanning(PATCH_SIZE)).astype(np.float32)
    fy = np.fft.fftfreq(PATCH_SIZE)
    fx = np.fft.rfftfreq(PATCH_SIZE)
    radial_frequency = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    radial_bin = np.rint(radial_frequency * PATCH_SIZE).astype(np.int32)
    maximum_bin = int(radial_bin.max())
    wavelengths = []

    for top, left in locations:
        patch = gray[top : top + PATCH_SIZE, left : left + PATCH_SIZE]
        low_frequency = cv2.GaussianBlur(
            patch,
            (0, 0),
            sigmaX=GAUSSIAN_SIGMA,
            sigmaY=GAUSSIAN_SIGMA,
            borderType=cv2.BORDER_REFLECT101,
        )
        residual = (patch - low_frequency) * window
        power = np.abs(np.fft.rfft2(residual)) ** 2
        radial_power = np.bincount(
            radial_bin.ravel(), weights=power.ravel(), minlength=maximum_bin + 1
        )
        radial_count = np.bincount(radial_bin.ravel(), minlength=maximum_bin + 1)
        radial_power /= np.maximum(radial_count, 1)
        frequency = np.arange(len(radial_power), dtype=np.float64) / PATCH_SIZE
        valid = (frequency >= 1.0 / WAVELENGTH_MAX) & (
            frequency <= 1.0 / WAVELENGTH_MIN
        )
        valid[0] = False
        if not valid.any():
            raise RuntimeError("empty radial-frequency band")
        peak = int(np.flatnonzero(valid)[np.argmax(radial_power[valid])])
        wavelengths.append(float(1.0 / frequency[peak]))
    return float(np.median(wavelengths))


def rank_scale(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, method="average").astype(np.float64)
    spread = float(ranked.std(ddof=0))
    if spread == 0:
        raise ValueError("constant rank vector")
    return (ranked - ranked.mean()) / spread


def design_matrix(bins: np.ndarray, fractions: np.ndarray, all_bins: list[str]) -> np.ndarray:
    fraction_rank = rank_scale(fractions)
    columns = [np.ones(len(bins), dtype=np.float64), fraction_rank]
    columns.extend((bins == value).astype(np.float64) for value in all_bins[1:])
    return np.column_stack(columns)


def partial_rank_fit(
    feature: np.ndarray,
    width: np.ndarray,
    fraction: np.ndarray,
    bins: np.ndarray,
    all_bins: list[str],
) -> dict:
    x = rank_scale(feature)
    y = rank_scale(width)
    nuisance = design_matrix(bins, fraction, all_bins)
    x_residual = x - nuisance @ np.linalg.lstsq(nuisance, x, rcond=None)[0]
    y_reduced_beta = np.linalg.lstsq(nuisance, y, rcond=None)[0]
    y_fitted = nuisance @ y_reduced_beta
    y_residual = y - y_fitted
    denominator = float(np.linalg.norm(x_residual) * np.linalg.norm(y_residual))
    if denominator == 0:
        raise ValueError("partial-rank residual is constant")
    partial_rho = float(x_residual @ y_residual / denominator)

    full = np.column_stack((nuisance, x))
    beta = np.linalg.lstsq(full, y, rcond=None)[0]
    residual = y - full @ beta
    degrees = max(1, len(y) - np.linalg.matrix_rank(full))
    sigma2 = float(residual @ residual / degrees)
    covariance = np.linalg.pinv(full.T @ full)
    standard_error = float(np.sqrt(max(0.0, sigma2 * covariance[-1, -1])))
    t_value = float(beta[-1] / standard_error) if standard_error > 0 else float("inf")
    return {
        "partial_rank_rho": partial_rho,
        "slope": float(beta[-1]),
        "t_statistic": t_value,
        "reduced_fitted": y_fitted,
        "reduced_residual": y_residual,
    }


def freedman_lane_pvalue(
    observed_t: float,
    fitted: np.ndarray,
    residual: np.ndarray,
    feature: np.ndarray,
    width: np.ndarray,
    fraction: np.ndarray,
    bins: np.ndarray,
    all_bins: list[str],
) -> float:
    rng = np.random.default_rng(PERMUTATION_SEED)
    members = [np.flatnonzero(bins == value) for value in all_bins]
    exceed = 0
    for _ in range(PERMUTATIONS):
        permuted_residual = residual.copy()
        for indices in members:
            permuted_residual[indices] = rng.permutation(residual[indices])
        permuted_width_ranks = fitted + permuted_residual
        # Re-rank the permuted response before applying the predeclared rank model.
        permuted_width = rankdata(permuted_width_ranks, method="average")
        fit = partial_rank_fit(feature, permuted_width, fraction, bins, all_bins)
        exceed += fit["t_statistic"] >= observed_t
    return float((exceed + 1) / (PERMUTATIONS + 1))


def grouped_bootstrap_interval(
    feature: np.ndarray,
    width: np.ndarray,
    fraction: np.ndarray,
    bins: np.ndarray,
    all_bins: list[str],
) -> tuple[list[float], int]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    n = len(feature)
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, n, n)
        try:
            result = partial_rank_fit(
                feature[indices], width[indices], fraction[indices], bins[indices], all_bins
            )
        except ValueError:
            continue
        values.append(result["partial_rank_rho"])
    if not values:
        raise RuntimeError("no valid specimen-group bootstrap replicates")
    interval = np.quantile(values, (0.025, 0.975))
    return [float(x) for x in interval], len(values)


def verify_lock(archive_path: Path) -> dict:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError(f"Model394 lock missing: {LOCK_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("train_zip_sha256") != sha256(archive_path):
        raise ValueError("train.zip does not match the Model394 pre-analysis lock")
    if lock.get("model393_result_sha256") != sha256(MODEL393_RESULT):
        raise ValueError("Model393 result differs from the Model394 lock")
    if lock.get("model393_lock_sha256") != sha256(MODEL393_LOCK):
        raise ValueError("Model393 receipt differs from the Model394 lock")
    for relative, expected in lock["sources_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Model394 source hash mismatch: {relative}")
    return lock


def run() -> dict:
    lock = verify_lock(ARCHIVE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite Model394 output: {OUTPUT}")
    source = json.loads(MODEL393_RESULT.read_text(encoding="utf-8"))
    source_rows = {row["image_id"]: row for row in source["per_image"]}
    if len(source_rows) != 70:
        raise ValueError(f"expected 70 Model393 image rows, found {len(source_rows)}")

    with zipfile.ZipFile(ARCHIVE) as archive:
        members = archive.namelist()
        image_members = [
            name
            for name in members
            if name.startswith(IMAGE_PREFIX) and name.endswith(IMAGE_SUFFIX)
        ]
        if len(image_members) != 70:
            raise ValueError(f"expected 70 train images, found {len(image_members)}")
        ids = [Path(name).name.removesuffix(IMAGE_SUFFIX) for name in image_members]
        if len(set(ids)) != 70 or set(ids) != set(source_rows):
            raise ValueError("train image IDs do not exactly match locked Model393 mask IDs")

        grouped: dict[str, list[dict]] = defaultdict(list)
        image_feature = {}
        for member, image_id in sorted(zip(image_members, ids), key=lambda pair: pair[1]):
            row = source_rows[image_id]
            with Image.open(BytesIO(archive.read(member))) as source_image:
                rgb = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
            if (rgb.shape[1], rgb.shape[0]) != (row["width"], row["height"]):
                raise ValueError(f"image/mask resolution mismatch for {image_id}")
            feature = apparent_wavelength(rgb)
            image_feature[image_id] = feature
            grouped[row["specimen_group"]].append({"row": row, "feature": feature})

    if len(grouped) != 59:
        raise ValueError(f"expected 59 specimen groups, found {len(grouped)}")

    group_rows = []
    for group, items in sorted(grouped.items()):
        raster_bins = {f"{x['row']['width']}x{x['row']['height']}" for x in items}
        if len(raster_bins) != 1:
            raise ValueError(f"specimen group {group} spans multiple raster bins")
        fractions = np.mean([item["row"]["phase_fraction"] for item in items], axis=0)
        record = {
            "specimen_group": group,
            "raster_bin": next(iter(raster_bins)),
            "images": len(items),
            "mean_apparent_wavelength_px": float(np.mean([x["feature"] for x in items])),
            "mean_phase_fraction": fractions.tolist(),
        }
        for class_name in ("Eutectic_Si", "Al3Ni"):
            q50_values = [
                item["row"]["distance_width_proxy"][class_name]["width_proxy_px_q"][0]
                for item in items
                if item["row"]["distance_width_proxy"][class_name]["width_proxy_px_q"][0]
                is not None
            ]
            if not q50_values:
                raise ValueError(f"{class_name} absent from every image in group {group}")
            record[f"{class_name}_mean_mask_width_q50_px"] = float(np.mean(q50_values))
        group_rows.append(record)

    bins = np.asarray([row["raster_bin"] for row in group_rows])
    all_bins = sorted(set(bins.tolist()))
    if len(all_bins) != 4:
        raise ValueError(f"expected four raster bins, found {len(all_bins)}")
    bin_counts = {value: int(np.count_nonzero(bins == value)) for value in all_bins}
    tests = {}
    for class_name in ("Eutectic_Si", "Al3Ni"):
        feature = np.asarray(
            [row["mean_apparent_wavelength_px"] for row in group_rows], dtype=np.float64
        )
        width = np.asarray(
            [row[f"{class_name}_mean_mask_width_q50_px"] for row in group_rows],
            dtype=np.float64,
        )
        class_index = CLASSES.index(class_name)
        fraction = np.asarray(
            [row["mean_phase_fraction"][class_index] for row in group_rows],
            dtype=np.float64,
        )
        fit = partial_rank_fit(feature, width, fraction, bins, all_bins)
        p_value = freedman_lane_pvalue(
            fit["t_statistic"],
            fit["reduced_fitted"],
            fit["reduced_residual"],
            feature,
            width,
            fraction,
            bins,
            all_bins,
        )
        interval, valid_replicates = grouped_bootstrap_interval(
            feature, width, fraction, bins, all_bins
        )
        tests[class_name] = {
            "role": "primary" if class_name == "Eutectic_Si" else "secondary_descriptive",
            "partial_rank_rho": fit["partial_rank_rho"],
            "slope_on_standardized_ranks": fit["slope"],
            "one_sided_freedman_lane_p": p_value,
            "group_bootstrap_percentile_95pct": interval,
            "bootstrap_valid_replicates": valid_replicates,
            "groups": len(group_rows),
        }

    result = {
        "model": 394,
        "stage": "posthoc_train_only_apparent_scale_association",
        "status": "exploratory; hypothesis selected after inspecting Model393 mask summaries and raster-bin associations",
        "scope": "train.zip image pixels + locked Model393 train-mask summaries only; no valid/test images or labels, OOF predictions, checkpoints, or model scores",
        "estimator": {
            "type": "dominant radial power-spectrum wavelength",
            "gray": "fixed BT.709 luminance",
            "native_resolution": True,
            "patch_px": PATCH_SIZE,
            "patch_locations": "four corners and center",
            "highpass_gaussian_sigma_px": GAUSSIAN_SIGMA,
            "radial_wavelength_band_px": [WAVELENGTH_MIN, WAVELENGTH_MAX],
            "image_summary": "median of five crop peak wavelengths",
        },
        "specimen_groups": len(group_rows),
        "raster_bin_group_counts": bin_counts,
        "tests": tests,
        "per_group": group_rows,
        "interpretation_guard": "Any association is not physical-scale recovery and cannot distinguish residual morphology, acquisition, illumination, or annotation confounding. It does not establish OOF error causation or a segmentation gain; it does not authorize training.",
        "model394_lock_sha256": sha256(LOCK_PATH),
        "model393_result_sha256": lock["model393_result_sha256"],
        "source_train_zip_sha256": lock["train_zip_sha256"],
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
    temporary.replace(OUTPUT)
    print(json.dumps({key: value for key, value in result.items() if key != "per_group"}, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    run()


if __name__ == "__main__":
    main()
