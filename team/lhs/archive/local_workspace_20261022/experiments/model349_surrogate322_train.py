#!/usr/bin/env python3
"""Train-only OOF linear surrogate for Model322 B4 probabilities.

This learns to approximate source 322 from the already-required 326 and 329
probability maps. It never reads train/valid segmentation masks. Four-fold
cross-fitting measures probability reconstruction before one fixed valid use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def prob_paths(root: Path, split: str) -> dict[str, Path]:
    names = {
        "322": "model322_team_b4_group_oof",
        "326": "model326_hybrid_fullstats_group_oof",
        "329": "model329_fullstats_width4000_uniform_group_oof",
    }
    suffix = "train_oof_probs.npz" if split == "train" else "valid_ensemble/probabilities.npz"
    return {key: root / "outputs" / folder / suffix for key, folder in names.items()}


def image_arrays(archives, name: str):
    arrays = {key: archives[key][name].astype(np.float32) for key in archives}
    shape = arrays["322"].shape
    if len(shape) != 3 or shape[0] != 4 or any(a.shape != shape for a in arrays.values()):
        raise ValueError(f"probability shape mismatch for {name}")
    return arrays


def feature_matrix(arrays, pixels=None):
    count = arrays["326"].shape[1] * arrays["326"].shape[2]
    indexes = np.arange(count) if pixels is None else pixels
    channels = [arrays[source].reshape(4, -1)[:, indexes].T
                for source in ("326", "329")]
    return np.column_stack((channels[0], channels[1],
                            np.ones(len(indexes), dtype=np.float32)))


def sample_indexes(target: np.ndarray, n_uniform: int, n_per_class: int,
                   rng: np.random.Generator) -> np.ndarray:
    flat = target.reshape(4, -1)
    pixels = flat.shape[1]
    chosen = [rng.integers(0, pixels, size=n_uniform)]
    labels = flat.argmax(0)
    for class_id in range(4):
        available = np.flatnonzero(labels == class_id)
        if len(available):
            chosen.append(rng.choice(available, n_per_class, replace=True))
    return np.concatenate(chosen)


def fit(archives, names: list[str], ridge: float, rng: np.random.Generator):
    xx = np.zeros((9, 9), dtype=np.float64)
    xy = np.zeros((9, 4), dtype=np.float64)
    samples = 0
    for name in names:
        arrays = image_arrays(archives, name)
        indexes = sample_indexes(arrays["322"], 5000, 1250, rng)
        x = feature_matrix(arrays, indexes).astype(np.float64)
        y = arrays["322"].reshape(4, -1)[:, indexes].T.astype(np.float64)
        xx += x.T @ x
        xy += x.T @ y
        samples += len(indexes)
    xx /= samples
    xy /= samples
    penalty = np.diag([ridge] * 8 + [0.0])
    return np.linalg.solve(xx + penalty, xy), samples


def predict(arrays, coefficients):
    shape = arrays["326"].shape
    x = feature_matrix(arrays)
    y = (x @ coefficients).T.reshape(shape)
    np.clip(y, 1e-6, 1.0, out=y)
    y /= y.sum(axis=0, keepdims=True)
    return y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=0.001)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.ridge < 0:
        raise ValueError("ridge must be nonnegative")
    train = {key: np.load(path) for key, path in prob_paths(args.root, "train").items()}
    names = sorted(train["322"].files)
    if len(names) != 70 or any(set(archive.files) != set(names) for archive in train.values()):
        raise ValueError("expected 70 paired train OOF probability maps")
    fold_names = {}
    for fold in range(1, 5):
        path = (args.root / "outputs" / "model322_team_b4_group_oof" /
                f"fold{fold}_oof/probabilities.npz")
        with np.load(path) as archive:
            fold_names[fold] = set(archive.files)
    if set.union(*fold_names.values()) != set(names) or sum(map(len, fold_names.values())) != 70:
        raise ValueError("OOF fold image partition mismatch")
    rng = np.random.default_rng(349322)
    checks = []
    for fold in range(1, 5):
        train_names = [name for name in names if name not in fold_names[fold]]
        coefficients, samples = fit(train, train_names, args.ridge, rng)
        mae_surrogate, mae_326 = [], []
        for name in sorted(fold_names[fold]):
            arrays = image_arrays(train, name)
            pred = predict(arrays, coefficients)
            mae_surrogate.append(float(np.abs(pred - arrays["322"]).mean()))
            mae_326.append(float(np.abs(arrays["326"] - arrays["322"]).mean()))
        checks.append({"fold": fold, "images": len(fold_names[fold]),
                       "fit_samples": samples,
                       "surrogate_mae": float(np.mean(mae_surrogate)),
                       "direct_326_mae": float(np.mean(mae_326))})
        print(f"fold {fold}: surrogate MAE={checks[-1]['surrogate_mae']:.6f}, "
              f"326 MAE={checks[-1]['direct_326_mae']:.6f}", flush=True)
    surrogate_mae = sum(row["images"] * row["surrogate_mae"] for row in checks) / 70
    direct_mae = sum(row["images"] * row["direct_326_mae"] for row in checks) / 70
    gate = (surrogate_mae < direct_mae and
            sum(row["surrogate_mae"] < row["direct_326_mae"] for row in checks) >= 3)
    args.output_dir.mkdir(parents=True)
    if not gate:
        (args.output_dir / "fit_summary.json").write_text(json.dumps({
            "method": "train-OOF ridge linear regression, input=326+329, target=322",
            "train_images": 70, "ridge": args.ridge,
            "crossfit_probability_mae": checks,
            "crossfit_mean_mae": {"surrogate": surrogate_mae, "direct_326": direct_mae},
            "preregistered_probability_gate_pass": False,
            "valid_read": False,
        }, indent=2) + "\n", encoding="utf-8")
        print("OOF reconstruction gate failed; valid probabilities not read", flush=True)
        return
    coefficients, samples = fit(train, names, args.ridge, rng)
    valid = {key: np.load(path) for key, path in prob_paths(args.root, "valid").items()}
    valid_names = sorted(valid["322"].files)
    if len(valid_names) != 20 or any(set(archive.files) != set(valid_names)
                                   for archive in valid.values()):
        raise ValueError("expected 20 paired valid probability maps")
    output_probs = {}
    for name in valid_names:
        arrays = image_arrays(valid, name)
        output_probs[name] = predict(arrays, coefficients).astype(np.float16)
        print(f"valid source surrogate: {name}", flush=True)
    np.savez_compressed(args.output_dir / "valid_surrogate322_probabilities.npz", **output_probs)
    result = {
        "method": "train-OOF ridge linear regression, input=326+329, target=322",
        "class_order": ["Al", "Al3Ni", "Primary Si", "Eutectic Si"],
        "train_images": 70, "valid_images": 20,
        "ridge": args.ridge, "fit_samples": samples,
        "crossfit_probability_mae": checks,
        "crossfit_mean_mae": {"surrogate": surrogate_mae, "direct_326": direct_mae},
        "preregistered_probability_gate_pass": True,
        "coefficients": coefficients.tolist(),
        "valid_labels_used": False,
    }
    (args.output_dir / "fit_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
