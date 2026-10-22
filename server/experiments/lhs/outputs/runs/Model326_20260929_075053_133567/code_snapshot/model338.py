"""Execute the recorded Model338 route, without rerunning a validation search."""
from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from runtime import ROOT, SOURCE, DATA, common, pairs, image_key, read_json, export_results

TEAM_TO_MAIN = np.array([1, 2, 0, 3])  # probability channel selection
TEAM_LABEL_TO_MAIN = np.array([2, 0, 1, 3], dtype=np.uint8)
MAIN_LABEL_TO_TEAM = np.array([1, 2, 0, 3], dtype=np.uint8)


def route_functions():
    sys.path.insert(0, str(SOURCE / "experiments/historical_scripts"))
    from model335_exhaustive_b4_route_search import make_mixed
    from model187_schedule_branch_blend import undo_model181_primary_blend
    from model198_fast_branch_search import bilateral_weights, predict
    return make_mixed, undo_model181_primary_blend, bilateral_weights, predict


def artifact_status(config):
    paths = {k: Path(v) if Path(v).is_absolute() else ROOT / v
             for k, v in config["artifacts"].items()}
    if set(paths) != {"entries", "269", "322", "326"}:
        raise ValueError("Model338 requires entries, 269, 322 and 326 artifacts")
    return {"files_present": all(p.is_file() for p in paths.values()),
            "paths": {k: str(p) for k, p in paths.items()},
            "missing": {k: str(p) for k, p in paths.items() if not p.is_file()}}


def validate_probs(value, shape, label):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (4, *shape) or not np.isfinite(a).all():
        raise ValueError(f"Invalid probability dimensions/values: {label}: {a.shape}")
    if a.min() < -1e-6 or a.max() > 1.0001 or not np.allclose(a.sum(0), 1, atol=2e-3):
        raise ValueError(f"Probabilities must be normalized: {label}")
    return a


def load_archive(path, shapes, team_order=False):
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        for raw_key in archive.files:
            key = image_key(raw_key)
            if key not in shapes or key in result:
                raise ValueError(f"Unknown/duplicate image key {raw_key}: {path}")
            a = validate_probs(archive[raw_key], shapes[key], f"{path}:{key}")
            result[key] = a[TEAM_TO_MAIN] if team_order else a
    if set(result) != set(shapes):
        raise ValueError(f"Archive has missing images: {path}: {set(shapes)-set(result)}")
    return result


def fingerprint(target):
    return target.shape, hashlib.sha256(np.asarray(target, dtype=np.uint8).tobytes()).hexdigest()


def align_entries(entries, items):
    """Legacy caches have no IDs: match mask bytes AND image guide, never list order."""
    targets, guides = {}, {}
    for ip, mp in items:
        key = image_key(ip)
        target = TEAM_LABEL_TO_MAIN[common.load_mask(mp)]
        targets.setdefault(fingerprint(target), []).append(key)
        gray = np.asarray(Image.open(ip).convert("RGB"), dtype=np.float32)[..., 0]
        guides[key] = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
    if len(entries) != len(items):
        raise ValueError(f"entries count differs: {len(entries)} versus {len(items)}")
    result = {}
    for item in entries:
        target = np.asarray(item["target"])
        if not np.isin(target, range(4)).all():
            raise ValueError("Invalid target labels in route cache")
        candidates = [key for key in targets.get(fingerprint(target), [])
                      if np.asarray(item["z"]).shape == guides[key].shape
                      and np.allclose(item["z"], guides[key], atol=1e-5, rtol=1e-5)]
        if len(candidates) != 1 or candidates[0] in result:
            raise ValueError("Cannot uniquely align route cache to local images; check dataset/cache provenance")
        key = candidates[0]
        shape = guides[key].shape
        for field in ("mixed", "extra"):
            validate_probs(item[field], shape, f"entries:{key}:{field}")
        q = np.asarray(item["q"])
        if q.shape != shape or not np.isfinite(q).all():
            raise ValueError(f"Invalid percentile guide: {key}")
        for field in ("prior_z_stats", "prior_q_stats"):
            stats = np.asarray(item[field])
            if stats.shape != (4, 2) or not np.isfinite(stats).all() or np.any(stats[:, 1] <= 0):
                raise ValueError(f"Invalid prior stats: {field}:{key}")
        for field in ("beta", "al_bias", "eut_bias"):
            if not np.isfinite(float(item["params"][field])):
                raise ValueError(f"Invalid route parameter: {field}:{key}")
        result[key] = item
    return result


def load_inputs(config, data=DATA):
    status = artifact_status(config)
    if not status["files_present"]:
        missing = "\n".join(f"  {k}: {v}" for k, v in status["missing"].items())
        raise FileNotFoundError("Model338 artifacts are missing (see MISSING_ARTIFACTS.md):\n" + missing)
    items = pairs(data, "valid")
    if len(items) != 20:
        raise ValueError("Expected 20 validation images")
    paths = status["paths"]
    # This is the original research cache format, supplied by the repository owner.
    with open(paths["entries"], "rb") as handle:
        entries = align_entries(pickle.load(handle), items)
    shapes = {k: item["z"].shape for k, item in entries.items()}
    sources = {k: load_archive(paths[k], shapes, team_order=k in ("322", "326"))
               for k in ("269", "322", "326")}
    return items, entries, sources


def infer(config, data=DATA):
    items, entries, archives = load_inputs(config, data)
    make_mixed, undo, weights, predict = route_functions()
    predictions = {}
    for ip, _ in tqdm(items, desc="Model338 fixed route", unit="image"):
        key = image_key(ip)
        # Ground truth is used only for cache identity and final scoring, never prediction.
        item = {k: v for k, v in entries[key].items() if k != "target"}
        sources = {k: archive[key] for k, archive in archives.items()}
        sources["b4mean"] = 0.5 * (sources["322"] + sources["326"])
        mixed = make_mixed({"base": undo(item), "sources": sources}, config["setting"])
        main_pred = predict(item, mixed, config["setting"], weights(item["z"]), exact=True)
        predictions[key] = MAIN_LABEL_TO_TEAM[main_pred]
    return predictions


def evaluate(config_path=ROOT / "configs/model338.json", data=DATA):
    config = read_json(config_path)
    predictions = infer(config, data)
    return export_results("lhs_Model338", predictions, data, {
        "mode": "recorded fixed route (no validation search)", "config": config,
        "source_commit": read_json(ROOT / "model_metadata.json")["source_commit"],
        "upstream_score_is_reference_only": True})
