#!/usr/bin/env python3
"""Rebuild model 182 and search calibration under both mIoU conventions."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from advanced_inference import predict_probabilities as predict_rgb
from aspp768_addition import cleanup_primary
from encoder_structure_search import build_model
from hard_source_search import route_params
from highres1024_search import MATRIX, choose_source
from improve_five_causes import collect_split, decode_mask
from model181_raw_zcdf_aspp import TeamRepresentationASPP, predict_probabilities as predict_rep, score
from prior_representation_search import fit_percentile_stats, percentile_image
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from tent_style_search import adapt_model


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def load_rep(path, device):
    model = TeamRepresentationASPP(pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def make_mixed(item, high, gamma=0.40):
    params = item["params"]
    style_high = (1.0 - params["style_weight"]) * high[0] + params["style_weight"] * high[1]
    base = MATRIX * (1.0 - params["high_weight"] - params["pair_weight"])
    mixed = np.asarray([
        sum(base[c, m] * item["branches"][m][c] for m in range(3))
        + params["high_weight"] * style_high[c]
        + params["pair_weight"] * item["pair"][c]
        for c in range(4)
    ], dtype=np.float32)
    mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
    extra = item["extra"]
    mixed[1] = (1.0 - gamma) * mixed[1] + gamma * extra[1]
    return mixed / np.maximum(mixed.sum(0, keepdims=True), 1e-7)


def prior_terms(item, mixed):
    stats_z, stats_q = [], []
    for class_id in range(4):
        weights = np.maximum(mixed[class_id], 1e-7) ** 2
        zm = float((weights * item["z"]).sum() / weights.sum())
        zs = float(np.sqrt((weights * (item["z"] - zm) ** 2).sum() / weights.sum()))
        qm = float((weights * item["q"]).sum() / weights.sum())
        qs = float(np.sqrt((weights * (item["q"] - qm) ** 2).sum() / weights.sum()))
        stats_z.append((zm, max(zs, 0.15)))
        stats_q.append((qm, max(qs, 0.02)))
    prior_z = gaussian_prior(item["z"], item["prior_z_stats"])
    prior_q = gaussian_prior(item["q"], item["prior_q_stats"])
    adaptive_z = gaussian_prior(item["z"], stats_z)
    adaptive_q = gaussian_prior(item["q"], stats_q)
    base_prior = 0.85 * prior_z + 0.15 * prior_q
    delta = 0.85 * (adaptive_z[0] - prior_z[0]) + 0.15 * (adaptive_q[0] - prior_q[0])
    return base_prior, delta


def prepare_smoothing_cache(item):
    """Cache only the Primary-Si terms used by Gaussian logit smoothing."""
    item["smooth_components"] = {}
    for smooth_sigma in (0.25, 0.50, 0.75):
        item["smooth_components"][smooth_sigma] = (
            gaussian_filter(item["log_mixed"][1], sigma=smooth_sigma, mode="reflect"),
            gaussian_filter(0.09 * item["base_prior"][1], sigma=smooth_sigma, mode="reflect"),
        )


def predict(item, prior, al_delta, eut_delta, primary_bias, sigma, cleanup_threshold,
            prior_scale=1.0, beta_scale=1.0):
    params = item["params"]
    # Keep the exact model-173/180 prior strength; prior_scale searches around
    # the established 0.09 coefficient rather than replacing it.
    logits = item["log_mixed"] + 0.09 * prior
    logits[1] += primary_bias
    logits[2] += params["al_bias"] + al_delta
    logits[3] += params["eut_bias"] + eut_delta
    if sigma > 0:
        # The Gaussian operator is linear.  Reusing the three precomputed
        # components avoids applying it to a full image for every setting.
        smoothed = item["smooth_components"][float(sigma)]
        logits[1] = (
            smoothed[0]
            + prior_scale * smoothed[1]
            + primary_bias
        )
    prediction = logits.argmax(0).astype(np.uint8)
    if cleanup_threshold >= 0:
        prediction = cleanup_primary(prediction, logits, cleanup_threshold)
    if float((prediction == 1).mean()) <= 0.01:
        mask = prediction == 1
        alternative = logits.copy()
        alternative[1] = -np.inf
        prediction[mask] = alternative[:, mask].argmax(0).astype(np.uint8)
    return prediction


def evaluate(entries, setting):
    predictions, targets = [], []
    prior_scale = setting["prior_scale"]
    beta_scale = setting["beta_scale"]
    for item in entries:
        # The adaptive prior correction is defined for Al3Ni only (class 0),
        # matching model 182.  Applying it to all four classes changes the
        # decision boundary and invalidates the comparison.
        prior = prior_scale * item["base_prior"].copy()
        prior[0] += (prior_scale * beta_scale * item["params"]["beta"]
                     * item["delta"])
        predictions.append(predict(item, prior, setting["al_delta"], setting["eut_delta"],
                                   setting["primary_bias"], setting["sigma"],
                                   setting["cleanup_threshold"], prior_scale, beta_scale))
        targets.append(item["target"])
    return score(predictions, targets)


def cache_entries(args, device):
    entries_path = Path(args.output_json).parent / "entries.pkl"
    if entries_path.exists():
        with entries_path.open("rb") as handle:
            entries = pickle.load(handle)
        for item in entries:
            prepare_smoothing_cache(item)
        print(f"loaded route cache {entries_path} ({len(entries)} images)", flush=True)
        return entries
    data = Path(args.data_dir)
    amp = device.type == "cuda"
    global_z, global_q = fit_global_stats(data), fit_percentile_stats(data)
    train_stats = []
    for image_path, _ in collect_split(data, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([value[0] for value in train_stats]))
    train_std = float(np.mean([value[1] for value in train_stats]))
    pair = load_aspp(args.pair_checkpoint, device)
    rep = load_rep(args.representation_checkpoint, device)
    source_paths = {"restart51": args.source_restart51, "model109": args.source_model109,
                    "seed456": args.source_seed456, "pairwise": args.source_pairwise,
                    "ema": args.source_ema}
    source_args = argparse.Namespace(**{f"source_{key}": value for key, value in source_paths.items()})
    entries = []
    for image_path, mask_path in collect_split(data, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gray = raw[..., 0].astype(np.float32)
        mean, std = float(gray.mean()), float(gray.std())
        source = choose_source(mean, source_args)
        pair_value = predict_rgb(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(source, raw, device, amp, 5, 1e-5)
        style = transform_image(raw, "standardize", train_mean, train_std)
        high768 = (predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian"),
                   predict_array(adapted, style, 768, 576, device, amp, 1, "d4", "gaussian"))
        high1024 = (predict_array(adapted, raw, 1024, 768, device, amp, 1, "d4", "gaussian"),
                    predict_array(adapted, style, 1024, 768, device, amp, 1, "d4", "gaussian"))
        high = tuple((1.0 - args.alpha_1024) * high768[index]
                     + args.alpha_1024 * high1024[index] for index in (0, 1))
        item = {
            "branches": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
            "pair": pair_value.astype(np.float32),
            "extra": predict_rep(rep, image_path, 512, 384, device, amp, 2),
            "z": (gray - mean) / max(std, 1e-5), "q": percentile_image(gray),
            "prior_z_stats": global_z, "prior_q_stats": global_q,
            "params": route_params(mean, std),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
        }
        item["mixed"] = make_mixed(item, high, gamma=0.40)
        item["base_prior"], item["delta"] = prior_terms(item, item["mixed"])
        item["log_mixed"] = np.log(np.maximum(item["mixed"], 1e-7)).astype(np.float32)
        # Precompute terms needed by the linear Gaussian smoothing search.
        prepare_smoothing_cache(item)
        entries.append(item)
        del adapted, high768, high1024, high
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"cached {image_path.name} source={source}", flush=True)
    entries_path.parent.mkdir(parents=True, exist_ok=True)
    with entries_path.open("wb") as handle:
        pickle.dump(entries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved route cache {entries_path}", flush=True)
    return entries


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = cache_entries(args, device)
    records = []
    coarse = itertools.product(
        (-0.20, -0.10, 0.0, 0.10, 0.20),
        (-0.40, -0.20, 0.0, 0.20, 0.40),
        (-0.10, 0.0, 0.10),
    )
    for al_delta, eut_delta, primary_bias in coarse:
        setting = {"stage": "coarse", "prior_scale": 1.0, "beta_scale": 1.0,
                   "al_delta": al_delta, "eut_delta": eut_delta,
                   "primary_bias": primary_bias, "sigma": 0.50,
                   # Negative means no connected-component pass.  It keeps
                   # the broad calibration search cheap; cleanup is tested
                   # only on the strongest candidates below.
                   "cleanup_threshold": -1}
        records.append({**setting, **evaluate(entries, setting)})
    # Keep the search focused: the route cache already contains the expensive
    # neural predictions, and the top few coarse settings are enough to tune
    # the small calibration terms without spending hours on dominated points.
    top = sorted(records, key=lambda row: row["image_mean_miou"], reverse=True)[:4]
    top += sorted(records, key=lambda row: row["image_mean_miou_exclude"], reverse=True)[:4]
    top += sorted(records, key=lambda row: row["global_miou"], reverse=True)[:4]
    seen = set()
    for parent in top:
        key = tuple(parent[name] for name in ("al_delta", "eut_delta", "primary_bias"))
        if key in seen:
            continue
        seen.add(key)
        for prior_scale, beta_scale, sigma in itertools.product(
                (0.90, 1.00, 1.10), (0.75, 1.00, 1.25),
                (0.25, 0.50, 0.75)):
            setting = {"stage": "refine", "prior_scale": prior_scale,
                       "beta_scale": beta_scale, "al_delta": parent["al_delta"],
                       "eut_delta": parent["eut_delta"],
                       "primary_bias": parent["primary_bias"], "sigma": sigma,
                       "cleanup_threshold": -1}
            records.append({**setting, **evaluate(entries, setting)})

    # Connected-component cleanup is useful for tiny Primary islands but is
    # much more expensive than the calibration arithmetic.  Evaluate it only
    # for the strongest settings under each metric, preserving all metrics in
    # the final ranking without spending minutes on dominated combinations.
    cleanup_parents = []
    cleanup_parents += sorted(records, key=lambda row: row["image_mean_miou"], reverse=True)[:6]
    cleanup_parents += sorted(records, key=lambda row: row["image_mean_miou_exclude"], reverse=True)[:6]
    cleanup_parents += sorted(records, key=lambda row: row["global_miou"], reverse=True)[:6]
    seen_settings = set()
    for parent in cleanup_parents:
        key = tuple(parent[name] for name in (
            "prior_scale", "beta_scale", "al_delta", "eut_delta",
            "primary_bias", "sigma"))
        if key in seen_settings:
            continue
        seen_settings.add(key)
        for cleanup_threshold in (64, 128, 256):
            setting = {**{name: parent[name] for name in (
                "prior_scale", "beta_scale", "al_delta", "eut_delta",
                "primary_bias", "sigma")},
                       "stage": "cleanup", "cleanup_threshold": cleanup_threshold}
            records.append({**setting, **evaluate(entries, setting)})
    best_image = max(records, key=lambda row: row["image_mean_miou"])
    best_exclude = max(records, key=lambda row: row["image_mean_miou_exclude"])
    best_global = max(records, key=lambda row: row["global_miou"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best_image": best_image, "best_exclude": best_exclude,
                                  "best_global": best_global, "records": records},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_image": best_image, "best_exclude": best_exclude,
                      "best_global": best_global}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--representation-checkpoint", required=True)
    parser.add_argument("--pair-checkpoint", required=True)
    parser.add_argument("--source-restart51", required=True)
    parser.add_argument("--source-model109", required=True)
    parser.add_argument("--source-seed456", required=True)
    parser.add_argument("--source-pairwise", required=True)
    parser.add_argument("--source-ema", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--alpha-1024", type=float, default=0.20)
    main(parser.parse_args())
