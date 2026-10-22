# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import numpy as np

import torch

from advanced_inference import predict_probabilities as predict_rgb

from encoder_structure_search import build_model

from highres1024_search import MATRIX

from model181_raw_zcdf_aspp import TeamRepresentationASPP

from model181_raw_zcdf_aspp import predict_probabilities as predict_rep

from tent_adaptive_prior_search import gaussian_prior

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
