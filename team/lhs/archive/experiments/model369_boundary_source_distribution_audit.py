#!/usr/bin/env python3
"""Label-free preflight for transferring a B4 boundary correction to Model360.

This compares raw predicted phase occupancy and Al-vs-Al3Ni decisions only.
It deliberately does not open validation masks and does not compute mIoU.
The Model360-side Al3Ni score is approximated by the Model322/326 mean; this
is a source-mismatch diagnostic, not an exact Model360 reconstruction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARCHIVES = {
    "322": "model322_team_b4_group_oof/valid_ensemble/probabilities.npz",
    "326": "model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz",
    "329": "model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz",
    "336": "model336_hrnet_best/valid_eval/valid_probabilities.npz",
    "347": "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz",
    "351": "model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz",
}
EPS = 1e-7


def entropy(probability: np.ndarray) -> np.ndarray:
    return -(probability * np.log(np.maximum(probability, EPS))).sum(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    paths = {key: args.root / "outputs" / value for key, value in ARCHIVES.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    archives = {key: np.load(path) for key, path in paths.items()}
    try:
        names = sorted(archives["329"].files)
        if len(names) != 20:
            raise ValueError(f"expected 20 validation probability keys, got {len(names)}")
        expected = set(names)
        for key, archive in archives.items():
            if set(archive.files) != expected:
                raise ValueError(f"Model{key} probability keys do not match Model329")

        rows = []
        for index, name in enumerate(names):
            p = {key: np.asarray(archive[name], dtype=np.float32)
                 for key, archive in archives.items()}
            shape = p["329"].shape
            if shape[0] != 4 or any(v.shape != shape for v in p.values()):
                raise ValueError(f"4-class probability shape mismatch for {name}")
            if any(not np.allclose(v.sum(0), 1.0, atol=3e-3) for v in p.values()):
                raise ValueError(f"unnormalized probability map in {name}")

            # Raw archives use team order [Al, Al3Ni, Primary, Eutectic].
            p336_gated = p["336"] + (entropy(p["351"]) < entropy(p["336"]))[None] * (
                p["351"] - p["336"])
            # Model360 uses gamma336(Al)=1, then gamma347(Al)=0.25.
            al360_proxy = 0.75 * p336_gated[0] + 0.25 * p["347"][0]
            # Al3Ni remains on the B4/Model270 route; 322/326 mean is a proxy.
            al3ni360_proxy = 0.5 * (p["322"][1] + p["326"][1])
            m329 = np.log(np.maximum(p["329"][0], EPS)) - np.log(
                np.maximum(p["329"][1], EPS))
            m360_proxy = np.log(np.maximum(al360_proxy, EPS)) - np.log(
                np.maximum(al3ni360_proxy, EPS))
            winner329 = m329 >= 0
            winner360_proxy = m360_proxy >= 0
            if np.std(m329) > 0 and np.std(m360_proxy) > 0:
                margin_correlation = float(np.corrcoef(m329.ravel(),
                                                       m360_proxy.ravel())[0, 1])
            else:
                margin_correlation = None
            rows.append({
                "image": name,
                "model329_al_area_fraction": float((p["329"].argmax(0) == 0).mean()),
                "model336_al_area_fraction": float((p["336"].argmax(0) == 0).mean()),
                "model336_gated_d4_al_area_fraction": float((p336_gated.argmax(0) == 0).mean()),
                "al360_source_blend_mean_score": float(al360_proxy.mean()),
                "model329_al_al3ni_pair_al_win_fraction": float(winner329.mean()),
                "hrnet_al_vs_b4mean_al3ni_proxy_win_fraction": float(winner360_proxy.mean()),
                "pair_decision_disagreement_fraction": float((winner329 != winner360_proxy).mean()),
                "pair_margin_correlation": margin_correlation,
            })
            print(f"audited probability-only image {index + 1}/{len(names)}: {name}",
                  flush=True)

        differences = np.asarray([
            row["hrnet_al_vs_b4mean_al3ni_proxy_win_fraction"]
            - row["model329_al_al3ni_pair_al_win_fraction"] for row in rows])
        disagreements = np.asarray([row["pair_decision_disagreement_fraction"]
                                    for row in rows])
        correlations = np.asarray([row["pair_margin_correlation"] for row in rows
                                   if row["pair_margin_correlation"] is not None])
        result = {
            "model": 369,
            "audit": "label-free probability-source mismatch preflight; not an mIoU evaluation",
            "uses_validation_masks": False,
            "class_order": ["Al", "Al3Ni", "Primary Si", "Eutectic Si"],
            "model360_proxy": "Al=0.75*gated336+0.25*347; Al3Ni=mean(322,326); excludes Model270, priors, and spatial refinements",
            "image_count": len(names),
            "median_pairwise_Al_win_fraction_difference": float(np.median(differences)),
            "images_proxy_has_less_Al_than_model329": int((differences < 0).sum()),
            "median_pairwise_decision_disagreement_fraction": float(np.median(disagreements)),
            "mean_pairwise_decision_disagreement_fraction": float(disagreements.mean()),
            "median_pair_margin_correlation": float(np.median(correlations)),
            "mean_pair_margin_correlation": float(correlations.mean()),
            "per_image": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({key: result[key] for key in (
            "median_pairwise_Al_win_fraction_difference",
            "images_proxy_has_less_Al_than_model329",
            "median_pairwise_decision_disagreement_fraction",
            "mean_pairwise_decision_disagreement_fraction",
            "median_pair_margin_correlation",
            "mean_pair_margin_correlation")}, indent=2), flush=True)
    finally:
        for archive in archives.values():
            archive.close()


if __name__ == "__main__":
    main()
