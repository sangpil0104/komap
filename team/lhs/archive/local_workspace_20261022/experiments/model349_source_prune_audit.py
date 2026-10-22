#!/usr/bin/env python3
"""Fixed-route Model349 source-removal diagnostic, not a route search.

Uses the original 20 validation targets only to measure predeclared removals.
The cached Model183 base is still required; this audit does not claim to
compress the models that produced that base cache or estimate test accuracy.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
for path in (HERE / "historical_scripts", HERE):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import model341_model329_joint_route_search as model341  # noqa: E402
from model1_unet_resnet34 import decode_mask  # noqa: E402
from model323_model270_branch_route_probe import canonical_image_key  # noqa: E402


def load_rows(root: Path, *, compact: bool = False, entries_path: Path | None = None):
    outputs = root / "outputs"
    source_paths = {
        "269": (outputs / "model269_fullstats_1x_extended/valid_probs.npz", False),
        "322": (outputs / "model322_team_b4_group_oof/valid_ensemble/probabilities.npz", True),
        "326": (outputs / "model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz", True),
        "329": (outputs / "model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz", True),
        "336": (outputs / "model336_hrnet_best/valid_eval/valid_probabilities.npz", True),
        "347": (outputs / "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz", True),
    }
    if compact:
        source_paths = {key: value for key, value in source_paths.items()
                        if key not in ("322", "347")}
    archives = {}
    for source, (path, team_order) in source_paths.items():
        _, archives[source] = model341.route.load_archive(path, team_order=team_order)
    names = sorted(next(iter(archives.values())))
    assert len(names) == 20
    assert all(set(values) == set(names) for values in archives.values())
    mask_paths = sorted((root / "data/valid/masks").glob("*_mask.png"))
    assert [canonical_image_key(path.name[:-len("_mask.png")]) for path in mask_paths] == names
    with (entries_path or outputs / "model183_metric_calibration/entries.pkl").open("rb") as stream:
        entries = pickle.load(stream)
    assert len(entries) == len(names)
    rows = []
    for name, item, mask_path in zip(names, entries, mask_paths):
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        assert np.array_equal(item["target"], target), name
        sources = {}
        for source, values in archives.items():
            array = values[name]
            assert array.shape == (4, *target.shape), (name, source, array.shape)
            assert np.allclose(array.sum(axis=0), 1.0, atol=3e-3), (name, source)
            sources[source] = array
        sources["b4mean"] = (sources["326"] if compact else
                             0.5 * (sources["322"] + sources["326"]))
        base = (item["preaux"] if "preaux" in item else
                model341.route.undo_model181_primary_blend(item))
        rows.append({"name": name, "item": item,
                     "base": base,
                     "sources": sources})
    return rows


def setting_variants(base_route: dict, gamma329, gamma336, gamma347):
    baseline = ("full_349", dict(base_route), gamma329, gamma336, gamma347, None)
    yield baseline
    zeros = (0.0,) * 4
    yield ("without_347", dict(base_route), gamma329, gamma336, zeros, None)
    yield ("without_336", dict(base_route), gamma329, zeros, gamma347, None)
    yield ("without_329", dict(base_route), zeros, gamma336, gamma347, None)
    yield ("without_336_347", dict(base_route), gamma329, zeros, zeros, None)
    yield ("without_329_347", dict(base_route), zeros, gamma336, zeros, None)
    yield ("without_329_336_347", dict(base_route), zeros, zeros, zeros, None)
    no_269 = dict(base_route, eutectic_gamma=0.0)
    yield ("without_269", no_269, gamma329, gamma336, gamma347, None)
    yield ("without_269_347", no_269, gamma329, gamma336, zeros, None)
    yield ("without_322", dict(base_route, al_source="326"),
           gamma329, gamma336, gamma347, "326")
    yield ("without_326", dict(base_route, primary_source="322"),
           gamma329, gamma336, gamma347, "322")
    yield ("without_322_347", dict(base_route, al_source="326"),
           gamma329, gamma336, zeros, "326")
    yield ("without_326_347", dict(base_route, primary_source="322"),
           gamma329, gamma336, zeros, "322")


def prediction(row, variant, guide):
    _, route, gamma329, gamma336, gamma347, b4_only = variant
    sources = dict(row["sources"])
    if b4_only is not None:
        sources["b4mean"] = sources[b4_only]
    model_row = {"base": row["base"], "sources": sources}
    mixed = model341.route.make_mixed(model_row, route)
    for source, gammas in (("329", gamma329), ("336", gamma336), ("347", gamma347)):
        for class_id, gamma in enumerate(gammas):
            if gamma:
                mixed = model341.route.replace_classes(
                    mixed, sources[source], (class_id,), gamma)
    return model341.route.predict(row["item"], mixed, route, guide, exact=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entries", type=Path, default=None,
                        help="alternate Model349-ready cache; may contain preaux without extra")
    parser.add_argument("--baseline-tolerance", type=float, default=1e-9,
                        help="absolute mIoU tolerance for a raw-image parity replay")
    parser.add_argument("--compact", action="store_true",
                        help="load only the four source archives needed after removing 322/347")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    outputs = args.root / "outputs"
    result = lambda model: json.loads((outputs / model / "results.json").read_text())
    route = result("model339_joint_gamma_route_search")["best_image_empty_zero"]
    gamma329 = tuple(result("model341_model329_joint_route_search")["best_image_empty_zero"]["gamma329"])
    gamma336 = tuple(result("model346_model336_into_341_route")["raw_best_gamma336"])
    result349 = result("model349_model347_into_346_route")
    gamma347 = tuple(result349["raw_best_gamma347"])
    rows = load_rows(args.root, compact=args.compact, entries_path=args.entries)
    variants = list(setting_variants(route, gamma329, gamma336, gamma347))
    if args.compact:
        variants = [variant for variant in variants if variant[0] == "without_322_347"]
    scores = {variant[0]: [] for variant in variants}
    for index, row in enumerate(rows, 1):
        guide = model341.route.bilateral_weights(row["item"]["z"])
        target = row["item"]["target"].astype(np.int64)
        for variant in variants:
            pred = prediction(row, variant, guide).astype(np.int64)
            confusion = np.bincount((target * 4 + pred).ravel(), minlength=16).reshape(4, 4)
            intersection = np.diag(confusion)
            union = confusion.sum(0) + confusion.sum(1) - intersection
            iou = np.divide(intersection, union, out=np.zeros(4, dtype=float), where=union > 0)
            scores[variant[0]].append(float(iou.mean()))
        print(f"{index}/{len(rows)} {row['name']}", flush=True)
    if args.compact:
        reference = json.loads((outputs / "model349_source_prune_audit/results.json").read_text())
        expected = float(reference["variants"]["without_322_347"]["score"])
        actual = float(np.mean(scores["without_322_347"]))
    else:
        expected = float(result349["raw_best_valid_image_mean_miou_empty0"])
        actual = float(np.mean(scores["full_349"]))
    if abs(actual - expected) > args.baseline_tolerance:
        raise AssertionError(f"baseline mismatch: {actual} != {expected}")
    baseline349 = float(result349["raw_best_valid_image_mean_miou_empty0"])
    report = {"metric": "image-wise four-class mIoU, empty-union=0",
              "evaluation_role": "fixed-route diagnostic on previously used valid; not independent model selection",
              "base_cache_caveat": "Model183 cached base requires separate upstream models; six loaded source files are not the complete inference model count",
              "loaded_source_archives": (["269", "326", "329", "336"] if args.compact else
                                         ["269", "322", "326", "329", "336", "347"]),
              "image_names": [row["name"] for row in rows],
              "baseline": baseline349,
              "replayed_baseline": actual,
              "replayed_baseline_delta_percentage_points": 100 * (actual - baseline349),
              "variants": {
                  name: {"score": float(np.mean(values)),
                         "delta_percentage_points": float(100 * (np.mean(values) - baseline349)),
                         "image_scores": values}
                  for name, values in scores.items()
              }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: [round(data["score"], 9), round(data["delta_percentage_points"], 4)]
                      for name, data in report["variants"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
