"""Small synthetic route-chain integration, including final export. No training."""
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/experiments"
sys.path.insert(0, str(SOURCE))
from local_result_export import PALETTE


def main():
    check = Path(tempfile.mkdtemp(prefix="routes_", dir=ROOT / "outputs/checks"))
    out = check / "outputs"
    for kind in ("images", "masks"):
        (check / "data/valid" / kind).mkdir(parents=True)
    rng = np.random.default_rng(349)
    names = sorted(f"A{i + 1}_step1_sample1" for i in range(20))
    sources = {k: {} for k in ("195", "217", "269", "322", "326", "329", "336", "347")}
    entries = []
    for name in names:
        g = rng.integers(0, 256, (24, 32), dtype=np.uint8)
        target = rng.integers(0, 4, g.shape, dtype=np.uint8)
        Image.fromarray(g).save(check / "data/valid/images" / f"{name}_image.png")
        Image.fromarray(PALETTE[target]).save(check / "data/valid/masks" / f"{name}_mask.png")
        a = rng.random((4, *g.shape), dtype=np.float32)
        a /= a.sum(0)
        entries.append({"target": target, "z": (g.astype(np.float32) - g.mean()) / g.std(),
                        "q": g.astype(np.float32) / 255, "mixed": a, "extra": np.full_like(a, .25),
                        "prior_z_stats": [(0, 1)] * 4, "prior_q_stats": [(.5, .3)] * 4,
                        "params": {"beta": .5, "al_bias": .1, "eut_bias": -.1}})
        for key in sources:
            p = rng.random(a.shape, dtype=np.float32)
            p /= p.sum(0)
            sources[key][name] = p
    paths = {
        "195": "model195_native_resume/valid_probs.npz",
        "217": "model220_native_tent_extended/valid_probs_06.npz",
        "269": "model269_fullstats_1x_extended/valid_probs.npz",
        "322": "model322_team_b4_group_oof/valid_ensemble/probabilities.npz",
        "326": "model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz",
        "329": "model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz",
        "336": "model336_hrnet_best/valid_eval/valid_probabilities.npz",
        "347": "model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz",
    }
    for key, path in paths.items():
        full = out / path
        full.parent.mkdir(parents=True, exist_ok=True)
        np.savez(full, **sources[key])
    entries_path = out / "model183_metric_calibration/entries.pkl"
    entries_path.parent.mkdir(parents=True)
    with entries_path.open("wb") as stream:
        pickle.dump(entries, stream)
    env = os.environ.copy()
    env.update(PYTHONPATH=f"{SOURCE}:{SOURCE / 'historical_scripts'}", CUDA_VISIBLE_DEVICES="",
               KOMAP_RESULTS_ROOT=str(check / "synthetic_results"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NO_ALBUMENTATIONS_UPDATE="1", PYTHONDONTWRITEBYTECODE="1")
    base = [sys.executable, "-B", "-u"]
    h = SOURCE / "historical_scripts"
    commands = [
        [str(h / "model335_exhaustive_b4_route_search.py"), "--entries", str(entries_path),
         "--model195", str(out / paths["195"]), "--tent20", str(out / paths["217"]),
         "--model269", str(out / paths["269"]), "--model322", str(out / paths["322"]),
         "--model326", str(out / paths["326"]), "--max-routes", "2", "--workers", "2",
         "--output", str(out / "model335_exhaustive_b4_routes/results.json")],
        [str(h / "model338_fine_gamma_route_search.py"), "--root", str(check), "--workers", "2"],
        [str(SOURCE / "model339_joint_gamma_route_search.py"), "--root", str(check), "--max-routes", "2", "--workers", "2"],
        [str(SOURCE / "model341_model329_joint_route_search.py"), "--root", str(check), "--max-routes", "2", "--workers", "2"],
        [str(SOURCE / "model346_model336_into_341_route.py"), "--root", str(check), "--max-routes", "2", "--workers", "2", "--bootstrap", "10"],
        [str(SOURCE / "model346_model336_into_341_route.py"), "--root", str(check), "--expert-id", "347", "--anchor-model346",
         "--max-routes", "2", "--workers", "2", "--bootstrap", "10"],
    ]
    for i, command in enumerate(commands):
        log = check / f"stage_{i}.log"
        with log.open("w") as stream:
            process = subprocess.run(base + command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        if process.returncode:
            print(log.read_text()[-6000:])
            raise RuntimeError(f"Route stage failed: {command[0]}, see {log}")
        print(f"OK route stage {i + 1}/6: {Path(command[0]).name}", flush=True)
    result = Path((check / "result_path.txt").read_text().strip())
    record = json.loads((result / "metrics.json").read_text())
    assert len(record["images"]) == 20
    assert len(list((result / "valid_mask_comparison").glob("*.png"))) == 20
    assert np.isclose(record["image_miou"], record["route"]["raw_best_valid_image_mean_miou_empty0"], atol=1e-10)
    report = {"training_started": False, "route_stages_passed": 6, "test_data": "synthetic 20x24x32",
              "candidate_grid": "2-route prefixes; Model338 full fine-gamma grid", "parallel_workers": 2,
              "final_export_matches_route": True, "check_directory": str(check)}
    (ROOT / "reports/route_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
