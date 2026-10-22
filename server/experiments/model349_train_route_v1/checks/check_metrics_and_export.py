"""Meaningful policy and export checks on synthetic data; no optimizer steps."""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/experiments"))
sys.path.insert(0, str(ROOT / "source/experiments/historical_scripts"))
from local_image_metrics import class_ious, summary
from local_result_export import NAMES, PALETTE, export_final


def main():
    # 8 TP + 1 FP + 1 FN per present class -> .8, .8, .8, 0 -> .6.
    target = np.repeat(np.arange(3, dtype=np.uint8), 9).reshape(3, 9)
    pred = target.copy()
    pred[:, -1] = [1, 2, 0]
    assert np.allclose(class_ious(pred, target), [.8, .8, .8, 0])
    assert np.isclose(summary([pred], [target], NAMES)["image_mean_miou"], .6)
    # Images carry equal weight, irrespective of their pixel counts.
    a = np.arange(4, dtype=np.uint8).reshape(2, 2)
    b = np.tile(a, (20, 20))
    s = summary([a, (b + 1) % 4], [a, b], NAMES)
    assert s["image_mean_miou"] == .5
    assert not any("global" in key for key in s)
    from model181_raw_zcdf_aspp import score
    from encoder_structure_search import metrics
    from resnet34_refinements import metrics as ema_metrics
    for function in (score, metrics, ema_metrics):
        result = function([pred], [target])
        assert np.isclose(result["image_mean_miou"], .6)
        assert np.isclose(result["image_miou_compat"], .6)
    for branch in ("model322_team_b4", "model326_hybrid_fullstats"):
        spec = importlib.util.spec_from_file_location(branch, ROOT / f"source/experiments/{branch}/KoMaP/common.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert np.isclose(module.image_miou(pred, target)[0], .6)
    parent = ROOT / "outputs/checks"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="export_", dir=parent))
    image_dir = run / "data/valid/images"
    image_dir.mkdir(parents=True)
    # Large enough for full score labels. Wrong pixels must use GT color.
    full_target = np.tile(target, (100, 70))
    full_pred = np.tile(pred, (100, 70))
    names = [f"A{i}_synthetic" for i in range(20)]
    rows = [{"item": {"target": full_target}} for _ in names]
    for name in names:
        Image.fromarray(np.full((*full_target.shape, 3), 96, dtype=np.uint8)).save(image_dir / f"{name}_image.png")
    os.environ["KOMAP_RESULTS_ROOT"] = str(run / "synthetic_results")
    result = export_final(rows, names, lambda row: full_pred, run,
                          {"raw_best_valid_image_mean_miou_empty0": .6})
    assert len(list((result / "valid_mask_comparison").glob("*.png"))) == 20
    assert np.isclose(json.loads((result / "metrics.json").read_text())["image_miou"], .6)
    panel = np.array(Image.open(result / "valid_mask_comparison" / f"{names[0]}.png"))
    h, w = full_target.shape
    error_panel = panel[76:76 + h, w:2 * w]
    correct = full_pred == full_target
    assert np.all(error_panel[correct] == 255)
    assert np.array_equal(error_panel[~correct], PALETTE[full_target[~correct]])
    # Atomic numbering must preserve existing results.
    from local_result_export import reserve_result
    assert reserve_result(run / "synthetic_results") != result
    report = {"training_started": False, "empty0_example": .6, "equal_image_weighting": .5,
              "metric_adapters": "passed", "panels": 20, "error_colors": "passed", "synthetic_result": str(result)}
    (ROOT / "reports/metric_export_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
