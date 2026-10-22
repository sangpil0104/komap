"""Verify smoke artifacts without training or changing existing results."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT.parent / "accuracy_v1/baseline/source/src")]
import torch
import pandas as pd
from alphase.utils import code_hash
from refinelab.workspace import plan, read_json, source_hash

torch.set_num_threads(4)
out = ROOT / "outputs"
suite = read_json(out / "smoke_suite_status.json")
assert suite["state"] == "completed" and len(suite["completed"]) == 10
results = {}
for name in plan():
    folder = out / "smoke" / name / "seed2026/official"
    status = read_json(folder / "status.json")
    assert status["state"] == "completed" and status["update"] == status["total_updates"] == 3
    state = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
    assert state["completed"] and state["optimizer_step"] == 3 and state["world_size"] == 8
    cfg = state["resolved_config"]
    assert cfg["refinement"]["source_sha256"] == source_hash()
    assert state["code_sha256"] == code_hash()
    assert len(state["train_ids"]) == 70 and len(state["validation_ids"]) == 20
    assert not set(state["train_ids"]) & set(state["validation_ids"])
    assert all(torch.isfinite(t).all() for t in state["model"].values())
    if "aspp" in name:
        assert float(state["model"]["context.project.weight"].abs().sum()) > 0
    for mode in ["plain", "d4"]:
        evaluation = out / "smoke_evaluations" / name / "seed2026" / mode
        summary = read_json(evaluation / "metrics_summary.json")
        assert summary["num_images"] == 20
        assert 0 <= summary["global_mIoU"] <= 100 and 0 <= summary["image_mIoU"] <= 100
        assert len(pd.read_csv(evaluation / "per_image_metrics.csv")) == 20
    results[name] = dict(completed=True, updates=3, architecture=cfg["model"]["name"])
    del state
assert not (out / "runs").exists()
assert not list(out.rglob("*.html"))
assert len(pd.read_csv(out / "reports/comparison.csv")) == 2
log = (out / "verification/smoke_suite.log").read_text()
assert "ETA=" in log and "Sequential experiments" in log
assert "Monitor:" not in log and "localhost" not in log
checks = dict(unit_tests_passed=16, experiments=results,
              full_experiments_started=False, terminal_only=True,
              mining=read_json(out / "verification/mining_check.json"),
              smoke_results_excluded_from_comparison=True)
(out / "verification/integration.json").write_text(json.dumps(checks, indent=2))
print("Integration: 10 smoke experiments PASS")
