import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from report import comparison, scores, seed_report, suite_report
from run import (BASE, HERE, RESULT, SEEDS, archive_before_append, initial_eta,
                 plan_existing, run_queue, sha256)
from runtime import read_json, write_json


def make_job(session, result, seed):
    run = session / f"seed_{seed:02d}"
    run.mkdir(parents=True)
    job_file = run / "job.json"
    write_json(job_file, dict(BASE, seed=seed, run=str(run), result=str(result)))
    return dict(seed=seed, state="queued", run=str(run), job_file=str(job_file))


def finish_fixture(result, job):
    run, seed = Path(job["run"]), job["seed"]
    rows = [dict(image=f"image_{i}.png", image_miou=0.5, image_miou_exclude=0.5,
                 image_miou_zero=0.5, class_iou=[0.5]*4) for i in range(20)]
    (run / "last.pt").write_bytes(b"fixture: no model")
    write_json(run / "status.json", dict(state="completed", elapsed_seconds=3600))
    seed_report(result, seed, rows, run)
    for folder in ["predictions", "valid_mask_comparison"]:
        dest = result / folder / f"seed_{seed:02d}"
        dest.mkdir(parents=True)
        for row in rows:
            (dest / row["image"]).write_bytes(b"fixture: no image")
    job["state"] = "completed"


def baseline_fixture(tmp_path):
    result = tmp_path / "results"
    (result / "metrics").mkdir(parents=True)
    jobs = [make_job(tmp_path / "original", result, s) for s in range(10)]
    for job in jobs:
        finish_fixture(result, job)
    write_json(result / "suite.json", dict(state="completed", jobs=jobs, elapsed_seconds=7200))
    suite_report(result, jobs, "completed", 7200)
    return result, jobs


def test_actual_training_source_conditions_and_plan_are_identical():
    baseline = read_json(HERE / "baseline_provenance.json")
    for filename in ["train.py", "runtime.py", "vendor/common.py"]:
        assert sha256(HERE / filename) == baseline["source_sha256"][filename]
    suite = read_json(HERE / "baseline_suite.json")
    for job in suite["jobs"]:
        cfg = read_json(job["job_file"])
        assert {k: v for k, v in cfg.items() if k not in {"seed", "run", "result"}} == BASE
    prior, pending, _ = plan_existing(RESULT)
    assert [j["seed"] for j in prior] == list(range(10))
    assert pending == SEEDS == list(range(10, 100))
    assert 14 * 3600 < initial_eta(prior, len(pending)) < 17 * 3600


def test_actual_latest_figure_layout_is_reproduced(tmp_path):
    # No model run: render saved seed0 masks to a temporary fixture path.
    name = "A71_step2_sample1_image.png"
    dest = tmp_path / "seed_00" / name
    dest.parent.mkdir()
    dataset = Path(BASE["data"])
    original = np.array(Image.open(dataset / "valid/images" / name).convert("RGB"))
    from comparison_layout import labels
    pred = labels(np.array(Image.open(RESULT / "predictions/seed_00" / name).convert("RGB")))
    gt = labels(np.array(Image.open(dataset / "valid/masks" / name.replace("_image", "_mask")).convert("RGB")))
    record = comparison(dest, original, pred, gt, scores(pred, gt)["image_miou"])
    assert np.array_equal(np.array(Image.open(dest)), np.array(Image.open(RESULT / "valid_mask_comparison/seed_00" / name)))
    assert record["error_rate_percent"] == pytest.approx(100 * np.mean(pred != gt))
    assert record["pixel_verification"] == "passed"


def test_refuse_missing_original_and_overwriting_metrics(tmp_path):
    result, jobs = baseline_fixture(tmp_path)
    rows = read_json(result / "metrics/seed_00.json")["per_image"]
    with pytest.raises(FileExistsError):
        seed_report(result, 0, rows, jobs[0]["run"])
    (result / "valid_mask_comparison/seed_00/image_0.png").unlink()
    with pytest.raises(ValueError, match="Original seed 0"):
        plan_existing(result)


def test_append_all_90_failure_retry_and_preserve_originals(tmp_path, capsys):
    result, original = baseline_fixture(tmp_path)
    preserved = {p: p.read_bytes() for p in result.rglob("*") if p.is_file() and
                 (p.parent.name.startswith("seed_") or p.parent.name == "metrics")}
    old_report = (result / "results.md").read_bytes()
    first_history = result / "report_history/first"
    archive_before_append(result, first_history, SEEDS)
    assert (first_history / "results.md").read_bytes() == old_report
    session = tmp_path / "extension"
    jobs = [make_job(session, result, seed) for seed in SEEDS]
    fake = tmp_path / "worker_fixture.py"
    fake.write_text('''
import json, os, pathlib, sys, time
job=json.loads(pathlib.Path(sys.argv[1]).read_text()); fail=int(sys.argv[2])
run=pathlib.Path(job['run']); result=pathlib.Path(job['result']); seed=job['seed']
start=time.time(); print('fixture seed',seed,flush=True); time.sleep(0.03)
(run/'seen.json').write_text(json.dumps(dict(gpu=os.environ['CUDA_VISIBLE_DEVICES'], start=start,end=time.time())))
if seed == fail:
    partial=result/'predictions'/f'seed_{seed:02d}'; partial.mkdir(parents=True)
    (partial/'partial.png').write_bytes(b'partial fixture')
    raise SystemExit(7)
rows=[dict(image=f'image_{i}.png',image_miou=0.5) for i in range(20)]
metric=dict(seed=seed,run=str(run),image_miou=0.5,image_miou_exclude=0.5,image_miou_zero=0.5,per_class_iou=[0.5]*4,per_image=rows)
(result/'metrics'/f'seed_{seed:02d}.json').write_text(json.dumps(metric))
for folder in ['predictions','valid_mask_comparison']:
    dest=result/folder/f'seed_{seed:02d}'; dest.mkdir(parents=True)
    for row in rows: (dest/row['image']).write_bytes(b'fixture')
(run/'last.pt').write_bytes(b'fixture')
(run/'status.json').write_text(json.dumps(dict(state='completed',elapsed_seconds=time.time()-start)))
''')
    code = run_queue(jobs, tmp_path, session, result, heartbeat=0.05,
                     prior_jobs=original, prior_elapsed=7200,
                     command_builder=lambda j, gpu: [sys.executable, str(fake), j["job_file"], "12"])
    assert code == 1
    registry = read_json(result / "suite.json")
    assert len(registry["jobs"]) == 100
    assert sum(j["state"] == "completed" for j in registry["jobs"]) == 99
    assert registry["invocation_seeds"] == list(range(10, 100))
    seen = [read_json(Path(j["run"]) / "seen.json") for j in jobs]
    assert {s["gpu"] for s in seen} == {str(i) for i in range(6)}
    for i, a in enumerate(seen):
        for b in seen[i+1:]:
            if a["gpu"] == b["gpu"]:
                assert a["end"] <= b["start"] or b["end"] <= a["start"]
    prior, pending, previous = plan_existing(result)
    assert len(prior) == 99 and pending == [12]
    second_history = result / "report_history/retry"
    archive_before_append(result, second_history, pending)
    assert (second_history / "partial_seeds/predictions/seed_12/partial.png").is_file()
    retry_session = tmp_path / "retry"
    retry = make_job(retry_session, result, 12)
    code = run_queue([retry], tmp_path, retry_session, result, heartbeat=0.05,
                     prior_jobs=prior, prior_elapsed=previous["elapsed_seconds"],
                     command_builder=lambda j, gpu: [sys.executable, str(fake), j["job_file"], "-1"])
    assert code == 0
    prior, pending, _ = plan_existing(result)
    assert len(prior) == 100 and pending == []
    report = (result / "results.md").read_text()
    assert "성공 100/100" in report and "현재 완료 2000장" in report
    assert "### seed 90–99" in report and "### seed 0–9" in report
    assert all(p.read_bytes() == before for p, before in preserved.items())
    output = capsys.readouterr().out
    assert "all_done=100/100" in output and "total_ETA=" in output
    assert all("fixture seed" in (Path(j["run"]) / "worker.log").read_text() for j in jobs)
