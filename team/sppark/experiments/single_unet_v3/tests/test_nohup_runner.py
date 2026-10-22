import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from nohup_runner import SuiteTracker, duration, seconds_per_update


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write(run / "config.json", dict(experiment="S0", scale=1, train=dict(updates=8000, effective_batch=8)))
    write(run / "latest.json", dict(update=2000, checkpoint="checkpoints/imported_update_002000.pt"))
    write(run / "migration.json", dict(imported_update=2000, source_world_size=1, target_world_size=6))
    history = [dict(update=i, elapsed_seconds=i * .8) for i in range(1800, 2001, 10)]
    (run / "training_before_migration.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
    return run


def metric(update, elapsed):
    return dict(update=update, elapsed_seconds=elapsed, loss=.4, ce=.2, dice=.2, encoder_lr=1e-4)


def test_initial_eta_preserves_remaining_budget_and_six_gpu_round_ratio(tmp_path):
    tracker = SuiteTracker(fixture_run(tmp_path), ["S1", "S2"], clock=lambda: 0.)
    estimate = tracker.estimate()
    # 0.8 s/update on 1 GPU -> 2/8 rounds -> 0.2 s/update on six GPUs.
    assert tracker.initial_rate == pytest.approx(.2)
    assert tracker.total_planned == 22000
    assert estimate["training_left"] == pytest.approx(6000 * .2 + 8000 * .8 + 8000 * 1.8)
    assert estimate["remaining"] == pytest.approx(22000 + 3 * (180 + 30))


def test_measured_rate_replaces_initial_estimate_and_all_metrics_are_logged(tmp_path, capsys):
    run = fixture_run(tmp_path)
    now = [0.]
    tracker = SuiteTracker(run, ["S1", "S2"], clock=lambda: now[0])
    tracker.accept_output(f"Run: {run}")
    rows = [metric(i, 1600 + (i - 2000) * .4) for i in range(2010, 2041, 10)]
    (run / "training.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    now[0] = 16
    tracker.poll()
    assert tracker.tasks[0].observed_rate == pytest.approx(.4)
    assert tracker.rate(tracker.tasks[1])[0] == pytest.approx(1.6)
    assert tracker.tasks[0].update == 2040
    output = capsys.readouterr().out
    assert output.count("[TRAIN]") == 4
    tracker.poll()
    assert "[TRAIN]" not in capsys.readouterr().out  # No duplicate history spam.
    write(run / "completed.json", dict(elapsed_seconds=4000))
    now[0] = 25
    tracker.poll()
    assert tracker.tasks[0].phase == "검증/보고서"
    now[0] = 70
    tracker.accept_output("Result report: /tmp/new_report")
    assert tracker.tasks[0].phase == "완료"
    assert tracker.postprocess_seconds == 45


def test_partial_jsonl_line_is_not_lost(tmp_path):
    run = fixture_run(tmp_path)
    tracker = SuiteTracker(run, [], clock=lambda: 0.)
    tracker.accept_output(f"Run: {run}")
    value = json.dumps(metric(2010, 1605))
    (run / "training.jsonl").write_text(value[:15])
    tracker.poll()
    assert tracker.tasks[0].update == 2000
    with (run / "training.jsonl").open("a") as f:
        f.write(value[15:] + "\n")
    tracker.poll()
    assert tracker.tasks[0].update == 2010


def test_short_rate_window_and_rollback_rows():
    assert seconds_per_update([metric(10, 4), metric(20, 8), metric(30, 12)]) == .4
    assert seconds_per_update([metric(20, 8), metric(10, 10), metric(20, 14)]) == .4
    assert duration(3661) == "01:01:01"


def test_plan_only_never_launches_training(tmp_path):
    run = fixture_run(tmp_path)
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, str(root / "nohup_runner.py"), "--run", str(run), "--plan-only"],
                          capture_output=True, text=True, check=True)
    assert "[PLAN_ONLY]" in done.stdout
    assert "[FINISH_ETA]" in done.stdout and "KST" in done.stdout and "UTC" in done.stdout
    assert "[PROCESS]" not in done.stdout
    assert not (run / "training.jsonl").exists()


@pytest.mark.parametrize("outcome,code,tag", [("complete", 0, "DONE"), ("pause", 0, "PAUSED"), ("fail", 7, "FAILED")])
def test_runner_with_harmless_suite_merges_output_tracks_stages_and_exit(tmp_path, outcome, code, tag):
    import shutil
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / "nohup_runner.py", tmp_path / "nohup_runner.py")
    run = fixture_run(tmp_path)
    # This replacement CLI only writes fixture JSON; no model, GPU or optimizer is used.
    (tmp_path / "run.py").write_text('''import json, os, sys
from pathlib import Path
assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,5,6,7"
assert os.environ["TQDM_DISABLE"] == "1"
assert sys.argv[1] == "continue-suite" and sys.argv[3:5] == ["--gpus", "6"]
run = Path(sys.argv[2])
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
print("stderr fixture", file=sys.stderr, flush=True)
for name in ["S0", "S1", "S2"]:
    if name != "S0":
        run = run.parent / name
        write(run / "config.json", dict(experiment=name, scale=int(name[1])+1))
    print(f"Run: {run}", flush=True)
    if os.environ["FIXTURE_OUTCOME"] == "pause":
        print("Saved and paused", flush=True)
        sys.exit(0)
    if os.environ["FIXTURE_OUTCOME"] == "fail":
        print("fixture failure", file=sys.stderr, flush=True)
        sys.exit(7)
    rows = [dict(update=i, elapsed_seconds=i*.5, loss=.4, ce=.2, dice=.2, encoder_lr=1e-4)
            for i in [7970, 7980, 7990, 8000]]
    (run / "training.jsonl").write_text("".join(json.dumps(r)+"\\n" for r in rows))
    write(run / "completed.json", dict(elapsed_seconds=4000))
    write(run / "evaluations/fixture/summary.json", dict(num_images=20, image_miou=75., miou_one=75., miou_zero=70., miou_exclude=72.))
    print(f"Result report: {run}/report", flush=True)
print("Continuation suite report: fixture_report", flush=True)
''')
    done = subprocess.run([sys.executable, str(tmp_path / "nohup_runner.py"), "--run", str(run), "--interval", "0.01"],
                          env={**os.environ, "FIXTURE_OUTCOME": outcome}, capture_output=True, text=True, timeout=15)
    assert done.returncode == code
    assert "stderr fixture" in done.stdout and not done.stderr
    assert f"[{tag}]" in done.stdout
    assert "[FINISH_ETA]" in done.stdout
    if outcome == "complete":
        assert done.stdout.count("[STAGE_DONE]") == 3
        assert done.stdout.count("[TRAIN]") == 12
        assert done.stdout.count("[METRICS]") == 3
        assert '"image_miou": 75.0' in done.stdout
        assert "Continuation suite report: fixture_report" in done.stdout
    else:
        assert "[DONE]" not in done.stdout


def test_nohup_script_merges_stdout_stderr_and_preserves_previous_logs(tmp_path):
    import shutil
    root = Path(__file__).resolve().parents[1]
    launcher = tmp_path / "start_nohup_6gpu.sh"
    shutil.copy2(root / launcher.name, launcher)
    # A harmless child, not the actual experiment CLI.
    (tmp_path / "nohup_runner.py").write_text("import sys\nprint('stdout message', flush=True)\nprint('stderr message', file=sys.stderr, flush=True)\n")
    env = {**os.environ, "KOMAP_PYTHON": sys.executable}
    first = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, check=True)
    logs = tmp_path / "outputs/logs"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        text = (logs / "latest.log").read_text()
        if "stdout message" in text and "stderr message" in text:
            break
        time.sleep(.05)
    else:
        pytest.fail("nohup child did not write both streams")
    old = (logs / "latest.log").resolve()
    old_bytes = old.read_bytes()
    # Wait for the inherited lock to be released, not an arbitrary sleep.
    while time.monotonic() < deadline:
        second = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
        if second.returncode == 0:
            break
        time.sleep(.05)
    assert second.returncode == 0
    assert (logs / "latest.log").resolve() != old
    assert old.read_bytes() == old_bytes
    assert "nohup 실행 PID" in first.stdout
