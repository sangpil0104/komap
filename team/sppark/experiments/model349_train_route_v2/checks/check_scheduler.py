"""Real subprocess tests of all 59 dependencies, 8-GPU dispatch and failure cleanup."""
import copy
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scheduler import gpu_list, render, run_tasks, validate_tasks


def simulate(tasks, directory, modes=None):
    modes = modes or {}
    directory.mkdir()
    fixture = []
    for t in tasks:
        row = copy.deepcopy(t)
        for key in ("requires", "produces"):
            row[key] = render(row.get(key, []), ROOT / "source", directory, 4)
        fixture.append(row)
    spec = directory / "fixture.json"
    spec.write_text(json.dumps(fixture))
    for row in fixture:
        row["argv"] = [str(ROOT / "checks/fake_worker.py"), modes.get(row["id"], "ok"), str(spec), row["id"]]
    return run_tasks(fixture, directory, ROOT / "source", list(range(8)),
                     {g: f"TEST-GPU-{g}" for g in range(8)}, sys.executable, os.environ.copy(),
                     poll_interval=.01, report_interval=120)


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def main():
    parent = ROOT / "outputs/checks"
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="scheduler_", dir=parent))
    recipe = json.loads((ROOT / "pipeline_tasks.json").read_text())
    tasks = recipe["tasks"]
    calls = [json.loads(line) for line in (ROOT / "reports/serial_commands_reference.jsonl").read_text().splitlines()]
    calls = [[a for a in call if a != "-u"] for call in calls]
    old_source = calls[1][0].split("/experiments/historical_scripts/")[0]
    old_run = calls[1][calls[1].index("--data-dir") + 1].removesuffix("/data")
    for task in tasks:
        # Same argv byte-for-byte when expanded to the checked v1 locations.
        assert render(task["argv"], old_source, old_run, 4) == calls[task["serial_index"]]
    events = simulate(tasks, root / "all_tasks")
    done, active, max_active, seen = set(), {}, 0, set()
    by_id = {t["id"]: t for t in tasks}
    for event in events:
        if event["event"] == "started":
            task = by_id[event["task"]]
            assert set(task["deps"]) <= done
            if task["kind"] == "gpu":
                gpu = event["gpu"]
                assert gpu not in active
                active[gpu] = task["id"]
                seen.add(gpu)
                max_active = max(max_active, len(active))
        elif event["event"] == "finished":
            done.add(event["task"])
            if event["gpu"] is not None:
                assert active.pop(event["gpu"]) == event["task"]
    assert len(done) == 59 and seen == set(range(8)) and max_active == 8
    first = [e for e in events if e["event"] == "started"][:8]
    assert [e["gpu"] for e in first] == list(range(8))
    base = [{"id": "bad", "serial_index": 0, "priority": 0, "kind": "gpu", "deps": [], "argv": [], "produces": []},
            {"id": "sibling", "serial_index": 1, "priority": 0, "kind": "gpu", "deps": [], "argv": [], "produces": []},
            {"id": "dependent", "serial_index": 2, "priority": 0, "kind": "gpu", "deps": ["bad"], "argv": [], "produces": []}]
    try:
        simulate(base, root / "failure", {"bad": "fail", "sibling": "linger"})
        raise AssertionError("Expected failure")
    except RuntimeError as error:
        assert "exit 7" in str(error)
    failure_events = [json.loads(line) for line in (root / "failure/events.jsonl").read_text().splitlines()]
    assert not any(e.get("task") == "dependent" for e in failure_events)
    for event in failure_events:
        if event["event"] == "started":
            assert not alive(event["pid"])
    child_file = root / "failure/fixture.child_pid"
    assert child_file.exists()
    child_pid = int(child_file.read_text())
    for _ in range(100):
        if not alive(child_pid):
            break
        time.sleep(.01)
    assert not alive(child_pid)
    missing = copy.deepcopy(base[:1])
    missing[0]["produces"] = ["{run}/missing.pt"]
    try:
        simulate(missing, root / "missing_output", {"bad": "missing"})
        raise AssertionError("Expected output validation failure")
    except RuntimeError as error:
        assert "required output" in str(error)
    for value in ("", "0,0", "-1", "a", "1,"):
        try:
            gpu_list(value)
            raise AssertionError(value)
        except ValueError:
            pass
    cyclic = copy.deepcopy(base)
    cyclic[0]["deps"] = ["dependent"]
    try:
        validate_tasks(cyclic)
        raise AssertionError("Expected cycle rejection")
    except ValueError:
        pass
    # Confirm source/model code and all previous v1 files remain untouched.
    old = ROOT.parent / "model349_train_route_v1"
    for row in json.loads((ROOT / "reports/v1_preservation.json").read_text()):
        assert hashlib.sha256((old / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    for path in (ROOT / "source").rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (old / "source" / path.relative_to(ROOT / "source")).read_bytes()
    result = {"training_started": False, "all_stages": 59, "gpu_ids_used": sorted(seen),
              "max_simultaneous_gpu_stages": max_active, "argv_matches_v1": True,
              "dependency_order_and_exclusive_gpu_assignment": "passed",
              "failure_and_descendant_cleanup": "passed", "missing_output_guard": "passed",
              "v1_preserved": True, "model_sources_unchanged": True, "check_directory": str(root)}
    (ROOT / "reports/scheduler_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
