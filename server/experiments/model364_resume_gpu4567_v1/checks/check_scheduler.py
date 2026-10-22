"""Real subprocess tests of all 31 dependencies, 4-GPU dispatch and failure cleanup."""
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
    # Seed only requirements from previously completed stages.
    produced = [p for row in fixture for p in row.get("produces", [])]
    for row in fixture:
        for name in row.get("requires", []):
            if not any(name == p or name.startswith(p + "/") for p in produced):
                path = Path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("reused completed artifact\n")
    spec = directory / "fixture.json"
    spec.write_text(json.dumps(fixture))
    for row in fixture:
        row["argv"] = [str(ROOT / "checks/fake_worker.py"), modes.get(row["id"], "ok"), str(spec), row["id"]]
    return run_tasks(fixture, directory, ROOT / "source", [4,5,6,7],
                     {g: f"TEST-GPU-{g}" for g in [4,5,6,7]}, sys.executable, os.environ.copy(),
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
    assert len(done) == 31 and seen == {4,5,6,7} and max_active == 4
    first = [e for e in events if e["event"] == "started"][:4]
    assert [e["gpu"] for e in first] == [4,5,6,7]
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
    result = {"training_started": False, "all_stages": 31, "gpu_ids_used": sorted(seen),
              "max_simultaneous_gpu_stages": max_active, 
              "dependency_order_and_exclusive_gpu_assignment": "passed",
              "failure_and_descendant_cleanup": "passed", "missing_output_guard": "passed",
               "check_directory": str(root)}
    (ROOT / "reports/scheduler_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
