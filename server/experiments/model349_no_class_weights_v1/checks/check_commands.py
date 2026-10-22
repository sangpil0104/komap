"""Validate the ablation recipe against real argparse interfaces; never run training."""
import argparse
import concurrent.futures
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_only(call):
    class Parsed(BaseException):
        pass
    original = argparse.ArgumentParser.parse_args
    def intercepted(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise Parsed()
    argparse.ArgumentParser.parse_args = intercepted
    sys.argv = call
    sys.path.insert(0, str(Path(call[0]).parent))
    try:
        runpy.run_path(call[0], run_name="__main__")
    except Parsed:
        print("PARSE_OK")
        return
    raise RuntimeError("Script finished without argparse interception")


def main():
    parent = ROOT / "outputs/checks"
    parent.mkdir(parents=True, exist_ok=True)
    check = Path(tempfile.mkdtemp(prefix="commands_", dir=parent))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "source/experiments") + ":" + str(ROOT / "source/experiments/historical_scripts"),
               CUDA_VISIBLE_DEVICES="", NO_ALBUMENTATIONS_UPDATE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    sys.path.insert(0, str(ROOT))
    from scheduler import render
    recipe = json.loads((ROOT / "pipeline_tasks.json").read_text())
    calls = [render(t["argv"], ROOT / "source", check / "dummy_run", 4) for t in recipe["tasks"]]
    calls.append(render(recipe["prepare_cv_argv"], ROOT / "source", check / "dummy_run", 4))
    unique = {}
    for call in calls:
        call = [a for a in call if a != "-u"]
        if not call or not call[0].endswith(".py"):
            continue
        key = (call[0], tuple(a for a in call[1:] if a.startswith("--")))
        unique.setdefault(key, call)
    def check_one(index_call):
        index, call = index_call
        spec = check / f"call_{index:02d}.json"
        spec.write_text(json.dumps(call))
        result = subprocess.run([sys.executable, "-B", __file__, "--parse", str(spec)],
                                env=env, capture_output=True, text=True, timeout=90)
        (check / f"call_{index:02d}.log").write_text(result.stdout + result.stderr)
        ok = result.returncode == 0 and "PARSE_OK" in result.stdout
        print(f"{'OK' if ok else 'FAIL'} {Path(call[0]).name}", flush=True)
        return {"script": call[0], "ok": ok, "log": str(check / f"call_{index:02d}.log")}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(check_one, enumerate(unique.values())))
    output = {"training_started": False, "recorded_commands": len(calls), "unique_interfaces": len(records),
              "all_passed": all(r["ok"] for r in records), "records": records}
    report = ROOT / "reports/command_validation.json"
    report.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "records"}), flush=True)
    if not output["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--parse":
        parse_only(json.loads(Path(sys.argv[2]).read_text()))
    else:
        main()
