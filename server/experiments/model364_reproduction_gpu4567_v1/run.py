"""Model364: GPU 4,5,6,7 for independent stages; training is user-started."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from preflight import ROOT, REPO, environment, prepare_data, verify_source, write_json
from scheduler import gpu_list, render, run_tasks, validate_tasks


def load_recipe():
    recipe = json.loads((ROOT / "pipeline_tasks.json").read_text())
    validate_tasks(recipe["tasks"])
    return recipe


def inventory():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"
    ], text=True)
    rows = {}
    for line in output.splitlines():
        index, uuid, name, memory = [part.strip() for part in line.split(",")]
        rows[int(index)] = {"uuid": uuid, "name": name, "memory_mib": int(memory)}
    return rows


def plan_info(recipe, gpus):
    ready = sorted((t for t in recipe["tasks"] if not t["deps"] and t["kind"] == "gpu"),
                   key=lambda t: (t["priority"], t["serial_index"]))
    return {"physical_gpus": gpus, "max_concurrent_gpu_stages": len(gpus),
            "gpu_stages": sum(t["kind"] == "gpu" for t in recipe["tasks"]),
            "cpu_stages": sum(t["kind"] == "cpu" for t in recipe["tasks"]),
            "initial_assignment": {str(g): t["id"] for g, t in zip(gpus, ready)},
            "scheduling": "one independent model/fold per GPU; dependencies respected",
            "training_started": False}


def prepare_run(kind, recipe, gpus, devices, workers):
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    parent = ROOT / "outputs" / ("runs" if kind == "train" else "checks")
    run = parent / f"model364_{kind}_{tag}"
    run.mkdir(parents=True, exist_ok=False)
    source = run / "source_snapshot"
    shutil.copytree(ROOT / "source", source)
    shutil.copytree(ROOT / "adapters", run / "adapters")
    launcher = run / "launcher_snapshot"
    launcher.mkdir()
    for filename in ("run.py", "run.sh", "start_nohup.sh", "preflight.py", "scheduler.py",
                     "pipeline_tasks.json", "README.md", "setup_environment.sh"):
        shutil.copy2(ROOT / filename, launcher / filename)
    for filename in ("provenance.json", "source_manifest.json"):
        shutil.copy2(ROOT / filename, run / filename)
    shutil.copy2(ROOT / "reports/local_patches.diff", run / "local_patches.diff")
    shutil.copy2(ROOT / "datasets/manifest.json", run / "dataset_manifest.json")
    (run / "environment.freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    write_json(run / "parallel_plan.json", {**plan_info(recipe, gpus), "gpu_inventory": devices, "route_workers": workers})
    write_json(run / "launcher_manifest.json", {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(launcher.iterdir())})
    data = run / "data"
    data.mkdir()
    for split in ("train", "valid"):
        with zipfile.ZipFile(ROOT / "datasets" / f"{split}.zip") as archive:
            for name in archive.namelist():
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError(f"Unsafe dataset archive member: {name}")
            archive.extractall(data)
    (run / "outputs").mkdir()
    for reference in (source / "results/outputs").glob("*/results.json"):
        if reference.parent.name in {"model339_joint_gamma_route_search", "model341_model329_joint_route_search", "model346_model336_into_341_route", "model349_model347_into_346_route", "model355_model349_spatial_audit", "model360_entropy_gated_d4_into_355"}:
            dest = run / "outputs" / reference.parent.name
            dest.mkdir()
            shutil.copy2(reference, dest / "results.json")
    env = os.environ.copy()
    env.update(PYTHONPATH=f"{run / 'adapters'}:{source / 'experiments'}:{source / 'experiments/historical_scripts'}",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", NO_ALBUMENTATIONS_UPDATE="1",
               KOMAP_RESULTS_ROOT=str(REPO / "results"), CUDA_VISIBLE_DEVICES="")
    prepare = [sys.executable, "-B", "-u"] + render(recipe["prepare_cv_argv"], source, run, workers)
    subprocess.run(prepare, env=env, cwd=run, check=True)
    write_json(run / "resolved_commands.json", [
        {"id": t["id"], "deps": t["deps"], "kind": t["kind"],
         "argv": render(t["argv"], source, run, workers)} for t in recipe["tasks"]])
    return run, source, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("check", "plan", "train"), default="check")
    parser.add_argument("--execute", action="store_true", help="User-only: actually start training")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("KOMAP_WORKERS", "4")),
                        help="CPU workers within a route stage")
    args = parser.parse_args()
    if args.execute and args.command != "train":
        parser.error("--execute requires train")
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        gpus = gpu_list(args.gpus)
    except ValueError as error:
        parser.error(str(error))
    recipe = load_recipe()
    plan = plan_info(recipe, gpus)
    print(json.dumps(plan, indent=2), flush=True)
    if args.command == "plan":
        return
    # The coordinator sees all physical devices; each child receives exactly
    # one UUID, avoiding dependence on CUDA's device-index ordering.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    verify_source()
    devices = inventory()
    if any(g not in devices for g in gpus):
        parser.error(f"GPU list is not present: requested={gpus}, available={list(devices)}")
    info = {"environment": environment(), "dataset": prepare_data(), "plan": plan}
    if args.command == "train" and not args.execute:
        print("DRY RUN: no training. Use ./run.sh train --execute to start.", flush=True)
        return
    if args.command == "check":
        run, source, env = prepare_run("preflight", recipe, gpus, devices, args.workers)
        write_json(run / "preflight.json", info)
        print(f"PREFLIGHT OK: {len(gpus)} GPUs, 57 stages, no training.\nCHECK_DIRECTORY={run}", flush=True)
        return
    (ROOT / "outputs").mkdir(exist_ok=True)
    with (ROOT / "outputs/pipeline.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Model364 v2 pipeline is active")
        run, source, env = prepare_run("train", recipe, gpus, devices, args.workers)
        write_json(run / "preflight.json", info)
        print(f"RUN_DIRECTORY={run}\nPHYSICAL_GPUS={gpus}", flush=True)
        run_tasks(recipe["tasks"], run, source, gpus, {g: devices[g]["uuid"] for g in gpus},
                  sys.executable, env, workers=args.workers, lock_fd=lock.fileno())
        result = Path((run / "result_path.txt").read_text().strip())
        print(f"COMPLETE: {run}\nRESULT_DIRECTORY={result}", flush=True)


if __name__ == "__main__":
    main()
