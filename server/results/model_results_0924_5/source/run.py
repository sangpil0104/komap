#!/usr/bin/env python3
import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import time

import torch

from sslab.common import (ROOT, config, digest, eta, read_json, signature, snapshot,
                          sources, stamp, write_json)
from sslab.engine import Context, budget_remaining, run_stage, seed_all
from sslab.evaluation import comparison_report, evaluate
from sslab.model import SmallUNet, prepare_weights


def master(ctx, function):
    """Propagate rank-0 setup/evaluation failures to waiting workers."""
    packet = None
    if ctx.rank == 0:
        try:
            packet = {"result": function()}
        except Exception as error:
            packet = {"error": f"{type(error).__name__}: {error}"}
    packet = ctx.broadcast(packet)
    if "error" in packet:
        raise RuntimeError(packet["error"])
    return packet["result"]


def prepare(cfg):
    checked = signature(cfg, verify=True)
    weight = prepare_weights(download=True) if cfg["model"]["pretrained"] else (None, None)
    path = ROOT / "verification" / f"prepare_{stamp()}.json"
    write_json(path, {"dataset": checked, "weights": str(weight[0]), "weights_sha256": weight[1],
                     "torch_version": str(torch.__version__), "source_hashes": sources(),
                     "training_started": False, "checked_originals": 100, "checked_masks": 90})
    print(f"PREPARED={path}\nNo training started.", flush=True)


def run_suite(cfg, ctx, output_root=None, resume=None, results_root=None):
    started = time.monotonic()

    def create_or_verify():
        if resume:
            suite = Path(resume).resolve()
            if read_json(suite / "config.json") != cfg:
                raise ValueError("Resume configuration differs from the original")
            if read_json(suite / "source_hashes.json") != sources():
                raise ValueError("Source changed after suite creation; preserve original sources for resume")
            if read_json(suite / "dataset.json") != signature(cfg, verify=True):
                raise ValueError("Dataset changed after suite creation")
            env = read_json(suite / "environment.json")
            if env["world_size"] != ctx.world:
                raise ValueError("Resume requires the original GPU count")
            if (suite / "completed.json").exists():
                return str(suite)
            (suite / "SAVE_AND_STOP").unlink(missing_ok=True)
        else:
            checked = signature(cfg, verify=True)
            weight = prepare_weights() if cfg["model"]["pretrained"] else (None, None)
            suite = Path(output_root or ROOT / "outputs") / f"suite_{stamp()}"
            suite.mkdir(parents=True, exist_ok=False)
            write_json(suite / "config.json", cfg)
            write_json(suite / "dataset.json", checked)
            snapshot(suite)
            write_json(suite / "environment.json", {"world_size": ctx.world, "torch": str(torch.__version__),
                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                       "weights_path": str(weight[0]), "weights_sha256": weight[1]})
            seed_all(cfg["seed"])
            model = SmallUNet(pretrained=cfg["model"]["pretrained"], freeze_bn=cfg["model"]["freeze_encoder_bn_stats"])
            torch.save({"model": model.state_dict(), "config": cfg}, suite / "initial.pt")
            del model
        return str(suite)

    suite = Path(master(ctx, create_or_verify))
    if ctx.rank == 0:
        print(f"SUITE={suite}\nOrder: A supervised/eval -> B SSL -> B finetune/eval -> C continuation/eval", flush=True)
    if (suite / "completed.json").exists():
        if ctx.rank == 0:
            print(f"Already completed: {read_json(suite / 'completed.json')['comparison']}", flush=True)
        return suite
    initial = suite / "initial.pt"
    stages, evaluations = {}, {}

    def stage(name, **kwargs):
        result = run_stage(suite, cfg, ctx, name, initial, overall_start=started, **kwargs)
        if result is None:
            if ctx.rank == 0:
                print(f"PAUSED after checkpoint: {suite}; resume with --resume-suite", flush=True)
            return None
        stages[name] = result
        return result

    def assess(arm, state):
        record = suite / f"evaluation_{arm}.json"
        def action():
            if record.exists():
                return read_json(record)
            result = evaluate(state["checkpoint"], cfg, ctx.device, arm, suite, results_root)
            write_json(record, result)
            return result
        result = master(ctx, action)
        evaluations[arm] = result
        return result

    a = stage("A_supervised", future_work={"updates": cfg['train']['updates'] + 2 * cfg['ssl']['updates'], "evaluations": 3})
    if a is None:
        return suite
    ae = assess("A", a)
    seg_step = a['wall_seconds'] / a['updates']
    ssl = stage("B_ssl", task="ssl", planned_steps=cfg['ssl']['updates'],
                future_work={"updates": cfg['train']['updates'] + cfg['ssl']['updates'], "seconds_per_update": seg_step,
                             "evaluations": 2, "evaluation_seconds": ae['wall_seconds']})
    if ssl is None:
        return suite
    expected_extra = math.ceil(ssl['compute_seconds'] / max(a['compute_seconds'] / a['updates'], 1e-9))
    b = stage("B_finetune", encoder_checkpoint=ssl['checkpoint'],
              future_work={"updates": expected_extra, "seconds_per_update": seg_step,
                           "evaluations": 2, "evaluation_seconds": ae['wall_seconds']})
    if b is None:
        return suite
    assess("B", b)
    budget = budget_remaining(a['compute_seconds'], ssl['compute_seconds'], b['compute_seconds'])
    planned = max(1, math.ceil(budget / max(a['compute_seconds'] / a['updates'], 1e-9)))
    if budget > 0:
        c = stage("C_extra", parent_checkpoint=a['checkpoint'], budget_seconds=budget, planned_steps=planned,
                  future_work={"updates": 0, "evaluations": 1, "evaluation_seconds": ae['wall_seconds']})
        if c is None:
            return suite
    else:
        c = {**a, "updates": 0, "compute_seconds": 0., "wall_seconds": 0., "stage": "C_extra",
             "note": "A already exhausted B total measured compute; no additional updates"}
        stages['C_extra'] = c
    assess("C", c)

    def finish():
        destination = comparison_report(suite, evaluations, stages, cfg, results_root)
        result = {"comparison": str(destination), "stages": stages,
                  "evaluations": evaluations, "finished_at": datetime.now().astimezone().isoformat(),
                  "this_invocation_wall_seconds": time.monotonic() - started}
        write_json(suite / "completed.json", result)
        print(f"DONE | COMPARISON={destination} | elapsed={time.monotonic()-started:.1f}s", flush=True)
        return result
    master(ctx, finish)
    return suite


def main():
    parser = argparse.ArgumentParser(description="Small-CNN masked reconstruction vs supervised A/B/C experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "suite", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path)
        if command == "suite":
            sub.add_argument("--resume-suite", type=Path)
        if command == "evaluate":
            sub.add_argument("--checkpoint", required=True, type=Path)
            sub.add_argument("--arm", choices=("A", "B", "C"), required=True)
    args = parser.parse_args()
    if args.command == "suite" and args.resume_suite:
        cfg = read_json(args.resume_suite / "config.json")
        if args.config and config(args.config) != cfg:
            parser.error("--config differs from saved suite config")
    else:
        cfg = config(args.config)
    if args.command == "prepare":
        prepare(cfg)
        return
    if args.command == "evaluate":
        saved_cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
        if args.config and cfg != saved_cfg:
            parser.error("Evaluation config must match checkpoint config")
        cfg = saved_cfg
    ctx = Context(cfg)
    try:
        if args.command == "suite":
            run_suite(cfg, ctx, resume=args.resume_suite)
        else:
            master(ctx, lambda: signature(cfg, verify=True))
            master(ctx, lambda: evaluate(args.checkpoint, cfg, ctx.device, args.arm, args.checkpoint.parents[2]))
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
