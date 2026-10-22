from __future__ import annotations

import argparse
import json
import os
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from lab.workspace import BASELINE, SOURCE, WORKSPACE, load_experiment, plan, read_json, run_dir, verify_preservation


def parser():
    cli = argparse.ArgumentParser(description="Isolated accuracy experiments; original models/source/results stay unchanged", allow_abbrev=False)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Show experiments and their differences")
    commands.add_parser("verify", help="Hash-check all original files and the frozen baseline")
    commands.add_parser("compare", help="Build CSV and browser comparison reports")
    p = commands.add_parser("train", help="Train a new experiment using the frozen model implementation", allow_abbrev=False)
    p.add_argument("experiment", choices=list(plan()["experiments"]))
    p.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument("--split", default="official", choices=["official", "all90"]+[f"fold{i}" for i in range(5)])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pretrained", help="Original official DINOv3-L pretrained .pth file")
    p.add_argument("--smoke", action="store_true", help="One update per stage; separate outputs/smoke directory")
    p.add_argument("--stop-after", type=int, help="Pause at a global optimizer update for resume verification")
    p.add_argument("--set", dest="overrides", nargs="*", default=[])
    p = commands.add_parser("suite", help="Run the four controlled CNN experiments, validation, seed ensemble and comparison")
    p.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], default=8)
    p = commands.add_parser("evaluate", help="Evaluate a checkpoint on its held-out images, or export test predictions")
    p.add_argument("experiment", choices=["baseline", *plan()["experiments"]])
    p.add_argument("--model-split", default="official", choices=["official", "all90"]+[f"fold{i}" for i in range(5)])
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--d4", action="store_true")
    p.add_argument("--scales", nargs="+", type=float, default=[1.])
    p.add_argument("--tag", help="Distinct output name for another inference experiment")
    p.add_argument("--smoke", action="store_true")
    p = commands.add_parser("ensemble", help="Average original-resolution class probabilities, then argmax")
    p.add_argument("--config", default=str(WORKSPACE/"configs/ensemble_seeds.yaml"))
    p.add_argument("--model-split", default="official", choices=["official", "all90"]+[f"fold{i}" for i in range(5)])
    p.add_argument("--split", default="valid", choices=["valid", "test"])
    p.add_argument("--smoke", action="store_true")
    p = commands.add_parser("monitor", help="Show the existing live training dashboard for a new experiment")
    p.add_argument("experiment", choices=["suite", *plan()["experiments"]])
    p.add_argument("--split", default="official")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    return cli


def main():
    import sys
    cli = parser()
    args = cli.parse_args()
    if args.command == "list":
        print(f"Workspace: {WORKSPACE}\nFrozen baseline: {BASELINE/'run/best.pt'}\n")
        for name, details in plan()["experiments"].items():
            cfg = load_experiment(name)
            print(f"{name:12} seed={cfg['seed']} updates={sum(s['max_updates'] for s in cfg['train']['stages'])}  {details['description']}")
        print("\nDefault suite excludes DINOv3, which requires official pretrained weights.")
    elif args.command == "verify":
        print(json.dumps(verify_preservation(), indent=2))
    elif args.command == "train":
        from lab.training import run_training
        if args.stop_after is not None and args.stop_after < 1:
            cli.error("--stop-after must be positive")
        # Fail before launching eight workers if the optional backbone file is absent.
        if args.experiment == "dino_large":
            cfg = load_experiment(args.experiment, args.overrides, args.pretrained)
            checkpoint = cfg["model"].get("pretrained_checkpoint")
            if not checkpoint or not Path(checkpoint).is_file():
                cli.error("DINOv3-L official weights are required: --pretrained /path/to/original_filename.pth")
        if "LOCAL_RANK" not in os.environ and args.gpus > 1:
            from lab.suite import launch_training
            launch_training(sys.argv[1:], args.gpus)
        else:
            run_training(args)
    elif args.command == "suite":
        from lab.suite import run_suite
        run_suite(args.gpus)
    elif args.command == "evaluate":
        from lab.inference import evaluate_one
        verify_preservation(full=False)
        folder, result = evaluate_one(args.experiment, args.model_split, args.split, args.d4, args.scales, args.smoke, args.tag)
        print(json.dumps(result, indent=2))
        print(f"Output: {folder}")
    elif args.command == "ensemble":
        from lab.inference import ensemble
        verify_preservation(full=False)
        folder, result = ensemble(args.config, args.model_split, args.split, args.smoke)
        print(json.dumps(result, indent=2))
        print(f"Output: {folder}")
    elif args.command == "compare":
        from lab.report import compare_results
        compare_results()
    elif args.command == "monitor":
        if args.experiment == "suite":
            from lab.monitor import monitor_suite
            monitor_suite(args.port)
            return
        from alphase.cli.monitor import Handler
        root = run_dir(args.experiment, args.split, args.smoke)
        if not (root/"index.html").exists():
            raise FileNotFoundError(f"Start this experiment first: {root}")
        server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(root)))
        print(f"Training monitor: http://127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
