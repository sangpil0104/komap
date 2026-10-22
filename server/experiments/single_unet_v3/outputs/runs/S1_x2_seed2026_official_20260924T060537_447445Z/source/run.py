#!/usr/bin/env python
"""Explicit CLI entry point. Import/preflight/prepare never starts training."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from singlelab.common import OUTPUTS, ROOT, config, dataset_signature, digest, read_json, records, snapshot, stamp, write_json


def device_for(value):
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    return device


def preflight(args):
    from singlelab.data import PatchDataset, labels
    from singlelab.model import EfficientUNet
    from torchvision.models import EfficientNet_B4_Weights
    cfg = config("S0")
    signature = dataset_signature(cfg, verify=True)
    rows = records(cfg)
    from tqdm import tqdm
    for row in tqdm(rows["train"] + rows["valid"], desc="Validate masks", unit="mask"):
        target = labels(row["mask_path"])
        if target.shape != (int(row["height"]), int(row["width"])):
            raise ValueError(f"Image/mask shape mismatch: {row['image_id']}")
    torch.set_num_threads(4)
    model = EfficientUNet(cfg, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 96))
    assert output.shape == (1, 4, 64, 96) and torch.isfinite(output).all()
    sample_shapes = {}
    for name in ("S0", "S1", "S2"):
        x, target, edge = PatchDataset(rows["train"], config(name))[0]
        sample_shapes[name] = list(x.shape)
        assert target.shape == edge.shape == x.shape[-2:] and (target != 255).any()
    weights = ROOT / "weights" / Path(EfficientNet_B4_Weights.IMAGENET1K_V1.url).name
    result = dict(status="passed", training_started=False, optimizer_steps=0,
                  counts={k: len(v) for k, v in rows.items()}, input_shapes=sample_shapes,
                  model_parameters=sum(p.numel() for p in model.parameters()), pretrained_ready=weights.is_file(),
                  empty_union=cfg["evaluation"]["empty_union"], torch=str(torch.__version__),
                  dataset=signature)
    if args.profile:
        from singlelab.inference import autocast
        from singlelab.loss import segmentation_loss
        device = device_for(args.device)
        if device.type != "cuda":
            raise ValueError("--profile requires CUDA")
        del model
        profiles = []
        for name in tqdm(("S0", "S1", "S2"), desc="Synthetic gradient memory check", unit="scale"):
            cfg = config(name)
            model = EfficientUNet(cfg, pretrained=False).to(device).train()
            size = (cfg["sampling"]["core"] + 2 * cfg["sampling"]["halo"]) * cfg["scale"]
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.rand(1, 3, size, size, device=device)
            target = torch.zeros((1, size, size), dtype=torch.long, device=device)
            edge = torch.zeros_like(target, dtype=torch.float32)
            with autocast(device, cfg["train"]["precision"]):
                logits = model(x)
                loss, _ = segmentation_loss(logits, target, edge, cfg["loss"])
            loss.backward()  # Synthetic derivative check only; no optimizer is created.
            torch.cuda.synchronize(device)
            profiles.append(dict(experiment=name, size=size, peak_gib=torch.cuda.max_memory_allocated(device) / 2**30))
            del model, logits, loss, x, target, edge
            torch.cuda.empty_cache()
        result["synthetic_gradient_memory"] = profiles
    destination = OUTPUTS / "verification" / f"preflight_{stamp()}.json"
    write_json(destination, result)
    print(json.dumps(result, indent=2))
    print(f"Saved: {destination}")


def prepare(args):
    from singlelab.model import pretrained_state
    state = pretrained_state()
    print(f"ImageNet B4 weights ready ({len(state)} tensors); no training was started.")


def new_run(args, experiment, seed):
    cfg = config(experiment, seed, args.scale if experiment == "S3" else None, args.split or "official")
    if args.empty_union:
        cfg["evaluation"]["empty_union"] = args.empty_union
    if args.updates is not None:
        if args.updates <= 0:
            raise ValueError("updates must be positive")
        cfg["train"]["updates"] = args.updates
        cfg["train"]["warmup"] = min(cfg["train"]["warmup"], max(1, args.updates // 10))
    signature = dataset_signature(cfg, verify=True)
    from singlelab.model import pretrained_state
    pretrained_state()  # Download once before launching DDP workers.
    run = OUTPUTS / "runs" / f"{experiment}_x{cfg['scale']}_seed{cfg['seed']}_{cfg['data']['split']}_{stamp()}"
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", cfg)
    write_json(run / "dataset.json", signature)
    snapshot(run)
    import platform
    import torchvision
    from torchvision.models import EfficientNet_B4_Weights
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1
    write_json(run / "environment.json", dict(python=platform.python_version(), torch=str(torch.__version__),
               torchvision=str(torchvision.__version__), cuda=torch.version.cuda,
               pretrained_url=weights.url, pretrained_sha256=digest(ROOT / "weights" / Path(weights.url).name)))
    return run


def launch(args, experiment=None, seed=None):
    from singlelab.evaluation import evaluate
    from singlelab.report import make_report
    if args.gpus < 1 or args.gpus > torch.cuda.device_count():
        raise ValueError("--gpus must be positive and not exceed visible CUDA devices")
    if getattr(args, "resume", None):
        run = Path(args.resume).resolve()
        cfg = read_json(run / "config.json")
        for supplied, expected, name in ((experiment, cfg["experiment"], "experiment"),
                                         (seed, cfg["seed"], "seed"),
                                         (args.scale, cfg["scale"], "scale"),
                                         (args.empty_union, cfg["evaluation"]["empty_union"], "empty policy"),
                                         (args.split, cfg["data"]["split"], "split"),
                                         (args.updates, cfg["train"]["updates"], "updates")):
            if supplied is not None and supplied != expected:
                raise ValueError(f"Cannot change {name} on resume")
        if not (run / "latest.json").is_file():
            raise ValueError("No saved checkpoint; start a new run")
    else:
        if experiment is None:
            raise ValueError("Specify S0, S1, S2, S3, or --resume RUN")
        if experiment != "S3" and args.scale is not None:
            raise ValueError("--scale is only for S3")
        run = new_run(args, experiment, seed)
    print(f"Run: {run}", flush=True)
    with (run / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (run / "completed.json").exists():
            for name in ("SAVE_AND_STOP", "paused.json"):
                marker = run / name
                if marker.exists():
                    marker.rename(run / (name + ".resumed_" + stamp()))
            command = [sys.executable]
            if args.gpus > 1:
                # End torchrun options before passing worker flags such as --run.
                command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}", "--"]
            command += [str(Path(__file__).resolve()), "_worker", "--run", str(run)]
            if getattr(args, "resume", None):
                command += ["--resume"]
            subprocess.run(command, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        if not (run / "completed.json").exists():
            print(f"Saved and paused: {run}; suite will not launch another model", flush=True)
            return run, None
        # Training worker has exited, releasing GPU optimizer/activation memory.
        folder = evaluate(run, device_for("cuda:0"))
        report = make_report([folder])
        print(f"Result report: {report}", flush=True)
    return run, folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Download pretrained weights only")
    p.set_defaults(action=prepare)
    p = commands.add_parser("preflight", help="Verify data/model; no optimizer updates")
    p.add_argument("--profile", action="store_true", help="Synthetic forward/backward GPU memory check, no optimizer")
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(action=preflight)
    for name in ("train", "suite"):
        p = commands.add_parser(name, help="Explicitly start training, then evaluate and publish a local report")
        if name == "train":
            p.add_argument("experiment", nargs="?", choices=("S0", "S1", "S2", "S3"))
            p.add_argument("--seed", type=int)
            p.add_argument("--resume", type=Path)
        else:
            p.add_argument("--experiments", nargs="+", default=["S0", "S1", "S2"], choices=("S0", "S1", "S2"))
            p.add_argument("--seeds", nargs="+", type=int, default=[2026])
        p.add_argument("--gpus", type=int, default=1)
        p.add_argument("--scale", type=int, choices=(1, 2, 3))
        p.add_argument("--split", choices=("official", "fold0", "fold1", "fold2", "fold3", "fold4"))
        p.add_argument("--empty-union", choices=("one", "zero", "exclude"))
        p.add_argument("--updates", type=int, help="Override common budget; apply identically to compared runs")
    p = commands.add_parser("_worker", help=argparse.SUPPRESS)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p = commands.add_parser("import-run", help="Copy a checkpoint into a new migration run; no training")
    p.add_argument("run", type=Path)
    p.add_argument("--gpus", type=int, default=6)
    p = commands.add_parser("continue-suite", help="Resume the imported model, then train the remaining experiments")
    p.add_argument("run", type=Path, nargs="?")
    p.add_argument("--gpus", type=int, default=6)
    p.add_argument("--remaining", choices=("S0", "S1", "S2"), nargs="*", default=["S1", "S2"])
    p = commands.add_parser("pause", help="Request a checkpoint and stop at the next completed update")
    p.add_argument("run", type=Path)
    for name in ("evaluate", "predict", "compare-borders"):
        p = commands.add_parser(name)
        p.add_argument("run", type=Path)
        p.add_argument("--device", default="cuda:0")
        if name != "compare-borders":
            p.add_argument("--mode", choices=("full", "tiled"))
            p.add_argument("--pad", type=int)
            p.add_argument("--halo", type=int)
            p.add_argument("--offset", type=int)
            p.add_argument("--d4", action="store_true")
    p = commands.add_parser("report")
    p.add_argument("evaluations", type=Path, nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if hasattr(args, "action"):
        args.action(args)
    elif args.command == "_worker":
        from singlelab.train import train
        train(args.run, args.resume)
    elif args.command == "train":
        launch(args, args.experiment, args.seed)
    elif args.command == "suite":
        if args.scale is not None:
            parser.error("suite uses fixed S0/S1/S2 scales; run S3 separately")
        from tqdm import tqdm
        from singlelab.report import make_report
        tasks = [(e, s) for s in args.seeds for e in args.experiments]
        evaluations = []
        for experiment, seed in tqdm(tasks, desc="Experiment suite", unit="run", dynamic_ncols=True):
            _, folder = launch(args, experiment, seed)
            if folder is None:
                return
            evaluations.append(folder)
        print(f"Combined report: {make_report(evaluations)}")
    elif args.command in ("evaluate", "predict", "compare-borders"):
        from singlelab.evaluation import border_comparison, evaluate
        from singlelab.report import make_report
        if args.command == "compare-borders":
            folders = border_comparison(args.run, device_for(args.device))
        else:
            overrides = {k: getattr(args, k) for k in ("mode", "pad", "halo", "offset") if getattr(args, k) is not None}
            if any(v < 0 for k, v in overrides.items() if k != "mode"):
                parser.error("pad/halo/offset must be nonnegative")
            overrides["d4"] = args.d4
            folders = [evaluate(args.run, device_for(args.device), overrides, split="test" if args.command == "predict" else "valid")]
        print(f"Result report: {make_report(folders)}")
    elif args.command == "report":
        from singlelab.report import make_report
        print(make_report(args.evaluations))
    elif args.command == "import-run":
        if args.gpus < 1:
            parser.error("--gpus must be positive")
        from singlelab.migration import import_run
        print(f"Ready (training not started): {import_run(args.run, args.gpus)}")
    elif args.command == "pause":
        run = args.run.resolve()
        if (run / "completed.json").exists():
            parser.error("Run is already complete")
        (run / "SAVE_AND_STOP").touch(exist_ok=False)
        print("Save-and-stop requested; wait for paused.json/checkpoint before terminating anything.")
    elif args.command == "continue-suite":
        from singlelab.report import make_report
        run = args.run or Path(read_json(OUTPUTS / "migration/ready.json")["run"])
        cfg = read_json(run / "config.json")
        options = argparse.Namespace(gpus=args.gpus, resume=run, scale=None, split=cfg["data"]["split"],
                                     empty_union=cfg["evaluation"]["empty_union"], updates=cfg["train"]["updates"])
        _, folder = launch(options)
        if folder is None:
            return
        evaluations = [folder]
        options.resume = None
        for experiment in args.remaining:
            if experiment == cfg["experiment"]:
                parser.error("--remaining must not repeat the resumed experiment")
            _, folder = launch(options, experiment, cfg["seed"])
            if folder is None:
                return
            evaluations.append(folder)
        print(f"Continuation suite report: {make_report(evaluations)}")


if __name__ == "__main__":
    main()
