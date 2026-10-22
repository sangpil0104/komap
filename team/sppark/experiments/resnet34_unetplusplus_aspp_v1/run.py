#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add package directory to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model_lab.workspace import load_config, OUTPUTS_DIR
from model_lab.models import build_model
from model_lab.trainer import train_runner
from model_lab.evaluation import evaluate_dataset
from model_lab.inference import run_inference_on_test
from model_lab.report import generate_report


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet34 U-Net++ + Residual ASPP CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--gpus", type=int, default=8, choices=[1, 2, 4, 8], help="Number of GPUs to use")
    train_parser.add_argument("--smoke", action="store_true", help="Run short smoke test (1-2 updates)")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    train_parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")
    train_parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate validation set")
    eval_parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to evaluate")
    eval_parser.add_argument("--d4", action="store_true", help="Use 8-way D4 test-time augmentation")

    # Predict
    pred_parser = subparsers.add_parser("predict", help="Generate predictions on test set")
    pred_parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to use")
    pred_parser.add_argument("--output", type=str, default=None, help="Directory to save masks")
    pred_parser.add_argument("--d4", action="store_true", default=True, help="Use D4 TTA")

    # Report
    subparsers.add_parser("report", help="Generate or update results/model_results_0921_1.md")

    # Test
    subparsers.add_parser("test", help="Run unit tests")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        cfg = load_config(args.config)
        if args.worker or args.gpus == 1:
            train_runner(cfg, smoke=args.smoke, resume_path=args.resume)
        else:
            # Multi-GPU launch via torchrun
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.gpus}",
                str(ROOT / "run.py"),
                "train",
                "--gpus",
                str(args.gpus),
                "--worker",
            ]
            if args.smoke:
                cmd.append("--smoke")
            if args.resume:
                cmd.extend(["--resume", str(args.resume)])
            if args.config:
                cmd.extend(["--config", str(args.config)])
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            subprocess.run(cmd, env=env, check=True)

    elif args.command == "evaluate":
        cfg = load_config()
        ckpt_path = Path(args.checkpoint) if args.checkpoint else (OUTPUTS_DIR / "runs/best.pt")
        if not ckpt_path.is_file():
            print(f"Error: Checkpoint not found at {ckpt_path}. Please train the model first.", file=sys.stderr)
            sys.exit(1)

        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("selected_model", ckpt.get("model", ckpt))
        model.load_state_dict(state_dict)

        mode_name = "d4" if args.d4 else "plain"
        out_dir = OUTPUTS_DIR / f"evaluations/{mode_name}"
        metrics = evaluate_dataset(model, d4=args.d4, device=device, out_dir=out_dir)
        print("\nEvaluation Summary:")
        print(f"Mode: {metrics['mode']}")
        print(f"Image-mean mIoU: {metrics['image_mean_miou']:.2f}%")
        print(f"Global mIoU:     {metrics['global_miou']:.2f}%")
        print("Class IoUs:")
        for cls_name, val in metrics["class_ious"].items():
            print(f"  {cls_name}: {val:.2f}%")
        print(f"Saved evaluation artifacts to {out_dir}")

    elif args.command == "predict":
        cfg = load_config()
        ckpt_path = Path(args.checkpoint) if args.checkpoint else (OUTPUTS_DIR / "runs/best.pt")
        if not ckpt_path.is_file():
            print(f"Error: Checkpoint not found at {ckpt_path}. Please train the model first.", file=sys.stderr)
            sys.exit(1)

        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("selected_model", ckpt.get("model", ckpt))
        model.load_state_dict(state_dict)

        out_dir = Path(args.output) if args.output else (OUTPUTS_DIR / "predictions")
        run_inference_on_test(model, output_dir=out_dir, d4=args.d4, device=device)

    elif args.command == "report":
        report_file = generate_report()
        print(f"Generated report at {report_file}")

    elif args.command == "test":
        import pytest
        sys.exit(pytest.main(["-v", str(ROOT / "tests")]))


if __name__ == "__main__":
    main()

