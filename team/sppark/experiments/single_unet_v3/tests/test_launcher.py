"""Regression coverage for real torchrun option parsing; never starts training."""
import argparse
import json
import os
import subprocess

import pytest
from torch.distributed.run import get_args_parser

import run as cli


@pytest.mark.parametrize("world", [1, 6])
def test_worker_flags_belong_to_training_script(tmp_path, monkeypatch, world):
    cfg = cli.config("S0")
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "latest.json").write_text(json.dumps(dict(update=2000, checkpoint="unused.pt")))
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 6)
    captured = []
    real_subprocess_run = subprocess.run

    def capture(command, **kwargs):
        captured.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", capture)
    args = argparse.Namespace(gpus=world, resume=tmp_path, scale=None, split=None,
                              empty_union=None, updates=None)
    run, evaluation = cli.launch(args)
    assert run == tmp_path and evaluation is None
    command, = captured
    expected_worker_flags = ["_worker", "--run", str(tmp_path), "--resume"]
    assert command[-4:] == expected_worker_flags
    if world == 6:
        parsed = get_args_parser().parse_args(command[3:])
        assert parsed.training_script == str(cli.ROOT / "run.py")
        assert parsed.training_script_args == expected_worker_flags
        # Six real launcher processes, but argparse --help exits before train() is reached.
        smoke = real_subprocess_run(command + ["--help"], capture_output=True, text=True,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                                    timeout=90)
        assert smoke.returncode == 0, smoke.stdout + smoke.stderr
        assert smoke.stdout.count("usage: run.py _worker") == 6
        assert "ambiguous option" not in smoke.stderr
    else:
        assert "torch.distributed.run" not in command
    assert not (tmp_path / "training.jsonl").exists()
    assert not (tmp_path / "worker.json").exists()
