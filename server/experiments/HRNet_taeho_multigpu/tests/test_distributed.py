import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runtime
from distributed_core import augmentation, batch_sizes, batch_slice, global_loss
from komap_seg.losses import ComboLoss, BoundaryHeadLoss


def test_six_gpu_batch_covers_eight_samples_once():
    assert batch_sizes(8, 6) == [2, 2, 1, 1, 1, 1]
    indices = list(range(8))
    assert [i for rank in range(6) for i in indices[batch_slice(8, 6, rank)]] == indices
    with pytest.raises(ValueError):
        batch_sizes(8, 9)


@pytest.mark.parametrize("target_mode", ["all_zero", "stripes", "random"])
def test_global_loss_matches_original_and_gradients(target_mode):
    torch.manual_seed(42)
    logits = torch.randn(8, 4, 16, 16, requires_grad=True)
    boundary = torch.randn(8, 1, 16, 16, requires_grad=True)
    if target_mode == "all_zero":
        target = torch.zeros(8, 16, 16, dtype=torch.int64)
    elif target_mode == "stripes":
        target = (torch.arange(16) // 4)[None, None].expand(8, 16, 16)
    else:
        target = torch.randint(0, 4, (8, 16, 16))
    expected = ComboLoss()(logits, target) + BoundaryHeadLoss()(boundary, target)
    expected_grad = torch.autograd.grad(expected, (logits, boundary))
    actual = global_loss(logits, boundary, target)
    actual_grad = torch.autograd.grad(actual, (logits, boundary))
    torch.testing.assert_close(actual, expected)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)


def test_ablation_retains_all_other_transforms_and_configs():
    baseline = augmentation("baseline").to_dict()
    expected = baseline["transform"]["transforms"]
    baseline["transform"]["transforms"] = [
        t
        for t in expected
        if t["__class_fullname__"] not in {"GaussNoise", "GaussianBlur"}
    ]
    assert augmentation("no_gaussian").to_dict() == baseline
    for variant, folder in [
        ("baseline", "HRNet_taeho"),
        ("no_gaussian", "HRNet_taeho_no_gaussian"),
    ]:
        before = json.loads((ROOT.parent / folder / "configs/default.json").read_text())
        after = json.loads((ROOT / "configs" / f"{variant}.json").read_text())
        assert before == after


@pytest.mark.parametrize("fail_first", [False, True])
def test_sequence_waits_and_stops_on_failure(tmp_path, fail_first):
    (tmp_path / "run_sequence.sh").write_bytes((ROOT / "run_sequence.sh").read_bytes())
    stub = tmp_path / "run_distributed.sh"
    stub.write_text(
        '#!/usr/bin/env bash\necho "$*" >> calls.txt\nif [[ "$*" == *baseline* && "${FAIL_FIRST:-0}" == 1 ]]; then exit 7; fi\n'
    )
    stub.chmod(0o755)
    env = dict(os.environ, FAIL_FIRST=str(int(fail_first)))
    dry = subprocess.run(
        ["bash", str(tmp_path / "run_sequence.sh"), "--dry-run"],
        env=env,
        capture_output=True,
    )
    assert dry.returncode == 0 and not (tmp_path / "calls.txt").exists()
    result = subprocess.run(
        ["bash", str(tmp_path / "run_sequence.sh")], env=env, capture_output=True
    )
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert calls[0] == "train --variant baseline --execute"
    if fail_first:
        assert result.returncode == 7 and len(calls) == 1
    else:
        assert (
            result.returncode == 0
            and calls[1] == "train --variant no_gaussian --execute"
        )


def test_launcher_assigns_six_workers_and_validates_gpu_ids(tmp_path):
    stub = tmp_path / "python_stub"
    log = tmp_path / "launch.json"
    stub.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$*" > "$LAUNCH_TEST_LOG"\n'
    )
    stub.chmod(0o755)
    env = dict(
        os.environ,
        HRNET_PYTHON=str(stub),
        HRNET_GPUS="0,1,2,3,4,5",
        LAUNCH_TEST_LOG=str(log),
    )
    result = subprocess.run(
        ["bash", str(ROOT / "run_distributed.sh"), "train"],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0
    lines = log.read_text().splitlines()
    assert lines[0] == "0,1,2,3,4,5"
    assert "--nproc_per_node=6" in lines[1]
    for invalid in ["0,0", "6", "", "0,1,"]:
        env["HRNET_GPUS"] = invalid
        # Empty override follows shell default intentionally; test empty component instead.
        if invalid == "":
            env["HRNET_GPUS"] = "0,,1"
        result = subprocess.run(
            ["bash", str(ROOT / "run_distributed.sh"), "train"],
            env=env,
            capture_output=True,
        )
        assert result.returncode == 2


@pytest.mark.parametrize("variant", ["baseline", "no_gaussian"])
def test_reports_keep_variant_identity_and_image_metric(tmp_path, variant):
    cfg = runtime.load_config(ROOT / "configs" / f"{variant}.json")
    row = {
        "image_id": "synthetic",
        "height": 1,
        "width": 2,
        **runtime.image_metrics(
            torch.tensor([[0, 2]]).numpy(), torch.tensor([[0, 0]]).numpy()
        ),
    }
    checkpoint = tmp_path / "stub.pt"
    checkpoint.write_bytes(b"synthetic")
    summary = runtime.write_report(tmp_path, [row], checkpoint, cfg)
    assert summary["model_name"] == cfg["model_name"]
    assert summary["image_miou"] == 0.625
    assert cfg["model_name"] in (tmp_path / (tmp_path.name + ".md")).read_text()
