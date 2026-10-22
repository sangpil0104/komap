"""Compare uneven-shard DDP gradients against an unpartitioned batch, no updates."""

import copy
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from distributed_core import batch_slice, global_loss
from runtime import ROOT, save_json
from komap_seg.losses import ComboLoss, BoundaryHeadLoss


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
        )
        self.seg = torch.nn.Conv2d(8, 4, 1)
        self.boundary = torch.nn.Conv2d(8, 1, 1)

    def forward(self, x):
        features = self.body(x)
        return self.seg(features), self.boundary(features)


def verify(cfg, rank, world, device):
    # Isolate reduction mathematics from TF32 kernel selection for batch 8 vs 1/2.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(11)
    reference = Tiny().to(device).train()
    distributed = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        copy.deepcopy(reference)
    )
    wrapped = DDP(distributed, device_ids=[device.index])
    cases = []
    for mode in ["all_zero", "stripes", "random"]:
        torch.manual_seed(41)
        x = torch.randn(8, 3, 32, 32, device=device)
        if mode == "all_zero":
            y = torch.zeros(8, 32, 32, device=device, dtype=torch.int64)
        elif mode == "stripes":
            y = (torch.arange(32, device=device) // 8)[None, None].expand(8, 32, 32)
        else:
            y = torch.randint(0, 4, (8, 32, 32), device=device)
        reference.zero_grad(set_to_none=True)
        wrapped.zero_grad(set_to_none=True)
        logits, boundary = reference(x)
        expected = ComboLoss()(logits, y) + BoundaryHeadLoss()(boundary, y)
        expected.backward()
        selected = batch_slice(8, world, rank)
        logits, boundary = wrapped(x[selected])
        actual = global_loss(logits, boundary, y[selected])
        actual.backward()
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        max_diff = 0.0
        for (name, param), (other_name, other) in zip(
            reference.named_parameters(), distributed.named_parameters()
        ):
            assert name == other_name
            torch.testing.assert_close(
                other.grad,
                param.grad,
                atol=2e-5,
                rtol=2e-4,
                msg=lambda message: name + ": " + message,
            )
            max_diff = max(max_diff, float((other.grad - param.grad).abs().max()))
        for (name, value), (other_name, other) in zip(
            reference.named_buffers(), distributed.named_buffers()
        ):
            assert name == other_name
            torch.testing.assert_close(other, value, atol=2e-6, rtol=2e-5)
        cases.append(
            {
                "target": mode,
                "loss_abs_difference": abs(
                    float(actual.detach()) - float(expected.detach())
                ),
                "max_parameter_gradient_abs_difference": max_diff,
            }
        )
    reports = [None] * world
    dist.all_gather_object(reports, {"rank": rank, "cases": cases})
    if rank == 0:
        report = {
            "status": "passed",
            "world_size": world,
            "global_batch_size": 8,
            "optimizer_updates": 0,
            "tf32_disabled_for_equivalence_check": True,
            "batchnorm_running_stats_match": True,
            "ranks": reports,
        }
        save_json(ROOT / "reports/distributed_equivalence.json", report)
        print(json.dumps(report, indent=2), flush=True)


class ConstantModel(torch.nn.Module):
    def forward(self, x):
        output = x.new_zeros((x.shape[0], 4, *x.shape[-2:]))
        output[:, 0] = 5
        return output


def verify_evaluation(cfg, rank, world, device, validate):
    """Run the complete parallel result exporter on temporary synthetic images."""
    from pathlib import Path
    import tempfile
    import numpy as np
    from PIL import Image
    import runtime
    from komap_seg.palette import PALETTE

    temp = tempfile.TemporaryDirectory(prefix="hrnet_ddp_eval_") if rank == 0 else None
    shared = [temp.name if rank == 0 else None]
    dist.broadcast_object_list(shared, src=0)
    root = Path(shared[0])
    if rank == 0:
        for folder in ["images", "masks"]:
            (root / "data/valid" / folder).mkdir(parents=True)
        for index in range(20):
            shape = (19 + index % 4, 27)
            Image.fromarray(np.full((*shape, 3), 100, dtype=np.uint8)).save(
                root / f"data/valid/images/test{index:02}_image.png"
            )
            Image.fromarray(PALETTE[np.zeros(shape, dtype=np.uint8)]).save(
                root / f"data/valid/masks/test{index:02}_mask.png"
            )
        (root / "synthetic.pt").write_bytes(
            b"synthetic fixture; predictions supplied by constant model"
        )
    dist.barrier()
    runtime.REPO = root
    test_cfg = dict(
        cfg, data_root=str(root / "data"), patch_size=32, input_scale=2.0, overlap=4
    )
    score, result = validate(
        ConstantModel().to(device), test_cfg, rank, world, device, root / "synthetic.pt"
    )
    assert score == 1.0
    if rank == 0:
        summary = json.loads((result / "summary.json").read_text())
        assert summary["num_images"] == 20 and summary["image_miou"] == 1.0
        assert len(list((result / "valid_mask_comparison").glob("*.png"))) == 20
        assert len(list((result / "predictions").glob("*.png"))) == 20
        for index in range(20):
            labels = runtime.strict_mask(
                result / f"predictions/test{index:02}_mask.png"
            )
            assert labels.shape == (19 + index % 4, 27) and not labels.any()
        report = {
            "status": "passed",
            "world_size": world,
            "synthetic_images": 20,
            "comparison_images": 20,
            "predictions": 20,
            "native_shapes_preserved": True,
            "temporary_results_only": True,
            "optimizer_updates": 0,
        }
        save_json(ROOT / "reports/distributed_evaluation.json", report)
        print(json.dumps(report, indent=2), flush=True)
    dist.barrier()
    if temp is not None:
        temp.cleanup()
