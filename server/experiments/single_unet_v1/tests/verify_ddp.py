"""Two-GPU synthetic gradient synchronization check. No optimizer or real data."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from singlelab.common import config
from singlelab.inference import autocast
from singlelab.loss import segmentation_loss
from singlelab.model import EfficientUNet


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    dist.init_process_group("nccl")
    try:
        cfg = config("S0")
        torch.manual_seed(2026)
        model = DistributedDataParallel(EfficientUNet(cfg).cuda(local), device_ids=[local])
        torch.manual_seed(2026 + local)
        for micro in tqdm(range(2), desc="DDP synthetic gradients (no updates)", disable=local != 0):
            from contextlib import nullcontext
            context = model.no_sync() if micro == 0 else nullcontext()
            with context:
                x = torch.rand(1, 3, 64, 96, device=local)
                target = torch.randint(4, (1, 64, 96), device=local)
                with autocast(torch.device("cuda", local), "bf16"):
                    loss, _ = segmentation_loss(model(x), target, torch.zeros_like(target), cfg["loss"])
                (loss / 2).backward()
        gradient = model.module.head.weight.grad
        assert torch.isfinite(gradient).all()
        gathered = [torch.zeros_like(gradient) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, gradient)
        for other in gathered:
            torch.testing.assert_close(gradient, other, rtol=0, atol=0)
        if local == 0:
            print("PASS: synchronized finite gradients; optimizer steps=0; training_started=False", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
