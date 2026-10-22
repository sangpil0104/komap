"""Bounded synthetic DDP check: no project images, models, or training runs."""
import json
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sslab.common import config
from sslab.engine import Context, seed_all
from sslab.loss import reconstruction_loss, segmentation_loss
from sslab.model import SmallUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--microbatches', type=int, default=1)
    args = parser.parse_args()
    ctx = Context(config())
    try:
        details = []
        for task in ('ssl', 'seg'):
            seed_all(100)
            net = SmallUNet(outputs=1 if task == 'ssl' else 4).to(ctx.device)
            wrapped = DDP(net, device_ids=[ctx.local])
            torch.manual_seed(1000 + ctx.rank)
            torch.cuda.reset_peak_memory_stats(ctx.device)
            for micro in range(args.microbatches):
                image = torch.rand(1, 3, args.size, args.size, device=ctx.device)
                sync = wrapped.no_sync() if micro < args.microbatches - 1 else nullcontext()
                with sync:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = wrapped(image)
                        if task == 'ssl':
                            target = torch.rand(1, 1, args.size, args.size, device=ctx.device)
                            hidden = torch.zeros_like(target)
                            hidden[..., ::2, :] = 1
                            loss = reconstruction_loss(output, target, hidden)
                        else:
                            target = torch.randint(0, 4, (1, args.size, args.size), device=ctx.device)
                            loss = segmentation_loss(output, target)
                    (loss / args.microbatches).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            torch.optim.AdamW(net.parameters(), lr=1e-4).step()
            stats = torch.stack([next(net.parameters()).sum(), next(net.parameters()).square().sum()])
            gathered = [torch.empty_like(stats) for _ in range(ctx.world)]
            dist.all_gather(gathered, stats)
            if not all(torch.equal(gathered[0], item) for item in gathered):
                raise AssertionError('DDP parameters differ across ranks')
            details.append({'task': task, 'finite_loss': bool(torch.isfinite(loss)),
                            'output_shape': list(output.shape), 'identical_parameters': True})
            details[-1]['peak_memory_MiB_rank0'] = torch.cuda.max_memory_allocated(ctx.device) / 1024**2
            del wrapped, net
        if ctx.rank == 0:
            print(json.dumps({'synthetic_only': True, 'world_size': ctx.world,
                              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                              'microbatches_per_rank': args.microbatches,
                              'checks': details}, indent=2), flush=True)
    finally:
        ctx.close()


if __name__ == '__main__':
    main()
