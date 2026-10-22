"""Verify real checkpoint/optimizer restoration on every GPU, without training."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from singlelab.common import read_json
from singlelab.model import EfficientUNet
from singlelab.parallel import DistributedUpdateIndices
from singlelab.train import restore_rng, seed_all


def main():
    run = Path(sys.argv[1]).resolve()
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    dist.init_process_group("nccl")
    try:
        cfg = read_json(run / "config.json")
        migration = read_json(run / "migration.json")
        assert world == migration["target_world_size"]
        state = torch.load(run / read_json(run / "latest.json")["checkpoint"], map_location="cpu", weights_only=False)
        raw = EfficientUNet(cfg).cuda(local)
        model = DistributedDataParallel(raw, device_ids=[local])
        raw.load_state_dict(state["model"])
        encoder = list(raw.encoder.parameters())
        ids = {id(p) for p in encoder}
        groups = [dict(params=encoder, lr=cfg["train"]["encoder_lr"]),
                  dict(params=[p for p in raw.parameters() if id(p) not in ids], lr=cfg["train"]["decoder_lr"])]
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg["train"]["weight_decay"])
        optimizer.load_state_dict(state["optimizer"])
        assert int(optimizer.state[encoder[0]]["step"]) == migration["imported_update"]
        assert torch.equal(raw.head.weight.cpu(), state["model"]["head.weight"])
        for group, original in zip(optimizer.param_groups, state["optimizer"]["param_groups"]):
            assert group["lr"] == original["lr"]
        if local < state["world_size"]:
            restore_rng(state["rng"][local])
        else:
            seed_all((cfg["seed"] + 10_000 + local + 1_000_003 * state["update"]) % 2**32)
        sampler = iter(DistributedUpdateIndices(state["update"] + 1, 8, local, world, state["update"]))
        slots = list(sampler)
        all_slots = [None] * world
        dist.all_gather_object(all_slots, slots)
        actual = sorted(index for rank in all_slots for index, real in rank if real)
        assert actual == list(range(state["update"] * 8, (state["update"] + 1) * 8))
        # Model is loaded and DDP constructed; no forward/backward or optimizer step.
        if local == 0:
            print(f"PASS: checkpoint update {state['update']}, optimizer step/LR/model/RNG restored on {world} GPUs; next samples {actual}; optimizer steps executed=0", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
