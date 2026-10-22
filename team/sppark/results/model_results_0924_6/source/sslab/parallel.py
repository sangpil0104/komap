"""Keep the effective batch unchanged when world size does not divide it."""
import math

from torch.utils.data import Dataset


class DistributedUpdateIndices:
    """Equal forward counts per rank, exactly B real samples per optimizer update.

    For B=8,W=6, slots are [0..5] then [6,7,dummy,dummy,dummy,dummy].
    Dummy forwards participate in DDP but contribute zero loss/gradient.
    """
    def __init__(self, updates, batch, rank, world, start=0):
        if batch < 1 or world < 1 or not 0 <= rank < world or not 0 <= start <= updates:
            raise ValueError("Invalid distributed batch/sampler settings")
        self.updates, self.batch, self.rank, self.world, self.start = updates, batch, rank, world, start
        self.rounds = math.ceil(batch / world)

    def __iter__(self):
        for update in range(self.start, self.updates):
            for micro in range(self.rounds):
                slot = micro * self.world + self.rank
                real = slot < self.batch
                index = update * self.batch + (slot if real else min(self.rank, self.batch - 1))
                yield index, real

    def __len__(self):
        return (self.updates - self.start) * self.rounds


class WeightedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        index, real = item
        return (*self.dataset[index], float(real))


def gradient_weight(batch, world, real):
    # DDP averages gradients over W ranks: (1/W) * sum(W/B * g_i) = mean_B(g).
    return float(real) * world / batch
