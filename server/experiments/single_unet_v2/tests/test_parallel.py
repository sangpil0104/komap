import math

import numpy as np
import pytest

from singlelab.parallel import DistributedUpdateIndices, gradient_weight


@pytest.mark.parametrize("world", [1, 2, 3, 4, 5, 6, 7, 8, 12])
def test_batch8_exact_coverage_and_weight(world):
    ranks = [list(DistributedUpdateIndices(5, 8, rank, world, start=2)) for rank in range(world)]
    rounds = math.ceil(8 / world)
    assert len({len(x) for x in ranks}) == 1
    for update in range(2, 5):
        section = [(index, real) for rank in ranks for index, real in
                   rank[(update - 2) * rounds:(update - 1) * rounds]]
        real_indices = [index for index, real in section if real]
        assert sorted(real_indices) == list(range(update * 8, (update + 1) * 8))
        averaged_gradient = sum(gradient_weight(8, world, real) * index for index, real in section) / world
        assert averaged_gradient == pytest.approx(np.mean(real_indices))


def test_six_gpu_slots_and_dummy_weights():
    first = [list(DistributedUpdateIndices(1, 8, rank, 6)) for rank in range(6)]
    assert [sum(real for _, real in row) for row in first] == [2, 2, 1, 1, 1, 1]
    assert gradient_weight(8, 6, True) == .75
    assert gradient_weight(8, 6, False) == 0
