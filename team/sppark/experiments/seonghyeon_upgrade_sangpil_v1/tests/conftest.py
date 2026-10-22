import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import pytest
from alphase.config import load_config

torch.set_num_threads(1)


@pytest.fixture
def cfg():
    return load_config("configs/examples/enhanced_candidate.yaml")
