import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def random_seed():
    np.random.seed(42)


@pytest.fixture
def sample_data():
    return {
        'x': np.random.rand(10, 10),
        'y': np.random.rand(10)
    }


@pytest.fixture
def sample_architecture():
    return {
        'layers': [32, 64, 128, 256],
        'activations': ['relu', 'relu', 'relu', 'relu'],
        'kernel_size': [3, 3, 3, 3],
        'pool_size': [2, 2, 2, 2]
    }


@pytest.fixture
def sample_metrics():
    class Metrics:
        def __init__(self):
            self.accuracy = 0.85
            self.latency = 15.5
            self.energy = 10.2
            self.flops = 1500000

    return Metrics()
