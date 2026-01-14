import pytest
import numpy as np
from src.nas.behavior_space import (
    BehaviorType, BehaviorDimension, BehaviorSpace,
    create_nas_behavior_space
)
from src.nas.characterization import (
    ArchitectureMetrics, StaticCharacterization,
    compute_diversity, compute_novelty
)


class TestBehaviorSpace:
    
    def test_behavior_dimension(self):
        dim = BehaviorDimension(
            name='test_dim',
            type=BehaviorType.CONTINUOUS,
            min_value=0.0,
            max_value=1.0,
            n_bins=10
        )
        assert dim.name == 'test_dim'
        assert dim.type == BehaviorType.CONTINUOUS
        assert dim.min_value == 0.0
        assert dim.max_value == 1.0
        assert dim.n_bins == 10

    def test_behavior_space(self):
        dimensions = [
            BehaviorDimension('latency', BehaviorType.CONTINUOUS, 0, 100, 10),
            BehaviorDimension('energy', BehaviorType.CONTINUOUS, 0, 50, 5),
            BehaviorDimension('complexity', BehaviorType.CONTINUOUS, 0, 1000, 20)
        ]
        bs = BehaviorSpace(dimensions)
        assert len(bs.dimensions) == 3

    def test_get_cell_key(self):
        dimensions = [
            BehaviorDimension('latency', BehaviorType.CONTINUOUS, 0, 100, 10),
            BehaviorDimension('energy', BehaviorType.CONTINUOUS, 0, 50, 5)
        ]
        bs = BehaviorSpace(dimensions)
        behavior_vector = [50.0, 25.0]
        cell_key = bs.get_cell_key(behavior_vector)
        assert isinstance(cell_key, tuple)
        assert len(cell_key) == 2

    def test_create_nas_behavior_space(self):
        bs = create_nas_behavior_space()
        assert bs is not None
        assert len(bs.dimensions) >= 3


class TestCharacterization:
    
    def test_architecture_metrics(self):
        metrics = ArchitectureMetrics(
            accuracy=0.85,
            latency=15.5,
            energy=10.2,
            flops=1500000,
            num_parameters=1000000
        )
        assert metrics.accuracy == 0.85
        assert metrics.latency == 15.5
        assert metrics.energy == 10.2
        assert metrics.flops == 1500000
        assert metrics.num_parameters == 1000000

    def test_behavior_vector(self):
        metrics = ArchitectureMetrics(
            accuracy=0.85,
            latency=15.5,
            energy=10.2,
            flops=1500000,
            num_parameters=1000000
        )
        behavior_vector = metrics.get_behavior_vector()
        assert isinstance(behavior_vector, list)
        assert len(behavior_vector) >= 2

    def test_static_characterization(self):
        char = StaticCharacterization()
        architecture = {
            'layers': [32, 64, 128],
            'activations': ['relu', 'relu', 'relu']
        }
        metrics = char.characterize(architecture)
        assert isinstance(metrics, ArchitectureMetrics)

    def test_compute_diversity(self):
        metrics_list = [
            ArchitectureMetrics(accuracy=0.85, latency=15.5, energy=10.2),
            ArchitectureMetrics(accuracy=0.82, latency=18.2, energy=11.5),
            ArchitectureMetrics(accuracy=0.88, latency=12.3, energy=9.8)
        ]
        diversity = compute_diversity(metrics_list)
        assert isinstance(diversity, float)
        assert diversity >= 0

    def test_compute_novelty(self):
        target_metrics = ArchitectureMetrics(accuracy=0.85, latency=15.5, energy=10.2)
        archive_metrics = [
            ArchitectureMetrics(accuracy=0.82, latency=18.2, energy=11.5),
            ArchitectureMetrics(accuracy=0.88, latency=12.3, energy=9.8)
        ]
        novelty = compute_novelty(target_metrics, archive_metrics)
        assert isinstance(novelty, float)
        assert novelty >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
