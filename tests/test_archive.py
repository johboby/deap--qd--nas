import pytest
import numpy as np
from src.nas.behavior_space import BehaviorSpace, BehaviorDimension, BehaviorType
from src.nas.characterization import ArchitectureMetrics, StaticCharacterization
from src.nas.archive import Archive, ArchiveEntry


class TestArchiveEntry:
    
    def test_creation(self):
        architecture = {'layers': [32, 64]}
        metrics = ArchitectureMetrics(accuracy=0.85, latency=15.5, energy=10.2)
        behavior_vector = [15.5, 10.2]
        cell_key = (5, 3)

        entry = ArchiveEntry(
            architecture=architecture,
            metrics=metrics,
            behavior_vector=behavior_vector,
            cell_key=cell_key,
            generation=10
        )

        assert entry.architecture == architecture
        assert entry.metrics == metrics
        assert entry.behavior_vector == behavior_vector
        assert entry.cell_key == cell_key
        assert entry.generation == 10


class TestArchive:
    
    @pytest.fixture
    def sample_archive(self):
        dimensions = [
            BehaviorDimension('latency', BehaviorType.CONTINUOUS, 0, 100, 10),
            BehaviorDimension('energy', BehaviorType.CONTINUOUS, 0, 50, 5)
        ]
        behavior_space = BehaviorSpace(dimensions)
        archive = Archive(behavior_space, optimize_for='accuracy')
        return archive

    def test_creation(self, sample_archive):
        assert sample_archive is not None
        assert sample_archive.optimize_for == 'accuracy'
        assert len(sample_archive.grid) == 0

    def test_insert(self, sample_archive):
        architecture = {'layers': [32, 64]}
        metrics = ArchitectureMetrics(accuracy=0.85, latency=15.5, energy=10.2)
        result = sample_archive.insert(architecture, metrics, generation=10)
        assert result == True
        assert len(sample_archive.grid) == 1

    def test_insert_duplicate(self, sample_archive):
        architecture1 = {'layers': [32, 64]}
        metrics1 = ArchitectureMetrics(accuracy=0.85, latency=15.5, energy=10.2)
        sample_archive.insert(architecture1, metrics1, generation=10)

        architecture2 = {'layers': [32, 128]}
        metrics2 = ArchitectureMetrics(accuracy=0.90, latency=15.5, energy=10.2)
        result = sample_archive.insert(architecture2, metrics2, generation=11)

        assert result == True
        assert len(sample_archive.grid) == 1
        entry = list(sample_archive.grid.values())[0]
        assert entry.metrics.accuracy == 0.90

    def test_get_best(self, sample_archive):
        architectures = [
            {'layers': [32, 64]},
            {'layers': [64, 128]},
            {'layers': [128, 256]}
        ]
        metrics = [
            ArchitectureMetrics(accuracy=0.85, latency=15.5, energy=10.2),
            ArchitectureMetrics(accuracy=0.90, latency=18.2, energy=11.5),
            ArchitectureMetrics(accuracy=0.82, latency=12.3, energy=9.8)
        ]

        for arch, met in zip(architectures, metrics):
            sample_archive.insert(arch, met)

        best = sample_archive.get_best()
        assert best is not None
        assert best.metrics.accuracy == 0.90

    def test_get_entries(self, sample_archive):
        architectures = [
            {'layers': [32, 64]},
            {'layers': [64, 128]},
            {'layers': [128, 256]}
        ]
        metrics = [
            ArchitectureMetrics(accuracy=0.85, latency=15.5, energy=10.2),
            ArchitectureMetrics(accuracy=0.90, latency=18.2, energy=11.5),
            ArchitectureMetrics(accuracy=0.82, latency=12.3, energy=9.8)
        ]

        for arch, met in zip(architectures, metrics):
            sample_archive.insert(arch, met)

        entries = sample_archive.get_entries()
        assert len(entries) == 3

    def test_get_coverage(self, sample_archive):
        for i in range(5):
            architecture = {'layers': [32 * (i+1)]}
            metrics = ArchitectureMetrics(
                accuracy=0.8 + i * 0.02,
                latency=10 + i * 5,
                energy=5 + i * 2
            )
            sample_archive.insert(architecture, metrics)

        coverage = sample_archive.get_coverage()
        assert isinstance(coverage, float)
        assert 0 <= coverage <= 1

    def test_get_diversity(self, sample_archive):
        architectures = [
            {'layers': [32, 64]},
            {'layers': [64, 128]},
            {'layers': [128, 256]}
        ]
        metrics = [
            ArchitectureMetrics(accuracy=0.85, latency=15.5, energy=10.2),
            ArchitectureMetrics(accuracy=0.90, latency=18.2, energy=11.5),
            ArchitectureMetrics(accuracy=0.82, latency=12.3, energy=9.8)
        ]

        for arch, met in zip(architectures, metrics):
            sample_archive.insert(arch, met)

        diversity = sample_archive.get_diversity()
        assert isinstance(diversity, float)
        assert diversity >= 0

    def test_get_statistics(self, sample_archive):
        architectures = [
            {'layers': [32, 64]},
            {'layers': [64, 128]},
            {'layers': [128, 256]}
        ]
        metrics = [
            ArchitectureMetrics(accuracy=0.85, latency=15.5, energy=10.2),
            ArchitectureMetrics(accuracy=0.90, latency=18.2, energy=11.5),
            ArchitectureMetrics(accuracy=0.82, latency=12.3, energy=9.8)
        ]

        for arch, met in zip(architectures, metrics):
            sample_archive.insert(arch, met)

        stats = sample_archive.get_statistics()
        assert 'size' in stats
        assert 'coverage' in stats
        assert 'diversity' in stats
        assert 'best_fitness' in stats
        assert stats['size'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
