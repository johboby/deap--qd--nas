"""
QD-NAS: Quality-Diversity NeuroArchitecture Search
质量-多样性神经架构搜索框架 - 主入口

整合所有组件，提供完整的NAS搜索能力
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging

from .behavior_space import BehaviorSpace, create_nas_behavior_space
from .characterization import (
    ArchitectureMetrics, BaseCharacterization,
    StaticCharacterization, HybridCharacterization,
    compute_diversity, compute_novelty
)
from .archive import Archive, ArchiveEntry
from .map_elites import (
    MAPElites, CMA_MAPElites,
    RandomSearchMAPElites, GradientGuidedMAPElites
)
from .multi_objective_nas import (
    MultiObjectiveNAS, ObjectiveType,
    Objective, Constraint, create_default_multi_objective_nas
)
from .search_space import (
    Architecture, Cell, SearchSpace,
    HierarchicalSearchSpace, OperationType
)
from .population_guided_search import (
    PopulationGuidedSearch, AdaptiveHybridSearch
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QDNASOptimizer:
    """
    QD-NAS优化器

    整合QD思想、多目标优化和种群引导搜索，
    提供强大的神经架构搜索能力。

    核心特性:
    1. Quality-Diversity优化：维护高质量和多样性
    2. 多目标多约束优化：精度、延迟、能耗等
    3. 种群引导搜索：减少人工设计
    4. 自适应搜索策略：自动调整搜索方向
    5. 行为特征映射：保持解的多样性

    使用流程:
        1. 创建优化器
        2. 初始化种群或归档
        3. 运行优化
        4. 获取Pareto前沿和最佳架构
    """

    def __init__(self,
                 search_space: Optional[SearchSpace] = None,
                 behavior_space: Optional[BehaviorSpace] = None,
                 characterizer: Optional[BaseCharacterization] = None,
                 optimization_mode: str = 'map_elites',
                 multi_objective: bool = False,
                 population_guided: bool = True):
        """
        初始化QD-NAS优化器

        Args:
            search_space: 搜索空间（默认创建）
            behavior_space: 行为空间（默认创建）
            characterizer: 特征提取器（默认创建）
            optimization_mode: 优化模式
                - 'map_elites': MAP-Elites算法
                - 'cma_map_elites': CMA-ES增强的MAP-Elites
                - 'random_map_elites': 随机搜索增强
                - 'gradient_map_elites': 梯度引导
                - 'multi_objective': 多目标优化
            multi_objective: 是否使用多目标优化
            population_guided: 是否使用种群引导搜索
        """
        # 创建或使用提供的搜索空间
        self.search_space = search_space or SearchSpace()

        # 创建或使用提供的行为空间
        self.behavior_space = behavior_space or create_nas_behavior_space()

        # 创建或使用提供的特征提取器
        self.characterizer = characterizer or StaticCharacterization()

        # 优化配置
        self.optimization_mode = optimization_mode
        self.multi_objective = multi_objective
        self.population_guided = population_guided

        # 初始化优化器
        self.optimizer = self._create_optimizer()

        # 种群引导搜索器
        self.pop_guided_search = None
        if population_guided:
            self.pop_guided_search = PopulationGuidedSearch(
                search_space=self.search_space,
                characterizer=self.characterizer
            )

        # 搜索历史
        self.history = []

        logger.info("🎯 QD-NAS优化器初始化完成")
        logger.info(f"   优化模式: {optimization_mode}")
        logger.info(f"   多目标优化: {multi_objective}")
        logger.info(f"   种群引导: {population_guided}")

    def _create_optimizer(self):
        """创建优化器"""
        if self.multi_objective:
            # 多目标优化
            return MultiObjectiveNAS(
                behavior_space=self.behavior_space,
                characterizer=self.characterizer
            )
        else:
            # 单目标QD优化
            if self.optimization_mode == 'map_elites':
                return MAPElites(
                    behavior_space=self.behavior_space,
                    characterizer=self.characterizer
                )
            elif self.optimization_mode == 'cma_map_elites':
                return CMA_MAPElites(
                    behavior_space=self.behavior_space,
                    characterizer=self.characterizer
                )
            elif self.optimization_mode == 'random_map_elites':
                return RandomSearchMAPElites(
                    behavior_space=self.behavior_space,
                    characterizer=self.characterizer
                )
            elif self.optimization_mode == 'gradient_map_elites':
                return GradientGuidedMAPElites(
                    behavior_space=self.behavior_space,
                    characterizer=self.characterizer
                )
            else:
                logger.warning(f"Unknown optimization mode: {self.optimization_mode}")
                return MAPElites(
                    behavior_space=self.behavior_space,
                    characterizer=self.characterizer
                )

    def initialize(self, initial_population: Optional[List[Architecture]] = None):
        """
        初始化优化器

        Args:
            initial_population: 初始种群（可选）
        """
        logger.info("🚀 初始化QD-NAS优化器")

        if self.pop_guided_search:
            # 使用种群引导搜索
            if initial_population is None:
                self.pop_guided_search.initialize_population()
            else:
                self.pop_guided_search.population = initial_population
                self.pop_guided_search.metrics = [
                    self.characterizer.characterize(arch)
                    for arch in initial_population
                ]
        elif hasattr(self.optimizer, 'initialize_archive'):
            # 使用MAP-Elites
            if initial_population is None:
                # 生成初始种群
                initial_population = [
                    self.search_space.random_sample()
                    for _ in range(100)
                ]
            self.optimizer.initialize_archive(initial_population)

        logger.info("✅ 优化器初始化完成")

    def optimize(self,
                 n_iterations: int = 1000,
                 batch_size: int = 100,
                 verbose: bool = True) -> Tuple[Any, List[ArchiveEntry]]:
        """
        运行优化

        Args:
            n_iterations: 迭代次数
            batch_size: 批处理大小
            verbose: 是否输出详细信息

        Returns:
            (归档/结果, Pareto前沿)
        """
        logger.info(f"🔥 开始优化，迭代次数: {n_iterations}")

        if self.multi_objective:
            # 多目标优化
            archive, pareto_front = self.optimizer.evolve(
                generate_function=self.search_space.random_sample,
                mutate_function=self._mutate,
                n_iterations=n_iterations,
                batch_size=batch_size,
                verbose=verbose
            )
            return archive, pareto_front
        else:
            # 单目标QD优化
            archive = self.optimizer.evolve(
                generate_function=self.search_space.random_sample,
                mutate_function=self._mutate,
                verbose=verbose
            )

            # 如果是MAP-Elites，Pareto前沿就是归档中的最佳个体
            pareto_front = [archive.get_best()] if archive.get_best() else []

            return archive, pareto_front

    def _mutate(self, architecture: Architecture) -> Architecture:
        """
        变异函数

        如果启用了种群引导搜索，使用引导变异
        否则使用标准变异
        """
        if self.pop_guided_search:
            # 使用种群引导的变异
            return self.pop_guided_search.guided_mutation(architecture)
        else:
            # 使用标准变异
            return self.search_space.mutate(architecture)

    def get_best_architecture(self) -> Optional[Architecture]:
        """
        获取最佳架构

        Returns:
            最佳架构
        """
        if self.multi_objective:
            # 多目标：返回Pareto前沿中的第一个
            pareto_front = self.optimizer.get_pareto_front()
            if pareto_front:
                return pareto_front[0].architecture
        else:
            # 单目标：返回归档中的最佳
            if hasattr(self.optimizer, 'get_best_architecture'):
                return self.optimizer.get_best_architecture()

        return None

    def get_pareto_front(self) -> List[Tuple[Architecture, ArchitectureMetrics]]:
        """
        获取Pareto前沿

        Returns:
            [(架构, 性能指标)] 列表
        """
        if self.multi_objective:
            pareto_entries = self.optimizer.get_pareto_front()
            return [(e.architecture, e.metrics) for e in pareto_entries]
        else:
            # 单目标：返回归档中的最佳个体
            best = self.optimizer.archive.get_best()
            if best:
                return [(best.architecture, best.metrics)]
        return []

    def get_archive(self) -> Archive:
        """
        获取归档

        Returns:
            归档对象
        """
        if self.multi_objective:
            return self.optimizer.archive
        else:
            return self.optimizer.archive

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        archive = self.get_archive()
        stats = archive.get_statistics()

        # 添加Pareto前沿统计
        if self.multi_objective:
            pareto_front = self.optimizer.get_pareto_front()
            stats['pareto_size'] = len(pareto_front)

        # 添加种群引导搜索统计
        if self.pop_guided_search:
            pop_stats = self.pop_guided_search.get_statistics()
            stats['population_stats'] = {
                'mean_accuracy': pop_stats.mean_accuracy,
                'std_accuracy': pop_stats.std_accuracy,
                'diversity': pop_stats.diversity,
            }

        return stats

    def visualize(self, save_path: Optional[str] = None):
        """
        可视化结果

        Args:
            save_path: 保存路径（可选）
        """
        archive = self.get_archive()
        archive.visualize(save_path)

    def save_results(self, filepath: str):
        """
        保存结果

        Args:
            filepath: 保存路径
        """
        archive = self.get_archive()
        archive.save(filepath)

        # 保存Pareto前沿
        if self.multi_objective:
            pareto_front = self.get_pareto_front()
            pareto_data = [
                {
                    'architecture': arch.to_dict(),
                    'metrics': metrics.to_dict()
                }
                for arch, metrics in pareto_front
            ]

            import json
            with open(filepath.replace('.pkl', '_pareto.json'), 'w') as f:
                json.dump(pareto_data, f, indent=2)

        logger.info(f"✅ 结果保存至: {filepath}")


def create_default_qd_nas(optimization_mode: str = 'map_elites',
                          multi_objective: bool = False,
                          population_guided: bool = True) -> QDNASOptimizer:
    """
    创建默认的QD-NAS优化器

    Args:
        optimization_mode: 优化模式
        multi_objective: 是否多目标优化
        population_guided: 是否种群引导搜索

    Returns:
        QD-NAS优化器
    """
    return QDNASOptimizer(
        optimization_mode=optimization_mode,
        multi_objective=multi_objective,
        population_guided=population_guided
    )


# ==================== 示例代码 ====================

def example_simple_nas():
    """简单的NAS示例"""
    # 创建优化器
    optimizer = create_default_qd_nas(
        optimization_mode='map_elites',
        multi_objective=False,
        population_guided=True
    )

    # 初始化
    optimizer.initialize()

    # 优化
    archive, pareto_front = optimizer.optimize(
        n_iterations=100,
        batch_size=20,
        verbose=True
    )

    # 获取最佳架构
    best_arch = optimizer.get_best_architecture()
    print(f"Best architecture: {best_arch.to_dict()}")

    # 获取统计信息
    stats = optimizer.get_statistics()
    print(f"Statistics: {stats}")

    # 可视化
    optimizer.visualize()

    # 保存结果
    optimizer.save_results('results/nas_results.pkl')


def example_multi_objective_nas():
    """多目标NAS示例"""
    # 创建优化器
    optimizer = create_default_qd_nas(
        optimization_mode='map_elites',
        multi_objective=True,
        population_guided=True
    )

    # 初始化
    optimizer.initialize()

    # 优化
    archive, pareto_front = optimizer.optimize(
        n_iterations=100,
        batch_size=20,
        verbose=True
    )

    # 获取Pareto前沿
    pareto = optimizer.get_pareto_front()
    print(f"Pareto front size: {len(pareto)}")

    for i, (arch, metrics) in enumerate(pareto[:5]):
        print(f"Solution {i+1}:")
        print(f"  Accuracy: {metrics.accuracy:.4f}")
        print(f"  Latency: {metrics.latency:.2f}ms")
        print(f"  Energy: {metrics.energy:.2f}mJ")

    # 可视化
    optimizer.visualize()


__all__ = [
    'QDNASOptimizer',
    'create_default_qd_nas',
    'example_simple_nas',
    'example_multi_objective_nas',
]
