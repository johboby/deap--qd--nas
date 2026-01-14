"""
MAP-Elites算法实现
Multi-Archive Map-Elites for QD-NAS
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
import logging

from .behavior_space import BehaviorSpace
from .archive import Archive, ArchiveEntry
from .characterization import ArchitectureMetrics, BaseCharacterization


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MAPElites:
    """
    MAP-Elites算法

    Quality-Diversity优化算法，通过行为空间网格维护高质量的多样化解。

    核心思想:
    1. 将行为空间划分为多个网格（cells）
    2. 每个cell保存最佳个体
    3. 通过变异和选择生成新个体
    4. 维护行为的多样性和质量

    参数:
    - behavior_space: 行为空间定义
    - characterizer: 特征提取器
    - n_iterations: 迭代次数
    - batch_size: 每次生成的个体数
    - archive_size: 归档最大大小
    """

    def __init__(self,
                 behavior_space: BehaviorSpace,
                 characterizer: BaseCharacterization,
                 optimize_for: str = 'accuracy',
                 n_iterations: int = 1000,
                 batch_size: int = 100,
                 archive_size: Optional[int] = None):
        """
        初始化MAP-Elites

        Args:
            behavior_space: 行为空间定义
            characterizer: 特征提取器
            optimize_for: 优化目标
            n_iterations: 迭代次数
            batch_size: 批处理大小
            archive_size: 归档最大大小
        """
        self.behavior_space = behavior_space
        self.characterizer = characterizer
        self.optimize_for = optimize_for
        self.n_iterations = n_iterations
        self.batch_size = batch_size

        # 创建归档
        self.archive = Archive(
            behavior_space=behavior_space,
            optimize_for=optimize_for,
            max_size=archive_size
        )

        # 搜索历史
        self.history = []

        logger.info("🗺️  MAP-Elites算法初始化完成")

    def initialize_archive(self, initial_population: List[Any]):
        """
        初始化归档

        Args:
            initial_population: 初始种群
        """
        logger.info(f"📦 初始化归档，种群大小: {len(initial_population)}")

        for arch in initial_population:
            metrics = self.characterizer.characterize(arch)
            self.archive.insert(arch, metrics, generation=0)

        stats = self.archive.get_statistics()
        logger.info(f"✅ 初始归档: {stats['size']} 个个体")

    def evolve(self,
              generate_function: Callable,
              mutate_function: Callable,
              verbose: bool = True) -> Archive:
        """
        运行MAP-Elites进化

        Args:
            generate_function: 生成新架构的函数
            mutate_function: 变异函数
            verbose: 是否输出详细信息

        Returns:
            最终归档
        """
        logger.info(f"🚀 开始MAP-Elites进化，迭代次数: {self.n_iterations}")

        for iteration in range(self.n_iterations):
            # 生成一批新个体
            batch = self._generate_batch(generate_function, mutate_function)

            # 评估并插入归档
            inserted_count = 0
            for arch in batch:
                metrics = self.characterizer.characterize(arch)
                success = self.archive.insert(arch, metrics, generation=iteration)
                if success:
                    inserted_count += 1

            # 记录历史
            stats = self.archive.get_statistics()
            self.history.append(stats)

            # 输出进度
            if verbose and (iteration + 1) % 10 == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{self.n_iterations} | "
                    f"Archive: {stats['size']} | "
                    f"Coverage: {stats['coverage']:.2%} | "
                    f"Diversity: {stats['diversity']:.4f} | "
                    f"Best: {stats['best_fitness']:.4f}"
                )

        logger.info("✅ MAP-Elites进化完成")
        return self.archive

    def _generate_batch(self,
                       generate_function: Callable,
                       mutate_function: Callable) -> List[Any]:
        """
        生成一批新个体

        Args:
            generate_function: 生成函数
            mutate_function: 变异函数

        Returns:
            新个体列表
        """
        batch = []

        for _ in range(self.batch_size):
            # 从归档中选择一个父本
            parent = self._select_parent()

            # 变异生成子代
            if parent is not None:
                child = mutate_function(parent)
            else:
                # 如果归档为空，随机生成
                child = generate_function()

            batch.append(child)

        return batch

    def _select_parent(self) -> Optional[Any]:
        """
        选择父本

        从归档中随机选择一个个体

        Returns:
            选中的架构
        """
        entry = self.archive.get_random()
        return entry.architecture if entry else None

    def get_best_architecture(self) -> Optional[Any]:
        """获取最佳架构"""
        entry = self.archive.get_best()
        return entry.architecture if entry else None

    def get_best_metrics(self) -> Optional[ArchitectureMetrics]:
        """获取最佳指标"""
        entry = self.archive.get_best()
        return entry.metrics if entry else None


class CMA_MAPElites(MAPElites):
    """
    CMA-ES增强的MAP-Elites

    使用CMA-ES（Covariance Matrix Adaptation Evolution Strategy）
    引导搜索，提高效率。
    """

    def __init__(self,
                 behavior_space: BehaviorSpace,
                 characterizer: BaseCharacterization,
                 optimize_for: str = 'accuracy',
                 n_iterations: int = 1000,
                 batch_size: int = 100,
                 archive_size: Optional[int] = None,
                 cma_population_size: int = 20):
        """
        初始化CMA-MAP-Elites

        Args:
            behavior_space: 行为空间定义
            characterizer: 特征提取器
            optimize_for: 优化目标
            n_iterations: 迭代次数
            batch_size: 批处理大小
            archive_size: 归档最大大小
            cma_population_size: CMA-ES种群大小
        """
        super().__init__(
            behavior_space=behavior_space,
            characterizer=characterizer,
            optimize_for=optimize_for,
            n_iterations=n_iterations,
            batch_size=batch_size,
            archive_size=archive_size
        )

        self.cma_population_size = cma_population_size
        self.cma_initialized = False

        logger.info("🧬 CMA-MAP-Elites算法初始化完成")

    def _select_parent(self) -> Optional[Any]:
        """
        选择父本

        使用CMA-ES选择策略
        """
        # 这里简化实现，实际应使用CMA-ES
        return super()._select_parent()


class RandomSearchMAPElites(MAPElites):
    """
    随机搜索增强的MAP-Elites

    结合随机搜索探索行为空间
    """

    def __init__(self,
                 behavior_space: BehaviorSpace,
                 characterizer: BaseCharacterization,
                 optimize_for: str = 'accuracy',
                 n_iterations: int = 1000,
                 batch_size: int = 100,
                 archive_size: Optional[int] = None,
                 random_search_ratio: float = 0.1):
        """
        初始化随机搜索MAP-Elites

        Args:
            behavior_space: 行为空间定义
            characterizer: 特征提取器
            optimize_for: 优化目标
            n_iterations: 迭代次数
            batch_size: 批处理大小
            archive_size: 归档最大大小
            random_search_ratio: 随机搜索比例
        """
        super().__init__(
            behavior_space=behavior_space,
            characterizer=characterizer,
            optimize_for=optimize_for,
            n_iterations=n_iterations,
            batch_size=batch_size,
            archive_size=archive_size
        )

        self.random_search_ratio = random_search_ratio
        logger.info("🎲 随机搜索MAP-Elites算法初始化完成")

    def _generate_batch(self,
                       generate_function: Callable,
                       mutate_function: Callable) -> List[Any]:
        """
        生成一批新个体

        混合使用变异和随机生成
        """
        batch = []

        for _ in range(self.batch_size):
            # 随机决定使用变异还是随机生成
            if np.random.random() < self.random_search_ratio:
                # 随机生成
                child = generate_function()
            else:
                # 变异生成
                parent = self._select_parent()
                if parent is not None:
                    child = mutate_function(parent)
                else:
                    child = generate_function()

            batch.append(child)

        return batch


class GradientGuidedMAPElites(MAPElites):
    """
    梯度引导的MAP-Elites

    使用梯度信息引导搜索方向
    """

    def __init__(self,
                 behavior_space: BehaviorSpace,
                 characterizer: BaseCharacterization,
                 optimize_for: str = 'accuracy',
                 n_iterations: int = 1000,
                 batch_size: int = 100,
                 archive_size: Optional[int] = None,
                 gradient_steps: int = 5):
        """
        初始化梯度引导MAP-Elites

        Args:
            behavior_space: 行为空间定义
            characterizer: 特征提取器
            optimize_for: 优化目标
            n_iterations: 迭代次数
            batch_size: 批处理大小
            archive_size: 归档最大大小
            gradient_steps: 梯度步数
        """
        super().__init__(
            behavior_space=behavior_space,
            characterizer=characterizer,
            optimize_for=optimize_for,
            n_iterations=n_iterations,
            batch_size=batch_size,
            archive_size=archive_size
        )

        self.gradient_steps = gradient_steps
        logger.info("📈 梯度引导MAP-Elites算法初始化完成")

    def _generate_batch(self,
                       generate_function: Callable,
                       mutate_function: Callable) -> List[Any]:
        """
        生成一批新个体

        使用梯度引导的变异
        """
        batch = []

        for _ in range(self.batch_size):
            parent = self._select_parent()

            if parent is not None:
                # 使用梯度引导的变异
                child = self._gradient_guided_mutate(parent, mutate_function)
            else:
                child = generate_function()

            batch.append(child)

        return batch

    def _gradient_guided_mutate(self,
                                  parent: Any,
                                  mutate_function: Callable) -> Any:
        """
        梯度引导的变异

        使用归档中的最佳个体信息引导变异方向
        """
        # 获取归档中最佳个体的行为特征
        best_entry = self.archive.get_best()

        if best_entry is not None:
            # 简化实现：向最佳个体的方向变异
            # 实际应该计算梯度
            return mutate_function(parent)

        return mutate_function(parent)


__all__ = [
    'MAPElites',
    'CMA_MAPElites',
    'RandomSearchMAPElites',
    'GradientGuidedMAPElites',
]
