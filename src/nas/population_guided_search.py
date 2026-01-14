"""
种群引导的结构化搜索 (Population-Guided Structured Search)
减少人工设计的算子和参数，使用种群信息引导搜索
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict
import logging

from .search_space import Architecture, SearchSpace
from .characterization import ArchitectureMetrics, BaseCharacterization


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PopulationStats:
    """种群统计信息"""
    mean_accuracy: float
    std_accuracy: float
    mean_latency: float
    std_latency: float
    mean_energy: float
    std_energy: float
    diversity: float
    behavior_variance: float


class PopulationGuidedSearch:
    """
    种群引导的结构化搜索

    核心思想:
    1. 使用种群统计信息引导搜索方向
    2. 自动学习有效操作和连接模式
    3. 减少对人工设计算子的依赖
    4. 自适应调整变异策略

    特性:
    - 自动学习操作偏好
    - 自适应变异率
    - 基于行为的引导搜索
    - 动态调整搜索空间
    """

    def __init__(self,
                 search_space: SearchSpace,
                 characterizer: BaseCharacterization,
                 population_size: int = 50,
                 adaptive_rate: float = 0.1):
        """
        初始化种群引导搜索

        Args:
            search_space: 搜索空间
            characterizer: 特征提取器
            population_size: 种群大小
            adaptive_rate: 自适应学习率
        """
        self.search_space = search_space
        self.characterizer = characterizer
        self.population_size = population_size
        self.adaptive_rate = adaptive_rate

        # 种群
        self.population: List[Architecture] = []
        self.metrics: List[ArchitectureMetrics] = []

        # 操作偏好统计
        self.operation_preferences: Dict[str, float] = defaultdict(float)
        self.connection_preferences: Dict[Tuple[int, int], float] = defaultdict(float)

        # 搜索历史
        self.history: List[PopulationStats] = []

        logger.info("🧭 种群引导的结构化搜索初始化完成")

    def initialize_population(self):
        """初始化种群"""
        logger.info(f"📦 初始化种群，大小: {self.population_size}")

        self.population = []
        self.metrics = []

        for _ in range(self.population_size):
            arch = self.search_space.random_sample()
            metrics = self.characterizer.characterize(arch)
            self.population.append(arch)
            self.metrics.append(metrics)

            # 更新操作偏好
            self._update_operation_preferences(arch, metrics)

        logger.info(f"✅ 种群初始化完成")

    def _update_operation_preferences(self,
                                      architecture: Architecture,
                                      metrics: ArchitectureMetrics):
        """
        更新操作偏好

        基于架构性能调整操作使用概率
        """
        # 简化的偏好更新：基于准确率
        weight = metrics.accuracy

        for cell in architecture.cells:
            for _, _, op in cell.edges:
                self.operation_preferences[op] += weight * self.adaptive_rate

    def get_statistics(self) -> PopulationStats:
        """
        计算种群统计信息

        Returns:
            种群统计信息
        """
        if not self.metrics:
            return PopulationStats(0, 0, 0, 0, 0, 0, 0, 0)

        accuracies = [m.accuracy for m in self.metrics]
        latencies = [m.latency for m in self.metrics]
        energies = [m.energy for m in self.metrics]

        # 计算多样性
        behavior_vectors = [m.get_behavior_vector() for m in self.metrics]
        behavior_array = np.array(behavior_vectors)
        behavior_variance = np.var(behavior_array, axis=0).mean()

        # 计算行为空间多样性（平均成对距离）
        if len(behavior_vectors) > 1:
            distances = []
            for i in range(len(behavior_vectors)):
                for j in range(i + 1, len(behavior_vectors)):
                    dist = np.linalg.norm(behavior_array[i] - behavior_array[j])
                    distances.append(dist)
            diversity = np.mean(distances) if distances else 0.0
        else:
            diversity = 0.0

        return PopulationStats(
            mean_accuracy=np.mean(accuracies),
            std_accuracy=np.std(accuracies),
            mean_latency=np.mean(latencies),
            std_latency=np.std(latencies),
            mean_energy=np.mean(energies),
            std_energy=np.std(energies),
            diversity=diversity,
            behavior_variance=behavior_variance,
        )

    def guided_mutation(self,
                        parent: Architecture,
                        guide_by: str = 'performance') -> Architecture:
        """
        引导变异

        Args:
            parent: 父本架构
            guide_by: 引导方式 ('performance', 'diversity', 'balanced')

        Returns:
            变异后的架构
        """
        # 选择引导策略
        if guide_by == 'performance':
            return self._performance_guided_mutation(parent)
        elif guide_by == 'diversity':
            return self._diversity_guided_mutation(parent)
        else:  # balanced
            if np.random.random() < 0.5:
                return self._performance_guided_mutation(parent)
            else:
                return self._diversity_guided_mutation(parent)

    def _performance_guided_mutation(self,
                                     parent: Architecture) -> Architecture:
        """
        性能引导的变异

        偏向使用高性能架构中的操作
        """
        new_arch = Architecture.from_dict(parent.to_dict())

        # 归一化操作偏好
        total_weight = sum(self.operation_preferences.values())
        operation_probs = {
            op: weight / total_weight
            for op, weight in self.operation_preferences.items()
        }

        # 根据偏好变异操作
        for cell in new_arch.cells:
            if np.random.random() < 0.3:  # 30%概率变异
                if cell.edges:
                    # 选择要变异的边
                    edge_idx = np.random.randint(0, len(cell.edges))
                    i, j, _ = cell.edges[edge_idx]

                    # 根据偏好选择新操作
                    ops = list(operation_probs.keys())
                    probs = list(operation_probs.values())
                    new_op = np.random.choice(ops, p=probs)

                    cell.edges[edge_idx] = (i, j, new_op)

        return new_arch

    def _diversity_guided_mutation(self,
                                    parent: Architecture) -> Architecture:
        """
        多样性引导的变异

        偏向使用少见的操作，增加多样性
        """
        new_arch = Architecture.from_dict(parent.to_dict())

        # 归一化操作偏好
        total_weight = sum(self.operation_preferences.values())
        operation_probs = {
            op: weight / total_weight
            for op, weight in self.operation_preferences.items()
        }

        # 计算多样性偏好（使用少的操作概率更高）
        diversity_probs = {
            op: 1.0 / (prob + 0.01)  # 反比
            for op, prob in operation_probs.items()
        }

        # 归一化
        total_diversity_weight = sum(diversity_probs.values())
        diversity_probs = {
            op: weight / total_diversity_weight
            for op, weight in diversity_probs.items()
        }

        # 根据多样性偏好变异操作
        for cell in new_arch.cells:
            if np.random.random() < 0.3:  # 30%概率变异
                if cell.edges:
                    edge_idx = np.random.randint(0, len(cell.edges))
                    i, j, _ = cell.edges[edge_idx]

                    ops = list(diversity_probs.keys())
                    probs = list(diversity_probs.values())
                    new_op = np.random.choice(ops, p=probs)

                    cell.edges[edge_idx] = (i, j, new_op)

        return new_arch

    def adaptive_mutation_rate(self,
                              generation: int,
                              max_generations: int) -> float:
        """
        自适应变异率

        Args:
            generation: 当前代数
            max_generations: 最大代数

        Returns:
            变异率
        """
        # 基于多样性的自适应
        stats = self.get_statistics()

        # 多样性越低，变异率越高
        base_mutation_rate = 0.2
        diversity_factor = 1.0 - (stats.diversity / (stats.diversity + 0.1))

        # 基于代数的自适应
        progress = generation / max_generations
        generation_factor = 1.0 - 0.5 * progress  # 后期变异率降低

        mutation_rate = base_mutation_rate * diversity_factor * generation_factor

        return np.clip(mutation_rate, 0.05, 0.5)

    def generate_offspring(self,
                          n_offspring: int,
                          generation: int = 0,
                          max_generations: int = 100) -> List[Architecture]:
        """
        生成子代

        Args:
            n_offspring: 子代数量
            generation: 当前代数
            max_generations: 最大代数

        Returns:
            子代架构列表
        """
        offspring = []

        # 获取自适应变异率
        mutation_rate = self.adaptive_mutation_rate(generation, max_generations)

        for _ in range(n_offspring):
            # 锦标赛选择父本
            parent = self._tournament_selection(k=3)

            # 引导变异
            child = self.guided_mutation(
                parent,
                guide_by=self._select_guide_strategy()
            )

            offspring.append(child)

        return offspring

    def _tournament_selection(self, k: int = 3) -> Architecture:
        """
        锦标赛选择

        Args:
            k: 锦标赛规模

        Returns:
            选中的架构
        """
        # 随机选择k个个体
        indices = np.random.choice(len(self.population), k, replace=False)
        candidates = [self.population[i] for i in indices]
        candidate_metrics = [self.metrics[i] for i in indices]

        # 选择最佳个体（基于准确率）
        best_idx = np.argmax([m.accuracy for m in candidate_metrics])
        return candidates[best_idx]

    def _select_guide_strategy(self) -> str:
        """
        选择引导策略

        根据种群多样性自动选择引导方式
        """
        stats = self.get_statistics()

        # 如果多样性低，使用多样性引导
        if stats.diversity < 0.3:
            return 'diversity'
        # 如果多样性正常，使用性能引导
        elif stats.diversity < 0.6:
            return 'performance'
        # 多样性高，平衡使用
        else:
            return 'balanced'

    def update_population(self,
                        new_architectures: List[Architecture],
                        new_metrics: List[ArchitectureMetrics]):
        """
        更新种群

        Args:
            new_architectures: 新架构列表
            new_metrics: 新架构的性能指标列表
        """
        # 合并新旧种群
        combined_arch = self.population + new_architectures
        combined_metrics = self.metrics + new_metrics

        # 选择最佳个体（基于准确率，保持多样性）
        selected_indices = self._environmental_selection(
            combined_arch,
            combined_metrics,
            self.population_size
        )

        # 更新种群
        self.population = [combined_arch[i] for i in selected_indices]
        self.metrics = [combined_metrics[i] for i in selected_indices]

        # 更新操作偏好
        for arch, metrics in zip(new_architectures, new_metrics):
            self._update_operation_preferences(arch, metrics)

        # 记录历史
        self.history.append(self.get_statistics())

    def _environmental_selection(self,
                                 architectures: List[Architecture],
                                 metrics: List[ArchitectureMetrics],
                                 n_select: int) -> List[int]:
        """
        环境选择

        选择最佳个体，同时保持多样性

        Args:
            architectures: 架构列表
            metrics: 性能指标列表
            n_select: 选择数量

        Returns:
            选中个体的索引列表
        """
        # 计算每个个体的综合分数
        scores = []
        for i, (arch, m) in enumerate(zip(architectures, metrics)):
            # 准确率分数
            accuracy_score = m.accuracy

            # 多样性分数（与已选个体的平均距离）
            diversity_score = 0.0
            if scores:
                selected_indices = [idx for idx, _ in scores]
                selected_behaviors = [architectures[idx].encode()
                                    for idx in selected_indices]
                behavior = arch.encode()
                distances = [np.linalg.norm(behavior - b)
                             for b in selected_behaviors]
                diversity_score = np.mean(distances) if distances else 0.0

            # 综合分数
            combined_score = 0.7 * accuracy_score + 0.3 * diversity_score
            scores.append((i, combined_score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 选择前n_select个
        selected_indices = [idx for idx, _ in scores[:n_select]]

        return selected_indices


class AdaptiveHybridSearch(PopulationGuidedSearch):
    """
    自适应混合搜索

    结合多种搜索策略，自动选择最优策略
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 搜索策略
        self.strategies = [
            'population_guided',
            'random_search',
            'local_search',
        ]

        # 策略性能记录
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)

    def generate_offspring(self,
                          n_offspring: int,
                          generation: int = 0,
                          max_generations: int = 100) -> List[Architecture]:
        """
        生成子代（自适应选择策略）
        """
        offspring = []

        # 根据历史性能选择策略
        strategy = self._select_strategy()

        for i in range(n_offspring):
            if strategy == 'population_guided':
                child = self._generate_population_guided(generation, max_generations)
            elif strategy == 'random_search':
                child = self.search_space.random_sample()
            else:  # local_search
                parent = self._tournament_selection(k=3)
                child = self._generate_local_search(parent)

            offspring.append(child)

        return offspring

    def _select_strategy(self) -> str:
        """选择最佳策略"""
        # 计算每个策略的平均性能
        strategy_scores = {}
        for strategy, scores in self.strategy_performance.items():
            if scores:
                strategy_scores[strategy] = np.mean(scores)
            else:
                strategy_scores[strategy] = 0.5  # 默认分数

        # 选择最佳策略
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        return best_strategy

    def _generate_population_guided(self,
                                   generation: int,
                                   max_generations: int) -> Architecture:
        """生成种群引导的个体"""
        parent = self._tournament_selection(k=3)
        return self.guided_mutation(parent)

    def _generate_local_search(self,
                               parent: Architecture) -> Architecture:
        """生成局部搜索的个体"""
        neighbors = self.search_space.local_search(parent, n_neighbors=5)
        # 返回最好的邻居
        return neighbors[0]


__all__ = [
    'PopulationStats',
    'PopulationGuidedSearch',
    'AdaptiveHybridSearch',
]
