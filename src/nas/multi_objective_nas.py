"""
多目标多约束NAS优化 (Multi-Objective Multi-Constraint NAS)
支持延迟、能耗、精度等多目标优化
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from .behavior_space import BehaviorSpace
from .archive import Archive, ArchiveEntry
from .characterization import ArchitectureMetrics, BaseCharacterization
from .map_elites import MAPElites


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """目标类型"""
    MAXIMIZE = "maximize"  # 最大化
    MINIMIZE = "minimize"  # 最小化


@dataclass
class Objective:
    """
    优化目标定义

    Args:
        name: 目标名称
        type: 目标类型（最大化/最小化）
        weight: 目标权重
        constraint: 约束阈值（可选）
    """
    name: str
    type: ObjectiveType
    weight: float = 1.0
    constraint: Optional[float] = None


@dataclass
class Constraint:
    """
    约束定义

    Args:
        name: 约束名称
        threshold: 约束阈值
        type: 约束类型（<=, >=, ==）
        penalty: 约束违反的惩罚系数
    """
    name: str
    threshold: float
    type: str = "<="
    penalty: float = 1000.0

    def is_satisfied(self, value: float) -> bool:
        """检查约束是否满足"""
        if self.type == "<=":
            return value <= self.threshold
        elif self.type == ">=":
            return value >= self.threshold
        elif self.type == "==":
            return abs(value - self.threshold) < 1e-6
        else:
            return True

    def penalty_value(self, value: float) -> float:
        """计算约束违反的惩罚值"""
        if self.is_satisfied(value):
            return 0.0
        else:
            violation = abs(value - self.threshold)
            return self.penalty * violation


class MultiObjectiveNAS:
    """
    多目标多约束NAS优化器

    支持多个优化目标和约束条件，使用Pareto支配和约束处理。

    核心特性:
    1. 多目标优化（精度、延迟、能耗等）
    2. 多约束处理（延迟约束、能耗约束、参数约束等）
    3. Pareto前沿维护
    4. 约束违反惩罚
    5. 自适应权重调整

    典型目标:
    - accuracy: 精度（最大化）
    - latency: 延迟（最小化）
    - energy: 能耗（最小化）
    - params: 参数量（最小化）

    典型约束:
    - latency <= 100ms
    - energy <= 1000mJ
    - params <= 5M
    """

    def __init__(self,
                 behavior_space: BehaviorSpace,
                 characterizer: BaseCharacterization,
                 objectives: List[Objective],
                 constraints: Optional[List[Constraint]] = None,
                 archive_size: Optional[int] = None):
        """
        初始化多目标NAS

        Args:
            behavior_space: 行为空间定义
            characterizer: 特征提取器
            objectives: 优化目标列表
            constraints: 约束列表（可选）
            archive_size: 归档最大大小
        """
        self.behavior_space = behavior_space
        self.characterizer = characterizer
        self.objectives = objectives
        self.constraints = constraints or []

        # 创建归档
        self.archive = Archive(
            behavior_space=behavior_space,
            optimize_for='multi_objective',  # 特殊标记
            max_size=archive_size
        )

        # Pareto前沿
        self.pareto_front: List[ArchiveEntry] = []

        logger.info(f"🎯 多目标NAS初始化完成")
        logger.info(f"   目标数量: {len(objectives)}")
        logger.info(f"   约束数量: {len(constraints)}")

    def compute_fitness(self,
                       metrics: ArchitectureMetrics,
                       return_details: bool = False) -> float:
        """
        计算综合适应度

        Args:
            metrics: 架构性能指标
            return_details: 是否返回详细计算

        Returns:
            综合适应度
        """
        # 计算目标分数
        objective_scores = {}
        total_score = 0.0

        for obj in self.objectives:
            value = self._get_objective_value(obj, metrics)

            # 根据类型调整方向
            if obj.type == ObjectiveType.MAXIMIZE:
                score = value
            else:  # MINIMIZE
                score = -value

            # 应用权重
            weighted_score = score * obj.weight
            total_score += weighted_score

            objective_scores[obj.name] = {
                'value': value,
                'score': score,
                'weighted_score': weighted_score,
            }

        # 计算约束惩罚
        penalty = 0.0
        constraint_scores = {}

        for constraint in self.constraints:
            value = self._get_constraint_value(constraint, metrics)
            constraint_penalty = constraint.penalty_value(value)
            penalty += constraint_penalty

            constraint_scores[constraint.name] = {
                'value': value,
                'satisfied': constraint.is_satisfied(value),
                'penalty': constraint_penalty,
            }

        # 综合适应度 = 目标分数 - 约束惩罚
        final_fitness = total_score - penalty

        if return_details:
            return final_fitness, {
                'objective_scores': objective_scores,
                'constraint_scores': constraint_scores,
                'total_objective_score': total_score,
                'penalty': penalty,
            }

        return final_fitness

    def _get_objective_value(self, objective: Objective, metrics: ArchitectureMetrics) -> float:
        """获取目标值"""
        objective_map = {
            'accuracy': metrics.accuracy,
            'latency': metrics.latency,
            'energy': metrics.energy,
            'params': metrics.parameters,
            'flops': metrics.flops,
            'memory': metrics.memory,
        }

        return objective_map.get(objective.name, 0.0)

    def _get_constraint_value(self, constraint: Constraint, metrics: ArchitectureMetrics) -> float:
        """获取约束值"""
        constraint_map = {
            'latency': metrics.latency,
            'energy': metrics.energy,
            'params': metrics.parameters,
            'flops': metrics.flops,
            'memory': metrics.memory,
        }

        return constraint_map.get(constraint.name, 0.0)

    def dominates(self, metrics1: ArchitectureMetrics, metrics2: ArchitectureMetrics) -> bool:
        """
        Pareto支配判断

        Args:
            metrics1: 架构1的性能指标
            metrics2: 架构2的性能指标

        Returns:
            metrics1是否支配metrics2
        """
        # 检查约束
        for constraint in self.constraints:
            value1 = self._get_constraint_value(constraint, metrics1)
            value2 = self._get_constraint_value(constraint, metrics2)

            satisfied1 = constraint.is_satisfied(value1)
            satisfied2 = constraint.is_satisfied(value2)

            # 如果一个满足约束，另一个不满足，则满足的支配
            if satisfied1 and not satisfied2:
                return True
            if not satisfied1 and satisfied2:
                return False

        # 对于约束都满足或都不满足的情况，比较目标
        at_least_one_better = False
        none_worse = True

        for obj in self.objectives:
            value1 = self._get_objective_value(obj, metrics1)
            value2 = self._get_objective_value(obj, metrics2)

            if obj.type == ObjectiveType.MAXIMIZE:
                if value1 > value2:
                    at_least_one_better = True
                elif value1 < value2:
                    none_worse = False
            else:  # MINIMIZE
                if value1 < value2:
                    at_least_one_better = True
                elif value1 > value2:
                    none_worse = False

        return at_least_one_better and none_worse

    def update_pareto_front(self, entry: ArchiveEntry):
        """
        更新Pareto前沿

        Args:
            entry: 新的归档条目
        """
        # 检查新个体是否被前沿中的个体支配
        dominated = False
        to_remove = []

        for i, front_entry in enumerate(self.pareto_front):
            if self.dominates(front_entry.metrics, entry.metrics):
                # 前沿中的个体支配新个体
                dominated = True
                break
            elif self.dominates(entry.metrics, front_entry.metrics):
                # 新个体支配前沿中的个体
                to_remove.append(i)

        # 如果新个体不被支配，添加到前沿
        if not dominated:
            # 移除被新个体支配的个体
            for i in sorted(to_remove, reverse=True):
                del self.pareto_front[i]

            # 添加新个体
            self.pareto_front.append(entry)

    def insert_with_multi_objective(self,
                                    architecture: Any,
                                    metrics: ArchitectureMetrics,
                                    generation: int = 0) -> bool:
        """
        使用多目标标准插入归档

        Args:
            architecture: 架构
            metrics: 性能指标
            generation: 发现代数

        Returns:
            是否成功插入
        """
        # 计算综合适应度
        fitness = self.compute_fitness(metrics)

        # 创建归档条目
        behavior_vector = metrics.get_behavior_vector()
        cell_key = self.behavior_space.get_cell_key(behavior_vector)
        entry = ArchiveEntry(
            architecture=architecture,
            metrics=metrics,
            behavior_vector=behavior_vector,
            cell_key=cell_key,
            generation=generation
        )

        # 检查约束
        for constraint in self.constraints:
            value = self._get_constraint_value(constraint, metrics)
            if not constraint.is_satisfied(value):
                # 约束违反，降低优先级
                # 但如果cell为空，仍然可以插入
                if cell_key in self.archive.grid:
                    return False

        # 插入逻辑
        if cell_key not in self.archive.grid:
            # Cell为空，插入
            self.archive.grid[cell_key] = entry
            self.update_pareto_front(entry)
            return True
        else:
            # Cell已有个体，比较性能
            current_entry = self.archive.grid[cell_key]
            current_fitness = self.compute_fitness(current_entry.metrics)

            if fitness > current_fitness:
                # 新个体更好
                self.archive.grid[cell_key] = entry
                self.update_pareto_front(entry)
                return True
            else:
                return False

    def evolve(self,
              generate_function: Callable,
              mutate_function: Callable,
              n_iterations: int = 1000,
              batch_size: int = 100,
              verbose: bool = True) -> Tuple[Archive, List[ArchiveEntry]]:
        """
        运行多目标NAS进化

        Args:
            generate_function: 生成函数
            mutate_function: 变异函数
            n_iterations: 迭代次数
            batch_size: 批处理大小
            verbose: 是否输出详细信息

        Returns:
            (归档, Pareto前沿)
        """
        logger.info(f"🚀 开始多目标NAS进化，迭代次数: {n_iterations}")

        for iteration in range(n_iterations):
            # 生成一批新个体
            batch = self._generate_batch(generate_function, mutate_function, batch_size)

            # 评估并插入归档
            inserted_count = 0
            for arch in batch:
                metrics = self.characterizer.characterize(arch)
                success = self.insert_with_multi_objective(arch, metrics, generation=iteration)
                if success:
                    inserted_count += 1

            # 输出进度
            if verbose and (iteration + 1) % 10 == 0:
                archive_stats = self.archive.get_statistics()
                logger.info(
                    f"Iteration {iteration + 1}/{n_iterations} | "
                    f"Archive: {archive_stats['size']} | "
                    f"Pareto: {len(self.pareto_front)} | "
                    f"Inserted: {inserted_count}"
                )

        logger.info("✅ 多目标NAS进化完成")
        return self.archive, self.pareto_front

    def _generate_batch(self,
                       generate_function: Callable,
                       mutate_function: Callable,
                       batch_size: int) -> List[Any]:
        """生成一批新个体"""
        batch = []

        for _ in range(batch_size):
            # 从归档或Pareto前沿选择父本
            if np.random.random() < 0.5 and self.pareto_front:
                # 从Pareto前沿选择
                parent = np.random.choice(self.pareto_front).architecture
            else:
                # 从归档中选择
                entry = self.archive.get_random()
                parent = entry.architecture if entry else None

            # 生成子代
            if parent is not None:
                child = mutate_function(parent)
            else:
                child = generate_function()

            batch.append(child)

        return batch

    def get_pareto_front(self) -> List[ArchiveEntry]:
        """获取Pareto前沿"""
        return self.pareto_front.copy()

    def get_pareto_front_metrics(self) -> List[ArchitectureMetrics]:
        """获取Pareto前沿的性能指标"""
        return [entry.metrics for entry in self.pareto_front]


def create_default_multi_objective_nas(behavior_space: BehaviorSpace,
                                      characterizer: BaseCharacterization,
                                      latency_constraint: float = 100.0,
                                      energy_constraint: float = 1000.0,
                                      params_constraint: float = 5.0) -> MultiObjectiveNAS:
    """
    创建默认的多目标NAS配置

    优化目标:
    - accuracy (权重: 0.6, 最大化)
    - latency (权重: 0.2, 最小化)
    - energy (权重: 0.2, 最小化)

    约束条件:
    - latency <= 100ms
    - energy <= 1000mJ
    - params <= 5M

    Args:
        behavior_space: 行为空间定义
        characterizer: 特征提取器
        latency_constraint: 延迟约束（ms）
        energy_constraint: 能耗约束（mJ）
        params_constraint: 参数约束（M）

    Returns:
        多目标NAS优化器
    """
    objectives = [
        Objective(name='accuracy', type=ObjectiveType.MAXIMIZE, weight=0.6),
        Objective(name='latency', type=ObjectiveType.MINIMIZE, weight=0.2),
        Objective(name='energy', type=ObjectiveType.MINIMIZE, weight=0.2),
    ]

    constraints = [
        Constraint(name='latency', threshold=latency_constraint, type="<="),
        Constraint(name='energy', threshold=energy_constraint, type="<="),
        Constraint(name='params', threshold=params_constraint, type="<="),
    ]

    return MultiObjectiveNAS(
        behavior_space=behavior_space,
        characterizer=characterizer,
        objectives=objectives,
        constraints=constraints,
    )


__all__ = [
    'ObjectiveType',
    'Objective',
    'Constraint',
    'MultiObjectiveNAS',
    'create_default_multi_objective_nas',
]
