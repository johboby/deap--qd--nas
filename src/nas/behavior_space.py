"""
行为空间定义 (Behavior Space)
用于QD-NAS中的行为特征映射和多样性维护
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class BehaviorType(Enum):
    """行为特征类型"""
    DISCRETE = "discrete"  # 离散值
    CONTINUOUS = "continuous"  # 连续值
    CATEGORICAL = "categorical"  # 类别值


@dataclass
class BehaviorDimension:
    """
    行为维度定义

    Args:
        name: 维度名称
        min_val: 最小值
        max_val: 最大值
        behavior_type: 行为类型
        n_bins: 离散化桶数（仅用于连续类型）
    """
    name: str
    min_val: float
    max_val: float
    behavior_type: BehaviorType = BehaviorType.CONTINUOUS
    n_bins: int = 10

    def __post_init__(self):
        """初始化后检查"""
        if self.min_val >= self.max_val:
            raise ValueError(f"Invalid range for {self.name}: [{self.min_val}, {self.max_val}]")

    def discretize(self, value: float) -> int:
        """将连续值离散化为bin索引"""
        if self.behavior_type == BehaviorType.DISCRETE:
            return int(value)
        elif self.behavior_type == BehaviorType.CONTINUOUS:
            # 归一化到[0, 1]
            normalized = (value - self.min_val) / (self.max_val - self.min_val + 1e-10)
            normalized = np.clip(normalized, 0, 1)
            # 映射到bin
            return int(normalized * (self.n_bins - 1))
        else:
            return int(value)

    def normalize(self, value: float) -> float:
        """将值归一化到[0, 1]"""
        return (value - self.min_val) / (self.max_val - self.min_val + 1e-10)


class BehaviorSpace:
    """
    行为空间

    定义神经架构的行为特征空间，用于QD算法的多样性维护。
    支持多个维度的行为特征，如网络深度、宽度、参数量、计算量等。

    Example:
        >>> behavior_space = BehaviorSpace()
        >>> behavior_space.add_dimension(BehaviorDimension("depth", 0, 100))
        >>> behavior_space.add_dimension(BehaviorDimension("width", 0, 1000))
        >>> behavior_space.add_dimension(BehaviorDimension("params", 0, 10e6))
        >>> cell_key = behavior_space.get_cell_key([5, 100, 1e6])
    """

    def __init__(self):
        """初始化行为空间"""
        self.dimensions: List[BehaviorDimension] = []
        self.dim_names: List[str] = []

    def add_dimension(self, dimension: BehaviorDimension):
        """
        添加行为维度

        Args:
            dimension: 行为维度定义
        """
        self.dimensions.append(dimension)
        self.dim_names.append(dimension.name)

    def get_cell_key(self, behavior_vector: List[float]) -> Tuple[int, ...]:
        """
        获取行为向量对应的网格cell key

        Args:
            behavior_vector: 行为特征向量

        Returns:
            cell key tuple
        """
        if len(behavior_vector) != len(self.dimensions):
            raise ValueError(f"Expected {len(self.dimensions)} dimensions, got {len(behavior_vector)}")

        cell_key = tuple(
            dim.discretize(value)
            for dim, value in zip(self.dimensions, behavior_vector)
        )
        return cell_key

    def normalize_vector(self, behavior_vector: List[float]) -> np.ndarray:
        """
        归一化行为向量到[0, 1]

        Args:
            behavior_vector: 行为特征向量

        Returns:
            归一化的行为向量
        """
        return np.array([
            dim.normalize(value)
            for dim, value in zip(self.dimensions, behavior_vector)
        ])

    def distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个行为向量之间的距离（欧氏距离）

        Args:
            vec1: 行为向量1
            vec2: 行为向量2

        Returns:
            归一化的欧氏距离
        """
        norm1 = self.normalize_vector(vec1)
        norm2 = self.normalize_vector(vec2)
        return np.linalg.norm(norm1 - norm2)

    def get_grid_size(self) -> Tuple[int, ...]:
        """
        获取行为空间的网格大小

        Returns:
            每个维度的bin数组成的元组
        """
        return tuple(dim.n_bins for dim in self.dimensions)

    def total_cells(self) -> int:
        """
        获取行为空间的总cell数量

        Returns:
            总cell数
        """
        grid_size = self.get_grid_size()
        total = 1
        for size in grid_size:
            total *= size
        return total

    def random_sample(self) -> List[float]:
        """
        随机采样一个行为向量

        Returns:
            随机行为向量
        """
        return [
            np.random.uniform(dim.min_val, dim.max_val)
            for dim in self.dimensions
        ]


# ==================== 预定义的行为空间 ====================

def create_nas_behavior_space() -> BehaviorSpace:
    """
    创建标准的NAS行为空间

    包含以下行为维度:
    - depth: 网络深度
    - width: 网络宽度（平均通道数）
    - params: 参数量（百万）
    - flops: 计算量（MFLOPs）

    Returns:
        NAS行为空间
    """
    behavior_space = BehaviorSpace()

    # 网络深度 (0-100层)
    behavior_space.add_dimension(BehaviorDimension(
        name="depth",
        min_val=0,
        max_val=100,
        behavior_type=BehaviorType.CONTINUOUS,
        n_bins=20
    ))

    # 网络宽度 (0-1000通道)
    behavior_space.add_dimension(BehaviorDimension(
        name="width",
        min_val=0,
        max_val=1000,
        behavior_type=BehaviorType.CONTINUOUS,
        n_bins=20
    ))

    # 参数量 (0-100M)
    behavior_space.add_dimension(BehaviorDimension(
        name="params",
        min_val=0,
        max_val=100e6,
        behavior_type=BehaviorType.CONTINUOUS,
        n_bins=20
    ))

    # 计算量 (0-10 GFLOPs)
    behavior_space.add_dimension(BehaviorDimension(
        name="flops",
        min_val=0,
        max_val=10e9,
        behavior_type=BehaviorType.CONTINUOUS,
        n_bins=20
    ))

    return behavior_space


def create_latency_energy_behavior_space() -> BehaviorSpace:
    """
    创建延迟-能耗行为空间

    包含以下行为维度:
    - latency: 延迟（ms）
    - energy: 能耗（mJ）

    Returns:
        延迟-能耗行为空间
    """
    behavior_space = BehaviorSpace()

    # 延迟 (0-1000ms)
    behavior_space.add_dimension(BehaviorDimension(
        name="latency",
        min_val=0,
        max_val=1000,
        behavior_type=BehaviorType.CONTINUOUS,
        n_bins=20
    ))

    # 能耗 (0-10000mJ)
    behavior_space.add_dimension(BehaviorDimension(
        name="energy",
        min_val=0,
        max_val=10000,
        behavior_type=BehaviorType.CONTINUOUS,
        n_bins=20
    ))

    return behavior_space


def create_complexity_behavior_space() -> BehaviorSpace:
    """
    创建复杂度行为空间

    包含以下行为维度:
    - operations: 操作类型多样性
    - skip_connections: 跳跃连接数量
    - parameters: 参数量
    - memory: 内存占用

    Returns:
        复杂度行为空间
    """
    behavior_space = BehaviorSpace()

    # 操作类型多样性 (0-10)
    behavior_space.add_dimension(BehaviorDimension(
        name="operations",
        min_val=0,
        max_val=10,
        behavior_type=BehaviorType.CONTINUOUS,
        n_bins=10
    ))

    # 跳跃连接数量 (0-50)
    behavior_space.add_dimension(BehaviorDimension(
        name="skip_connections",
        min_val=0,
        max_val=50,
        behavior_type=BehaviorType.CONTINUOUS,
        n_bins=10
    ))

    # 参数量 (0-100M)
    behavior_space.add_dimension(BehaviorDimension(
        name="parameters",
        min_val=0,
        max_val=100e6,
        behavior_type=BehaviorType.CONTINUOUS,
        n_bins=20
    ))

    # 内存占用 (0-1000MB)
    behavior_space.add_dimension(BehaviorDimension(
        name="memory",
        min_val=0,
        max_val=1000,
        behavior_type=BehaviorType.CONTINUOUS,
        n_bins=20
    ))

    return behavior_space


__all__ = [
    'BehaviorType',
    'BehaviorDimension',
    'BehaviorSpace',
    'create_nas_behavior_space',
    'create_latency_energy_behavior_space',
    'create_complexity_behavior_space',
]
