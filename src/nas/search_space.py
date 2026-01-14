"""
NAS搜索空间和架构编码 (Search Space & Architecture Encoding)
定义神经架构搜索空间和编码方式
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import json


class OperationType(Enum):
    """操作类型"""
    CONV3X3 = "conv3x3"
    CONV5X5 = "conv5x5"
    CONV7X7 = "conv7x7"
    DILATED_CONV = "dilated_conv"
    SEPARABLE_CONV = "separable_conv"
    MAX_POOL3X3 = "max_pool3x3"
    AVG_POOL3X3 = "avg_pool3x3"
    IDENTITY = "identity"
    SKIP_CONNECTION = "skip_connection"
    ZEROIZE = "zeroize"


@dataclass
class Cell:
    """
    神经单元(Normal Cell/Reduction Cell)

    Args:
        nodes: 节点数量
        edges: 边定义 [(node_i, node_j, operation)]
        cell_type: cell类型
    """
    nodes: int = 4
    edges: List[Tuple[int, int, str]] = field(default_factory=list)
    cell_type: str = "normal"  # normal or reduction

    def __post_init__(self):
        """初始化后处理"""
        if not self.edges:
            self._generate_random_edges()

    def _generate_random_edges(self):
        """生成随机边"""
        operations = [op.value for op in OperationType]

        for i in range(self.nodes + 2):  # +2 for input nodes
            for j in range(i + 1, self.nodes + 2):
                op = random.choice(operations)
                self.edges.append((i, j, op))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'cell_type': self.cell_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Cell':
        """从字典创建"""
        return cls(
            nodes=data['nodes'],
            edges=data['edges'],
            cell_type=data['cell_type'],
        )


@dataclass
class Architecture:
    """
    神经架构表示

    Args:
        n_cells: cell数量
        n_nodes: 每个cell的节点数
        n_channels: 初始通道数
        stem_channels: stem网络通道数
        stem_type: stem类型
        cells: cell列表
        reduction_indices: reduction cell的索引列表
    """
    n_cells: int = 8
    n_nodes: int = 4
    n_channels: int = 16
    stem_channels: int = 32
    stem_type: str = "conv3x3_bn_relu"
    cells: List[Cell] = field(default_factory=list)
    reduction_indices: List[int] = field(default_factory=lambda: [2, 5])

    def __post_init__(self):
        """初始化后处理"""
        if not self.cells:
            self._generate_random_cells()

    def _generate_random_cells(self):
        """生成随机cell"""
        for i in range(self.n_cells):
            cell_type = "reduction" if i in self.reduction_indices else "normal"
            cell = Cell(nodes=self.n_nodes, cell_type=cell_type)
            self.cells.append(cell)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'n_cells': self.n_cells,
            'n_nodes': self.n_nodes,
            'n_channels': self.n_channels,
            'stem_channels': self.stem_channels,
            'stem_type': self.stem_type,
            'cells': [cell.to_dict() for cell in self.cells],
            'reduction_indices': self.reduction_indices,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Architecture':
        """从字典创建"""
        return cls(
            n_cells=data['n_cells'],
            n_nodes=data['n_nodes'],
            n_channels=data['n_channels'],
            stem_channels=data['stem_channels'],
            stem_type=data['stem_type'],
            cells=[Cell.from_dict(cell_data) for cell_data in data['cells']],
            reduction_indices=data['reduction_indices'],
        )

    def encode(self) -> np.ndarray:
        """
        编码为向量表示

        Returns:
            编码向量
        """
        encoding = []

        # 编码全局参数
        encoding.extend([self.n_cells, self.n_nodes, self.n_channels, self.stem_channels])

        # 编码stem类型（one-hot）
        stem_types = ["conv3x3_bn_relu", "conv3x3_bn", "conv5x5_bn_relu"]
        stem_encoding = [1.0 if self.stem_type == st else 0.0 for st in stem_types]
        encoding.extend(stem_encoding)

        # 编码每个cell
        for cell in self.cells:
            # 编码cell类型
            encoding.append(1.0 if cell.cell_type == "normal" else 0.0)

            # 编码节点数
            encoding.append(cell.nodes)

            # 编码操作（简化：使用操作ID）
            operations = [op.value for op in OperationType]
            for _, _, op in cell.edges:
                op_id = operations.index(op) if op in operations else 0
                encoding.append(op_id)

        return np.array(encoding, dtype=np.float32)

    @classmethod
    def decode(cls, encoding: np.ndarray) -> 'Architecture':
        """
        从向量解码为架构

        Args:
            encoding: 编码向量

        Returns:
            架构对象
        """
        # 解码全局参数
        n_cells = int(encoding[0])
        n_nodes = int(encoding[1])
        n_channels = int(encoding[2])
        stem_channels = int(encoding[3])

        # 解码stem类型
        stem_types = ["conv3x3_bn_relu", "conv3x3_bn", "conv5x5_bn_relu"]
        stem_idx = np.argmax(encoding[4:4+len(stem_types)])
        stem_type = stem_types[int(stem_idx)]

        # 解码cells
        cells = []
        ptr = 4 + len(stem_types)

        for _ in range(n_cells):
            cell_type = "normal" if encoding[ptr] > 0.5 else "reduction"
            ptr += 1

            nodes = int(encoding[ptr])
            ptr += 1

            cell = Cell(nodes=nodes, cell_type=cell_type)

            # 解码边
            operations = [op.value for op in OperationType]
            for _ in range(nodes + 2):
                for _ in range(nodes + 2):
                    if ptr < len(encoding):
                        op_id = int(encoding[ptr])
                        op = operations[op_id] if op_id < len(operations) else operations[0]
                        cell.edges.append((0, 0, op))  # 简化
                        ptr += 1

            cells.append(cell)

        return cls(
            n_cells=n_cells,
            n_nodes=n_nodes,
            n_channels=n_channels,
            stem_channels=stem_channels,
            stem_type=stem_type,
            cells=cells,
        )


class SearchSpace:
    """
    NAS搜索空间

    定义神经架构的搜索空间，支持生成和变异操作
    """

    def __init__(self,
                 n_cells_range: Tuple[int, int] = (6, 12),
                 n_nodes_range: Tuple[int, int] = (4, 8),
                 n_channels_range: Tuple[int, int] = (8, 64),
                 stem_channels_range: Tuple[int, int] = (16, 64),
                 available_operations: Optional[List[OperationType]] = None):
        """
        初始化搜索空间

        Args:
            n_cells_range: cell数量范围
            n_nodes_range: 每个cell节点数范围
            n_channels_range: 初始通道数范围
            stem_channels_range: stem通道数范围
            available_operations: 可用操作列表
        """
        self.n_cells_range = n_cells_range
        self.n_nodes_range = n_nodes_range
        self.n_channels_range = n_channels_range
        self.stem_channels_range = stem_channels_range
        self.available_operations = available_operations or list(OperationType)

    def random_sample(self) -> Architecture:
        """
        随机采样一个架构

        Returns:
            随机架构
        """
        n_cells = random.randint(*self.n_cells_range)
        n_nodes = random.randint(*self.n_nodes_range)
        n_channels = random.randint(*self.n_channels_range)
        stem_channels = random.randint(*self.stem_channels_range)

        arch = Architecture(
            n_cells=n_cells,
            n_nodes=n_nodes,
            n_channels=n_channels,
            stem_channels=stem_channels,
        )

        return arch

    def mutate(self,
               architecture: Architecture,
               mutation_rate: float = 0.2) -> Architecture:
        """
        变异一个架构

        Args:
            architecture: 原始架构
            mutation_rate: 变异率

        Returns:
            变异后的架构
        """
        # 深拷贝架构
        new_arch = Architecture.from_dict(architecture.to_dict())

        # 随机选择变异类型
        mutation_types = [
            'mutate_n_cells',
            'mutate_n_nodes',
            'mutate_channels',
            'mutate_operations',
        ]

        mutation_type = random.choice(mutation_types)

        if mutation_type == 'mutate_n_cells' and random.random() < mutation_rate:
            # 变cell数量
            delta = random.choice([-1, 1])
            new_n_cells = np.clip(new_arch.n_cells + delta, *self.n_cells_range)
            new_arch.n_cells = int(new_n_cells)

        elif mutation_type == 'mutate_n_nodes' and random.random() < mutation_rate:
            # 变节点数量
            delta = random.choice([-1, 1])
            new_n_nodes = np.clip(new_arch.n_nodes + delta, *self.n_nodes_range)
            new_arch.n_nodes = int(new_n_nodes)

        elif mutation_type == 'mutate_channels' and random.random() < mutation_rate:
            # 变通道数
            if random.random() < 0.5:
                delta = random.choice([-8, -4, 4, 8])
                new_n_channels = np.clip(new_arch.n_channels + delta, *self.n_channels_range)
                new_arch.n_channels = int(new_n_channels)
            else:
                delta = random.choice([-16, -8, 8, 16])
                new_stem_channels = np.clip(new_arch.stem_channels + delta, *self.stem_channels_range)
                new_arch.stem_channels = int(new_stem_channels)

        elif mutation_type == 'mutate_operations' and random.random() < mutation_rate:
            # 变操作
            for cell in new_arch.cells:
                if random.random() < mutation_rate:
                    # 随机修改一条边
                    if cell.edges:
                        edge_idx = random.randint(0, len(cell.edges) - 1)
                        i, j, _ = cell.edges[edge_idx]
                        new_op = random.choice([op.value for op in self.available_operations])
                        cell.edges[edge_idx] = (i, j, new_op)

        return new_arch

    def crossover(self,
                 parent1: Architecture,
                 parent2: Architecture) -> Architecture:
        """
        交叉两个架构

        Args:
            parent1: 父本1
            parent2: 父本2

        Returns:
            子代架构
        """
        # 简化的交叉：随机选择cell数量和通道数
        child_arch = Architecture.from_dict(parent1.to_dict())

        # 随机选择继承父本1或父本2的cell
        child_arch.cells = []
        min_cells = min(len(parent1.cells), len(parent2.cells))
        for i in range(min_cells):
            if random.random() < 0.5:
                child_arch.cells.append(Cell.from_dict(parent1.cells[i].to_dict()))
            else:
                child_arch.cells.append(Cell.from_dict(parent2.cells[i].to_dict()))

        return child_arch

    def local_search(self,
                    architecture: Architecture,
                    n_neighbors: int = 10) -> List[Architecture]:
        """
        局部搜索：生成邻域架构

        Args:
            architecture: 当前架构
            n_neighbors: 邻域大小

        Returns:
            邻域架构列表
        """
        neighbors = []

        for _ in range(n_neighbors):
            neighbor = self.mutate(architecture, mutation_rate=0.1)
            neighbors.append(neighbor)

        return neighbors


class HierarchicalSearchSpace(SearchSpace):
    """
    分层搜索空间

    支持粗粒度到细粒度的分层搜索
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_level = 0
        self.levels = [
            {'n_cells_range': (6, 10), 'n_nodes_range': (4, 6)},  # 粗粒度
            {'n_cells_range': (8, 12), 'n_nodes_range': (5, 8)},  # 中等粒度
            {'n_cells_range': (10, 20), 'n_nodes_range': (6, 10)},  # 细粒度
        ]

    def advance_level(self):
        """推进到下一层（更细粒度）"""
        if self.current_level < len(self.levels) - 1:
            self.current_level += 1
            # 更新搜索空间范围
            level_config = self.levels[self.current_level]
            self.n_cells_range = level_config['n_cells_range']
            self.n_nodes_range = level_config['n_nodes_range']
            print(f"📈 推进到搜索层级 {self.current_level + 1}")
        else:
            print(f"⚠️  已经在最细粒度层级")

    def current_config(self) -> Dict[str, Tuple[int, int]]:
        """获取当前层级的配置"""
        return self.levels[self.current_level]


__all__ = [
    'OperationType',
    'Cell',
    'Architecture',
    'SearchSpace',
    'HierarchicalSearchSpace',
]
