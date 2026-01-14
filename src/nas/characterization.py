"""
行为特征提取 (Behavior Characterization)
从神经架构中提取行为特征，用于QD-NAS的多样性评估
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


@dataclass
class ArchitectureMetrics:
    """
    神经架构的度量指标

    包含精度、延迟、能耗、复杂度等多种指标
    """
    accuracy: float  # 准确率
    latency: float  # 延迟 (ms)
    energy: float  # 能耗 (mJ)
    parameters: float  # 参数量 (M)
    flops: float  # 计算量 (MFLOPs)
    memory: float  # 内存占用 (MB)

    # 行为特征
    depth: float  # 网络深度
    width: float  # 网络宽度（平均通道数）
    n_layers: int  # 层数
    n_params: int  # 参数数
    n_operations: int  # 操作类型数
    n_skip_connections: int  # 跳跃连接数

    # 约束
    latency_constraint: bool = False  # 是否满足延迟约束
    energy_constraint: bool = False  # 是否满足能耗约束
    param_constraint: bool = False  # 是否满足参数约束

    def get_behavior_vector(self) -> List[float]:
        """
        获取行为特征向量

        用于QD算法的行为空间映射

        Returns:
            行为特征向量 [depth, width, params, flops]
        """
        return [
            self.depth,
            self.width,
            self.parameters,
            self.flops,
        ]

    def get_latency_energy_vector(self) -> List[float]:
        """
        获取延迟-能耗向量

        Returns:
            [latency, energy]
        """
        return [self.latency, self.energy]

    def get_complexity_vector(self) -> List[float]:
        """
        获取复杂度向量

        Returns:
            [operations, skip_connections, parameters, memory]
        """
        return [
            self.n_operations,
            self.n_skip_connections,
            self.parameters,
            self.memory,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'accuracy': self.accuracy,
            'latency': self.latency,
            'energy': self.energy,
            'parameters': self.parameters,
            'flops': self.flops,
            'memory': self.memory,
            'depth': self.depth,
            'width': self.width,
            'n_layers': self.n_layers,
            'n_params': self.n_params,
            'n_operations': self.n_operations,
            'n_skip_connections': self.n_skip_connections,
            'latency_constraint': self.latency_constraint,
            'energy_constraint': self.energy_constraint,
            'param_constraint': self.param_constraint,
        }


class BaseCharacterization(ABC):
    """
    行为特征提取基类

    从不同的角度提取神经架构的行为特征
    """

    @abstractmethod
    def characterize(self, architecture: Any, dataset: str = 'cifar10') -> ArchitectureMetrics:
        """
        提取行为特征

        Args:
            architecture: 神经架构
            dataset: 数据集名称

        Returns:
            架构度量指标
        """
        pass


class StaticCharacterization(BaseCharacterization):
    """
    静态特征提取

    从架构结构本身提取特征，不需要训练
    """

    def __init__(self, latency_model: Optional[Any] = None):
        """
        初始化静态特征提取器

        Args:
            latency_model: 延迟预测模型（可选）
        """
        self.latency_model = latency_model

    def characterize(self, architecture: Any, dataset: str = 'cifar10') -> ArchitectureMetrics:
        """
        提取静态行为特征

        Args:
            architecture: 神经架构
            dataset: 数据集名称

        Returns:
            架构度量指标
        """
        # 提取架构结构特征
        structure_info = self._extract_structure_info(architecture)

        # 计算静态指标
        metrics = ArchitectureMetrics(
            accuracy=0.0,  # 需要训练后获得
            latency=self._estimate_latency(structure_info),
            energy=self._estimate_energy(structure_info),
            parameters=structure_info['params'],
            flops=structure_info['flops'],
            memory=self._estimate_memory(structure_info),
            depth=structure_info['depth'],
            width=structure_info['width'],
            n_layers=structure_info['n_layers'],
            n_params=structure_info['n_params'],
            n_operations=structure_info['n_operations'],
            n_skip_connections=structure_info['n_skip_connections'],
        )

        return metrics

    def _extract_structure_info(self, architecture: Any) -> Dict[str, Any]:
        """
        提取架构结构信息

        Args:
            architecture: 神经架构

        Returns:
            结构信息字典
        """
        # 如果是PyTorch模型
        if isinstance(architecture, nn.Module):
            return self._extract_pytorch_structure(architecture)

        # 如果是字典表示
        elif isinstance(architecture, dict):
            return self._extract_dict_structure(architecture)

        # 其他情况，返回默认值
        else:
            return {
                'depth': 10,
                'width': 64,
                'n_layers': 10,
                'n_params': 1000000,
                'n_operations': 5,
                'n_skip_connections': 2,
                'params': 1.0,  # Million
                'flops': 100.0,  # MFLOPs
            }

    def _extract_pytorch_structure(self, model: nn.Module) -> Dict[str, Any]:
        """提取PyTorch模型的结构信息"""
        layers = list(model.modules())

        n_layers = len(layers)
        total_params = sum(p.numel() for p in model.parameters())
        total_params_m = total_params / 1e6

        # 估计深度（卷积+线性层数）
        depth = sum(1 for layer in layers
                    if isinstance(layer, (nn.Conv2d, nn.Linear)))

        # 估计宽度（平均通道数）
        channels = []
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                channels.append(layer.out_channels)
        width = np.mean(channels) if channels else 64

        # 估计FLOPs（简化）
        flops = total_params * 2  # 每个参数大约2次操作

        return {
            'depth': float(depth),
            'width': float(width),
            'n_layers': n_layers,
            'n_params': int(total_params),
            'n_operations': 5,  # 假设有5种操作
            'n_skip_connections': 2,  # 假设有2个跳跃连接
            'params': total_params_m,
            'flops': flops / 1e6,  # MFLOPs
        }

    def _extract_dict_structure(self, arch_dict: Dict) -> Dict[str, Any]:
        """从字典表示中提取结构信息"""
        # 根据不同的NAS表示方法提取特征
        # 这里提供通用的提取方法

        depth = arch_dict.get('depth', 10)
        width = arch_dict.get('width', 64)
        n_layers = arch_dict.get('n_layers', 10)
        n_operations = arch_dict.get('n_operations', 5)
        n_skip_connections = arch_dict.get('n_skip_connections', 2)

        # 估计参数量和FLOPs
        params = self._estimate_params_from_dict(arch_dict)
        flops = self._estimate_flops_from_dict(arch_dict)

        return {
            'depth': float(depth),
            'width': float(width),
            'n_layers': n_layers,
            'n_params': int(params * 1e6),
            'n_operations': n_operations,
            'n_skip_connections': n_skip_connections,
            'params': params,
            'flops': flops,
        }

    def _estimate_params_from_dict(self, arch_dict: Dict) -> float:
        """从字典估计参数量"""
        depth = arch_dict.get('depth', 10)
        width = arch_dict.get('width', 64)

        # 简化的参数量估计
        # 假设每层大约 width * width * 3 * 3 的参数
        params_per_layer = width * width * 9
        total_params = depth * params_per_layer
        return total_params / 1e6  # Million

    def _estimate_flops_from_dict(self, arch_dict: Dict) -> float:
        """从字典估计FLOPs"""
        depth = arch_dict.get('depth', 10)
        width = arch_dict.get('width', 64)

        # 简化的FLOPs估计
        flops_per_layer = width * width * 9
        total_flops = depth * flops_per_layer * 2  # 输入和输出
        return total_flops / 1e6  # MFLOPs

    def _estimate_latency(self, structure_info: Dict) -> float:
        """估计延迟（ms）"""
        if self.latency_model is not None:
            # 使用延迟预测模型
            features = np.array([
                structure_info['depth'],
                structure_info['width'],
                structure_info['flops'],
            ])
            return self.latency_model.predict(features)[0]
        else:
            # 简化的延迟估计
            flops = structure_info['flops']
            # 假设每MFLOPs需要0.1ms
            return flops * 0.1

    def _estimate_energy(self, structure_info: Dict) -> float:
        """估计能耗（mJ）"""
        flops = structure_info['flops']
        # 假设每MFLOPs需要1mJ
        return flops * 1.0

    def _estimate_memory(self, structure_info: Dict) -> float:
        """估计内存占用（MB）"""
        params = structure_info['params']
        activations = structure_info['flops'] * 4  # 假设activations大小
        return params * 4 + activations / 8  # MB


class DynamicCharacterization(BaseCharacterization):
    """
    动态特征提取

    需要训练后才能提取的行为特征
    """

    def __init__(self, train_config: Optional[Dict] = None):
        """
        初始化动态特征提取器

        Args:
            train_config: 训练配置
        """
        self.train_config = train_config or {}

    def characterize(self, architecture: Any, dataset: str = 'cifar10') -> ArchitectureMetrics:
        """
        提取动态行为特征

        需要先训练架构，然后提取特征

        Args:
            architecture: 神经架构
            dataset: 数据集名称

        Returns:
            架构度量指标
        """
        # 先获取静态指标
        static_characterizer = StaticCharacterization()
        metrics = static_characterizer.characterize(architecture, dataset)

        # 训练并获取动态指标
        training_result = self._train_architecture(architecture, dataset)

        # 更新精度等动态指标
        metrics.accuracy = training_result['accuracy']

        return metrics

    def _train_architecture(self, architecture: Any, dataset: str) -> Dict[str, Any]:
        """
        训练架构

        Args:
            architecture: 神经架构
            dataset: 数据集名称

        Returns:
            训练结果
        """
        # 这里需要实现实际的训练逻辑
        # 由于训练可能很耗时，通常使用proxy或early stopping

        # 返回模拟结果
        return {
            'accuracy': np.random.uniform(0.8, 0.95),
        }


class HybridCharacterization(BaseCharacterization):
    """
    混合特征提取

    结合静态和动态特征提取
    """

    def __init__(self,
                 static_characterizer: Optional[StaticCharacterization] = None,
                 dynamic_characterizer: Optional[DynamicCharacterization] = None):
        """
        初始化混合特征提取器

        Args:
            static_characterizer: 静态特征提取器
            dynamic_characterizer: 动态特征提取器
        """
        self.static_characterizer = static_characterizer or StaticCharacterization()
        self.dynamic_characterizer = dynamic_characterizer

    def characterize(self, architecture: Any, dataset: str = 'cifar10') -> ArchitectureMetrics:
        """
        提取混合行为特征

        Args:
            architecture: 神经架构
            dataset: 数据集名称

        Returns:
            架构度量指标
        """
        # 先获取静态特征
        metrics = self.static_characterizer.characterize(architecture, dataset)

        # 如果有动态特征提取器，获取动态特征
        if self.dynamic_characterizer is not None:
            dynamic_metrics = self.dynamic_characterizer.characterize(architecture, dataset)
            metrics.accuracy = dynamic_metrics.accuracy

        return metrics


# ==================== 辅助函数 ====================

def compute_diversity(architectures: List[ArchitectureMetrics],
                      characterizer: BaseCharacterization) -> float:
    """
    计算一组架构的多样性

    使用平均行为空间距离

    Args:
        architectures: 架构列表
        characterizer: 特征提取器

    Returns:
        多样性分数
    """
    if len(architectures) < 2:
        return 0.0

    behavior_vectors = [arch.get_behavior_vector() for arch in architectures]

    distances = []
    for i in range(len(behavior_vectors)):
        for j in range(i + 1, len(behavior_vectors)):
            vec1 = np.array(behavior_vectors[i])
            vec2 = np.array(behavior_vectors[j])
            dist = np.linalg.norm(vec1 - vec2)
            distances.append(dist)

    return np.mean(distances) if distances else 0.0


def compute_novelty(architecture: ArchitectureMetrics,
                    population: List[ArchitectureMetrics],
                    k: int = 5) -> float:
    """
    计算架构的新颖性

    与种群中k个最近邻居的平均距离

    Args:
        architecture: 目标架构
        population: 种群
        k: 近邻数

    Returns:
        新颖性分数
    """
    if len(population) < k:
        return 1.0

    target_vector = np.array(architecture.get_behavior_vector())
    population_vectors = [np.array(arch.get_behavior_vector()) for arch in population]

    # 计算到所有个体的距离
    distances = [np.linalg.norm(target_vector - vec) for vec in population_vectors]

    # 找到k个最近邻居
    distances.sort()
    k_nearest = distances[:k]

    # 返回平均距离
    return np.mean(k_nearest)


__all__ = [
    'ArchitectureMetrics',
    'BaseCharacterization',
    'StaticCharacterization',
    'DynamicCharacterization',
    'HybridCharacterization',
    'compute_diversity',
    'compute_novelty',
]
