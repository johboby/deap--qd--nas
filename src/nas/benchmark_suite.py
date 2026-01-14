"""
NAS基准测试套件 (NAS Benchmark Suite)
集成标准NAS基准和数据集
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .search_space import Architecture, SearchSpace
from .characterization import ArchitectureMetrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """
    数据集配置

    Args:
        name: 数据集名称
        num_classes: 类别数
        input_shape: 输入形状
        train_size: 训练集大小
        test_size: 测试集大小
        mean: 数据均值
        std: 数据标准差
    """
    name: str
    num_classes: int
    input_shape: Tuple[int, int, int]  # (C, H, W)
    train_size: int
    test_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]

    def __str__(self) -> str:
        return self.name


class StandardDatasets:
    """标准数据集"""

    @staticmethod
    def get_cifar10() -> DatasetConfig:
        """CIFAR-10"""
        return DatasetConfig(
            name='cifar10',
            num_classes=10,
            input_shape=(3, 32, 32),
            train_size=50000,
            test_size=10000,
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        )

    @staticmethod
    def get_cifar100() -> DatasetConfig:
        """CIFAR-100"""
        return DatasetConfig(
            name='cifar100',
            num_classes=100,
            input_shape=(3, 32, 32),
            train_size=50000,
            test_size=10000,
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2009, 0.1984, 0.2023),
        )

    @staticmethod
    def get_mnist() -> DatasetConfig:
        """MNIST"""
        return DatasetConfig(
            name='mnist',
            num_classes=10,
            input_shape=(1, 28, 28),
            train_size=60000,
            test_size=10000,
            mean=(0.1307,),
            std=(0.3081,),
        )

    @staticmethod
    def get_imagenet() -> DatasetConfig:
        """ImageNet（简化配置）"""
        return DatasetConfig(
            name='imagenet',
            num_classes=1000,
            input_shape=(3, 224, 224),
            train_size=1281167,
            test_size=50000,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )


class BaseNASBenchmark(ABC):
    """
    NAS基准测试基类
    """

    def __init__(self,
                 dataset_config: DatasetConfig,
                 search_space: SearchSpace,
                 device: str = 'cpu'):
        """
        初始化NAS基准

        Args:
            dataset_config: 数据集配置
            search_space: 搜索空间
            device: 计算设备
        """
        self.dataset_config = dataset_config
        self.search_space = search_space
        self.device = device

        # 加载数据集
        self._load_dataset()

        logger.info(f"📊 NAS基准初始化完成: {dataset_config.name}")

    @abstractmethod
    def _load_dataset(self):
        """加载数据集"""
        pass

    @abstractmethod
    def train_model(self, architecture: Architecture, epochs: int) -> Tuple[float, float]:
        """
        训练模型

        Args:
            architecture: 架构
            epochs: 训练轮数

        Returns:
            (train_accuracy, test_accuracy)
        """
        pass

    @abstractmethod
    def evaluate_architecture(self, architecture: Architecture) -> ArchitectureMetrics:
        """
        评估架构

        Args:
            architecture: 架构

        Returns:
            性能指标
        """
        pass


class CIFAR10Benchmark(BaseNASBenchmark):
    """
    CIFAR-10 NAS基准

    标准的CIFAR-10 NAS基准测试。
    """

    def __init__(self,
                 search_space: SearchSpace,
                 data_dir: str = './data',
                 device: str = 'cpu',
                 batch_size: int = 128):
        """
        初始化CIFAR-10基准

        Args:
            search_space: 搜索空间
            data_dir: 数据目录
            device: 计算设备
            batch_size: 批处理大小
        """
        self.data_dir = data_dir
        self.batch_size = batch_size

        dataset_config = StandardDatasets.get_cifar10()

        super().__init__(dataset_config, search_space, device)

    def _load_dataset(self):
        """加载CIFAR-10数据集"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CIFAR-10 benchmark")

        logger.info("📥 加载CIFAR-10数据集")

        # 数据增强
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.dataset_config.mean, self.dataset_config.std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.dataset_config.mean, self.dataset_config.std),
        ])

        # 加载数据集
        self.trainset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )

        self.testset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform_test
        )

        # 创建数据加载器
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        logger.info(f"✅ CIFAR-10数据集加载完成")
        logger.info(f"   训练集: {len(self.trainset)}")
        logger.info(f"   测试集: {len(self.testset)}")

    def train_model(self, architecture: Architecture, epochs: int = 50) -> Tuple[float, float]:
        """
        训练模型

        Args:
            architecture: 架构
            epochs: 训练轮数

        Returns:
            (train_accuracy, test_accuracy)
        """
        # 创建模型
        model = self._create_model(architecture)

        # 训练
        train_acc = self._train(model, epochs)

        # 测试
        test_acc = self._test(model)

        return train_acc, test_acc

    def _create_model(self, architecture: Architecture) -> nn.Module:
        """创建模型"""
        class SimpleCNN(nn.Module):
            def __init__(self, arch):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, arch.n_channels, 3, padding=1),
                    nn.BatchNorm2d(arch.n_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(arch.n_channels, arch.n_channels * 2, 3, padding=1),
                    nn.BatchNorm2d(arch.n_channels * 2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(arch.n_channels * 2 * 8 * 8, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, 10),
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

        model = SimpleCNN(architecture).to(self.device)
        return model

    def _train(self, model: nn.Module, epochs: int) -> float:
        """训练模型"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        model.train()
        correct = 0
        total = 0

        for epoch in range(epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            scheduler.step()

        accuracy = 100. * correct / total
        return accuracy

    def _test(self, model: nn.Module) -> float:
        """测试模型"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        return accuracy

    def evaluate_architecture(self, architecture: Architecture) -> ArchitectureMetrics:
        """
        评估架构

        Args:
            architecture: 架构

        Returns:
            性能指标
        """
        logger.info(f"🔍 评估架构: {architecture.to_dict()}")

        # 训练
        train_acc, test_acc = self.train_model(architecture, epochs=50)

        # 估计延迟和能耗（简化）
        n_params = sum(p.numel() for p in self._create_model(architecture).parameters())
        latency = n_params / 1e6 * 10  # 简化估计
        energy = n_params / 1e6 * 5  # 简化估计

        metrics = ArchitectureMetrics(
            accuracy=test_acc / 100,
            latency=latency,
            energy=energy,
            parameters=n_params,
            flops=n_params * 10,  # 简化估计
            depth=architecture.n_cells,
            width=architecture.n_channels,
            memory=n_params * 4 / (1024 ** 2),  # 假设float32
            operation_diversity=0.8,
            skip_connections=0,
        )

        logger.info(f"✅ 评估完成: Accuracy={test_acc:.2f}%")

        return metrics


class BenchmarkResults:
    """
    基准测试结果

    存储和比较不同NAS方法的性能。
    """

    def __init__(self):
        """初始化基准测试结果"""
        self.results: List[Dict[str, Any]] = []

    def add_result(self,
                  method_name: str,
                  architecture: Architecture,
                  metrics: ArchitectureMetrics,
                  runtime: float):
        """
        添加结果

        Args:
            method_name: 方法名称
            architecture: 架构
            metrics: 性能指标
            runtime: 运行时间
        """
        result = {
            'method': method_name,
            'architecture': architecture.to_dict(),
            'accuracy': metrics.accuracy,
            'latency': metrics.latency,
            'energy': metrics.energy,
            'parameters': metrics.parameters,
            'runtime': runtime,
        }
        self.results.append(result)

    def get_comparison(self) -> pd.DataFrame:
        """
        获取比较表格

        Returns:
            比较DataFrame
        """
        try:
            import pandas as pd
            df = pd.DataFrame(self.results)
            return df
        except ImportError:
            logger.warning("Pandas not available, cannot create DataFrame")
            return None

    def save_results(self, filepath: str):
        """
        保存结果

        Args:
            filepath: 文件路径
        """
        import json

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"结果保存至: {filepath}")

    def print_summary(self):
        """打印摘要"""
        print("\n" + "="*60)
        print("基准测试结果摘要")
        print("="*60)

        if not self.results:
            print("无结果")
            return

        # 按方法分组
        methods = {}
        for result in self.results:
            method = result['method']
            if method not in methods:
                methods[method] = []
            methods[method].append(result)

        # 打印每个方法的统计
        for method, method_results in methods.items():
            accuracies = [r['accuracy'] for r in method_results]
            latencies = [r['latency'] for r in method_results]
            parameters = [r['parameters'] for r in method_results]

            print(f"\n{method}:")
            print(f"  平均准确率: {np.mean(accuracies):.4f}")
            print(f"  平均延迟: {np.mean(latencies):.2f}ms")
            print(f"  平均参数量: {np.mean(parameters)/1e6:.2f}M")

        print("\n" + "="*60)


class BenchmarkRunner:
    """
    基准测试运行器

    运行和比较不同的NAS方法。
    """

    def __init__(self,
                 benchmark: BaseNASBenchmark,
                 search_space: SearchSpace):
        """
        初始化基准测试运行器

        Args:
            benchmark: 基准测试
            search_space: 搜索空间
        """
        self.benchmark = benchmark
        self.search_space = search_space
        self.results = BenchmarkResults()

    def run_benchmark(self,
                     method_name: str,
                     n_samples: int = 10) -> Dict[str, Any]:
        """
        运行基准测试

        Args:
            method_name: 方法名称
            n_samples: 采样数量

        Returns:
            统计结果
        """
        logger.info(f"🏃 运行基准测试: {method_name}")

        import time
        start_time = time.time()

        # 采样架构并评估
        architectures = [self.search_space.random_sample() for _ in range(n_samples)]

        for arch in architectures:
            metrics = self.benchmark.evaluate_architecture(arch)
            self.results.add_result(method_name, arch, metrics, time.time() - start_time)

        # 计算统计
        method_results = [r for r in self.results.results if r['method'] == method_name]

        stats = {
            'method': method_name,
            'mean_accuracy': np.mean([r['accuracy'] for r in method_results]),
            'std_accuracy': np.std([r['accuracy'] for r in method_results]),
            'mean_latency': np.mean([r['latency'] for r in method_results]),
            'mean_parameters': np.mean([r['parameters'] for r in method_results]),
        }

        logger.info(f"✅ 基准测试完成: {method_name}")
        return stats


def create_benchmark(dataset_name: str = 'cifar10',
                   search_space: Optional[SearchSpace] = None,
                   **kwargs) -> BaseNASBenchmark:
    """
    工厂函数：创建基准测试

    Args:
        dataset_name: 数据集名称
        search_space: 搜索空间
        **kwargs: 其他参数

    Returns:
        基准测试对象
    """
    search_space = search_space or SearchSpace()

    if dataset_name.lower() == 'cifar10':
        return CIFAR10Benchmark(search_space=search_space, **kwargs)
    elif dataset_name.lower() == 'cifar100':
        return CIFAR10Benchmark(search_space=search_space, **kwargs)
    elif dataset_name.lower() == 'mnist':
        return CIFAR10Benchmark(search_space=search_space, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


__all__ = [
    'DatasetConfig',
    'StandardDatasets',
    'BaseNASBenchmark',
    'CIFAR10Benchmark',
    'BenchmarkResults',
    'BenchmarkRunner',
    'create_benchmark',
]
