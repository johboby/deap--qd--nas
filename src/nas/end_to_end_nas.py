"""
端到端NAS流程 (End-to-End NAS Pipeline)
完整的数据加载、训练、评估、导出流程
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
import json
import os
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from .qd_nas import QDNASOptimizer
from .search_space import Architecture
from .characterization import ArchitectureMetrics
from .benchmark_suite import StandardDatasets
from .distributed_computing import create_evaluator, BaseEvaluator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NASConfig:
    """
    NAS配置

    Args:
        optimization_mode: 优化模式
        multi_objective: 是否多目标优化
        population_guided: 是否种群引导搜索
        n_iterations: 迭代次数
        batch_size: 批处理大小
        dataset: 数据集名称
        epochs: 训练轮数
        early_stopping: 是否早停
        patience: 早停耐心值
        save_dir: 保存目录
        device: 计算设备
    """
    optimization_mode: str = 'map_elites'
    multi_objective: bool = False
    population_guided: bool = True
    n_iterations: int = 1000
    batch_size: int = 100
    dataset: str = 'cifar10'
    epochs: int = 50
    early_stopping: bool = True
    patience: int = 5
    save_dir: str = './nas_results'
    device: str = 'cpu'

    def save(self, filepath: str):
        """保存配置"""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
        logger.info(f"配置保存至: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'NASConfig':
        """加载配置"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class NASResult:
    """
    NAS结果

    存储完整的NAS搜索结果。
    """
    best_architecture: Optional[Architecture] = None
    best_metrics: Optional[ArchitectureMetrics] = None
    pareto_front: List[Tuple[Architecture, ArchitectureMetrics]] = field(default_factory=list)
    archive_statistics: Dict[str, Any] = field(default_factory=dict)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    total_time: float = 0.0
    config: Optional[NASConfig] = None

    def save(self, save_dir: str):
        """保存结果"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存最佳架构
        if self.best_architecture:
            with open(f'{save_dir}/best_architecture.json', 'w') as f:
                json.dump(self.best_architecture.to_dict(), f, indent=2)

        # 保存Pareto前沿
        if self.pareto_front:
            pareto_data = []
            for arch, metrics in self.pareto_front:
                pareto_data.append({
                    'architecture': arch.to_dict(),
                    'metrics': metrics.to_dict(),
                })
            with open(f'{save_dir}/pareto_front.json', 'w') as f:
                json.dump(pareto_data, f, indent=2)

        # 保存历史
        with open(f'{save_dir}/optimization_history.json', 'w') as f:
            json.dump(self.optimization_history, f, indent=2)

        # 保存统计
        with open(f'{save_dir}/statistics.json', 'w') as f:
            json.dump(self.archive_statistics, f, indent=2)

        # 保存配置
        if self.config:
            self.config.save(f'{save_dir}/config.json')

        logger.info(f"结果保存至: {save_dir}")

    def generate_report(self) -> str:
        """生成报告"""
        report = []
        report.append("="*60)
        report.append("NAS优化报告")
        report.append("="*60)

        if self.best_architecture and self.best_metrics:
            report.append("\n最佳架构:")
            report.append(f"  准确率: {self.best_metrics.accuracy:.4f}")
            report.append(f"  延迟: {self.best_metrics.latency:.2f}ms")
            report.append(f"  能耗: {self.best_metrics.energy:.2f}mJ")
            report.append(f"  参数量: {self.best_metrics.parameters/1e6:.2f}M")
            report.append(f"  计算量: {self.best_metrics.flops/1e6:.2f}M")

        if self.pareto_front:
            report.append(f"\nPareto前沿大小: {len(self.pareto_front)}")

        if self.archive_statistics:
            report.append("\n归档统计:")
            report.append(f"  归档大小: {self.archive_statistics.get('size', 0)}")
            report.append(f"  覆盖率: {self.archive_statistics.get('coverage', 0):.2%}")
            report.append(f"  多样性: {self.archive_statistics.get('diversity', 0):.4f}")

        report.append(f"\n总运行时间: {self.total_time:.2f}s")
        report.append("="*60)

        return "\n".join(report)


class DataPipeline:
    """
    数据管道

    处理数据加载、预处理和增强。
    """

    def __init__(self, dataset_config, batch_size: int = 128):
        """
        初始化数据管道

        Args:
            dataset_config: 数据集配置
            batch_size: 批处理大小
        """
        self.dataset_config = dataset_config
        self.batch_size = batch_size

        self._load_dataset()

    def _load_dataset(self):
        """加载数据集"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for data pipeline")

        import torchvision
        import torchvision.transforms as transforms

        logger.info(f"📥 加载数据集: {self.dataset_config.name}")

        # 数据增强
        transform_train = transforms.Compose([
            transforms.RandomCrop(self.dataset_config.input_shape[1], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.dataset_config.mean, self.dataset_config.std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.dataset_config.mean, self.dataset_config.std),
        ])

        # 加载数据集
        if self.dataset_config.name.lower() == 'cifar10':
            self.trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train
            )
            self.testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test
            )
        elif self.dataset_config.name.lower() == 'cifar100':
            self.trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform_train
            )
            self.testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform_test
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_config.name}")

        # 创建数据加载器
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

        logger.info("✅ 数据集加载完成")

    def get_train_loader(self):
        """获取训练数据加载器"""
        return self.trainloader

    def get_test_loader(self):
        """获取测试数据加载器"""
        return self.testloader


class Trainer:
    """
    训练器

    负责模型训练和评估。
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        初始化训练器

        Args:
            model: 模型
            device: 计算设备
        """
        self.model = model
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
        }

    def setup_optimizer(self, lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 5e-4):
        """设置优化器"""
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    def setup_scheduler(self, epochs: int):
        """设置学习率调度器"""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)

    def train_epoch(self, trainloader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def test(self, testloader) -> Tuple[float, float]:
        """测试模型"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = test_loss / len(testloader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    def train(self,
              trainloader,
              testloader,
              epochs: int = 50,
              verbose: bool = True) -> Dict[str, Any]:
        """
        训练模型

        Args:
            trainloader: 训练数据加载器
            testloader: 测试数据加载器
            epochs: 训练轮数
            verbose: 是否输出详细信息

        Returns:
            训练历史
        """
        self.setup_scheduler(epochs)

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(trainloader)

            # 测试
            test_loss, test_acc = self.test(testloader)

            # 学习率调度
            self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)

            # 早停检查
            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                    f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
                    f"Best: {best_acc:.2f}%"
                )

            if patience_counter >= 5:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return self.history


class EndToEndNAS:
    """
    端到端NAS

    完整的NAS搜索流程。
    """

    def __init__(self, config: NASConfig):
        """
        初始化端到端NAS

        Args:
            config: NAS配置
        """
        self.config = config

        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)

        # 创建优化器
        self.optimizer = QDNASOptimizer(
            optimization_mode=config.optimization_mode,
            multi_objective=config.multi_objective,
            population_guided=config.population_guided
        )

        # 初始化数据管道
        dataset_config = StandardDatasets.get_cifar10()
        if config.dataset.lower() == 'cifar100':
            dataset_config = StandardDatasets.get_cifar100()
        elif config.dataset.lower() == 'mnist':
            dataset_config = StandardDatasets.get_mnist()

        self.data_pipeline = DataPipeline(
            dataset_config=dataset_config,
            batch_size=config.batch_size
        )

        # 创建评估器
        self.evaluator = create_evaluator(
            evaluate_function=self._evaluate_architecture,
            use_multiprocessing=True,
            n_workers=4
        )

        logger.info("🚀 端到端NAS初始化完成")

    def _evaluate_architecture(self, architecture: Architecture) -> ArchitectureMetrics:
        """评估架构"""
        # 创建模型
        model = self._create_model(architecture)

        # 训练
        trainer = Trainer(model, device=self.config.device)
        trainer.setup_optimizer()
        history = trainer.train(
            self.data_pipeline.get_train_loader(),
            self.data_pipeline.get_test_loader(),
            epochs=self.config.epochs,
            verbose=False
        )

        # 测试
        test_acc = np.max(history['test_acc'])

        # 估计性能指标
        n_params = sum(p.numel() for p in model.parameters())
        latency = n_params / 1e6 * 10
        energy = n_params / 1e6 * 5

        metrics = ArchitectureMetrics(
            accuracy=test_acc / 100,
            latency=latency,
            energy=energy,
            parameters=n_params,
            flops=n_params * 10,
            depth=architecture.n_cells,
            width=architecture.n_channels,
            memory=n_params * 4 / (1024 ** 2),
            operation_diversity=0.8,
            skip_connections=0,
        )

        return metrics

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

        model = SimpleCNN(architecture).to(self.config.device)
        return model

    def run(self) -> NASResult:
        """
        运行端到端NAS

        Returns:
            NAS结果
        """
        logger.info("🚀 开始端到端NAS搜索")

        start_time = time.time()

        # 初始化优化器
        self.optimizer.initialize()

        # 运行优化
        archive, pareto_front = self.optimizer.optimize(
            n_iterations=self.config.n_iterations,
            batch_size=self.config.batch_size,
            verbose=True
        )

        # 获取最佳架构
        best_arch = self.optimizer.get_best_architecture()
        best_metrics = None
        if best_arch:
            best_metrics = self.evaluator.evaluate([best_arch])[0]

        # 记录历史
        history = []
        for i, stats in enumerate(archive.get('history', [])):
            history.append({
                'iteration': i,
                'size': stats.get('size', 0),
                'coverage': stats.get('coverage', 0),
                'best_fitness': stats.get('best_fitness', 0),
            })

        # 创建结果
        result = NASResult(
            best_architecture=best_arch,
            best_metrics=best_metrics,
            pareto_front=pareto_front,
            archive_statistics=archive.get_statistics(),
            optimization_history=history,
            total_time=time.time() - start_time,
            config=self.config
        )

        # 保存结果
        result.save(self.config.save_dir)

        # 生成报告
        report = result.generate_report()
        print(report)

        # 保存报告
        with open(f'{self.config.save_dir}/report.txt', 'w') as f:
            f.write(report)

        logger.info(f"✅ 端到端NAS完成，耗时: {result.total_time:.2f}s")

        return result


def create_end_to_end_nas(config: NASConfig) -> EndToEndNAS:
    """
    工厂函数：创建端到端NAS

    Args:
        config: NAS配置

    Returns:
        端到端NAS对象
    """
    return EndToEndNAS(config)


__all__ = [
    'NASConfig',
    'NASResult',
    'DataPipeline',
    'Trainer',
    'EndToEndNAS',
    'create_end_to_end_nas',
]
