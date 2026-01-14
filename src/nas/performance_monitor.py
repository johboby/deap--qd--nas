"""
性能监控和分析工具 (Performance Monitor and Analyzer)
实时监控、性能指标收集和可视化
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import time
import logging
from abc import ABC, abstractmethod

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    性能指标

    Args:
        timestamp: 时间戳
        iteration: 迭代次数
        fitness: 适应度
        diversity: 多样性
        coverage: 覆盖率
        latency: 延迟
        memory_usage: 内存使用
        cpu_usage: CPU使用率
    """
    timestamp: float
    iteration: int
    fitness: float
    diversity: float
    coverage: float
    latency: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'iteration': self.iteration,
            'fitness': self.fitness,
            'diversity': self.diversity,
            'coverage': self.coverage,
            'latency': self.latency,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
        }


class MetricCollector(ABC):
    """
    指标收集器基类
    """

    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """收集指标"""
        pass


class SystemMetricCollector(MetricCollector):
    """
    系统指标收集器
    """

    def __init__(self):
        """初始化系统指标收集器"""
        self.start_time = time.time()

    def collect(self) -> Dict[str, Any]:
        """收集系统指标"""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        return {
            'memory_usage': process.memory_info().rss / (1024 ** 2),  # MB
            'cpu_usage': process.cpu_percent(),
            'elapsed_time': time.time() - self.start_time,
        }


class PerformanceMonitor:
    """
    性能监控器

    实时监控优化过程，收集和记录性能指标。

    核心特性:
    1. 实时指标收集
    2. 滑动窗口统计
    3. 性能趋势分析
    4. 异常检测
    """

    def __init__(self,
                 window_size: int = 100,
                 enable_system_metrics: bool = True):
        """
        初始化性能监控器

        Args:
            window_size: 滑动窗口大小
            enable_system_metrics: 是否启用系统指标
        """
        self.window_size = window_size
        self.enable_system_metrics = enable_system_metrics

        # 指标历史
        self.metrics_history: List[PerformanceMetrics] = []

        # 滑动窗口
        self.fitness_window = deque(maxlen=window_size)
        self.diversity_window = deque(maxlen=window_size)
        self.coverage_window = deque(maxlen=window_size)

        # 系统指标收集器
        if enable_system_metrics:
            try:
                self.system_collector = SystemMetricCollector()
            except ImportError:
                logger.warning("psutil not available, system metrics disabled")
                self.enable_system_metrics = False
                self.system_collector = None
        else:
            self.system_collector = None

        # 异常检测阈值
        self.fitness_stagnation_threshold = 50  # 迭代
        self.diversity_drop_threshold = 0.1  # 多样性下降阈值

        logger.info(f"📊 性能监控器初始化完成")
        logger.info(f"   窗口大小: {window_size}")

    def record(self,
              iteration: int,
              fitness: float,
              diversity: float,
              coverage: float,
              latency: Optional[float] = None):
        """
        记录性能指标

        Args:
            iteration: 迭代次数
            fitness: 适应度
            diversity: 多样性
            coverage: 覆盖率
            latency: 延迟
        """
        # 收集系统指标
        memory_usage = None
        cpu_usage = None

        if self.system_collector:
            sys_metrics = self.system_collector.collect()
            memory_usage = sys_metrics.get('memory_usage')
            cpu_usage = sys_metrics.get('cpu_usage')

        # 创建指标对象
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            iteration=iteration,
            fitness=fitness,
            diversity=diversity,
            coverage=coverage,
            latency=latency,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )

        # 记录历史
        self.metrics_history.append(metrics)

        # 更新滑动窗口
        self.fitness_window.append(fitness)
        self.diversity_window.append(diversity)
        self.coverage_window.append(coverage)

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前指标"""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]

    def get_window_statistics(self) -> Dict[str, Any]:
        """获取滑动窗口统计"""
        if not self.fitness_window:
            return {}

        return {
            'fitness': {
                'mean': np.mean(self.fitness_window),
                'std': np.std(self.fitness_window),
                'min': np.min(self.fitness_window),
                'max': np.max(self.fitness_window),
            },
            'diversity': {
                'mean': np.mean(self.diversity_window),
                'std': np.std(self.diversity_window),
                'min': np.min(self.diversity_window),
                'max': np.max(self.diversity_window),
            },
            'coverage': {
                'mean': np.mean(self.coverage_window),
                'std': np.std(self.coverage_window),
                'min': np.min(self.coverage_window),
                'max': np.max(self.coverage_window),
            },
        }

    def detect_anomalies(self) -> List[str]:
        """
        检测异常

        Returns:
            异常消息列表
        """
        anomalies = []

        # 检查适应度停滞
        if len(self.fitness_window) >= self.fitness_stagnation_threshold:
            recent_fitness = list(self.fitness_window)[-self.fitness_stagnation_threshold:]
            improvement = max(recent_fitness) - min(recent_fitness)
            if improvement < 0.001:
                anomalies.append(
                    f"Fitness stagnation detected: {improvement:.6f} improvement "
                    f"over {self.fitness_stagnation_threshold} iterations"
                )

        # 检查多样性下降
        if len(self.diversity_window) > 1:
            diversity_drop = self.diversity_window[0] - self.diversity_window[-1]
            if diversity_drop > self.diversity_drop_threshold:
                anomalies.append(
                    f"Diversity drop detected: {diversity_drop:.4f} "
                    f"(threshold: {self.diversity_drop_threshold})"
                )

        return anomalies

    def get_convergence_analysis(self) -> Dict[str, Any]:
        """
        收敛性分析

        Returns:
            收敛性分析结果
        """
        if len(self.metrics_history) < 2:
            return {}

        fitness_values = [m.fitness for m in self.metrics_history]

        # 计算收敛率
        initial_fitness = fitness_values[0]
        final_fitness = fitness_values[-1]
        total_improvement = final_fitness - initial_fitness
        convergence_rate = total_improvement / len(fitness_values) if len(fitness_values) > 0 else 0

        # 计算收敛速度
        early_fitness = np.mean(fitness_values[:len(fitness_values)//4])
        late_fitness = np.mean(fitness_values[-len(fitness_values)//4:])
        convergence_speed = (late_fitness - early_fitness) / len(fitness_values) if len(fitness_values) > 0 else 0

        return {
            'initial_fitness': initial_fitness,
            'final_fitness': final_fitness,
            'total_improvement': total_improvement,
            'convergence_rate': convergence_rate,
            'convergence_speed': convergence_speed,
        }


class PerformanceAnalyzer:
    """
    性能分析器

    对收集的性能指标进行深度分析。

    核心特性:
    1. 统计分析
    2. 趋势分析
    3. 相关性分析
    4. 可视化
    """

    def __init__(self, metrics_history: List[PerformanceMetrics]):
        """
        初始化性能分析器

        Args:
            metrics_history: 指标历史
        """
        self.metrics_history = metrics_history

    def analyze(self) -> Dict[str, Any]:
        """
        全面分析性能

        Returns:
            分析结果
        """
        analysis = {
            'basic_statistics': self._basic_statistics(),
            'trend_analysis': self._trend_analysis(),
            'correlation_analysis': self._correlation_analysis(),
            'phase_analysis': self._phase_analysis(),
        }

        return analysis

    def _basic_statistics(self) -> Dict[str, Any]:
        """基础统计分析"""
        if not self.metrics_history:
            return {}

        fitness = [m.fitness for m in self.metrics_history]
        diversity = [m.diversity for m in self.metrics_history]
        coverage = [m.coverage for m in self.metrics_history]

        return {
            'fitness': {
                'mean': float(np.mean(fitness)),
                'std': float(np.std(fitness)),
                'min': float(np.min(fitness)),
                'max': float(np.max(fitness)),
                'median': float(np.median(fitness)),
            },
            'diversity': {
                'mean': float(np.mean(diversity)),
                'std': float(np.std(diversity)),
                'min': float(np.min(diversity)),
                'max': float(np.max(diversity)),
                'median': float(np.median(diversity)),
            },
            'coverage': {
                'mean': float(np.mean(coverage)),
                'std': float(np.std(coverage)),
                'min': float(np.min(coverage)),
                'max': float(np.max(coverage)),
                'median': float(np.median(coverage)),
            },
        }

    def _trend_analysis(self) -> Dict[str, Any]:
        """趋势分析"""
        if not self.metrics_history:
            return {}

        fitness = [m.fitness for m in self.metrics_history]
        diversity = [m.diversity for m in self.metrics_history]

        # 计算线性趋势
        iterations = np.arange(len(fitness))
        fitness_trend = np.polyfit(iterations, fitness, 1)[0]
        diversity_trend = np.polyfit(iterations, diversity, 1)[0]

        return {
            'fitness_trend': float(fitness_trend),
            'diversity_trend': float(diversity_trend),
            'fitness_improving': fitness_trend > 0,
            'diversity_maintained': diversity_trend >= 0,
        }

    def _correlation_analysis(self) -> Dict[str, Any]:
        """相关性分析"""
        if not self.metrics_history:
            return {}

        fitness = [m.fitness for m in self.metrics_history]
        diversity = [m.diversity for m in self.metrics_history]
        coverage = [m.coverage for m in self.metrics_history]

        # 计算相关系数
        fitness_diversity_corr = np.corrcoef(fitness, diversity)[0, 1]
        fitness_coverage_corr = np.corrcoef(fitness, coverage)[0, 1]
        diversity_coverage_corr = np.corrcoef(diversity, coverage)[0, 1]

        return {
            'fitness_diversity_correlation': float(fitness_diversity_corr),
            'fitness_coverage_correlation': float(fitness_coverage_corr),
            'diversity_coverage_correlation': float(diversity_coverage_corr),
        }

    def _phase_analysis(self) -> Dict[str, Any]:
        """阶段分析"""
        if not self.metrics_history:
            return {}

        n_total = len(self.metrics_history)

        # 分为三个阶段
        phase1 = self.metrics_history[:n_total//3]
        phase2 = self.metrics_history[n_total//3:2*n_total//3]
        phase3 = self.metrics_history[2*n_total//3:]

        def phase_stats(phase):
            fitness = [m.fitness for m in phase]
            return {
                'mean_fitness': float(np.mean(fitness)),
                'std_fitness': float(np.std(fitness)),
                'iterations': len(phase),
            }

        return {
            'phase1': phase_stats(phase1),
            'phase2': phase_stats(phase2),
            'phase3': phase_stats(phase3),
        }

    def visualize(self, save_path: Optional[str] = None):
        """
        可视化性能分析

        Args:
            save_path: 保存路径（可选）
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping visualization")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 适应度曲线
        iterations = [m.iteration for m in self.metrics_history]
        fitness = [m.fitness for m in self.metrics_history]
        axes[0, 0].plot(iterations, fitness, linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].set_title('Fitness Over Time')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 多样性曲线
        diversity = [m.diversity for m in self.metrics_history]
        axes[0, 1].plot(iterations, diversity, linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Diversity')
        axes[0, 1].set_title('Diversity Over Time')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 覆盖率曲线
        coverage = [m.coverage for m in self.metrics_history]
        axes[1, 0].plot(iterations, coverage, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Coverage')
        axes[1, 0].set_title('Coverage Over Time')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 散点图（Fitness vs Diversity）
        axes[1, 1].scatter(diversity, fitness, alpha=0.5)
        axes[1, 1].set_xlabel('Diversity')
        axes[1, 1].set_ylabel('Fitness')
        axes[1, 1].set_title('Fitness vs Diversity')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def export_to_csv(self, filepath: str):
        """
        导出到CSV文件

        Args:
            filepath: 文件路径
        """
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available, skipping CSV export")
            return

        data = [m.to_dict() for m in self.metrics_history]
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Performance metrics exported to {filepath}")


__all__ = [
    'PerformanceMetrics',
    'MetricCollector',
    'SystemMetricCollector',
    'PerformanceMonitor',
    'PerformanceAnalyzer',
]
