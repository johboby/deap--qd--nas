"""
实用工具模块 (Utilities)
日志系统、配置管理、数据管道等
"""

import os
import sys
import json
import logging
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import numpy as np


class ColorFormatter(logging.Formatter):
    """彩色日志格式化器"""

    COLORS = {
        'DEBUG': '\033[36m',  # 青色
        'INFO': '\033[32m',  # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',  # 红色
        'CRITICAL': '\033[35m',  # 紫色
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['INFO'])
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class LoggerManager:
    """
    日志管理器

    统一管理日志配置和输出。
    """

    def __init__(self,
                 name: str,
                 level: str = 'INFO',
                 log_file: Optional[str] = None,
                 use_color: bool = True):
        """
        初始化日志管理器

        Args:
            name: 日志器名称
            level: 日志级别
            log_file: 日志文件路径（可选）
            use_color: 是否使用彩色输出
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # 清除现有处理器
        self.logger.handlers = []

        # 创建格式化器
        if use_color and sys.stdout.isatty():
            formatter = ColorFormatter(
                fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 文件处理器
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """获取日志器"""
        return self.logger


@dataclass
class ExperimentConfig:
    """
    实验配置

    Args:
        name: 实验名称
        description: 实验描述
        seed: 随机种子
        output_dir: 输出目录
        tags: 标签
        metadata: 元数据
    """
    name: str
    description: str = ""
    seed: int = 42
    output_dir: str = './experiments'
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_output_path(self, subpath: str = '') -> str:
        """获取输出路径"""
        path = os.path.join(self.output_dir, self.name, subpath)
        os.makedirs(path, exist_ok=True)
        return path

    def save(self, filepath: str):
        """保存配置"""
        with open(filepath, 'w') as f:
            json.dump({
                'name': self.name,
                'description': self.description,
                'seed': self.seed,
                'tags': self.tags,
                'metadata': self.metadata,
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """加载配置"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class Timer:
    """
    计时器

    简单的计时和性能测量工具。
    """

    def __init__(self, name: str = "Timer"):
        """
        初始化计时器

        Args:
            name: 计时器名称
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_times = []

    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self

    def stop(self) -> float:
        """停止计时"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.elapsed_times.append(elapsed)
        self.start_time = None
        return elapsed

    def reset(self):
        """重置计时器"""
        self.start_time = None
        self.end_time = None
        self.elapsed_times = []

    @property
    def elapsed(self) -> Optional[float]:
        """已用时间"""
        if self.start_time is not None:
            return time.time() - self.start_time
        elif len(self.elapsed_times) > 0:
            return self.elapsed_times[-1]
        return None

    @property
    def mean_elapsed(self) -> float:
        """平均已用时间"""
        if not self.elapsed_times:
            return 0.0
        return np.mean(self.elapsed_times)

    @property
    def total_elapsed(self) -> float:
        """总已用时间"""
        return np.sum(self.elapsed_times)

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


class ProgressBar:
    """
    进度条

    显示进度条。
    """

    def __init__(self, total: int, desc: str = 'Progress'):
        """
        初始化进度条

        Args:
            total: 总任务数
            desc: 描述
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """
        更新进度

        Args:
            n: 增加的进度数
        """
        self.current = min(self.current + n, self.total)
        self._display()

    def _display(self):
        """显示进度条"""
        percent = self.current / self.total
        filled_length = int(50 * percent)
        bar = '█' * filled_length + '-' * (50 - filled_length)

        elapsed = time.time() - self.start_time
        eta = elapsed / percent * (1 - percent) if percent > 0 else 0

        sys.stdout.write(
            f'\r{self.desc}: |{bar}| {percent:.1%} '
            f'({self.current}/{self.total}) '
            f'[{elapsed:.1f}s, ETA: {eta:.1f}s]'
        )
        sys.stdout.flush()

    def close(self):
        """关闭进度条"""
        sys.stdout.write('\n')
        sys.stdout.flush()


class CheckpointManager:
    """
    检查点管理器

    管理训练和优化过程中的检查点保存和加载。
    """

    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        """
        初始化检查点管理器

        Args:
            save_dir: 保存目录
            max_checkpoints: 最大检查点数量
        """
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, data: Dict[str, Any], epoch: int, name: str = 'checkpoint'):
        """
        保存检查点

        Args:
            data: 检查点数据
            epoch: 当前epoch
            name: 检查点名称
        """
        filename = f"{name}_epoch_{epoch}.pth"
        filepath = os.path.join(self.save_dir, filename)

        import torch
        torch.save(data, filepath)

        # 管理检查点数量
        self._cleanup_old_checkpoints(name, epoch)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            filepath: 检查点文件路径

        Returns:
            检查点数据
        """
        import torch
        return torch.load(filepath)

    def list_checkpoints(self) -> List[str]:
        """列出所有检查点"""
        files = []
        for file in os.listdir(self.save_dir):
            if file.endswith('.pth'):
                files.append(os.path.join(self.save_dir, file))
        return sorted(files, reverse=True)

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点"""
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None

    def _cleanup_old_checkpoints(self, name: str, current_epoch: int):
        """清理旧检查点"""
        checkpoints = self.list_checkpoints()
        name_checkpoints = [c for c in checkpoints if name in os.path.basename(c)]

        if len(name_checkpoints) > self.max_checkpoints:
            # 删除最旧的检查点
            for old_checkpoint in name_checkpoints[self.max_checkpoints:]:
                os.remove(old_checkpoint)


class MetricsTracker:
    """
    指标跟踪器

    跟踪和记录各种指标。
    """

    def __init__(self, metrics_names: List[str]):
        """
        初始化指标跟踪器

        Args:
            metrics_names: 指标名称列表
        """
        self.metrics = {name: [] for name in metrics_names}
        self.steps = []

    def update(self, step: int, **kwargs):
        """
        更新指标

        Args:
            step: 当前步数
            **kwargs: 指标值
        """
        self.steps.append(step)
        for name, value in kwargs.items():
            if name in self.metrics:
                self.metrics[name].append(value)

    def get_metric(self, name: str) -> List[float]:
        """获取指定指标的历史"""
        return self.metrics.get(name, [])

    def get_latest(self, name: str) -> Optional[float]:
        """获取指标的最新值"""
        values = self.get_metric(name)
        return values[-1] if values else None

    def get_mean(self, name: str, last_n: Optional[int] = None) -> float:
        """获取指标的平均值"""
        values = self.get_metric(name)
        if last_n is not None:
            values = values[-last_n:]
        return np.mean(values) if values else 0.0

    def get_best(self, name: str, mode: str = 'max') -> Optional[float]:
        """
        获取指标的极值

        Args:
            name: 指标名称
            mode: 'max' 或 'min'
        """
        values = self.get_metric(name)
        if not values:
            return None

        if mode == 'max':
            return max(values)
        else:
            return min(values)

    def save_to_csv(self, filepath: str):
        """
        保存到CSV文件

        Args:
            filepath: 文件路径
        """
        try:
            import pandas as pd

            data = {'step': self.steps}
            data.update(self.metrics)

            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

        except ImportError:
            # 回退到CSV直接写入
            with open(filepath, 'w') as f:
                header = ['step'] + list(self.metrics.keys())
                f.write(','.join(header) + '\n')

                for i, step in enumerate(self.steps):
                    row = [str(step)]
                    for name in self.metrics.keys():
                        row.append(str(self.metrics[name][i]))
                    f.write(','.join(row) + '\n')


class ConfigManager:
    """
    配置管理器

    管理实验和NAS配置。
    """

    def __init__(self, config_dir: str = './configs'):
        """
        初始化配置管理器

        Args:
            config_dir: 配置目录
        """
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)

    def save_config(self, name: str, config: Dict[str, Any]):
        """
        保存配置

        Args:
            name: 配置名称
            config: 配置字典
        """
        filepath = os.path.join(self.config_dir, f'{name}.json')

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    def load_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        加载配置

        Args:
            name: 配置名称

        Returns:
            配置字典
        """
        filepath = os.path.join(self.config_dir, f'{name}.json')

        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            return json.load(f)

    def list_configs(self) -> List[str]:
        """列出所有配置"""
        configs = []
        for file in os.listdir(self.config_dir):
            if file.endswith('.json'):
                configs.append(file[:-5])  # 移除.json扩展名
        return configs


def set_random_seed(seed: int):
    """
    设置随机种子

    Args:
        seed: 随机种子
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger = logging.getLogger(__name__)
    logger.info(f"✅ 随机种子设置为: {seed}")


def count_parameters(model) -> int:
    """
    计算模型参数量

    Args:
        model: PyTorch模型

    Returns:
        参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    格式化时间

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_number(n: int) -> str:
    """
    格式化数字

    Args:
        n: 数字

    Returns:
        格式化的字符串
    """
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(n)


__all__ = [
    'ColorFormatter',
    'LoggerManager',
    'ExperimentConfig',
    'Timer',
    'ProgressBar',
    'CheckpointManager',
    'MetricsTracker',
    'ConfigManager',
    'set_random_seed',
    'count_parameters',
    'format_time',
    'format_number',
]
