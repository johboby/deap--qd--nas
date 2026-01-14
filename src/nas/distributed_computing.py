"""
分布式计算模块 (Distributed Computing)
支持多进程评估和GPU加速
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from abc import ABC, abstractmethod
import logging
import time
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ray import serve, remote, init as ray_init, get_actor
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """
    工作进程配置

    Args:
        n_workers: 工作进程数
        use_gpu: 是否使用GPU
        gpu_ids: GPU ID列表
        use_ray: 是否使用Ray
        max_tasks_per_worker: 每个worker的最大任务数
    """
    n_workers: int = None  # None表示使用所有CPU核心
    use_gpu: bool = False
    gpu_ids: List[int] = None
    use_ray: bool = False
    max_tasks_per_worker: int = 10

    def __post_init__(self):
        """初始化后处理"""
        if self.n_workers is None:
            self.n_workers = mp.cpu_count()

        if self.use_gpu and not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, GPU acceleration disabled")
            self.use_gpu = False

        if self.use_ray and not RAY_AVAILABLE:
            logger.warning("Ray not available, falling back to multiprocessing")
            self.use_ray = False


class BaseEvaluator(ABC):
    """
    评估器基类
    """

    @abstractmethod
    def evaluate(self, items: List[Any], **kwargs) -> List[Any]:
        """
        评估一批项目

        Args:
            items: 待评估的项目列表
            **kwargs: 其他参数

        Returns:
            评估结果列表
        """
        pass


class SerialEvaluator(BaseEvaluator):
    """
    串行评估器

    单进程顺序评估，适用于小规模任务。
    """

    def __init__(self, evaluate_function: Callable[[Any], Any]):
        """
        初始化串行评估器

        Args:
            evaluate_function: 评估函数
        """
        self.evaluate_function = evaluate_function

    def evaluate(self, items: List[Any], **kwargs) -> List[Any]:
        """
        评估一批项目

        Args:
            items: 待评估的项目列表
            **kwargs: 其他参数

        Returns:
            评估结果列表
        """
        logger.info(f"🔄 串行评估 {len(items)} 个项目")

        results = []
        for i, item in enumerate(items):
            result = self.evaluate_function(item)
            results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(f"  进度: {i + 1}/{len(items)}")

        logger.info(f"✅ 串行评估完成")
        return results


class MultiProcessEvaluator(BaseEvaluator):
    """
    多进程评估器

    使用ProcessPoolExecutor进行并行评估。
    """

    def __init__(self,
                 evaluate_function: Callable[[Any], Any],
                 config: WorkerConfig):
        """
        初始化多进程评估器

        Args:
            evaluate_function: 评估函数
            config: 工作配置
        """
        self.evaluate_function = evaluate_function
        self.config = config

        logger.info(f"⚡ 初始化多进程评估器")
        logger.info(f"   工作进程数: {config.n_workers}")
        logger.info(f"   GPU加速: {config.use_gpu}")

    def evaluate(self, items: List[Any], **kwargs) -> List[Any]:
        """
        评估一批项目

        Args:
            items: 待评估的项目列表
            **kwargs: 其他参数

        Returns:
            评估结果列表
        """
        logger.info(f"⚡ 并行评估 {len(items)} 个项目（{self.config.n_workers} 个进程）")

        start_time = time.time()

        # 创建结果列表（保持顺序）
        results = [None] * len(items)

        # 创建ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            # 提交所有任务
            future_to_index = {}
            for i, item in enumerate(items):
                future = executor.submit(self._worker, item, self.config)
                future_to_index[future] = i

            # 收集结果
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    completed += 1

                    if completed % max(1, len(items) // 10) == 0:
                        logger.info(f"  进度: {completed}/{len(items)}")

                except Exception as e:
                    logger.error(f"  任务 {index} 失败: {e}")
                    results[index] = None

        elapsed = time.time() - start_time
        logger.info(f"✅ 并行评估完成，耗时: {elapsed:.2f}s")

        return results

    def _worker(self, item: Any, config: WorkerConfig) -> Any:
        """
        工作进程函数

        Args:
            item: 待评估的项目
            config: 工作配置

        Returns:
            评估结果
        """
        # 设置GPU
        if config.use_gpu and config.gpu_ids:
            import os
            gpu_id = os.getpid() % len(config.gpu_ids)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_ids[gpu_id])

        # 执行评估
        return self.evaluate_function(item)


class GPUAcceleratedEvaluator(BaseEvaluator):
    """
    GPU加速评估器

    使用GPU进行加速评估。
    """

    def __init__(self,
                 evaluate_function: Callable[[Any, str], Any],
                 config: WorkerConfig):
        """
        初始化GPU加速评估器

        Args:
            evaluate_function: 评估函数（需要device参数）
            config: 工作配置
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GPU acceleration")

        self.evaluate_function = evaluate_function
        self.config = config

        # 检查GPU可用性
        if torch.cuda.is_available():
            self.n_gpus = torch.cuda.device_count()
            logger.info(f"🎮 检测到 {self.n_gpus} 个GPU")
        else:
            logger.warning("未检测到CUDA，GPU加速不可用")
            self.n_gpus = 0

        # 分配GPU
        if config.use_gpu and config.gpu_ids:
            self.allocated_gpus = config.gpu_ids
        elif config.use_gpu and self.n_gpus > 0:
            self.allocated_gpus = list(range(self.n_gpus))
        else:
            self.allocated_gpus = []

    def evaluate(self, items: List[Any], **kwargs) -> List[Any]:
        """
        评估一批项目（使用GPU）

        Args:
            items: 待评估的项目列表
            **kwargs: 其他参数

        Returns:
            评估结果列表
        """
        if not self.allocated_gpus:
            logger.warning("没有可用的GPU，回退到CPU评估")
            return self._evaluate_cpu(items)

        logger.info(f"🎮 GPU评估 {len(items)} 个项目（{len(self.allocated_gpus)} 个GPU）")

        start_time = time.time()

        # 分配任务到GPU
        results = self._evaluate_on_gpus(items)

        elapsed = time.time() - start_time
        logger.info(f"✅ GPU评估完成，耗时: {elapsed:.2f}s")

        return results

    def _evaluate_cpu(self, items: List[Any]) -> List[Any]:
        """CPU评估"""
        results = []
        for item in items:
            result = self.evaluate_function(item, 'cpu')
            results.append(result)
        return results

    def _evaluate_on_gpus(self, items: List[Any]) -> List[Any]:
        """在GPU上评估"""
        n_gpus = len(self.allocated_gpus)

        # 为每个GPU创建线程池
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = []

            for i, item in enumerate(items):
                # 分配到GPU
                gpu_id = self.allocated_gpus[i % n_gpus]
                device = f'cuda:{gpu_id}'

                future = executor.submit(self.evaluate_function, item, device)
                futures.append((future, i))

            # 收集结果
            results = [None] * len(items)
            completed = 0

            for future, index in futures:
                try:
                    result = future.result()
                    results[index] = result
                    completed += 1

                    if completed % max(1, len(items) // 10) == 0:
                        logger.info(f"  进度: {completed}/{len(items)}")

                except Exception as e:
                    logger.error(f"  GPU任务 {index} 失败: {e}")
                    results[index] = None

        return results


class DistributedNASOptimizer:
    """
    分布式NAS优化器

    支持多进程和GPU加速的NAS优化。
    """

    def __init__(self,
                 optimizer: Any,  # QDNASOptimizer或其他优化器
                 evaluator: BaseEvaluator,
                 batch_size: int = 100):
        """
        初始化分布式NAS优化器

        Args:
            optimizer: 基础优化器
            evaluator: 评估器
            batch_size: 批处理大小
        """
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.batch_size = batch_size

        logger.info(f"🚀 分布式NAS优化器初始化完成")
        logger.info(f"   批处理大小: {batch_size}")

    def optimize_distributed(self,
                           n_iterations: int = 1000,
                           verbose: bool = True) -> Tuple[Any, List]:
        """
        分布式优化

        Args:
            n_iterations: 迭代次数
            verbose: 是否输出详细信息

        Returns:
            (archive, pareto_front)
        """
        logger.info(f"🚀 开始分布式优化，迭代次数: {n_iterations}")

        # 初始化
        self.optimizer.initialize()

        for iteration in range(n_iterations):
            # 生成一批候选架构
            candidates = self._generate_candidates(self.batch_size)

            # 分布式评估
            metrics_list = self.evaluator.evaluate(candidates)

            # 处理评估结果
            self._process_evaluation_results(candidates, metrics_list, iteration)

            # 输出进度
            if verbose and (iteration + 1) % 10 == 0:
                stats = self.optimizer.get_statistics()
                logger.info(
                    f"Iteration {iteration + 1}/{n_iterations} | "
                    f"Archive: {stats['size']} | "
                    f"Coverage: {stats['coverage']:.2%} | "
                    f"Best: {stats['best_fitness']:.4f}"
                )

        logger.info("✅ 分布式优化完成")

        # 返回结果
        archive = self.optimizer.get_archive()
        pareto_front = self.optimizer.get_pareto_front()

        return archive, pareto_front

    def _generate_candidates(self, batch_size: int) -> List[Any]:
        """生成候选架构"""
        candidates = []
        for _ in range(batch_size):
            if hasattr(self.optimizer, 'search_space'):
                candidate = self.optimizer.search_space.random_sample()
            else:
                candidate = self.optimizer.search_space.random_sample()

            candidates.append(candidate)

        return candidates

    def _process_evaluation_results(self,
                                     candidates: List[Any],
                                     metrics_list: List[Any],
                                     generation: int):
        """处理评估结果"""
        for candidate, metrics in zip(candidates, metrics_list):
            if metrics is not None:
                if hasattr(self.optimizer, 'archive'):
                    self.optimizer.archive.insert(
                        architecture=candidate,
                        metrics=metrics,
                        generation=generation
                    )
                elif hasattr(self.optimizer, 'optimizer') and hasattr(self.optimizer.optimizer, 'archive'):
                    self.optimizer.optimizer.archive.insert(
                        architecture=candidate,
                        metrics=metrics,
                        generation=generation
                    )


class BatchProcessor:
    """
    批处理器

    高效处理大批量任务。
    """

    def __init__(self,
                 process_function: Callable[[List[Any]], List[Any]],
                 batch_size: int = 100,
                 n_workers: int = None):
        """
        初始化批处理器

        Args:
            process_function: 处理函数
            batch_size: 批处理大小
            n_workers: 工作进程数
        """
        self.process_function = process_function
        self.batch_size = batch_size
        self.n_workers = n_workers or mp.cpu_count()

    def process(self, items: List[Any]) -> List[Any]:
        """
        处理一批项目

        Args:
            items: 待处理的项目列表

        Returns:
            处理结果列表
        """
        logger.info(f"📦 处理 {len(items)} 个项目（批大小: {self.batch_size}）")

        results = []
        batches = self._create_batches(items, self.batch_size)

        for i, batch in enumerate(batches):
            batch_result = self.process_function(batch)
            results.extend(batch_result)

            if (i + 1) % 10 == 0:
                logger.info(f"  批次: {i + 1}/{len(batches)}")

        logger.info(f"✅ 批处理完成")
        return results

    def _create_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """创建批次"""
        n_batches = (len(items) + batch_size - 1) // batch_size
        return [items[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]


def create_evaluator(evaluate_function: Callable,
                     use_multiprocessing: bool = False,
                     use_gpu: bool = False,
                     n_workers: Optional[int] = None) -> BaseEvaluator:
    """
    工厂函数：创建评估器

    Args:
        evaluate_function: 评估函数
        use_multiprocessing: 是否使用多进程
        use_gpu: 是否使用GPU
        n_workers: 工作进程数

    Returns:
        评估器
    """
    config = WorkerConfig(
        n_workers=n_workers,
        use_gpu=use_gpu
    )

    if use_gpu and TORCH_AVAILABLE:
        return GPUAcceleratedEvaluator(evaluate_function, config)
    elif use_multiprocessing:
        return MultiProcessEvaluator(evaluate_function, config)
    else:
        return SerialEvaluator(evaluate_function)


__all__ = [
    'WorkerConfig',
    'BaseEvaluator',
    'SerialEvaluator',
    'MultiProcessEvaluator',
    'GPUAcceleratedEvaluator',
    'DistributedNASOptimizer',
    'BatchProcessor',
    'create_evaluator',
]
