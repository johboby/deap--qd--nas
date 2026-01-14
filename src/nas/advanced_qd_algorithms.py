"""
高级QD算法 (Advanced QD Algorithms)
实现CVT-MAP-Elites和Diverse Quality算法
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from .behavior_space import BehaviorSpace
from .archive import Archive, ArchiveEntry
from .characterization import ArchitectureMetrics, BaseCharacterization


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CVTMAPElites:
    """
    CVT-MAP-Elites (Centroidal Voronoi Tessellation MAP-Elites)

    使用CVT划分行为空间，提供更均匀的解分布。

    核心特性:
    1. CVT划分行为空间
    2. 每个Voronoi单元保存最佳个体
    3. 更好的空间覆盖
    4. 更高效的多样性维护

    参考文献:
    Vassiliades, V., et al. (2020). Using centroidal voronoi tessellations to scale
    up the multidimensional archive of phenotypic elites algorithm. IEEE TEC.
    """

    def __init__(self,
                 behavior_space: BehaviorSpace,
                 characterizer: BaseCharacterization,
                 n_cells: int = 1000,
                 optimize_for: str = 'accuracy',
                 batch_size: int = 100):
        """
        初始化CVT-MAP-Elites

        Args:
            behavior_space: 行为空间定义
            characterizer: 特征提取器
            n_cells: CVT单元数量
            optimize_for: 优化目标
            batch_size: 批处理大小
        """
        self.behavior_space = behavior_space
        self.characterizer = characterizer
        self.n_cells = n_cells
        self.optimize_for = optimize_for
        self.batch_size = batch_size

        # CVT centroids
        self.centroids = None

        # 归档: {cell_index: ArchiveEntry}
        self.archive: Dict[int, ArchiveEntry] = {}

        # 性能跟踪
        self.best_fitness = -np.inf
        self.best_architecture = None

        # 统计信息
        self.total_insertions = 0
        self.total_rejections = 0

        # 初始化CVT
        self._initialize_cvt()

        logger.info(f"🗺️  CVT-MAP-Elites初始化完成")
        logger.info(f"   CVT单元数: {n_cells}")

    def _initialize_cvt(self):
        """初始化CVT centroids"""
        logger.info("🔄 初始化CVT centroids")

        # 生成随机采样点
        n_samples = self.n_cells * 100
        samples = []

        for _ in range(n_samples):
            # 生成随机行为向量
            behavior_vector = []
            for dim in self.behavior_space.dimensions:
                value = np.random.uniform(dim.min_val, dim.max_val)
                behavior_vector.append(value)

            samples.append(behavior_vector)

        samples = np.array(samples)

        # K-means聚类
        kmeans = KMeans(n_clusters=self.n_cells, random_state=42, n_init=10)
        self.centroids = kmeans.fit_predict(samples, sample_weight=None)
        self.centroids = kmeans.cluster_centers_

        logger.info(f"✅ CVT centroids初始化完成")

    def _get_cell_index(self, behavior_vector: List[float]) -> int:
        """
        获取行为向量对应的cell索引

        Args:
            behavior_vector: 行为特征向量

        Returns:
            CVT cell索引
        """
        # 计算到所有centroids的距离
        distances = cdist([behavior_vector], self.centroids, 'euclidean')
        cell_index = np.argmin(distances[0])

        return cell_index

    def _get_fitness(self, metrics: ArchitectureMetrics) -> float:
        """获取适应度值"""
        if self.optimize_for == 'accuracy':
            return metrics.accuracy
        elif self.optimize_for == 'latency':
            return -metrics.latency
        elif self.optimize_for == 'energy':
            return -metrics.energy
        else:
            return metrics.accuracy

    def insert(self,
              architecture: Any,
              metrics: ArchitectureMetrics,
              generation: int = 0) -> bool:
        """
        插入一个架构到归档

        Args:
            architecture: 架构表示
            metrics: 性能指标
            generation: 发现代数

        Returns:
            是否成功插入
        """
        # 获取行为特征
        behavior_vector = metrics.get_behavior_vector()

        # 获取cell索引
        cell_index = self._get_cell_index(behavior_vector)

        # 获取适应度
        fitness = self._get_fitness(metrics)

        # 检查是否应该插入
        should_insert = False

        if cell_index not in self.archive:
            # Cell为空，直接插入
            should_insert = True
        else:
            # Cell已有个体，比较性能
            current_fitness = self._get_fitness(self.archive[cell_index].metrics)
            if fitness > current_fitness:
                should_insert = True

        if should_insert:
            # 创建归档条目
            entry = ArchiveEntry(
                architecture=architecture,
                metrics=metrics,
                behavior_vector=behavior_vector,
                cell_key=(cell_index,),  # CVT用单个索引作为key
                generation=generation
            )

            # 插入归档
            self.archive[cell_index] = entry
            self.total_insertions += 1

            # 更新最佳个体
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_architecture = architecture

            return True
        else:
            self.total_rejections += 1
            return False

    def evolve(self,
              generate_function: Callable,
              mutate_function: Callable,
              n_iterations: int = 1000,
              verbose: bool = True) -> Archive:
        """
        运行CVT-MAP-Elites进化

        Args:
            generate_function: 生成函数
            mutate_function: 变异函数
            n_iterations: 迭代次数
            verbose: 是否输出详细信息

        Returns:
            归档对象
        """
        logger.info(f"🚀 开始CVT-MAP-Elites进化，迭代次数: {n_iterations}")

        for iteration in range(n_iterations):
            # 生成一批新个体
            batch = []
            for _ in range(self.batch_size):
                # 从归档中随机选择父代
                if self.archive and np.random.random() < 0.9:
                    parent = np.random.choice(list(self.archive.values()))
                    child = mutate_function(parent.architecture)
                else:
                    child = generate_function()

                batch.append(child)

            # 评估并插入归档
            inserted_count = 0
            for arch in batch:
                metrics = self.characterizer.characterize(arch)
                if self.insert(arch, metrics, generation=iteration):
                    inserted_count += 1

            # 输出进度
            if verbose and (iteration + 1) % 10 == 0:
                stats = self.get_statistics()
                logger.info(
                    f"Iteration {iteration + 1}/{n_iterations} | "
                    f"Archive: {stats['size']} | "
                    f"Coverage: {stats['coverage']:.2%} | "
                    f"Best: {stats['best_fitness']:.4f}"
                )

        logger.info("✅ CVT-MAP-Elites进化完成")

        # 返回归档（转换为标准Archive对象）
        return self._convert_to_archive()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.archive:
            return {
                'size': 0,
                'coverage': 0.0,
                'best_fitness': -np.inf,
            }

        entries = list(self.archive.values())

        # 计算多样性
        behavior_vectors = [e.behavior_vector for e in entries]
        vectors = np.array(behavior_vectors)

        diversity = 0.0
        if len(vectors) > 1:
            distances = cdist(vectors, vectors, 'euclidean')
            diversity = np.mean(distances[distances > 0])

        return {
            'size': len(self.archive),
            'coverage': len(self.archive) / self.n_cells,
            'diversity': float(diversity),
            'best_fitness': self.best_fitness,
            'total_insertions': self.total_insertions,
            'total_rejections': self.total_rejections,
        }

    def _convert_to_archive(self) -> Archive:
        """转换为标准Archive对象"""
        archive = Archive(
            behavior_space=self.behavior_space,
            optimize_for=self.optimize_for
        )

        for entry in self.archive.values():
            archive.insert(
                architecture=entry.architecture,
                metrics=entry.metrics,
                generation=entry.generation
            )

        return archive


class DiverseQualityArchive:
    """
    Diverse Quality Archive (DQ-Archive)

    同时考虑质量和多样性的归档方法。

    核心特性:
    1. 基于质量的排序
    2. 基于多样性的排序
    3. 平衡质量和多样性
    4. 自适应选择策略
    """

    def __init__(self,
                 behavior_space: BehaviorSpace,
                 optimize_for: str = 'accuracy',
                 max_size: int = 100,
                 diversity_weight: float = 0.5):
        """
        初始化DQ归档

        Args:
            behavior_space: 行为空间定义
            optimize_for: 优化目标
            max_size: 最大归档大小
            diversity_weight: 多样性权重 [0, 1]
        """
        self.behavior_space = behavior_space
        self.optimize_for = optimize_for
        self.max_size = max_size
        self.diversity_weight = diversity_weight

        # 归档列表
        self.entries: List[ArchiveEntry] = []

        # 质量和多样性分数缓存
        self._quality_scores = {}
        self._diversity_scores = {}

        logger.info(f"🎯 DQ归档初始化完成")
        logger.info(f"   最大大小: {max_size}")
        logger.info(f"   多样性权重: {diversity_weight}")

    def _compute_quality_score(self, metrics: ArchitectureMetrics) -> float:
        """计算质量分数"""
        if self.optimize_for == 'accuracy':
            return metrics.accuracy
        elif self.optimize_for == 'latency':
            # 归一化到[0, 1]
            return max(0, 1 - metrics.latency / 1000)
        elif self.optimize_for == 'energy':
            return max(0, 1 - metrics.energy / 1000)
        else:
            return metrics.accuracy

    def _compute_diversity_score(self, behavior_vector: List[float]) -> float:
        """计算多样性分数"""
        if not self.entries:
            return 1.0

        # 计算到最近邻居的距离
        distances = []
        for entry in self.entries:
            dist = self.behavior_space.distance(behavior_vector, entry.behavior_vector)
            distances.append(dist)

        # 使用到最近邻居的距离作为多样性分数
        min_distance = min(distances)
        return min(min_distance, 1.0)

    def insert(self,
              architecture: Any,
              metrics: ArchitectureMetrics,
              generation: int = 0) -> bool:
        """
        插入一个架构到归档

        Args:
            architecture: 架构表示
            metrics: 性能指标
            generation: 发现代数

        Returns:
            是否成功插入
        """
        # 获取行为特征
        behavior_vector = metrics.get_behavior_vector()

        # 计算质量分数
        quality_score = self._compute_quality_score(metrics)

        # 计算多样性分数
        diversity_score = self._compute_diversity_score(behavior_vector)

        # 综合分数
        combined_score = (1 - self.diversity_weight) * quality_score + \
                        self.diversity_weight * diversity_score

        # 创建归档条目
        entry = ArchiveEntry(
            architecture=architecture,
            metrics=metrics,
            behavior_vector=behavior_vector,
            cell_key=(),  # DQ不使用网格
            generation=generation
        )

        # 如果归档未满，直接插入
        if len(self.entries) < self.max_size:
            self.entries.append(entry)
            self._quality_scores[entry] = quality_score
            self._diversity_scores[entry] = diversity_score
            return True

        # 如果归档已满，替换综合分数最低的
        min_combined_score = float('inf')
        min_entry = None

        for e in self.entries:
            q_score = self._quality_scores[e]
            d_score = self._diversity_scores[e]
            combined = (1 - self.diversity_weight) * q_score + \
                       self.diversity_weight * d_score

            if combined < min_combined_score:
                min_combined_score = combined
                min_entry = e

        if combined_score > min_combined_score:
            # 替换
            self.entries.remove(min_entry)
            del self._quality_scores[min_entry]
            del self._diversity_scores[min_entry]

            self.entries.append(entry)
            self._quality_scores[entry] = quality_score
            self._diversity_scores[entry] = diversity_score
            return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.entries:
            return {
                'size': 0,
                'mean_quality': 0.0,
                'mean_diversity': 0.0,
            }

        mean_quality = np.mean(list(self._quality_scores.values()))
        mean_diversity = np.mean(list(self._diversity_scores.values()))

        return {
            'size': len(self.entries),
            'mean_quality': float(mean_quality),
            'mean_diversity': float(mean_diversity),
        }

    def get_entries(self) -> List[ArchiveEntry]:
        """获取所有条目"""
        return self.entries.copy()


def create_cvt_map_elites(behavior_space: BehaviorSpace,
                         characterizer: BaseCharacterization,
                         n_cells: int = 1000,
                         **kwargs) -> CVTMAPElites:
    """
    工厂函数：创建CVT-MAP-Elites优化器

    Args:
        behavior_space: 行为空间定义
        characterizer: 特征提取器
        n_cells: CVT单元数量
        **kwargs: 其他参数

    Returns:
        CVT-MAP-Elites优化器
    """
    return CVTMAPElites(
        behavior_space=behavior_space,
        characterizer=characterizer,
        n_cells=n_cells,
        **kwargs
    )


def create_dq_archive(behavior_space: BehaviorSpace,
                     max_size: int = 100,
                     diversity_weight: float = 0.5,
                     **kwargs) -> DiverseQualityArchive:
    """
    工厂函数：创建DQ归档

    Args:
        behavior_space: 行为空间定义
        max_size: 最大归档大小
        diversity_weight: 多样性权重
        **kwargs: 其他参数

    Returns:
        DQ归档
    """
    return DiverseQualityArchive(
        behavior_space=behavior_space,
        max_size=max_size,
        diversity_weight=diversity_weight,
        **kwargs
    )


__all__ = [
    'CVTMAPElites',
    'DiverseQualityArchive',
    'create_cvt_map_elites',
    'create_dq_archive',
]
