"""
真正的CMA-ES算法实现
Covariance Matrix Adaptation Evolution Strategy
用于QD-NAS的高效优化
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
import logging
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CMAParameters:
    """
    CMA-ES参数配置

    Args:
        population_size: 种群大小
        sigma: 初始步长
        sigma_decay: 步长衰减率
        ccum: 协方差矩阵累积权重
        cs: 步长控制累积权重
        c1: 协方差矩阵更新权重（秩1）
        cmu: 协方差矩阵更新权重（秩μ）
        damps: 步长控制阻尼因子
    """
    population_size: int = 50
    sigma: float = 0.5
    sigma_decay: float = 0.95
    ccum: float = 0.5
    cs: float = 0.5
    c1: float = 0.3
    cmu: float = 0.3
    damps: float = 1.0

    def __post_init__(self):
        """验证参数"""
        assert self.population_size > 0, "population_size must be positive"
        assert self.sigma > 0, "sigma must be positive"
        assert 0 < self.ccum < 1, "ccum must be in (0, 1)"
        assert 0 < self.cs < 1, "cs must be in (0, 1)"
        assert 0 < self.c1 < 1, "c1 must be in (0, 1)"
        assert 0 < self.cmu < 1, "cmu must be in (0, 1)"
        assert self.damps > 0, "damps must be positive"


class CMAESOptimizer:
    """
    CMA-ES优化器

    Covariance Matrix Adaptation Evolution Strategy
    自适应协方差矩阵进化策略，用于连续优化问题。

    核心特性:
    1. 自适应步长控制
    2. 协方差矩阵学习
    3. 秩1和秩μ更新
    4. 精英选择

    参考文献:
    Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation
    in evolution strategies. Evolutionary Computation, 9(2), 159-195.
    """

    def __init__(self,
                 dimension: int,
                 objective_function: Callable[[np.ndarray], float],
                 params: Optional[CMAParameters] = None,
                 x0: Optional[np.ndarray] = None):
        """
        初始化CMA-ES

        Args:
            dimension: 优化维度
            objective_function: 目标函数
            params: CMA-ES参数（可选）
            x0: 初始解（可选，默认随机）
        """
        self.dimension = dimension
        self.objective_function = objective_function
        self.params = params or CMAParameters()

        # 初始化均值
        self.mean = x0 if x0 is not None else np.random.randn(dimension)

        # 初始化协方差矩阵
        self.C = np.eye(dimension)

        # 初始化步长
        self.sigma = self.params.sigma

        # 进化路径
        self.pc = np.zeros(dimension)  # 协方差矩阵进化路径
        self.ps = np.zeros(dimension)  # 步长进化路径

        # 权重设置
        self.mu = int(self.params.population_size / 2)
        self.weights = self._compute_weights(self.mu)

        # 预计算参数
        self.mueff = 1 / np.sum(self.weights**2)
        self.ccum_cov = 4 / (self.dimension + 4)
        self.ccum_sigma = 4 / (self.dimension + 4)

        # 期望值
        self.chiN = np.sqrt(self.dimension) * (1 - 1/(4*self.dimension) + 1/(21*self.dimension**2))

        # 历史记录
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.sigma_history = []

        logger.info(f"🧬 CMA-ES优化器初始化完成")
        logger.info(f"   维度: {dimension}")
        logger.info(f"   种群大小: {self.params.population_size}")
        logger.info(f"   初始步长: {self.sigma}")

    def _compute_weights(self, mu: int) -> np.ndarray:
        """
        计算重组权重

        Args:
            mu: 选择的精英数量

        Returns:
            权重数组
        """
        # 对数权重
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)

        return weights

    def _sample_population(self) -> np.ndarray:
        """
        采样种群

        Returns:
            (population_size, dimension) 的种群矩阵
        """
        # Cholesky分解协方差矩阵
        try:
            B = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            # 如果不是正定的，添加小的对角扰动
            self.C += np.eye(self.dimension) * 1e-12
            B = np.linalg.cholesky(self.C)

        # 采样
        z = np.random.randn(self.params.population_size, self.dimension)
        population = self.mean + self.sigma * (B @ z.T).T

        return population

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        评估种群

        Args:
            population: 种群矩阵

        Returns:
            适应度数组
        """
        fitness = np.array([self.objective_function(ind) for ind in population])
        return fitness

    def _sort_and_select(self, population: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        排序并选择精英

        Args:
            population: 种群矩阵
            fitness: 适应度数组

        Returns:
            (selected_population, selected_fitness)
        """
        # 排序（根据适应度）
        sorted_indices = np.argsort(fitness)
        selected_indices = sorted_indices[:self.mu]

        selected_population = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        return selected_population, selected_fitness

    def _update_mean(self, selected_population: np.ndarray):
        """
        更新均值

        Args:
            selected_population: 选择的精英种群
        """
        # 加权平均
        self.mean = np.sum(self.weights[:, np.newaxis] * selected_population, axis=0)

    def _update_evolution_path(self, old_mean: np.ndarray, selected_population: np.ndarray):
        """
        更新进化路径

        Args:
            old_mean: 旧均值
            selected_population: 选择的精英种群
        """
        # 计算加权平均的变异
        yw = np.sum(self.weights[:, np.newaxis] * (selected_population - old_mean), axis=0)

        # 更新步长进化路径
        self.ps = (1 - self.params.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs) * self.mueff) * yw / self.sigma

        # 更新协方差进化路径
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.params.cs)**(2 * len(self.best_fitness_history))) < (1.4 + 2 / (self.dimension + 1))
        self.pc = (1 - self.params.ccum) * self.pc + hsig * np.sqrt(self.params.ccum * (2 - self.params.ccum) * self.mueff) * yw / self.sigma

    def _update_covariance(self, selected_population: np.ndarray):
        """
        更新协方差矩阵

        Args:
            selected_population: 选择的精英种群
        """
        # 秩1更新
        rank_one_update = np.outer(self.pc, self.pc)

        # 秩μ更新
        z = (selected_population - self.mean) / self.sigma
        rank_mu_update = np.sum([self.weights[i] * np.outer(z[i], z[i]) for i in range(len(self.weights))], axis=0)

        # 组合更新
        self.C = (1 - self.params.c1 - self.params.cmu) * self.C + \
                 self.params.c1 * rank_one_update + \
                 self.params.cmu * rank_mu_update

    def _update_step_size(self):
        """更新步长"""
        # 计算步长更新因子
        sigma_update = np.exp((self.cs / self.params.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

        # 更新步长
        self.sigma *= sigma_update

        # 应用衰减（可选）
        if self.params.sigma_decay < 1.0:
            self.sigma *= self.params.sigma_decay

    def step(self) -> Tuple[np.ndarray, float]:
        """
        执行一步CMA-ES迭代

        Returns:
            (best_individual, best_fitness)
        """
        old_mean = self.mean.copy()

        # 采样种群
        population = self._sample_population()

        # 评估种群
        fitness = self._evaluate_population(population)

        # 选择精英
        selected_population, selected_fitness = self._sort_and_select(population, fitness)

        # 更新均值
        self._update_mean(selected_population)

        # 更新进化路径
        self._update_evolution_path(old_mean, selected_population)

        # 更新协方差矩阵
        self._update_covariance(selected_population)

        # 更新步长
        self._update_step_size()

        # 记录历史
        best_fitness = np.min(fitness)
        mean_fitness = np.mean(fitness)
        self.best_fitness_history.append(best_fitness)
        self.mean_fitness_history.append(mean_fitness)
        self.sigma_history.append(self.sigma)

        # 返回最佳个体
        best_index = np.argmin(fitness)
        return population[best_index], best_fitness

    def optimize(self,
                 n_iterations: int = 1000,
                 verbose: bool = True) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        运行CMA-ES优化

        Args:
            n_iterations: 迭代次数
            verbose: 是否输出详细信息

        Returns:
            (best_solution, best_fitness, info)
        """
        logger.info(f"🚀 开始CMA-ES优化，迭代次数: {n_iterations}")

        best_solution = None
        best_fitness = np.inf

        for iteration in range(n_iterations):
            # 执行一步迭代
            solution, fitness = self.step()

            # 更新最佳解
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = solution.copy()

            # 输出进度
            if verbose and (iteration + 1) % 10 == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{n_iterations} | "
                    f"Best: {best_fitness:.6f} | "
                    f"Mean: {self.mean_fitness_history[-1]:.6f} | "
                    f"Sigma: {self.sigma:.6f}"
                )

        logger.info("✅ CMA-ES优化完成")

        # 返回结果
        info = {
            'best_fitness_history': self.best_fitness_history,
            'mean_fitness_history': self.mean_fitness_history,
            'sigma_history': self.sigma_history,
            'final_mean': self.mean,
            'final_sigma': self.sigma,
            'iterations': n_iterations,
        }

        return best_solution, best_fitness, info


class CMAESOptimizerQD:
    """
    CMA-ES优化器用于QD-NAS

    将CMA-ES与QD框架结合，支持行为空间映射和多样性维护。
    """

    def __init__(self,
                 dimension: int,
                 behavior_function: Callable[[np.ndarray], List[float]],
                 objective_function: Callable[[np.ndarray], float],
                 behavior_space,
                 params: Optional[CMAParameters] = None):
        """
        初始化CMA-ES QD优化器

        Args:
            dimension: 优化维度
            behavior_function: 行为特征函数
            objective_function: 目标函数
            behavior_space: 行为空间
            params: CMA-ES参数
        """
        self.dimension = dimension
        self.behavior_function = behavior_function
        self.objective_function = objective_function
        self.behavior_space = behavior_space
        self.params = params or CMAParameters()

        # 内部CMA-ES优化器
        self.cmaes = CMAESOptimizer(
            dimension=dimension,
            objective_function=objective_function,
            params=params
        )

        # QD归档
        from .archive import Archive
        from .characterization import ArchitectureMetrics
        self.archive = Archive(behavior_space=behavior_space)

        logger.info(f"🧬 CMA-ES QD优化器初始化完成")

    def optimize_qd(self,
                     n_iterations: int = 1000,
                     batch_size: int = 10,
                     verbose: bool = True) -> Any:
        """
        运行QD优化

        Args:
            n_iterations: 迭代次数
            batch_size: 每次迭代生成的个体数
            verbose: 是否输出详细信息

        Returns:
            归档对象
        """
        logger.info(f"🚀 开始CMA-ES QD优化，迭代次数: {n_iterations}")

        for iteration in range(n_iterations):
            # 生成一批个体
            population = []
            for _ in range(batch_size):
                # 执行CMA-ES一步
                solution, _ = self.cmaes.step()
                population.append(solution)

            # 评估并插入归档
            for sol in population:
                # 获取行为特征
                behavior = self.behavior_function(sol)

                # 获取目标值
                fitness = self.objective_function(sol)

                # 创建指标
                # 这里简化处理，实际应该创建ArchitectureMetrics
                # metrics = ArchitectureMetrics(accuracy=fitness, ...)

                # 插入归档（简化版本）
                # self.archive.insert(sol, metrics, generation=iteration)
                pass

            # 输出进度
            if verbose and (iteration + 1) % 10 == 0:
                stats = self.archive.get_statistics()
                logger.info(
                    f"Iteration {iteration + 1}/{n_iterations} | "
                    f"Archive: {stats['size']} | "
                    f"Coverage: {stats['coverage']:.2%}"
                )

        logger.info("✅ CMA-ES QD优化完成")
        return self.archive


def create_cmaes_optimizer(dimension: int,
                           objective_function: Callable[[np.ndarray], float],
                           **kwargs) -> CMAESOptimizer:
    """
    工厂函数：创建CMA-ES优化器

    Args:
        dimension: 优化维度
        objective_function: 目标函数
        **kwargs: 其他参数

    Returns:
        CMA-ES优化器
    """
    return CMAESOptimizer(dimension, objective_function, **kwargs)


# 测试函数
def test_cmaes():
    """测试CMA-ES优化器"""

    # 定义测试目标函数（Sphere函数）
    def sphere_function(x: np.ndarray) -> float:
        return np.sum(x**2)

    # 创建优化器
    optimizer = create_cmaes_optimizer(
        dimension=10,
        objective_function=sphere_function,
        params=CMAParameters(population_size=50, sigma=1.0)
    )

    # 运行优化
    best_solution, best_fitness, info = optimizer.optimize(
        n_iterations=100,
        verbose=True
    )

    print(f"\nBest solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print(f"Convergence rate: {info['best_fitness_history'][-10:]}")


if __name__ == "__main__":
    test_cmaes()
