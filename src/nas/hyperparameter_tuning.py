"""
超参数调优模块 (Hyperparameter Tuning)
实现自适应参数调整和贝叶斯优化
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Hyperparameter:
    """
    超参数定义

    Args:
        name: 参数名称
        type: 参数类型 ('continuous', 'discrete', 'categorical')
        min_val: 最小值（连续/离散）
        max_val: 最大值（连续/离散）
        choices: 可选值（类别型）
        default: 默认值
    """
    name: str
    type: str  # 'continuous', 'discrete', 'categorical'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None

    def __post_init__(self):
        """验证参数"""
        if self.type not in ['continuous', 'discrete', 'categorical']:
            raise ValueError(f"Invalid type: {self.type}")

        if self.type == 'categorical':
            if self.choices is None:
                raise ValueError("Categorical parameter must have choices")
        elif self.type in ['continuous', 'discrete']:
            if self.min_val is None or self.max_val is None:
                raise ValueError("Continuous/discrete parameter must have min_val and max_val")

    def sample(self) -> Any:
        """采样一个值"""
        if self.type == 'continuous':
            return np.random.uniform(self.min_val, self.max_val)
        elif self.type == 'discrete':
            return np.random.randint(int(self.min_val), int(self.max_val) + 1)
        elif self.type == 'categorical':
            return np.random.choice(self.choices)
        else:
            raise ValueError(f"Unknown type: {self.type}")


class BaseHyperparameterOptimizer(ABC):
    """
    超参数优化器基类
    """

    def __init__(self,
                 hyperparameters: List[Hyperparameter],
                 objective_function: Callable[[Dict[str, Any]], float]):
        """
        初始化超参数优化器

        Args:
            hyperparameters: 超参数列表
            objective_function: 目标函数
        """
        self.hyperparameters = hyperparameters
        self.objective_function = objective_function

        # 优化历史
        self.history = []

    @abstractmethod
    def optimize(self, n_iterations: int = 100) -> Tuple[Dict[str, Any], float]:
        """
        优化超参数

        Args:
            n_iterations: 迭代次数

        Returns:
            (best_params, best_score)
        """
        pass


class RandomSearchOptimizer(BaseHyperparameterOptimizer):
    """
    随机搜索优化器

    简单的随机搜索基线方法。
    """

    def optimize(self, n_iterations: int = 100) -> Tuple[Dict[str, Any], float]:
        """
        优化超参数

        Args:
            n_iterations: 迭代次数

        Returns:
            (best_params, best_score)
        """
        logger.info(f"🎲 开始随机搜索优化，迭代次数: {n_iterations}")

        best_params = None
        best_score = -np.inf

        for iteration in range(n_iterations):
            # 采样参数
            params = {hp.name: hp.sample() for hp in self.hyperparameters}

            # 评估
            score = self.objective_function(params)
            self.history.append((params, score))

            # 更新最佳
            if score > best_score:
                best_score = score
                best_params = params

            # 输出进度
            if (iteration + 1) % 10 == 0:
                logger.info(f"Iteration {iteration + 1}/{n_iterations} | Best: {best_score:.6f}")

        logger.info("✅ 随机搜索优化完成")
        return best_params, best_score


class AdaptiveParameterTuner:
    """
    自适应参数调整器

    基于优化过程中的反馈自动调整参数。

    核心特性:
    1. 多样性自适应变异率
    2. 性能自适应选择压力
    3. 学习率自适应调整
    4. 批次大小自适应
    """

    def __init__(self,
                 initial_params: Dict[str, float],
                 adaptation_rate: float = 0.1):
        """
        初始化自适应参数调整器

        Args:
            initial_params: 初始参数
            adaptation_rate: 调整速率
        """
        self.params = initial_params.copy()
        self.adaptation_rate = adaptation_rate

        # 历史记录
        self.score_history = []
        self.param_history = []

        # 统计信息
        self.improvement_count = 0
        self.stagnation_count = 0

        logger.info(f"⚙️  自适应参数调整器初始化完成")
        logger.info(f"   初始参数: {initial_params}")

    def adapt_mutation_rate(self, diversity: float, target_diversity: float = 0.5) -> float:
        """
        自适应调整变异率

        Args:
            diversity: 当前多样性
            target_diversity: 目标多样性

        Returns:
            调整后的变异率
        """
        current_rate = self.params.get('mutation_rate', 0.1)

        # 如果多样性低，增加变异率
        if diversity < target_diversity:
            new_rate = min(current_rate * (1 + self.adaptation_rate), 0.5)
        else:
            new_rate = max(current_rate * (1 - self.adaptation_rate), 0.01)

        self.params['mutation_rate'] = new_rate
        return new_rate

    def adapt_selection_pressure(self, convergence_rate: float) -> float:
        """
        自适应调整选择压力

        Args:
            convergence_rate: 收敛速率

        Returns:
            调整后的选择压力
        """
        current_pressure = self.params.get('selection_pressure', 0.7)

        # 如果收敛太快，降低选择压力
        if convergence_rate > 0.9:
            new_pressure = max(current_pressure * (1 - self.adaptation_rate), 0.5)
        else:
            new_pressure = min(current_pressure * (1 + self.adaptation_rate), 0.95)

        self.params['selection_pressure'] = new_pressure
        return new_pressure

    def adapt_learning_rate(self, score_improvement: float) -> float:
        """
        自适应调整学习率

        Args:
            score_improvement: 分数改善量

        Returns:
            调整后的学习率
        """
        current_lr = self.params.get('learning_rate', 0.01)

        # 如果改善小，降低学习率
        if score_improvement < 0.001:
            new_lr = max(current_lr * 0.9, 0.0001)
        else:
            new_lr = min(current_lr * 1.1, 0.1)

        self.params['learning_rate'] = new_lr
        return new_lr

    def update(self, score: float, diversity: float = None, **metrics):
        """
        更新参数

        Args:
            score: 当前分数
            diversity: 多样性
            **metrics: 其他指标
        """
        self.score_history.append(score)
        self.param_history.append(self.params.copy())

        # 计算改善
        if len(self.score_history) > 1:
            improvement = score - self.score_history[-2]

            if improvement > 0:
                self.improvement_count += 1
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            # 自适应调整学习率
            self.adapt_learning_rate(improvement)

        # 自适应调整变异率
        if diversity is not None:
            self.adapt_mutation_rate(diversity)

    def get_params(self) -> Dict[str, float]:
        """获取当前参数"""
        return self.params.copy()


class BayesianOptimizer(BaseHyperparameterOptimizer):
    """
    贝叶斯优化器

    使用高斯过程代理模型进行超参数优化。

    核心特性:
    1. 高斯过程代理模型
    2. 采集函数优化（EI、UCB）
    3. 高效的全局搜索
    4. 样本效率高

    参考文献:
    Brochu, E., et al. (2010). A tutorial on Bayesian optimization of
    expensive cost functions, with application to active user modeling
    and hierarchical reinforcement learning.
    """

    def __init__(self,
                 hyperparameters: List[Hyperparameter],
                 objective_function: Callable[[Dict[str, Any]], float],
                 acquisition: str = 'ei',
                 n_warmup: int = 10000,
                 n_iter: int = 10):
        """
        初始化贝叶斯优化器

        Args:
            hyperparameters: 超参数列表
            objective_function: 目标函数
            acquisition: 采集函数 ('ei', 'ucb', 'pi')
            n_warmup: 随机热身采样数
            n_iter: 采集函数优化迭代数
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Bayesian optimization")

        super().__init__(hyperparameters, objective_function)

        self.acquisition = acquisition
        self.n_warmup = n_warmup
        self.n_iter = n_iter

        # 高斯过程模型
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

        # 评估历史
        self.X = []
        self.y = []

        logger.info(f"🎯 贝叶斯优化器初始化完成")
        logger.info(f"   采集函数: {acquisition}")

    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """将参数字典转换为向量"""
        vector = []
        for hp in self.hyperparameters:
            value = params[hp.name]
            if hp.type == 'categorical':
                # One-hot编码
                one_hot = [1.0 if v == value else 0.0 for v in hp.choices]
                vector.extend(one_hot)
            else:
                # 归一化到[0, 1]
                normalized = (value - hp.min_val) / (hp.max_val - hp.min_val)
                vector.append(normalized)

        return np.array(vector)

    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, Any]:
        """将向量转换为参数字典"""
        params = {}
        idx = 0

        for hp in self.hyperparameters:
            if hp.type == 'categorical':
                # 从one-hot解码
                one_hot = vector[idx:idx + len(hp.choices)]
                choice_idx = np.argmax(one_hot)
                params[hp.name] = hp.choices[choice_idx]
                idx += len(hp.choices)
            else:
                # 从归一化值解码
                normalized = vector[idx]
                value = hp.min_val + normalized * (hp.max_val - hp.min_val)

                # 离散化（如果需要）
                if hp.type == 'discrete':
                    value = int(round(value))

                params[hp.name] = value
                idx += 1

        return params

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """计算采集函数值"""
        # 预测均值和标准差
        y_mean, y_std = self.gp.predict(X, return_std=True)

        if self.acquisition == 'ei':
            # Expected Improvement
            y_best = np.max(self.y) if self.y else 0
            z = (y_mean - y_best) / (y_std + 1e-9)
            ei = (y_mean - y_best) * norm.cdf(z) + y_std * norm.pdf(z)
            return ei
        elif self.acquisition == 'ucb':
            # Upper Confidence Bound
            kappa = 2.576  # 99% confidence
            ucb = y_mean + kappa * y_std
            return ucb
        elif self.acquisition == 'pi':
            # Probability of Improvement
            y_best = np.max(self.y) if self.y else 0
            z = (y_mean - y_best - 0.01) / (y_std + 1e-9)
            pi = norm.cdf(z)
            return pi
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition}")

    def optimize(self, n_iterations: int = 100) -> Tuple[Dict[str, Any], float]:
        """
        优化超参数

        Args:
            n_iterations: 迭代次数

        Returns:
            (best_params, best_score)
        """
        logger.info(f"🎯 开始贝叶斯优化，迭代次数: {n_iterations}")

        best_params = None
        best_score = -np.inf

        for iteration in range(n_iterations):
            # 随机热身或建议采样
            if len(self.y) < 5:  # 前5次随机采样
                params = {hp.name: hp.sample() for hp in self.hyperparameters}
            else:
                # 拟合GP
                X_array = np.array(self.X)
                y_array = np.array(self.y)
                self.gp.fit(X_array, y_array)

                # 优化采集函数
                def objective(x):
                    return -self._acquisition_function(x.reshape(1, -1))[0]

                bounds = [(0, 1)] * len(self._params_to_vector(
                    {hp.name: hp.default for hp in self.hyperparameters}
                ))

                # 简单随机搜索（实际应该用更复杂的优化器）
                best_x = None
                best_acq = -np.inf

                for _ in range(self.n_warmup):
                    x = np.random.rand(len(bounds))
                    acq_value = self._acquisition_function(x.reshape(1, -1))[0]

                    if acq_value > best_acq:
                        best_acq = acq_value
                        best_x = x

                # 转换为参数
                params = self._vector_to_params(best_x)

            # 评估
            score = self.objective_function(params)

            # 记录
            x_vector = self._params_to_vector(params)
            self.X.append(x_vector)
            self.y.append(score)
            self.history.append((params, score))

            # 更新最佳
            if score > best_score:
                best_score = score
                best_params = params

            # 输出进度
            if (iteration + 1) % 10 == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{n_iterations} | "
                    f"Best: {best_score:.6f} | "
                    f"Current: {score:.6f}"
                )

        logger.info("✅ 贝叶斯优化完成")
        return best_params, best_score


def create_hyperparameter_optimizer(hyperparameters: List[Hyperparameter],
                                  objective_function: Callable[[Dict[str, Any]], float],
                                  method: str = 'bayesian',
                                  **kwargs) -> BaseHyperparameterOptimizer:
    """
    工厂函数：创建超参数优化器

    Args:
        hyperparameters: 超参数列表
        objective_function: 目标函数
        method: 优化方法 ('random', 'bayesian')
        **kwargs: 其他参数

    Returns:
        超参数优化器
    """
    if method == 'random':
        return RandomSearchOptimizer(hyperparameters, objective_function)
    elif method == 'bayesian':
        return BayesianOptimizer(hyperparameters, objective_function, **kwargs)
    else:
        raise ValueError(f"Unknown optimization method: {method}")


__all__ = [
    'Hyperparameter',
    'BaseHyperparameterOptimizer',
    'RandomSearchOptimizer',
    'AdaptiveParameterTuner',
    'BayesianOptimizer',
    'create_hyperparameter_optimizer',
]
