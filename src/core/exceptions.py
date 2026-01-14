"""
DEAP多目标优化框架 - 自定义异常类
提供具体的异常类型，改进异常处理
"""

from typing import Optional, Any, List


# ==================== 基础异常类 ====================

class DEAPError(Exception):
    """DEAP框架基础异常类"""

    def __init__(self, message: str, error_code: Optional[int] = None):
        """
        初始化异常

        Args:
            message: 错误消息
            error_code: 错误代码（可选）
        """
        super().__init__(message)
        self.error_code = error_code

    def __str__(self) -> str:
        if self.error_code is not None:
            return f"[Error {self.error_code}] {super().__str__()}"
        return super().__str__()


# ==================== 算法异常 ====================

class AlgorithmError(DEAPError):
    """算法相关异常"""
    pass


class AlgorithmNotInitializedError(AlgorithmError):
    """算法未初始化异常"""

    def __init__(self, algorithm_name: str):
        super().__init__(
            f"Algorithm '{algorithm_name}' is not initialized. Call initialize() first.",
            error_code=1001
        )
        self.algorithm_name = algorithm_name


class AlgorithmNotImplementedError(AlgorithmError):
    """算法未实现异常"""

    def __init__(self, algorithm_name: str, method_name: Optional[str] = None):
        if method_name:
            message = f"Method '{method_name}' not implemented for algorithm '{algorithm_name}'"
        else:
            message = f"Algorithm '{algorithm_name}' is not implemented"
        super().__init__(message, error_code=1002)
        self.algorithm_name = algorithm_name
        self.method_name = method_name


class AlgorithmConvergenceError(AlgorithmError):
    """算法收敛失败异常"""

    def __init__(self, algorithm_name: str, generations: int):
        super().__init__(
            f"Algorithm '{algorithm_name}' failed to converge after {generations} generations",
            error_code=1003
        )
        self.algorithm_name = algorithm_name
        self.generations = generations


class InvalidParameterError(AlgorithmError):
    """无效参数异常"""

    def __init__(self, parameter_name: str, parameter_value: Any, reason: Optional[str] = None):
        message = f"Invalid value '{parameter_value}' for parameter '{parameter_name}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, error_code=1004)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value


# ==================== 问题定义异常 ====================

class ProblemDefinitionError(DEAPError):
    """问题定义相关异常"""
    pass


class InvalidFunctionError(ProblemDefinitionError):
    """无效函数异常"""

    def __init__(self, function_name: str, reason: Optional[str] = None):
        message = f"Invalid function '{function_name}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, error_code=2001)
        self.function_name = function_name


class DimensionMismatchError(ProblemDefinitionError):
    """维度不匹配异常"""

    def __init__(self, expected_dim: int, actual_dim: int):
        super().__init__(
            f"Dimension mismatch: expected {expected_dim}, got {actual_dim}",
            error_code=2002
        )
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim


class BoundsError(ProblemDefinitionError):
    """边界异常"""

    def __init__(self, dimension: int, bounds: List[tuple], reason: Optional[str] = None):
        message = f"Invalid bounds for dimension {dimension}: {bounds}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, error_code=2003)
        self.dimension = dimension
        self.bounds = bounds


class EvaluationError(ProblemDefinitionError):
    """评估异常"""

    def __init__(self, problem_func: str, individual: List[Any], original_error: Exception):
        super().__init__(
            f"Failed to evaluate problem '{problem_func}' on individual: {original_error}",
            error_code=2004
        )
        self.problem_func = problem_func
        self.individual = individual
        self.original_error = original_error


# ==================== 优化异常 ====================

class OptimizationError(DEAPError):
    """优化相关异常"""
    pass


class PopulationEmptyError(OptimizationError):
    """种群为空异常"""

    def __init__(self):
        super().__init__(
            "Population is empty. Cannot perform optimization operation.",
            error_code=3001
        )


class InvalidPopulationSizeError(OptimizationError):
    """无效种群大小异常"""

    def __init__(self, population_size: int, min_size: int, max_size: Optional[int] = None):
        if max_size:
            message = f"Invalid population size {population_size}. Must be between {min_size} and {max_size}"
        else:
            message = f"Invalid population size {population_size}. Must be at least {min_size}"
        super().__init__(message, error_code=3002)
        self.population_size = population_size
        self.min_size = min_size
        self.max_size = max_size


class FitnessEvaluationError(OptimizationError):
    """适应度评估异常"""

    def __init__(self, individual: List[Any], reason: Optional[str] = None):
        message = f"Failed to evaluate fitness for individual {individual}"
        if reason:
            message += f": {reason}"
        super().__init__(message, error_code=3003)
        self.individual = individual


class ConvergenceNotReachedError(OptimizationError):
    """未收敛异常"""

    def __init__(self, best_fitness: float, threshold: float):
        super().__init__(
            f"Optimization did not converge. Best fitness: {best_fitness}, threshold: {threshold}",
            error_code=3004
        )
        self.best_fitness = best_fitness
        self.threshold = threshold


# ==================== 性能指标异常 ====================

class MetricsError(DEAPError):
    """性能指标相关异常"""
    pass


class EmptyParetoFrontError(MetricsError):
    """空帕累托前沿异常"""

    def __init__(self):
        super().__init__(
            "Cannot calculate metrics for empty Pareto front",
            error_code=4001
        )


class InvalidMetricError(MetricsError):
    """无效指标异常"""

    def __init__(self, metric_name: str):
        super().__init__(
            f"Unknown metric '{metric_name}'",
            error_code=4002
        )
        self.metric_name = metric_name


class MetricCalculationError(MetricsError):
    """指标计算异常"""

    def __init__(self, metric_name: str, reason: str):
        super().__init__(
            f"Failed to calculate metric '{metric_name}': {reason}",
            error_code=4003
        )
        self.metric_name = metric_name


# ==================== 配置异常 ====================

class ConfigurationError(DEAPError):
    """配置相关异常"""
    pass


class ConfigFileNotFoundError(ConfigurationError):
    """配置文件未找到异常"""

    def __init__(self, config_path: str):
        super().__init__(
            f"Configuration file not found: {config_path}",
            error_code=5001
        )
        self.config_path = config_path


class InvalidConfigError(ConfigurationError):
    """无效配置异常"""

    def __init__(self, config_key: str, reason: Optional[str] = None):
        message = f"Invalid configuration for key '{config_key}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, error_code=5002)
        self.config_key = config_key


class MissingConfigError(ConfigurationError):
    """缺失配置异常"""

    def __init__(self, config_key: str):
        super().__init__(
            f"Missing required configuration: '{config_key}'",
            error_code=5003
        )
        self.config_key = config_key


# ==================== 数据异常 ====================

class DataError(DEAPError):
    """数据相关异常"""
    pass


class DataFormatError(DataError):
    """数据格式异常"""

    def __init__(self, expected_format: str, actual_data: Any):
        super().__init__(
            f"Data format error: expected {expected_format}, got {type(actual_data).__name__}",
            error_code=6001
        )
        self.expected_format = expected_format
        self.actual_data = actual_data


class DataLoadError(DataError):
    """数据加载异常"""

    def __init__(self, data_path: str, reason: str):
        super().__init__(
            f"Failed to load data from {data_path}: {reason}",
            error_code=6002
        )
        self.data_path = data_path


class DataSaveError(DataError):
    """数据保存异常"""

    def __init__(self, data_path: str, reason: str):
        super().__init__(
            f"Failed to save data to {data_path}: {reason}",
            error_code=6003
        )
        self.data_path = data_path


# ==================== 约束处理异常 ====================

class ConstraintError(DEAPError):
    """约束处理相关异常"""
    pass


class ConstraintViolationError(ConstraintError):
    """约束违反异常"""

    def __init__(self, individual: List[Any], violations: List[float]):
        super().__init__(
            f"Constraint violation detected for individual: {violations}",
            error_code=7001
        )
        self.individual = individual
        self.violations = violations


class InfeasibleSolutionError(ConstraintError):
    """不可行解异常"""

    def __init__(self, individual: List[Any], reason: str):
        super().__init__(
            f"Infeasible solution: {reason}",
            error_code=7002
        )
        self.individual = individual


# ==================== 分布式计算异常 ====================

class DistributedError(DEAPError):
    """分布式计算相关异常"""
    pass


class ClusterInitializationError(DistributedError):
    """集群初始化异常"""

    def __init__(self, reason: str):
        super().__init__(
            f"Failed to initialize distributed cluster: {reason}",
            error_code=8001
        )


class TaskExecutionError(DistributedError):
    """任务执行异常"""

    def __init__(self, task_id: str, reason: str):
        super().__init__(
            f"Failed to execute task '{task_id}': {reason}",
            error_code=8002
        )
        self.task_id = task_id


class NodeConnectionError(DistributedError):
    """节点连接异常"""

    def __init__(self, node_id: str, reason: str):
        super().__init__(
            f"Failed to connect to node '{node_id}': {reason}",
            error_code=8003
        )
        self.node_id = node_id


# ==================== GPU加速异常 ====================

class GPUError(DEAPError):
    """GPU加速相关异常"""
    pass


class GPUNotAvailableError(GPUError):
    """GPU不可用异常"""

    def __init__(self, reason: Optional[str] = None):
        message = "GPU is not available"
        if reason:
            message += f": {reason}"
        super().__init__(message, error_code=9001)


class GPUInitializationError(GPUError):
    """GPU初始化异常"""

    def __init__(self, reason: str):
        super().__init__(
            f"Failed to initialize GPU: {reason}",
            error_code=9002
        )


class GPUMemoryError(GPUError):
    """GPU内存异常"""

    def __init__(self, required_memory: float, available_memory: float):
        super().__init__(
            f"Insufficient GPU memory: required {required_memory}GB, available {available_memory}GB",
            error_code=9003
        )
        self.required_memory = required_memory
        self.available_memory = available_memory


# ==================== 导出 ====================

__all__ = [
    # 基础异常
    'DEAPError',

    # 算法异常
    'AlgorithmError',
    'AlgorithmNotInitializedError',
    'AlgorithmNotImplementedError',
    'AlgorithmConvergenceError',
    'InvalidParameterError',

    # 问题定义异常
    'ProblemDefinitionError',
    'InvalidFunctionError',
    'DimensionMismatchError',
    'BoundsError',
    'EvaluationError',

    # 优化异常
    'OptimizationError',
    'PopulationEmptyError',
    'InvalidPopulationSizeError',
    'FitnessEvaluationError',
    'ConvergenceNotReachedError',

    # 性能指标异常
    'MetricsError',
    'EmptyParetoFrontError',
    'InvalidMetricError',
    'MetricCalculationError',

    # 配置异常
    'ConfigurationError',
    'ConfigFileNotFoundError',
    'InvalidConfigError',
    'MissingConfigError',

    # 数据异常
    'DataError',
    'DataFormatError',
    'DataLoadError',
    'DataSaveError',

    # 约束处理异常
    'ConstraintError',
    'ConstraintViolationError',
    'InfeasibleSolutionError',

    # 分布式计算异常
    'DistributedError',
    'ClusterInitializationError',
    'TaskExecutionError',
    'NodeConnectionError',

    # GPU加速异常
    'GPUError',
    'GPUNotAvailableError',
    'GPUInitializationError',
    'GPUMemoryError',
]
