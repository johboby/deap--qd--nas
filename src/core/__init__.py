"""
DEAP核心模块
智能多目标优化框架的核心组件
"""

import sys
import os

# 添加当前目录到路径以支持相对导入
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 使用绝对导入（相对于包）
from .framework import MultiObjectiveFramework
from .base_algorithms import NSGA2, MOEAD, SPEA2, IBEA, ClassicalEvolution
from .test_functions import TestFunctionLibrary
from .metrics import PerformanceMetrics

# 实验管理器 - 直接导入（已解决循环依赖）
from .experiment_manager import SimpleExperimentManager, RobustExperimentManager

# 智能框架组件
from .lightweight_intelligent_framework import LightweightIntelligentFramework, OptimizationConfig, OptimizationMode
from .intelligent_framework import IntelligentDEAPFramework

# 常量定义
from .constants import (
    OptimizationConstants, MetricsConstants, AnalysisConstants,
    AdaptiveConstants, ConstraintConstants, ParallelConstants,
    VisualizationConstants, LoggingConstants, ExperimentConstants,
    MetaLearningConstants, DistributedConstants, GPUConstants,
    BoundConstants, AlgorithmConstants,
    # 便捷常量
    DEFAULT_POPULATION_SIZE, DEFAULT_GENERATIONS,
    DEFAULT_CROSSOVER_PROB, DEFAULT_MUTATION_PROB,
    CONVERGENCE_THRESHOLD, DEFAULT_HYPERVOLUME_REFERENCE
)

# 自定义异常
from .exceptions import (
    DEAPError,
    # 算法异常
    AlgorithmError, AlgorithmNotInitializedError, AlgorithmNotImplementedError,
    AlgorithmConvergenceError, InvalidParameterError,
    # 问题定义异常
    ProblemDefinitionError, InvalidFunctionError, DimensionMismatchError,
    BoundsError, EvaluationError,
    # 优化异常
    OptimizationError, PopulationEmptyError, InvalidPopulationSizeError,
    FitnessEvaluationError, ConvergenceNotReachedError,
    # 性能指标异常
    MetricsError, EmptyParetoFrontError, InvalidMetricError, MetricCalculationError,
    # 配置异常
    ConfigurationError, ConfigFileNotFoundError, InvalidConfigError, MissingConfigError,
    # 数据异常
    DataError, DataFormatError, DataLoadError, DataSaveError,
    # 约束处理异常
    ConstraintError, ConstraintViolationError, InfeasibleSolutionError,
    # 分布式计算异常
    DistributedError, ClusterInitializationError, TaskExecutionError, NodeConnectionError,
    # GPU加速异常
    GPUError, GPUNotAvailableError, GPUInitializationError, GPUMemoryError,
)

# 高级功能组件（可选导入）
try:
    from .distributed_computing import DistributedIntelligentFramework, DistributedConfig, create_distributed_framework
except ImportError:
    DistributedIntelligentFramework = None
    DistributedConfig = None
    create_distributed_framework = None

try:
    from .gpu_acceleration import GPUAcceleratedFramework, GPUConfig, create_gpu_framework
except ImportError:
    GPUAcceleratedFramework = None
    GPUConfig = None
    create_gpu_framework = None

try:
    from .meta_learning_automl import create_meta_learning_framework, AutoMLEngine, AlgorithmSelector
except ImportError:
    create_meta_learning_framework = None
    AutoMLEngine = None
    AlgorithmSelector = None

try:
    from .advanced_integration import AdvancedIntelligentFramework, AdvancedConfig, create_advanced_framework
except ImportError:
    AdvancedIntelligentFramework = None
    AdvancedConfig = None
    create_advanced_framework = None

__all__ = [
    # 基础框架
    'MultiObjectiveFramework',
    'OptimizationConfig',
    'OptimizationMode',

    # 算法
    'NSGA2', 'MOEAD', 'SPEA2', 'IBEA', 'ClassicalEvolution',

    # 测试和评估
    'TestFunctionLibrary',
    'PerformanceMetrics',

    # 智能框架
    'LightweightIntelligentFramework',
    'IntelligentDEAPFramework',

    # 常量类
    'OptimizationConstants', 'MetricsConstants', 'AnalysisConstants',
    'AdaptiveConstants', 'ConstraintConstants', 'ParallelConstants',
    'VisualizationConstants', 'LoggingConstants', 'ExperimentConstants',
    'MetaLearningConstants', 'DistributedConstants', 'GPUConstants',
    'BoundConstants', 'AlgorithmConstants',

    # 便捷常量
    'DEFAULT_POPULATION_SIZE', 'DEFAULT_GENERATIONS',
    'DEFAULT_CROSSOVER_PROB', 'DEFAULT_MUTATION_PROB',
    'CONVERGENCE_THRESHOLD', 'DEFAULT_HYPERVOLUME_REFERENCE',

    # 自定义异常
    'DEAPError',
    'AlgorithmError', 'AlgorithmNotInitializedError', 'AlgorithmNotImplementedError',
    'AlgorithmConvergenceError', 'InvalidParameterError',
    'ProblemDefinitionError', 'InvalidFunctionError', 'DimensionMismatchError',
    'BoundsError', 'EvaluationError',
    'OptimizationError', 'PopulationEmptyError', 'InvalidPopulationSizeError',
    'FitnessEvaluationError', 'ConvergenceNotReachedError',
    'MetricsError', 'EmptyParetoFrontError', 'InvalidMetricError', 'MetricCalculationError',
    'ConfigurationError', 'ConfigFileNotFoundError', 'InvalidConfigError', 'MissingConfigError',
    'DataError', 'DataFormatError', 'DataLoadError', 'DataSaveError',
    'ConstraintError', 'ConstraintViolationError', 'InfeasibleSolutionError',
    'DistributedError', 'ClusterInitializationError', 'TaskExecutionError', 'NodeConnectionError',
    'GPUError', 'GPUNotAvailableError', 'GPUInitializationError', 'GPUMemoryError',

    # 高级功能（可能为None）
    'DistributedIntelligentFramework',
    'DistributedConfig',
    'create_distributed_framework',
    'GPUAcceleratedFramework',
    'GPUConfig',
    'create_gpu_framework',
    'create_meta_learning_framework',
    'AutoMLEngine',
    'AlgorithmSelector',
    'AdvancedIntelligentFramework',
    'AdvancedConfig',
    'create_advanced_framework',
]

print("🎉 DEAP智能优化框架核心模块加载完成")
print("📊 基础功能: 完全可用")
print("🚀 高级功能: 框架就绪 (部分需要额外依赖)")
print("✅ 生产就绪: 智能优化平台已就绪")