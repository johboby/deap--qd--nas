"""
QD-NAS: Quality-Diversity NeuroArchitecture Search
质量-多样性神经架构搜索框架

核心思想:
1. 使用QD（Quality-Diversity）思想优化NAS的搜索过程
2. 通过行为特征映射保持解的多样性
3. 减少人工设计的算子和参数，使用种群引导的搜索
4. 支持多目标多约束优化（延迟、能耗、精度等）
5. 自适应搜索策略和理论导向设计

主要组件:
- BehaviorSpace: 行为空间定义
- Characterization: 行为特征提取
- Archive: 归档管理
- MAPElites: MAP-Elites算法
- MultiObjectiveNAS: 多目标NAS
- QDNASOptimizer: QD-NAS主优化器
"""

__version__ = "4.0.0"
__author__ = "QD-NAS Research Team"

# 行为空间
from .behavior_space import (
    BehaviorType, BehaviorDimension, BehaviorSpace,
    create_nas_behavior_space, create_latency_energy_behavior_space,
    create_complexity_behavior_space
)

# 特征提取
from .characterization import (
    ArchitectureMetrics, BaseCharacterization,
    StaticCharacterization, DynamicCharacterization, HybridCharacterization,
    compute_diversity, compute_novelty
)

# 动态特征提取（v3.1新增）
try:
    from .dynamic_characterization import (
        TrainingConfig, DatasetConfig, DynamicCharacterizer,
        create_dynamic_characterizer
    )
    DYNAMIC_CHAR_AVAILABLE = True
except ImportError:
    DYNAMIC_CHAR_AVAILABLE = False

# 归档管理
from .archive import ArchiveEntry, Archive

# QD算法
from .map_elites import (
    MAPElites, CMA_MAPElites,
    RandomSearchMAPElites, GradientGuidedMAPElites
)

# CMA-ES算法（v3.1新增）
try:
    from .cma_es import (
        CMAESOptimizer, CMAESOptimizerQD, CMAParameters,
        create_cmaes_optimizer
    )
    CMA_ES_AVAILABLE = True
except ImportError:
    CMA_ES_AVAILABLE = False

# 高级QD算法（v3.1新增）
from .advanced_qd_algorithms import (
    CVTMAPElites, DiverseQualityArchive,
    create_cvt_map_elites, create_dq_archive
)

# 超参数调优（v3.1新增）
try:
    from .hyperparameter_tuning import (
        Hyperparameter, BaseHyperparameterOptimizer,
        RandomSearchOptimizer, AdaptiveParameterTuner,
        BayesianOptimizer, create_hyperparameter_optimizer
    )
    HYPERPARAMETER_TUNING_AVAILABLE = True
except ImportError:
    HYPERPARAMETER_TUNING_AVAILABLE = False

# 性能监控（v3.1新增）
from .performance_monitor import (
    PerformanceMetrics, MetricCollector, SystemMetricCollector,
    PerformanceMonitor, PerformanceAnalyzer
)

# 分布式计算（v4.0新增）
try:
    from .distributed_computing import (
        WorkerConfig, BaseEvaluator,
        SerialEvaluator, MultiProcessEvaluator, GPUAcceleratedEvaluator,
        DistributedNASOptimizer, BatchProcessor, create_evaluator
    )
    DISTRIBUTED_COMPUTING_AVAILABLE = True
except ImportError:
    DISTRIBUTED_COMPUTING_AVAILABLE = False

# NAS基准测试（v4.0新增）
try:
    from .benchmark_suite import (
        DatasetConfig, StandardDatasets,
        BaseNASBenchmark, CIFAR10Benchmark,
        BenchmarkResults, BenchmarkRunner, create_benchmark
    )
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# 端到端NAS（v4.0新增）
try:
    from .end_to_end_nas import (
        NASConfig, NASResult,
        DataPipeline, Trainer,
        EndToEndNAS, create_end_to_end_nas
    )
    END_TO_END_NAS_AVAILABLE = True
except ImportError:
    END_TO_END_NAS_AVAILABLE = False

# 工具（v4.0新增）
from .utils import (
    ColorFormatter, LoggerManager, ExperimentConfig,
    Timer, ProgressBar, CheckpointManager, MetricsTracker, ConfigManager,
    set_random_seed, count_parameters, format_time, format_number
)

# 错误处理（v4.0新增）
from .error_handling import (
    RecoveryAction, RecoveryStrategy,
    RetryStrategy, CheckpointRecoveryStrategy, FallbackStrategy,
    ErrorHandler, retry, safe_execute, error_context,
    CircuitBreaker, ErrorRecoveryManager, handle_errors
)

# 多目标优化
from .multi_objective_nas import (
    ObjectiveType, Objective, Constraint,
    MultiObjectiveNAS, create_default_multi_objective_nas
)

# 搜索空间
from .search_space import (
    OperationType, Cell, Architecture,
    SearchSpace, HierarchicalSearchSpace
)

# 种群引导搜索
from .population_guided_search import (
    PopulationStats, PopulationGuidedSearch, AdaptiveHybridSearch
)

# 主优化器
from .qd_nas import (
    QDNASOptimizer, create_default_qd_nas,
    example_simple_nas, example_multi_objective_nas
)

__all__ = [
    # 版本信息
    '__version__', '__author__',

    # 行为空间
    'BehaviorType', 'BehaviorDimension', 'BehaviorSpace',
    'create_nas_behavior_space', 'create_latency_energy_behavior_space',
    'create_complexity_behavior_space',

    # 特征提取
    'ArchitectureMetrics', 'BaseCharacterization',
    'StaticCharacterization', 'DynamicCharacterization', 'HybridCharacterization',
    'compute_diversity', 'compute_novelty',

    # 动态特征提取（v3.1新增）
    'TrainingConfig', 'DatasetConfig', 'DynamicCharacterizer',
    'create_dynamic_characterizer',

    # 归档管理
    'ArchiveEntry', 'Archive',

    # QD算法
    'MAPElites', 'CMA_MAPElites',
    'RandomSearchMAPElites', 'GradientGuidedMAPElites',

    # CMA-ES算法（v3.1新增）
    'CMAESOptimizer', 'CMAESOptimizerQD', 'CMAParameters',
    'create_cmaes_optimizer',

    # 高级QD算法（v3.1新增）
    'CVTMAPElites', 'DiverseQualityArchive',
    'create_cvt_map_elites', 'create_dq_archive',

    # 超参数调优（v3.1新增）
    'Hyperparameter', 'BaseHyperparameterOptimizer',
    'RandomSearchOptimizer', 'AdaptiveParameterTuner',
    'BayesianOptimizer', 'create_hyperparameter_optimizer',

    # 性能监控（v3.1新增）
    'PerformanceMetrics', 'MetricCollector', 'SystemMetricCollector',
    'PerformanceMonitor', 'PerformanceAnalyzer',

    # 分布式计算（v4.0新增）
    'WorkerConfig', 'BaseEvaluator',
    'SerialEvaluator', 'MultiProcessEvaluator', 'GPUAcceleratedEvaluator',
    'DistributedNASOptimizer', 'BatchProcessor', 'create_evaluator',

    # NAS基准测试（v4.0新增）
    'DatasetConfig', 'StandardDatasets',
    'BaseNASBenchmark', 'CIFAR10Benchmark',
    'BenchmarkResults', 'BenchmarkRunner', 'create_benchmark',

    # 端到端NAS（v4.0新增）
    'NASConfig', 'NASResult',
    'DataPipeline', 'Trainer',
    'EndToEndNAS', 'create_end_to_end_nas',

    # 工具（v4.0新增）
    'ColorFormatter', 'LoggerManager', 'ExperimentConfig',
    'Timer', 'ProgressBar', 'CheckpointManager', 'MetricsTracker', 'ConfigManager',
    'set_random_seed', 'count_parameters', 'format_time', 'format_number',

    # 错误处理（v4.0新增）
    'RecoveryAction', 'RecoveryStrategy',
    'RetryStrategy', 'CheckpointRecoveryStrategy', 'FallbackStrategy',
    'ErrorHandler', 'retry', 'safe_execute', 'error_context',
    'CircuitBreaker', 'ErrorRecoveryManager', 'handle_errors',

    # 多目标优化
    'ObjectiveType', 'Objective', 'Constraint',
    'MultiObjectiveNAS', 'create_default_multi_objective_nas',

    # 搜索空间
    'OperationType', 'Cell', 'Architecture',
    'SearchSpace', 'HierarchicalSearchSpace',

    # 种群引导搜索
    'PopulationStats', 'PopulationGuidedSearch', 'AdaptiveHybridSearch',

    # 主优化器
    'QDNASOptimizer', 'create_default_qd_nas',
    'example_simple_nas', 'example_multi_objective_nas',
]
