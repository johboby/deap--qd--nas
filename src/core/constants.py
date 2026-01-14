"""
DEAP多目标优化框架 - 常量定义
集中管理所有魔法数字，提高代码可维护性
"""

from dataclasses import dataclass
from typing import Tuple, Dict


# ==================== 优化参数常量 ====================

class OptimizationConstants:
    """优化算法相关常量"""

    # 种群规模
    DEFAULT_POPULATION_SIZE = 100
    MIN_POPULATION_SIZE = 20
    MAX_POPULATION_SIZE = 1000

    # 迭代次数
    DEFAULT_GENERATIONS = 100
    MIN_GENERATIONS = 10
    MAX_GENERATIONS = 10000

    # 交叉概率
    DEFAULT_CROSSOVER_PROBABILITY = 0.8
    MIN_CROSSOVER_PROBABILITY = 0.5
    MAX_CROSSOVER_PROBABILITY = 0.95

    # 变异概率
    DEFAULT_MUTATION_PROBABILITY = 0.1
    MIN_MUTATION_PROBABILITY = 0.01
    MAX_MUTATION_PROBABILITY = 0.3

    # 变异分布指数
    DEFAULT_ETA_C = 15.0
    DEFAULT_ETA_M = 20.0

    # 锦标赛选择大小
    DEFAULT_TOURNAMENT_SIZE = 3


# ==================== 性能评估常量 ====================

class MetricsConstants:
    """性能评估指标相关常量"""

    # 超体积参考点
    DEFAULT_HYPERVOLUME_REFERENCE = (1.1, 1.1)

    # 收敛阈值
    CONVERGENCE_THRESHOLD = 1e-10
    DEFAULT_TOLERANCE = 1e-6

    # IGD和GD权重
    DEFAULT_IGD_WEIGHT = 1.0
    DEFAULT_GD_WEIGHT = 1.0

    # Spread参数
    MIN_SPREAD = 0.0
    MAX_SPREAD = 1.0


# ==================== 问题分析常量 ====================

class AnalysisConstants:
    """问题分析相关常量"""

    # 维度分类
    LOW_DIMENSIONAL_THRESHOLD = 5
    MEDIUM_DIMENSIONAL_THRESHOLD = 20
    HIGH_DIMENSIONAL_THRESHOLD = 50

    # 多模态判断阈值
    MODALITY_THRESHOLD = 0.5

    # 约束违规阈值
    CONSTRAINT_VIOLATION_THRESHOLD = 1e-6


# ==================== 自适应参数常量 ====================

class AdaptiveConstants:
    """自适应算法相关常量"""

    # 学习率
    DEFAULT_LEARNING_RATE = 0.1
    MIN_LEARNING_RATE = 0.01
    MAX_LEARNING_RATE = 0.5

    # 动量系数
    DEFAULT_MOMENTUM = 0.9

    # 记忆窗口大小
    MEMORY_WINDOW_SIZE = 10

    # 自适应频率
    ADAPTATION_FREQUENCY = 5


# ==================== 约束处理常量 ====================

class ConstraintConstants:
    """约束处理相关常量"""

    # 罚函数系数
    DEFAULT_PENALTY_COEFFICIENT = 1000.0
    MIN_PENALTY_COEFFICIENT = 100.0
    MAX_PENALTY_COEFFICIENT = 10000.0

    # 障碍法参数
    DEFAULT_BARRIER_PARAMETER = 1.0
    BARRIER_DECAY_RATE = 0.9

    # 可行性阈值
    FEASIBILITY_THRESHOLD = 1e-4


# ==================== 并行计算常量 ====================

class ParallelConstants:
    """并行计算相关常量"""

    # 默认进程数
    DEFAULT_N_PROCESSES = 4
    MIN_N_PROCESSES = 1
    MAX_N_PROCESSES = 16

    # 批处理大小
    DEFAULT_BATCH_SIZE = 100

    # 任务队列大小
    TASK_QUEUE_SIZE = 1000


# ==================== 可视化常量 ====================

class VisualizationConstants:
    """可视化相关常量"""

    # 图表尺寸
    DEFAULT_FIGURE_SIZE = (10, 8)
    DPI = 100

    # 颜色方案
    COLOR_SCHEMES = {
        'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'pastel': ['#a8dadc', '#f1faee', '#457b9d', '#1d3557', '#e63946'],
        'dark': ['#2d3436', '#636e72', '#b2bec3', '#dfe6e9', '#74b9ff']
    }

    # 线型
    LINE_STYLES = ['-', '--', ':', '-.']

    # 标记样式
    MARKER_STYLES = ['o', 's', '^', 'v', 'D']


# ==================== 日志常量 ====================

class LoggingConstants:
    """日志相关常量"""

    # 日志级别
    LOG_LEVELS = {
        'debug': 10,
        'info': 20,
        'warning': 30,
        'error': 40,
        'critical': 50
    }

    # 日志格式
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    # 日志文件大小限制 (MB)
    MAX_LOG_FILE_SIZE = 10
    BACKUP_COUNT = 5


# ==================== 实验管理常量 ====================

class ExperimentConstants:
    """实验管理相关常量"""

    # 默认运行次数
    DEFAULT_RUNS = 10
    MIN_RUNS = 1
    MAX_RUNS = 100

    # 结果保存路径
    DEFAULT_RESULTS_DIR = "results"

    # 随机种子
    DEFAULT_RANDOM_SEED = 42


# ==================== 元学习常量 ====================

class MetaLearningConstants:
    """元学习相关常量"""

    # 任务数量
    DEFAULT_N_TASKS = 10
    MIN_N_TASKS = 2
    MAX_N_TASKS = 50

    # 元学习率
    DEFAULT_META_LR = 0.001

    # 内部优化步数
    INNER_STEPS = 5


# ==================== 分布式计算常量 ====================

class DistributedConstants:
    """分布式计算相关常量"""

    # Ray默认配置
    RAY_INIT_TIMEOUT = 30  # 秒
    RAY_NUM_CPUS = None  # 自动检测
    RAY_OBJECT_STORE_MEMORY = None  # 自动配置

    # 任务分配策略
    TASK_ALLOCATION_STRATEGIES = ['round_robin', 'least_loaded', 'random']
    DEFAULT_ALLOCATION_STRATEGY = 'least_loaded'


# ==================== GPU加速常量 ====================

class GPUConstants:
    """GPU加速相关常量"""

    # CUDA设备
    DEFAULT_CUDA_DEVICE = 0

    # 批处理大小
    GPU_DEFAULT_BATCH_SIZE = 256
    GPU_MIN_BATCH_SIZE = 32
    GPU_MAX_BATCH_SIZE = 1024

    # 内存限制 (GB)
    GPU_MEMORY_LIMIT = 8.0


# ==================== 默认边界常量 ====================

class BoundConstants:
    """边界相关常量"""

    # 默认搜索空间
    DEFAULT_BOUNDS = (-5.0, 5.0)

    # ZDT测试函数边界
    ZDT_DEFAULT_BOUNDS = (0.0, 1.0)

    # DTLZ测试函数边界
    DTLZ_DEFAULT_BOUNDS = (0.0, 1.0)

    # Sphere函数边界
    SPHERE_DEFAULT_BOUNDS = (-5.0, 5.0)

    # Rastrigin函数边界
    RASTRIGIN_DEFAULT_BOUNDS = (-5.12, 5.12)


# ==================== 算法特定常量 ====================

class AlgorithmConstants:
    """特定算法常量"""

    # NSGA-II
    NSGA2_CROWDING_DISTANCE_POWER = 2

    # MOEA/D
    MOEAD_T_NEIGHBORHOOD_SIZE = 20
    MOEAD_DELTA = 0.9
    MOEAD_NR = 2

    # SPEA2
    SPEA2_K_NEAREST_NEIGHBORS = 1
    SPEA2_ARCHIVE_SIZE = 100

    # IBEA
    IBEA_KAPPA = 0.05


# ==================== 统一导出 ====================

__all__ = [
    # 优化参数
    'OptimizationConstants',

    # 性能评估
    'MetricsConstants',

    # 问题分析
    'AnalysisConstants',

    # 自适应参数
    'AdaptiveConstants',

    # 约束处理
    'ConstraintConstants',

    # 并行计算
    'ParallelConstants',

    # 可视化
    'VisualizationConstants',

    # 日志
    'LoggingConstants',

    # 实验管理
    'ExperimentConstants',

    # 元学习
    'MetaLearningConstants',

    # 分布式计算
    'DistributedConstants',

    # GPU加速
    'GPUConstants',

    # 边界
    'BoundConstants',

    # 算法特定
    'AlgorithmConstants',
]


# 便捷常量（向后兼容）
DEFAULT_POPULATION_SIZE = OptimizationConstants.DEFAULT_POPULATION_SIZE
DEFAULT_GENERATIONS = OptimizationConstants.DEFAULT_GENERATIONS
DEFAULT_CROSSOVER_PROB = OptimizationConstants.DEFAULT_CROSSOVER_PROBABILITY
DEFAULT_MUTATION_PROB = OptimizationConstants.DEFAULT_MUTATION_PROBABILITY
CONVERGENCE_THRESHOLD = MetricsConstants.CONVERGENCE_THRESHOLD
DEFAULT_HYPERVOLUME_REFERENCE = MetricsConstants.DEFAULT_HYPERVOLUME_REFERENCE
