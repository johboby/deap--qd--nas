"""
DEAP多目标优化研究框架

一个现代化、模块化的进化算法研究平台，提供从基础到前沿的多目标优化算法实现。

架构层次:
- core/: 核心框架层（基础算法、测试函数、评估指标）
- algorithms/: 算法实现层（NSGA-II/III、MOEA/D、混合算法等）
- advanced/: 高级特性层（自适应算法、约束处理、并行计算等）
- utils/: 工具层（可视化、日志、分析等）
- applications/: 应用层（工程优化、机器学习超参优化等）
"""

__version__ = "2.0.0"
__author__ = "DEAP Research Team"

# 导出主要接口
from .core import (
    MultiObjectiveFramework,
    NSGA2, MOEAD,
    TestFunctionLibrary,
    PerformanceMetrics,
    SimpleExperimentManager,
    RobustExperimentManager
)

from .algorithms.nsga2 import NSGA2Algorithm

# 为向后兼容，提供别名
ExperimentManager = SimpleExperimentManager

__all__ = [
    # 版本信息
    '__version__', '__author__',

    # 核心组件
    'MultiObjectiveFramework',
    'NSGA2', 'MOEAD',
    'TestFunctionLibrary',
    'PerformanceMetrics',
    'SimpleExperimentManager',
    'RobustExperimentManager',
    'ExperimentManager',  # 别名

    # 算法实现
    'NSGA2Algorithm'
]