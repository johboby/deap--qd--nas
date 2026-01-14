"""
智能DEAP框架 - 兼容版本
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .base_algorithms import BaseMultiObjectiveAlgorithm, MOEAD
from .test_functions import TestFunctionLibrary
from .metrics import PerformanceMetrics
from .experiment_manager import SimpleExperimentManager
from .lightweight_intelligent_framework import LightweightIntelligentFramework, OptimizationConfig, OptimizationMode

# 简化的约束处理器（内联实现）
class SimpleConstraintHandler:
    """简化的约束处理器"""
    
    def __init__(self):
        pass
    
    def handle_constraints(self, solution, constraint_violations):
        """处理约束违反"""
        return solution, sum(max(0, v) for v in constraint_violations) if constraint_violations else 0.0

class IntelligentDEAPFramework:
    """智能DEAP框架 - 兼容版本"""
    
    def __init__(self):
        self.lightweight_framework = LightweightIntelligentFramework()
        print("🎉 智能DEAP框架初始化完成 (兼容模式)")
    
    def optimize(self, problem_func, n_dim, bounds, algorithm_name="NSGA2", **kwargs):
        """优化接口 - 委托给轻量级框架"""
        result = self.lightweight_framework.intelligent_hybrid_optimize(
            problem_func=problem_func,
            dim=n_dim,
            bounds=bounds,
            mode=OptimizationMode.INTELLIGENT_HYBRID
        )
        
        return {
            'population': [result['best_solution']],
            'pareto_front': result['pareto_front'],
            'execution_time': result['execution_time'],
            'metrics': result['metrics']
        }


__all__ = ['IntelligentDEAPFramework', 'OptimizationConfig', 'OptimizationMode']