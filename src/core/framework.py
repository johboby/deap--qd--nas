"""
多目标优化框架 - 兼容版本
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable

from .base_algorithms import BaseMultiObjectiveAlgorithm, MOEAD
from .test_functions import TestFunctionLibrary
from .metrics import PerformanceMetrics
from .experiment_manager import SimpleExperimentManager
from .lightweight_intelligent_framework import LightweightIntelligentFramework, OptimizationConfig, OptimizationMode


class MultiObjectiveFramework:
    """多目标优化框架 - 兼容版本"""
    
    def __init__(self):
        self.lightweight_framework = LightweightIntelligentFramework()
        print("🎉 多目标优化框架初始化完成 (兼容模式)")
    
    def setup_problem(self, problem_name, n_dim, bounds):
        """设置问题"""
        library = TestFunctionLibrary()
        if problem_name in library.functions:
            return library.create_function(problem_name)
        else:
            # 返回简单的测试函数
            def simple_func(x):
                return sum(xi**2 for xi in x), []
            return simple_func
    
    def run_optimization(self, problem_func, algorithm_name="NSGA2", n_dim=10, bounds=None, **kwargs):
        """运行优化"""
        if bounds is None:
            bounds = [(-5, 5)] * n_dim
        
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
    
    def benchmark(self, problems, algorithms, runs=3):
        """基准测试"""
        results = []
        
        for problem_name in problems:
            for algorithm_name in algorithms:
                for run in range(runs):
                    try:
                        problem_func = self.setup_problem(problem_name, 10, [(-5, 5)] * 10)
                        result = self.run_optimization(problem_func, algorithm_name, 10)
                        
                        results.append({
                            'problem': problem_name,
                            'algorithm': algorithm_name,
                            'run': run,
                            'hypervolume': result['metrics']['hypervolume'],
                            'execution_time': result['execution_time']
                        })
                        
                    except Exception as e:
                        print(f"基准测试失败 {problem_name}-{algorithm_name}-{run}: {e}")
        
        return results


__all__ = ['MultiObjectiveFramework']