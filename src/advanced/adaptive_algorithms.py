"""
自适应算法模块
包含自适应参数调整、自适应NSGA-III等高级算法
"""

import numpy as np
from typing import List, Callable, Dict, Any, Optional
from abc import ABC, abstractmethod
import random

from ..core.base_algorithms import BaseMultiObjectiveAlgorithm
from ..core.metrics import PerformanceMetrics

class AdaptiveAlgorithm(BaseMultiObjectiveAlgorithm):
    """自适应算法基类"""
    
    def __init__(self, adaptation_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        
    @abstractmethod
    def adapt_parameters(self, generation: int, performance_metrics: Dict[str, float]):
        """自适应调整算法参数"""
        pass

class AdaptiveNSGA2(AdaptiveAlgorithm):
    """自适应NSGA-II算法"""
    
    def __init__(self, initial_crossover_prob: float = 0.9, 
                 initial_mutation_prob: float = 0.1,
                 min_crossover: float = 0.7, max_crossover: float = 0.95,
                 min_mutation: float = 0.05, max_mutation: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.crossover_prob = initial_crossover_prob
        self.mutation_prob = initial_mutation_prob
        self.min_crossover = min_crossover
        self.max_crossover = max_crossover
        self.min_mutation = min_mutation
        self.max_mutation = max_mutation
        
    def adapt_parameters(self, generation: int, performance_metrics: Dict[str, float]):
        """基于性能自适应调整交叉和变异概率"""
        if 'hypervolume' in performance_metrics:
            hv = performance_metrics['hypervolume']
            self.performance_history.append(hv)
            
            # 简单的自适应策略：如果性能停滞，增加探索
            if len(self.performance_history) >= 5:
                recent_performance = self.performance_history[-5:]
                improvement = recent_performance[-1] - recent_performance[0]
                
                if improvement < 0.01:  # 性能改善很小
                    # 增加变异概率以增强探索
                    self.mutation_prob = min(self.max_mutation, 
                                           self.mutation_prob + self.adaptation_rate * 0.1)
                    self.crossover_prob = max(self.min_crossover,
                                             self.crossover_prob - self.adaptation_rate * 0.05)
                else:
                    # 性能良好，保持平衡
                    self.mutation_prob = max(self.min_mutation,
                                             min(self.max_mutation, self.mutation_prob))
                    self.crossover_prob = max(self.min_crossover,
                                             min(self.max_crossover, self.crossover_prob))
    
    def optimize(self, problem_func: Callable, n_gen: int = 100, 
                pop_size: int = 100, n_dim: int = 10,
                bounds_low: List[float] = None, bounds_high: List[float] = None) -> Dict[str, Any]:
        """运行自适应NSGA-II优化"""
        if bounds_low is None:
            bounds_low = [0.0] * n_dim
        if bounds_high is None:
            bounds_high = [1.0] * n_dim
            
        # 设置工具箱
        toolbox = self._setup_toolbox(problem_func, bounds_low, bounds_high, n_dim)
        
        # 注册遗传算子
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                       low=bounds_low, up=bounds_high, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, 
                        low=bounds_low, up=bounds_high, eta=20.0, 
                        indpb=1.0/n_dim)
        toolbox.register("select", tools.selNSGA2)
        
        # 创建初始种群
        pop = toolbox.population(n=pop_size)
        
        # 评估初始种群
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            
        # 进化过程
        for gen in range(n_gen):
            # 生成后代
            offspring = algorithms.varAnd(pop, toolbox, self.crossover_prob, self.mutation_prob)
            
            # 评估后代
            fits = list(map(toolbox.evaluate, offspring))
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
                
            # 环境选择
            combined = pop + offspring
            pop = toolbox.select(combined, pop_size)
            
            # 自适应调整参数
            if gen % 10 == 0:  # 每10代评估一次
                pareto_front = tools.sortLogNondominated(pop, k=len(pop), first_front_only=True)
                front_points = [ind.fitness.values for ind in pareto_front]
                metrics = PerformanceMetrics.calculate_all_metrics(front_points)
                self.adapt_parameters(gen, metrics)
            
            # 记录统计信息
            self._record_statistics(pop, gen)
            
        # 提取Pareto前沿
        pareto_front = tools.sortLogNondominated(pop, k=len(pop), first_front_only=True)
        
        return {
            'population': pop,
            'pareto_front': pareto_front,
            'history': self.history,
            'logbook': self._create_logbook(),
            'final_params': {
                'crossover_prob': self.crossover_prob,
                'mutation_prob': self.mutation_prob
            }
        }

class PopulationAdaptationStrategy(ABC):
    """种群自适应策略基类"""
    
    @abstractmethod
    def adapt_population_size(self, current_size: int, generation: int, 
                           performance: Dict[str, float]) -> int:
        """自适应调整种群大小"""
        pass

class DiversityBasedAdaptation(PopulationAdaptationStrategy):
    """基于多样性的种群自适应"""
    
    def __init__(self, min_size: int = 50, max_size: int = 300):
        self.min_size = min_size
        self.max_size = max_size
        
    def adapt_population_size(self, current_size: int, generation: int,
                           performance: Dict[str, float]) -> int:
        """基于多样性指标调整种群大小"""
        if 'spread' in performance:
            spread = performance['spread']
            
            # 如果分布太差，增加种群大小
            if spread > 0.8:
                new_size = min(self.max_size, int(current_size * 1.1))
            # 如果分布很好且收敛稳定，可以减少种群大小
            elif spread < 0.3:
                new_size = max(self.min_size, int(current_size * 0.9))
            else:
                new_size = current_size
                
            return new_size
        return current_size

# 导入必要的DEAP组件
from deap import base, creator, tools, algorithms