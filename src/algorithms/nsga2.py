"""
NSGA-II算法实现
"""

import random
import numpy as np
from deap import base, creator, tools, algorithms
from typing import List, Callable, Tuple, Dict, Any
from ..core.base_algorithms import BaseMultiObjectiveAlgorithm

class NSGA2Algorithm(BaseMultiObjectiveAlgorithm):
    """NSGA-II算法类"""
    
    def __init__(self, crossover_prob: float = 0.9, mutation_prob: float = 0.1, 
                 eta: float = 20.0, selection_size: int = None, **kwargs):
        super().__init__(**kwargs)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta = eta
        self.selection_size = selection_size
        
    def optimize(self, problem_func: Callable, n_gen: int = 100, 
                pop_size: int = 100, n_dim: int = 10, 
                bounds_low: List[float] = None, bounds_high: List[float] = None) -> Dict[str, Any]:
        """运行NSGA-II优化"""
        if bounds_low is None:
            bounds_low = [0.0] * n_dim
        if bounds_high is None:
            bounds_high = [1.0] * n_dim
            
        # 设置工具箱
        toolbox = self._setup_toolbox(problem_func, bounds_low, bounds_high, n_dim)
        
        # 注册遗传算子
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                       low=bounds_low, up=bounds_high, eta=self.eta)
        toolbox.register("mutate", tools.mutPolynomialBounded, 
                        low=bounds_low, up=bounds_high, eta=self.eta, 
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
                
            # 环境选择（合并父代和子代）
            combined = pop + offspring
            if self.selection_size:
                pop = toolbox.select(combined, self.selection_size)
            else:
                pop = toolbox.select(combined, pop_size)
            
            # 记录统计信息
            self._record_statistics(pop, gen)
            
        # 提取Pareto前沿
        pareto_front = tools.sortLogNondominated(pop, k=len(pop), first_front_only=True)
        
        return {
            'population': pop,
            'pareto_front': pareto_front,
            'history': self.history,
            'logbook': self._create_logbook(),
            'metrics': self._calculate_final_metrics(pareto_front)
        }
        
    def _record_statistics(self, population: List, generation: int):
        """记录统计信息"""
        fronts = tools.sortLogNondominated(population, k=len(population), first_front_only=False)
        first_front = fronts[0] if fronts else []
        
        if first_front:
            fitness_values = [ind.fitness.values for ind in first_front]
            avg_fitness = np.mean(fitness_values, axis=0)
            min_fitness = np.min(fitness_values, axis=0)
            max_fitness = np.max(fitness_values, axis=0)
        else:
            avg_fitness = min_fitness = max_fitness = [0.0] * 2
            
        self.history.append({
            'generation': generation,
            'avg_fitness': avg_fitness.tolist(),
            'min_fitness': min_fitness.tolist(),
            'max_fitness': max_fitness.tolist(),
            'population_size': len(population),
            'first_front_size': len(first_front)
        })
        
    def _create_logbook(self):
        """创建日志簿"""
        return {
            'history': self.history,
            'stats': {
                'total_generations': len(self.history),
                'final_population_size': self.history[-1]['population_size'] if self.history else 0
            }
        }
        
    def _calculate_final_metrics(self, pareto_front: List) -> Dict:
        """计算最终性能指标"""
        if not pareto_front:
            return {}
            
        from ..core.metrics import PerformanceMetrics
        front_points = [ind.fitness.values for ind in pareto_front]
        
        return {
            'hypervolume': PerformanceMetrics.hypervolume(front_points),
            'spread': PerformanceMetrics.spread(front_points),
            'pareto_front_size': len(pareto_front)
        }