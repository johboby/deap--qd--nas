"""
基础算法实现
提供轻量级的进化算法基类实现
"""

import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Any
import time


class BaseMultiObjectiveAlgorithm(ABC):
    """多目标优化算法基类"""
    
    def __init__(self, problem, population_size=100):
        self.problem = problem
        self.population_size = population_size
        self.population = None
        self.fitness_values = None
        self.start_time = None
        self.history = {'fitness': [], 'population': []}
    
    @abstractmethod
    def evolve(self, generations=100):
        """进化过程"""
        pass
    
    def evaluate(self, population):
        """评估种群"""
        fitness_values = []
        for individual in population:
            try:
                result = self.problem(individual)
                if isinstance(result, tuple):
                    fitness = result[0]
                else:
                    fitness = result
                fitness_values.append(fitness)
            except:
                fitness_values.append(float('inf'))
        return fitness_values
    
    def select_parents(self, population, fitness_values, num_parents):
        """选择父代"""
        # 锦标赛选择
        parents = []
        for _ in range(num_parents):
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            parents.append(population[winner_idx])
        return parents
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        alpha = random.random()
        child1 = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
        child2 = [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(parent1, parent2)]
        return child1, child2
    
    def mutate(self, individual, mutation_rate=0.1, mutation_strength=0.1):
        """变异操作"""
        mutated = []
        for gene in individual:
            if random.random() < mutation_rate:
                mutated.append(gene + random.gauss(0, mutation_strength))
            else:
                mutated.append(gene)
        return mutated
    
    def get_statistics(self):
        """获取统计信息"""
        if not self.fitness_values:
            return {}
        
        valid_fitness = [f for f in self.fitness_values if f != float('inf')]
        if not valid_fitness:
            return {'best_fitness': float('inf'), 'mean_fitness': float('inf')}
        
        return {
            'best_fitness': min(valid_fitness),
            'mean_fitness': np.mean(valid_fitness),
            'worst_fitness': max(valid_fitness),
            'std_fitness': np.std(valid_fitness)
        }


# 具体算法实现 - 使用正确的类名
class NSGA2(BaseMultiObjectiveAlgorithm):
    """NSGA-II算法实现"""
    
    def __init__(self, problem, population_size=100):
        super().__init__(problem, population_size)
        self.name = "NSGA2"
    
    def evolve(self, generations=100):
        """NSGA-II进化过程"""
        dim = len(self.problem.bounds) if hasattr(self.problem, 'bounds') else 10
        self.population = [[random.uniform(-5, 5) for _ in range(dim)] 
                          for _ in range(self.population_size)]
        
        self.start_time = time.time()
        
        for gen in range(generations):
            # 评估
            self.fitness_values = self.evaluate(self.population)
            
            # 选择父代
            parents = self.select_parents(self.population, self.fitness_values, self.population_size)
            
            # 交叉变异产生子代
            offspring = []
            for i in range(0, len(parents)-1, 2):
                child1, child2 = self.crossover(parents[i], parents[i+1])
                offspring.append(self.mutate(child1))
                offspring.append(self.mutate(child2))
            
            # 环境选择 (简化版)
            combined_pop = self.population + offspring
            combined_fitness = self.evaluate(combined_pop)
            
            # 选择最好的个体
            sorted_indices = np.argsort(combined_fitness)
            self.population = [combined_pop[i] for i in sorted_indices[:self.population_size]]
            self.fitness_values = [combined_fitness[i] for i in sorted_indices[:self.population_size]]
            
            self.history['fitness'].append(self.get_statistics())
        
        return self.get_best_solution()
    
    def get_best_solution(self):
        """获取最佳解"""
        if not self.fitness_values:
            return None
        
        best_idx = np.argmin(self.fitness_values)
        return {
            'solution': self.population[best_idx],
            'fitness': self.fitness_values[best_idx],
            'execution_time': time.time() - self.start_time if self.start_time else 0
        }


class MOEAD(BaseMultiObjectiveAlgorithm):
    """MOEA/D算法实现"""
    
    def __init__(self, problem, population_size=100):
        super().__init__(problem, population_size)
        self.name = "MOEAD"
    
    def evolve(self, generations=100):
        """MOEA/D进化过程"""
        dim = len(self.problem.bounds) if hasattr(self.problem, 'bounds') else 10
        self.population = [[random.uniform(-5, 5) for _ in range(dim)] 
                          for _ in range(self.population_size)]
        
        self.start_time = time.time()
        
        for gen in range(generations):
            # 简化的MOEA/D实现
            self.fitness_values = self.evaluate(self.population)
            
            # 邻域更新
            for i in range(self.population_size):
                # 选择邻居
                neighbors = random.sample(range(self.population_size), min(10, self.population_size))
                
                # 从邻居中选择最佳
                neighbor_fitness = [self.fitness_values[j] for j in neighbors]
                best_neighbor_idx = neighbors[np.argmin(neighbor_fitness)]
                
                # 变异当前个体向最佳邻居靠近
                parent = self.population[best_neighbor_idx]
                child = self.mutate(parent, mutation_rate=0.2, mutation_strength=0.05)
                child_fitness = self.evaluate([child])[0]
                
                if child_fitness < self.fitness_values[i]:
                    self.population[i] = child
                    self.fitness_values[i] = child_fitness
            
            self.history['fitness'].append(self.get_statistics())
        
        return self.get_best_solution()
    
    def get_best_solution(self):
        """获取最佳解"""
        if not self.fitness_values:
            return None
        
        best_idx = np.argmin(self.fitness_values)
        return {
            'solution': self.population[best_idx],
            'fitness': self.fitness_values[best_idx],
            'execution_time': time.time() - self.start_time if self.start_time else 0
        }


class SPEA2(BaseMultiObjectiveAlgorithm):
    """SPEA2算法实现"""
    
    def __init__(self, problem, population_size=100):
        super().__init__(problem, population_size)
        self.name = "SPEA2"
        self.archive_size = population_size
    
    def evolve(self, generations=100):
        """SPEA2进化过程"""
        dim = len(self.problem.bounds) if hasattr(self.problem, 'bounds') else 10
        
        # 初始化种群和存档
        population = [[random.uniform(-5, 5) for _ in range(dim)] 
                     for _ in range(self.population_size)]
        archive = []
        
        self.start_time = time.time()
        
        for gen in range(generations):
            # 合并种群和存档
            combined = population + archive
            combined_fitness = self.evaluate(combined)
            
            # 计算强度值 (简化版)
            strength = []
            for i, fit in enumerate(combined_fitness):
                if fit != float('inf'):
                    # 计算支配的个体数量作为强度
                    dominated_count = sum(1 for other_fit in combined_fitness 
                                        if other_fit != float('inf') and other_fit > fit)
                    strength.append(dominated_count)
                else:
                    strength.append(0)
            
            # 环境选择
            if len(combined) <= self.archive_size:
                new_population = combined
            else:
                # 基于强度的选择
                sorted_indices = np.argsort(strength)[::-1]  # 降序排列
                new_population = [combined[i] for i in sorted_indices[:self.archive_size]]
            
            # 更新种群
            population = new_population[:self.population_size]
            archive = new_population[self.population_size:] if len(new_population) > self.population_size else []
            
            # 变异产生下一代
            offspring = []
            for indiv in population:
                child = self.mutate(indiv, mutation_rate=0.15, mutation_strength=0.1)
                offspring.append(child)
            
            population = offspring
            self.fitness_values = self.evaluate(population)
            self.population = population
            
            self.history['fitness'].append(self.get_statistics())
        
        return self.get_best_solution()
    
    def get_best_solution(self):
        """获取最佳解"""
        if not self.fitness_values:
            return None
        
        best_idx = np.argmin(self.fitness_values)
        return {
            'solution': self.population[best_idx],
            'fitness': self.fitness_values[best_idx],
            'execution_time': time.time() - self.start_time if self.start_time else 0
        }


class IBEA(BaseMultiObjectiveAlgorithm):
    """IBEA算法实现"""
    
    def __init__(self, problem, population_size=100):
        super().__init__(problem, population_size)
        self.name = "IBEA"
        self.kappa = 0.05  # 缩放参数
    
    def evolve(self, generations=100):
        """IBEA进化过程"""
        dim = len(self.problem.bounds) if hasattr(self.problem, 'bounds') else 10
        self.population = [[random.uniform(-5, 5) for _ in range(dim)] 
                          for _ in range(self.population_size)]
        
        self.start_time = time.time()
        
        for gen in range(generations):
            self.fitness_values = self.evaluate(self.population)
            
            # 计算适应度 (基于指标的适应度分配)
            indicator_fitness = self._calculate_indicator_fitness()
            
            # 环境选择
            if len(self.population) > self.population_size:
                # 移除最差的个体
                worst_indices = np.argsort(indicator_fitness)[::-1]  # 降序排列
                keep_indices = worst_indices[:self.population_size]
                self.population = [self.population[i] for i in keep_indices]
                self.fitness_values = [self.fitness_values[i] for i in keep_indices]
            
            # 产生子代
            offspring = []
            for _ in range(self.population_size):
                parents = self.select_parents(self.population, self.fitness_values, 2)
                child1, child2 = self.crossover(parents[0], parents[1])
                offspring.append(self.mutate(child1))
            
            self.population = offspring
            self.history['fitness'].append(self.get_statistics())
        
        return self.get_best_solution()
    
    def _calculate_indicator_fitness(self):
        """计算基于指标的适应度"""
        # 简化的IBEA适应度计算
        fitness = []
        for i, indiv_fitness in enumerate(self.fitness_values):
            if indiv_fitness == float('inf'):
                fitness.append(float('-inf'))
                continue
            
            # 计算与其他个体的距离作为适应度指标
            distances = []
            for j, other_fitness in enumerate(self.fitness_values):
                if i != j and other_fitness != float('inf'):
                    distance = abs(indiv_fitness - other_fitness)
                    distances.append(distance)
            
            if distances:
                # 基于距离的适应度 (距离越大越好)
                avg_distance = np.mean(distances)
                indicator_fit = avg_distance / (1 + np.exp(-self.kappa * avg_distance))
                fitness.append(indicator_fit)
            else:
                fitness.append(0)
        
        return fitness
    
    def get_best_solution(self):
        """获取最佳解"""
        if not self.fitness_values:
            return None
        
        best_idx = np.argmin(self.fitness_values)
        return {
            'solution': self.population[best_idx],
            'fitness': self.fitness_values[best_idx],
            'execution_time': time.time() - self.start_time if self.start_time else 0
        }


class ClassicalEvolution(BaseMultiObjectiveAlgorithm):
    """经典进化算法实现"""
    
    def __init__(self, problem, population_size=100):
        super().__init__(problem, population_size)
        self.name = "ClassicalEvolution"
    
    def evolve(self, generations=100):
        """经典进化过程"""
        dim = len(self.problem.bounds) if hasattr(self.problem, 'bounds') else 10
        self.population = [[random.uniform(-5, 5) for _ in range(dim)] 
                          for _ in range(self.population_size)]
        
        self.start_time = time.time()
        
        for gen in range(generations):
            self.fitness_values = self.evaluate(self.population)
            
            # 轮盘赌选择
            probabilities = []
            valid_fitness = [f for f in self.fitness_values if f != float('inf')]
            if valid_fitness:
                max_fitness = max(valid_fitness)
                for fit in self.fitness_values:
                    if fit == float('inf'):
                        probabilities.append(0)
                    else:
                        probabilities.append((max_fitness - fit + 1e-6) / (max_fitness + 1e-6))
                
                # 归一化
                prob_sum = sum(probabilities)
                if prob_sum > 0:
                    probabilities = [p / prob_sum for p in probabilities]
            else:
                probabilities = [1.0/len(self.fitness_values)] * len(self.fitness_values)
            
            # 选择父代
            parent_indices = np.random.choice(
                len(self.population), size=self.population_size, p=probabilities
            )
            parents = [self.population[i] for i in parent_indices]
            
            # 交叉变异
            offspring = []
            for i in range(0, len(parents)-1, 2):
                child1, child2 = self.crossover(parents[i], parents[i+1])
                offspring.append(self.mutate(child1))
                offspring.append(self.mutate(child2))
            
            # 如果种群大小为奇数，添加一个额外的子代
            if len(parents) % 2 == 1 and offspring:
                extra_child = self.mutate(parents[-1])
                offspring.append(extra_child)
            
            self.population = offspring[:self.population_size]
            self.history['fitness'].append(self.get_statistics())
        
        return self.get_best_solution()
    
    def get_best_solution(self):
        """获取最佳解"""
        if not self.fitness_values:
            return None
        
        best_idx = np.argmin(self.fitness_values)
        return {
            'solution': self.population[best_idx],
            'fitness': self.fitness_values[best_idx],
            'execution_time': time.time() - self.start_time if self.start_time else 0
        }


# 导出正确的类名供外部使用
__all__ = ['BaseMultiObjectiveAlgorithm', 'NSGA2', 'MOEAD', 'SPEA2', 'IBEA', 'ClassicalEvolution']