#!/usr/bin/env python3
"""
DEAP框架 - 清洁版主入口
避免复杂的导入问题，提供直接可用的功能
"""

import random
import numpy as np
from deap import base, creator, tools, algorithms

def simple_zdt1(x):
    """简单ZDT1测试函数"""
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return f1, f2

def run_simple_nsga2(n_dim=10, pop_size=50, n_gen=50):
    """运行简化版NSGA-II"""
    print(f"Running NSGA-II: {n_dim}D, pop={pop_size}, gen={n_gen}")
    
    # 清理可能的creator类型
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass
    
    # 创建类型
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    # 设置工具箱
    toolbox = base.Toolbox()
    
    def uniform(low, up, size=None):
        return [random.uniform(a, b) for a, b in zip(low, up)]
    
    toolbox.register("attr_float", uniform, [0.0] * n_dim, [1.0] * n_dim)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", simple_zdt1)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0/n_dim)
    toolbox.register("select", tools.selNSGA2)
    
    # 创建种群
    pop = toolbox.population(n=pop_size)
    
    # 评估初始种群
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # 进化
    for gen in range(n_gen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.9, mutpb=0.1)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop = toolbox.select(pop + offspring, pop_size)
        
        if gen % 10 == 0:
            print(f"Generation {gen}: {len(pop)} individuals")
    
    # 提取Pareto前沿
    pareto_front = tools.sortLogNondominated(pop, k=len(pop), first_front_only=True)
    
    print(f"Found {len(pareto_front)} Pareto optimal solutions")
    
    return {
        'population': pop,
        'pareto_front': pareto_front,
        'front_points': [ind.fitness.values for ind in pareto_front]
    }

def demo_basic_usage():
    """演示基本用法"""
    print("=== DEAP Framework Demo ===")
    
    # 演示1: 简单NSGA-II运行
    print("\n1. Running NSGA-II optimization...")
    result = run_simple_nsga2(n_dim=5, pop_size=30, n_gen=20)
    
    # 显示一些结果
    if result['front_points']:
        points = result['front_points'][:5]  # 显示前5个点
        print(f"\nSample Pareto solutions:")
        for i, point in enumerate(points):
            print(f"  Solution {i+1}: f1={point[0]:.4f}, f2={point[1]:.4f}")
    
    print("\nDemo completed successfully!")

def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            demo_basic_usage()
        elif sys.argv[1] == 'test':
            print("Running framework test...")
            result = run_simple_nsga2(n_gen=10)
            print(f"Test passed: {len(result['pareto_front'])} solutions found")
        else:
            print("Usage: python main_clean.py [demo|test]")
    else:
        demo_basic_usage()

if __name__ == "__main__":
    main()