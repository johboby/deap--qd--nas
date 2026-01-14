"""
多目标优化测试函数库
包含ZDT、DTLZ、WFG系列函数和约束测试函数
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Any
import math


class TestFunctionLibrary:
    """测试函数库 - 统一的测试函数接口"""

    # ===== ZDT系列测试函数 =====
    
    @staticmethod
    def zdt1(x: List[float]) -> Tuple[float, float]:
        """
        ZDT1测试函数 - 凸Pareto前沿
        
        参考文献: Zitzler, E., et al. (2000). Comparison of multiobjective evolutionary algorithms.
        """
        n = len(x)
        f1 = x[0]
        g = 1 + 9 * sum(x[1:]) / (n - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return f1, f2
    
    @staticmethod
    def zdt2(x: List[float]) -> Tuple[float, float]:
        """ZDT2测试函数 - 非凸Pareto前沿"""
        n = len(x)
        f1 = x[0]
        g = 1 + 9 * sum(x[1:]) / (n - 1)
        f2 = g * (1 - (f1 / g) ** 2)
        return f1, f2
    
    @staticmethod
    def zdt3(x: List[float]) -> Tuple[float, float]:
        """ZDT3测试函数 - 断续的Pareto前沿"""
        n = len(x)
        f1 = x[0]
        g = 1 + 9 * sum(x[1:]) / (n - 1)
        f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
        return f1, f2
    
    @staticmethod
    def zdt4(x: List[float]) -> Tuple[float, float]:
        """ZDT4测试函数 - 多模态（21^9个局部Pareto前沿）"""
        n = len(x)
        f1 = x[0]
        g = 1 + 10 * (n - 1) + sum(xi ** 2 - 10 * np.cos(4 * np.pi * xi) for xi in x[1:])
        f2 = g * (1 - np.sqrt(f1 / g))
        return f1, f2
    
    @staticmethod
    def zdt6(x: List[float]) -> Tuple[float, float]:
        """ZDT6测试函数 - 非均匀搜索空间"""
        n = len(x)
        f1 = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0]) ** 6
        g = 1 + 9 * (sum(x[1:]) / (n - 1)) ** 0.25
        f2 = g * (1 - (f1 / g) ** 2)
        return f1, f2

    # ===== DTLZ系列测试函数 =====
    
    @staticmethod
    def dtlz1(x: List[float], n_obj: int = 2) -> Tuple:
        """
        DTLZ1测试函数 - 线性Pareto前沿
        
        参考文献: Deb, K., et al. (2002). Scalable multi-objective optimization test problems.
        """
        n = len(x)
        k = n - n_obj + 1
        g = 100 * (k + sum((xi - 0.5) ** 2 - np.cos(20 * np.pi * (xi - 0.5)) for xi in x[n_obj-1:]))
        
        objectives = []
        for i in range(n_obj):
            prod = 0.5 * (1 + g)
            for j in range(n_obj - i - 1):
                prod *= x[j]
            if i > 0:
                prod *= (1 - x[n_obj - i - 1])
            objectives.append(prod)
        
        return tuple(objectives)
    
    @staticmethod
    def dtlz2(x: List[float], n_obj: int = 2) -> Tuple:
        """DTLZ2测试函数 - 球形Pareto前沿"""
        n = len(x)
        k = n - n_obj + 1
        g = sum((xi - 0.5) ** 2 for xi in x[n_obj-1:])
        
        objectives = []
        for i in range(n_obj):
            theta = []
            for j in range(n_obj - 1):
                theta.append(x[j])
            if i > 0:
                theta.append(0.5 * (1 + g))
                
            cos_prod = 1.0
            for t in theta[:n_obj-i-1]:
                cos_prod *= np.cos(t * np.pi / 2)
                
            if i == 0:
                objective = (1 + g) * cos_prod
            else:
                sin_term = np.sin(x[n_obj - i - 1] * np.pi / 2)
                objective = (1 + g) * cos_prod * sin_term
                
            objectives.append(objective)
            
        return tuple(objectives)
    
    @staticmethod
    def dtlz3(x: List[float], n_obj: int = 2) -> Tuple:
        """DTLZ3测试函数 - 多模态"""
        n = len(x)
        k = n - n_obj + 1
        g = 100 * (k + sum((xi - 0.5) ** 2 - np.cos(20 * np.pi * (xi - 0.5)) for xi in x[n_obj-1:]))
        
        objectives = []
        for i in range(n_obj):
            prod = (1 + g)
            for j in range(n_obj - i - 1):
                prod *= np.cos(x[j] * np.pi / 2)
            if i > 0:
                prod *= np.sin(x[n_obj - i - 1] * np.pi / 2)
            objectives.append(prod)
            
        return tuple(objectives)
    
    @staticmethod
    def dtlz4(x: List[float], n_obj: int = 2, alpha: float = 100) -> Tuple:
        """DTLZ4测试函数 - 可变偏置"""
        n = len(x)
        k = n - n_obj + 1
        g = sum((xi - 0.5) ** 2 for xi in x[n_obj-1:])
        
        objectives = []
        for i in range(n_obj):
            xx = [x[j] ** alpha for j in range(n_obj - 1)]
            if i > 0:
                xx.append(x[n_obj - i - 1] ** alpha)
                
            cos_prod = 1.0
            for t in xx[:n_obj-i-1]:
                cos_prod *= np.cos(t * np.pi / 2)
                
            if i == 0:
                objective = (1 + g) * cos_prod
            else:
                sin_term = np.sin(x[n_obj - i - 1] ** alpha * np.pi / 2)
                objective = (1 + g) * cos_prod * sin_term
                
            objectives.append(objective)
            
        return tuple(objectives)
    
    @staticmethod
    def dtlz5(x: List[float], n_obj: int = 2) -> Tuple:
        """DTLZ5测试函数 - 参数化偏置"""
        n = len(x)
        k = n - n_obj + 1
        
        g = sum((xi - 0.5) ** 2 for xi in x[n_obj-1:])
        
        theta = []
        for i in range(n_obj - 1):
            if g > 0:
                theta_i = x[i] * np.pi / (4 * (1 + g))
            else:
                theta_i = x[i] * np.pi / 4
            theta.append(theta_i)
            
        objectives = []
        for i in range(n_obj):
            prod = 1.0
            for j in range(n_obj - i - 1):
                prod *= np.cos(theta[j])
            if i > 0:
                prod *= np.sin(theta[n_obj - i - 1])
            objectives.append((1 + g) * prod)
            
        return tuple(objectives)
    
    @staticmethod
    def dtlz6(x: List[float], n_obj: int = 2) -> Tuple:
        """DTLZ6测试函数 - 简化版本"""
        n = len(x)
        k = n - n_obj + 1
        g = sum(x[n_obj-1:] ** 0.1)
        
        theta = []
        for i in range(n_obj - 1):
            theta_i = (np.pi / (4 * (1 + g))) * x[i]
            theta.append(theta_i)
            
        objectives = []
        for i in range(n_obj):
            prod = 1.0
            for j in range(n_obj - i - 1):
                prod *= np.cos(theta[j])
            if i > 0:
                prod *= np.sin(theta[n_obj - i - 1])
            objectives.append((1 + g) * prod)
            
        return tuple(objectives)
    
    @staticmethod
    def dtlz7(x: List[float], n_obj: int = 2) -> Tuple:
        """DTLZ7测试函数 - 混合特征"""
        n = len(x)
        
        objectives = []
        for i in range(n_obj - 1):
            objectives.append(x[i])
            
        g = 1 + 9 * sum(x[n_obj-1:]) / (n - n_obj + 1)
        h = n_obj - sum(xi / (1 + g) * (1 + np.sin(3 * np.pi * xi)) for xi in objectives)
        f_n = (1 + g) * h
        
        objectives.append(f_n)
        return tuple(objectives)

    # ===== WFG系列测试函数 =====
    
    @staticmethod
    def wfg1(x: List[float], n_obj: int = 2) -> Tuple:
        """WFG1测试函数 - 基于形状函数的多目标问题"""
        n = len(x)
        k = n - n_obj + 1
        
        t1 = []
        for i in range(k):
            t1.append(x[i])
        for i in range(k, n):
            t1.append(x[i])
            
        g = 1 + 9 * sum(t1[k:]) / (n - k)
        
        objectives = []
        for i in range(n_obj):
            if i == 0:
                objectives.append(t1[0] * g)
            else:
                prod = 1.0
                for j in range(i):
                    prod *= np.sin(t1[j] * np.pi / 2)
                if i > 0:
                    prod *= np.cos(t1[i] * np.pi / 2)
                objectives.append((1 + g) * prod)
                
        return tuple(objectives)

    # ===== 约束测试函数 =====
    
    @staticmethod
    def constrained_zdt1(x: List[float]) -> Tuple[Tuple[float, float], List[float]]:
        """带约束的ZDT1"""
        f1, f2 = TestFunctionLibrary.zdt1(x)
        
        constraint1 = 1.2 - (x[0] + x[1])
        constraint2 = sum(x) - 0.5
        
        constraints = [constraint1, constraint2]
        return (f1, f2), constraints
    
    @staticmethod
    def srn_constrained(x: List[float]) -> Tuple[Tuple[float, float], List[float]]:
        """SRN约束测试函数"""
        f1 = 2 + (x[0] - 2) ** 2 + (x[1] - 1) ** 2
        f2 = 9 * x[0] - (x[1] - 1) ** 2
        
        constraint1 = 225 - (x[0] ** 2 + x[1] ** 2)
        constraint2 = 7 - (x[0] - 3) ** 2 - x[1]
        
        constraints = [constraint1, constraint2]
        return (f1, f2), constraints
    
    @staticmethod
    def bnh_constrained(x: List[float]) -> Tuple[Tuple[float, float], List[float]]:
        """BNH约束测试函数"""
        f1 = 4 * x[0] ** 2 + 4 * x[1] ** 2
        f2 = (x[0] - 5) ** 2 + (x[1] - 5) ** 2
        
        constraint1 = 25 - (x[0] - 5) ** 2 - x[1] ** 2
        constraint2 = 7.7 - (x[0] - 8) ** 2 - (x[1] + 3) ** 2
        
        constraints = [constraint1, constraint2]
        return (f1, f2), constraints
    
    @staticmethod
    def circle_constrained(x: List[float]) -> Tuple[Tuple[float, float], List[float]]:
        """圆形约束测试函数"""
        f1 = x[0] ** 2 + x[1] ** 2
        f2 = (x[0] - 2) ** 2 + (x[1] - 2) ** 2
        
        constraint = x[0] ** 2 + x[1] ** 2 - 1
        
        constraints = [constraint]
        return (f1, f2), constraints
    
    @staticmethod
    def triangle_constrained(x: List[float]) -> Tuple[Tuple[float, float], List[float]]:
        """三角形约束测试函数"""
        f1 = x[0]
        f2 = x[1]
        
        constraint1 = x[0] + x[1] - 1
        constraint2 = 0.5 - x[0]
        constraint3 = 0.5 - x[1]
        
        constraints = [constraint1, constraint2, constraint3]
        return (f1, f2), constraints

    # ===== 单目标测试函数 =====
    
    @staticmethod
    def sphere(x: List[float]) -> float:
        """Sphere单目标测试函数"""
        return sum(xi ** 2 for xi in x)
    
    @staticmethod
    def rastrigin(x: List[float]) -> float:
        """Rastrigin单目标测试函数 - 多模态"""
        A = 10
        n = len(x)
        return A * n + sum(xi ** 2 - A * np.cos(2 * np.pi * xi) for xi in x)
    
    @staticmethod
    def rosenbrock(x: List[float]) -> float:
        """Rosenbrock单目标测试函数 - 香蕉函数"""
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
    
    @staticmethod
    def ackley(x: List[float]) -> float:
        """Ackley单目标测试函数 - 多峰"""
        n = len(x)
        sum_sq = sum(xi ** 2 for xi in x)
        sum_cos = sum(np.cos(2 * np.pi * xi) for xi in x)
        
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        
        return term1 + term2 + 20 + np.e
    
    @staticmethod
    def griewank(x: List[float]) -> float:
        """Griewank单目标测试函数"""
        sum_sq = sum(xi ** 2 for xi in x) / 4000
        prod = 1.0
        for i, xi in enumerate(x, 1):
            prod *= np.cos(xi / np.sqrt(i))
        
        return sum_sq - prod + 1

    # ===== 动态优化测试函数 =====
    
    @staticmethod
    def dynamic_zdt1(x: List[float], time: int = 0) -> Tuple[float, float]:
        """动态ZDT1 - 随时间变化的测试函数"""
        phase_shift = 0.1 * np.sin(time * 0.1)
        
        n = len(x)
        f1 = x[0] + phase_shift
        g = 1 + 9 * sum(x[1:]) / (n - 1)
        f2 = g * (1 - np.sqrt(max(0, f1 - phase_shift) / g))
        
        return f1, f2
    
    @staticmethod
    def deceptive_function(x: List[float]) -> Tuple[float, float]:
        """欺骗性测试函数 - 具有误导性的局部最优"""
        f1 = sum((xi - 0.5) ** 2 for xi in x)
        f2 = sum((xi - 0.8) ** 2 for xi in x)
        
        if 0.3 < x[0] < 0.7 and 0.3 < x[1] < 0.7:
            f1 *= 0.1
            f2 *= 0.1
            
        return f1, f2


# ===== 测试函数集合 =====

# ZDT函数集合
ZDT_FUNCTIONS = {
    'zdt1': TestFunctionLibrary.zdt1,
    'zdt2': TestFunctionLibrary.zdt2, 
    'zdt3': TestFunctionLibrary.zdt3,
    'zdt4': TestFunctionLibrary.zdt4,
    'zdt6': TestFunctionLibrary.zdt6
}

# DTLZ函数集合
DTLZ_FUNCTIONS = {
    'dtlz1': lambda x: TestFunctionLibrary.dtlz1(x, 2),
    'dtlz2': lambda x: TestFunctionLibrary.dtlz2(x, 2),
    'dtlz3': lambda x: TestFunctionLibrary.dtlz3(x, 2),
    'dtlz4': lambda x: TestFunctionLibrary.dtlz4(x, 2),
    'dtlz5': lambda x: TestFunctionLibrary.dtlz5(x, 2),
    'dtlz6': lambda x: TestFunctionLibrary.dtlz6(x, 2),
    'dtlz7': lambda x: TestFunctionLibrary.dtlz7(x, 2)
}

# WFG函数集合
WFG_FUNCTIONS = {
    'wfg1': TestFunctionLibrary.wfg1
}

# 约束函数集合
CONSTRAINED_FUNCTIONS = {
    'constrained_zdt1': TestFunctionLibrary.constrained_zdt1,
    'srn_constrained': TestFunctionLibrary.srn_constrained,
    'bnh_constrained': TestFunctionLibrary.bnh_constrained,
    'circle_constrained': TestFunctionLibrary.circle_constrained,
    'triangle_constrained': TestFunctionLibrary.triangle_constrained
}

# 单目标函数集合
SINGLE_OBJECTIVE_FUNCTIONS = {
    'sphere': TestFunctionLibrary.sphere,
    'rastrigin': TestFunctionLibrary.rastrigin,
    'rosenbrock': TestFunctionLibrary.rosenbrock,
    'ackley': TestFunctionLibrary.ackley,
    'griewank': TestFunctionLibrary.griewank
}

# 动态函数集合
DYNAMIC_FUNCTIONS = {
    'dynamic_zdt1': TestFunctionLibrary.dynamic_zdt1,
    'deceptive_function': TestFunctionLibrary.deceptive_function
}

# 所有函数集合
ALL_TEST_FUNCTIONS = {
    **ZDT_FUNCTIONS,
    **DTLZ_FUNCTIONS,
    **WFG_FUNCTIONS,
    **CONSTRAINED_FUNCTIONS,
    **SINGLE_OBJECTIVE_FUNCTIONS,
    **DYNAMIC_FUNCTIONS
}


def get_test_function(name: str) -> Callable:
    """获取测试函数"""
    if name in ALL_TEST_FUNCTIONS:
        return ALL_TEST_FUNCTIONS[name]
    else:
        raise ValueError(f"Unknown test function: {name}")


def list_test_functions(category: str = None) -> List[str]:
    """列出测试函数"""
    if category is None:
        return list(ALL_TEST_FUNCTIONS.keys())
    elif category == 'zdt':
        return list(ZDT_FUNCTIONS.keys())
    elif category == 'dtlz':
        return list(DTLZ_FUNCTIONS.keys())
    elif category == 'wfg':
        return list(WFG_FUNCTIONS.keys())
    elif category == 'constrained':
        return list(CONSTRAINED_FUNCTIONS.keys())
    elif category == 'single':
        return list(SINGLE_OBJECTIVE_FUNCTIONS.keys())
    elif category == 'dynamic':
        return list(DYNAMIC_FUNCTIONS.keys())
    else:
        raise ValueError(f"Unknown category: {category}")
