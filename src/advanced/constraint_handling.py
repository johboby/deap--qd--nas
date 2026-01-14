"""
约束处理模块
提供多种约束处理方法用于多目标优化
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Any
from abc import ABC, abstractmethod

class ConstraintHandler(ABC):
    """约束处理器基类"""
    
    @abstractmethod
    def handle_constraints(self, individual: List[float], problem_func: Callable) -> Tuple[List[float], bool]:
        """处理约束违规
        
        Args:
            individual: 候选解
            problem_func: 问题函数（可能返回约束违反值）
            
        Returns:
            Tuple[修正后的个体, 是否满足约束]
        """
        pass

class PenaltyMethod(ConstraintHandler):
    """罚函数法"""
    
    def __init__(self, penalty_factor: float = 1e6):
        self.penalty_factor = penalty_factor
        
    def handle_constraints(self, individual: List[float], problem_func: Callable) -> Tuple[List[float], bool]:
        """使用罚函数处理约束"""
        try:
            # 假设问题函数返回 (objectives, constraints) 或仅 objectives
            result = problem_func(individual)
            
            if isinstance(result, tuple) and len(result) == 2:
                objectives, constraints = result
                # 检查约束违反
                violation = sum(max(0, c) for c in constraints)
                if violation > 0:
                    # 应用罚函数
                    penalized_objectives = [obj + self.penalty_factor * violation for obj in objectives]
                    return individual, False
                return individual, True
            else:
                # 无约束问题
                return individual, True
        except Exception:
            return individual, False

class BarrierMethod(ConstraintHandler):
    """障碍函数法"""
    
    def __init__(self, barrier_coefficient: float = 1.0):
        self.barrier_coefficient = barrier_coefficient
        
    def handle_constraints(self, individual: List[float], problem_func: Callable) -> Tuple[List[float], bool]:
        """使用障碍函数处理约束"""
        try:
            result = problem_func(individual)
            
            if isinstance(result, tuple) and len(result) == 2:
                objectives, constraints = result
                # 检查约束违反
                violation = sum(max(0, c) for c in constraints)
                if violation > 0:
                    # 应用障碍函数（阻止搜索进入不可行域）
                    barrier_penalty = self.barrier_coefficient / (violation + 1e-10)
                    penalized_objectives = [obj + barrier_penalty for obj in objectives]
                    return individual, False
                return individual, True
            else:
                return individual, True
        except Exception:
            return individual, False

class RepairMethod(ConstraintHandler):
    """修复方法"""
    
    def __init__(self, repair_threshold: float = 0.01):
        self.repair_threshold = repair_threshold
        
    def handle_constraints(self, individual: List[float], problem_func: Callable) -> Tuple[List[float], bool]:
        """尝试修复约束违规的解"""
        try:
            result = problem_func(individual)
            
            if isinstance(result, tuple) and len(result) == 2:
                objectives, constraints = result
                violation = sum(max(0, c) for c in constraints)
                
                if violation > self.repair_threshold:
                    # 简单的投影修复（需要根据具体问题定制）
                    repaired = self._repair_individual(individual, constraints)
                    return repaired, True
                return individual, True
            else:
                return individual, True
        except Exception:
            return individual, False
    
    def _repair_individual(self, individual: List[float], constraints: List[float]) -> List[float]:
        """修复个体（需要根据具体问题实现）"""
        # 这里提供一个通用的边界修复
        repaired = []
        for val in individual:
            # 简单的边界约束修复
            if val < 0:
                repaired.append(0.0)
            elif val > 1:
                repaired.append(1.0)
            else:
                repaired.append(val)
        return repaired

class DebConstraintHandling(ConstraintHandler):
    """Deb约束处理方法"""
    
    def handle_constraints(self, individual: List[float], problem_func: Callable) -> Tuple[List[float], bool]:
        """Deb约束处理（可行性规则）"""
        try:
            result = problem_func(individual)
            
            if isinstance(result, tuple) and len(result) == 2:
                objectives, constraints = result
                
                # 可行性规则：可行解优于不可行解，约束违反小的优于大的
                feasible = all(c <= 0 for c in constraints)
                total_violation = sum(max(0, c) for c in constraints)
                
                # 返回一个标记可行性的特殊目标
                extended_objectives = list(objectives) + [total_violation]
                return individual, feasible
            else:
                return individual, True
        except Exception:
            return individual, False

# 预定义的约束处理器
CONSTRAINT_HANDLERS = {
    'penalty': PenaltyMethod,
    'barrier': BarrierMethod,
    'repair': RepairMethod,
    'deb': DebConstraintHandling
}

def create_constraint_handler(method: str = 'penalty', **kwargs) -> ConstraintHandler:
    """创建约束处理器"""
    if method not in CONSTRAINT_HANDLERS:
        raise ValueError(f"Unknown constraint handling method: {method}. "
                        f"Available methods: {list(CONSTRAINT_HANDLERS.keys())}")
    return CONSTRAINT_HANDLERS[method](**kwargs)