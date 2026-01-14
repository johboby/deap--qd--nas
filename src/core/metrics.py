"""
多目标优化性能评估指标
"""

import numpy as np
from typing import List, Tuple, Callable
from scipy.spatial.distance import cdist

class PerformanceMetrics:
    """性能评估指标计算器"""
    
    @staticmethod
    def hypervolume(front: List[Tuple], reference_point: Tuple = None) -> float:
        """计算超体积指标"""
        if not front:
            return 0.0
            
        # 简化的超体积计算
        front_array = np.array(front)
        if reference_point is None:
            reference_point = np.max(front_array, axis=0) + 1
            
        # 对于2目标问题，使用矩形面积近似
        if len(reference_point) == 2:
            sorted_front = sorted(front, key=lambda x: x[0])
            volume = 0.0
            prev_x = 0.0
            
            for point in sorted_front:
                width = reference_point[0] - point[0]
                height = reference_point[1] - point[1]
                volume += width * height
                prev_x = point[0]
                
            return volume
            
        return 0.0  # 高维情况暂不支持
    
    @staticmethod
    def igd(true_pareto: List[Tuple], approx_pareto: List[Tuple]) -> float:
        """计算倒世代距离(Inverted Generational Distance)"""
        if not true_pareto or not approx_pareto:
            return float('inf')
            
        distances = []
        for tp in true_pareto:
            min_dist = min(np.linalg.norm(np.array(tp) - np.array(ap)) 
                          for ap in approx_pareto)
            distances.append(min_dist)
            
        return np.mean(distances)
    
    @staticmethod
    def spread(front: List[Tuple]) -> float:
        """计算分布均匀性(Spread)"""
        if len(front) <= 2:
            return 0.0
            
        front_array = np.array(front)
        distances = []
        
        # 计算相邻点之间的距离
        sorted_front = front_array[np.argsort(front_array[:, 0])]
        for i in range(len(sorted_front) - 1):
            dist = np.linalg.norm(sorted_front[i+1] - sorted_front[i])
            distances.append(dist)
            
        if not distances:
            return 0.0
            
        d_avg = np.mean(distances)
        d_f = np.linalg.norm(sorted_front[0] - sorted_front[1])
        d_l = np.linalg.norm(sorted_front[-1] - sorted_front[-2])
        
        spread_value = (sum(abs(d_i - d_avg) for d_i in distances) + abs(d_f - d_avg) + abs(d_l - d_avg)) \
                     / (sum(distances) + d_f + d_l)
                      
        return spread_value
    
    @staticmethod
    def generational_distance(front: List[Tuple], true_pareto: List[Tuple]) -> float:
        """计算世代距离(Generational Distance)"""
        if not front or not true_pareto:
            return float('inf')
            
        distances = []
        for ap in front:
            min_dist = min(np.linalg.norm(np.array(ap) - np.array(tp)) 
                          for tp in true_pareto)
            distances.append(min_dist)
            
        return np.sqrt(np.mean(np.array(distances) ** 2))
    
    @staticmethod
    def epsilon_indicator(front: List[Tuple], reference_front: List[Tuple]) -> float:
        """计算epsilon指标"""
        if not front or not reference_front:
            return float('inf')
            
        epsilons = []
        for ref_point in reference_front:
            min_ratio = min(max(ref_point[i] / point[i] if point[i] > 0 else float('inf') 
                              for i in range(len(point))) 
                          for point in front)
            epsilons.append(min_ratio)
            
        return max(epsilons) if epsilons else float('inf')
    
    @staticmethod
    def calculate_all_metrics(front: List[Tuple], reference_set: List[Tuple] = None) -> dict:
        """计算所有可用指标"""
        metrics = {}
        
        if reference_set:
            metrics['igd'] = PerformanceMetrics.igd(reference_set, front)
            metrics['gd'] = PerformanceMetrics.generational_distance(front, reference_set)
            
        metrics['hypervolume'] = PerformanceMetrics.hypervolume(front)
        metrics['spread'] = PerformanceMetrics.spread(front)
        
        return metrics