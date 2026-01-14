"""
结构优化应用
提供桁架结构、框架结构等的多目标优化
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Any
from dataclasses import dataclass

from ...core.base_algorithms import BaseMultiObjectiveAlgorithm
from ...core.test_functions import TestFunctionLibrary

@dataclass
class MaterialProperties:
    """材料属性"""
    density: float  # 密度 (kg/m³)
    young_modulus: float  # 杨氏模量 (Pa)
    yield_strength: float  # 屈服强度 (Pa)
    poisson_ratio: float = 0.3  # 泊松比

@dataclass
class LoadCase:
    """载荷工况"""
    forces: List[Tuple[float, float, float]]  # 力向量列表 [(Fx, Fy, Fz), ...]
    positions: List[Tuple[float, float, float]]  # 作用位置 [(x, y, z), ...]
    load_factor: float = 1.0  # 载荷因子

class TrussStructure:
    """桁架结构类"""
    
    def __init__(self, nodes: List[Tuple[float, float, float]],
                 elements: List[Tuple[int, int]],
                 material: MaterialProperties):
        self.nodes = nodes
        self.elements = elements
        self.material = material
        self.n_nodes = len(nodes)
        self.n_elements = len(elements)
        
    def analyze_displacements(self, cross_sections: List[float], 
                            loads: LoadCase) -> List[float]:
        """简化的位移分析（梁单元模型）"""
        # 这是一个简化的分析，实际应用中应使用有限元分析
        displacements = []
        
        for i, node in enumerate(self.nodes):
            # 简化的位移计算：基于节点受力和截面刚度
            node_force_sum = np.zeros(3)
            
            # 累加作用在该节点的力
            for force, pos in zip(loads.forces, loads.positions):
                if np.allclose(pos, node, atol=1e-6):
                    node_force_sum += np.array(force) * loads.load_factor
            
            # 简化的刚度计算
            stiffness = self._calculate_node_stiffness(i, cross_sections)
            
            # 位移 = 力 / 刚度
            if stiffness > 1e-10:
                displacement = node_force_sum / stiffness
            else:
                displacement = node_force_sum * 1e-6  # 极小刚度情况下的近似
                
            displacements.extend(displacement)
            
        return displacements
    
    def _calculate_node_stiffness(self, node_idx: int, cross_sections: List[float]) -> float:
        """计算节点刚度（简化）"""
        # 找到与该节点相连的所有杆件
        connected_elements = []
        for elem_idx, (i, j) in enumerate(self.elements):
            if i == node_idx or j == node_idx:
                connected_elements.append(elem_idx)
                
        # 简化的刚度计算：基于相连杆件的截面和长度
        total_stiffness = 0.0
        for elem_idx in connected_elements:
            if elem_idx < len(cross_sections):
                # 简化的轴向刚度 EA/L
                i, j = self.elements[elem_idx]
                length = np.linalg.norm(np.array(self.nodes[i]) - np.array(self.nodes[j]))
                area = cross_sections[elem_idx]
                
                if length > 0:
                    stiffness = (self.material.young_modulus * area) / length
                    total_stiffness += stiffness
                    
        return total_stiffness
    
    def calculate_volume(self, cross_sections: List[float]) -> float:
        """计算结构总体积"""
        total_volume = 0.0
        
        for elem_idx, (i, j) in enumerate(self.elements):
            if elem_idx < len(cross_sections):
                length = np.linalg.norm(np.array(self.nodes[i]) - np.array(self.nodes[j]))
                volume = cross_sections[elem_idx] * length
                total_volume += volume
                
        # 转换为质量
        return total_volume * self.material.density

class StructuralOptimization:
    """结构优化器"""
    
    def __init__(self):
        self.truss_structures = {}
        
    def define_truss_problem(self, name: str, nodes: List[Tuple[float, float, float]],
                           elements: List[Tuple[int, int]],
                           material: MaterialProperties,
                           loads: LoadCase,
                           design_vars_bounds: Tuple[float, float] = (0.001, 0.1)):
        """定义桁架优化问题"""
        structure = TrussStructure(nodes, elements, material)
        self.truss_structures[name] = {
            'structure': structure,
            'loads': loads,
            'design_vars_bounds': design_vars_bounds
        }
        
    def weight_minimization_objective(self, design_variables: List[float], 
                                      problem_name: str) -> Tuple[float, float]:
        """重量最小化目标函数"""
        if problem_name not in self.truss_structures:
            raise ValueError(f"Unknown problem: {problem_name}")
            
        problem = self.truss_structures[problem_name]
        structure = problem['structure']
        
        # 目标1: 最小化重量
        weight = structure.calculate_volume(design_variables)
        
        # 目标2: 最小化最大位移
        displacements = structure.analyze_displacements(design_variables, problem['loads'])
        max_displacement = max(abs(d) for d in displacements) if displacements else 0.0
        
        return weight, max_displacement
    
    def stress_constraint_objective(self, design_variables: List[float],
                                   problem_name: str) -> Tuple[float, float]:
        """考虑应力的多目标优化"""
        if problem_name not in self.truss_structures:
            raise ValueError(f"Unknown problem: {problem_name}")
            
        problem = self.truss_structures[problem_name]
        structure = problem['structure']
        
        # 简化的应力计算
        stresses = self._calculate_element_stresses(structure, design_variables, problem['loads'])
        max_stress = max(stresses) if stresses else 0.0
        
        # 归一化应力（相对于屈服强度）
        normalized_stress = max_stress / structure.material.yield_strength
        
        # 重量仍然是一个目标
        weight = structure.calculate_volume(design_variables)
        
        return weight, normalized_stress
    
    def _calculate_element_stresses(self, structure: TrussStructure,
                                  cross_sections: List[float],
                                  loads: LoadCase) -> List[float]:
        """计算杆件应力（简化）"""
        stresses = []
        
        for elem_idx, (i, j) in enumerate(structure.elements):
            if elem_idx < len(cross_sections) and cross_sections[elem_idx] > 0:
                # 简化的应力计算：假设轴向受力
                # 实际应用中需要求解结构内力
                force_magnitude = 1000.0  # N (简化值)
                stress = force_magnitude / cross_sections[elem_idx]
                stresses.append(stress)
            else:
                stresses.append(0.0)
                
        return stresses
    
    def create_optimization_problem(self, problem_name: str, 
                                   objective_type: str = 'weight_minimization') -> Callable:
        """创建优化问题函数"""
        if problem_name not in self.truss_structures:
            raise ValueError(f"Unknown problem: {problem_name}")
            
        if objective_type == 'weight_minimization':
            return lambda x: self.weight_minimization_objective(x, problem_name)
        elif objective_type == 'stress_constraint':
            return lambda x: self.stress_constraint_objective(x, problem_name)
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")

# 预定义的结构优化问题
class StandardTrussProblems:
    """标准桁架问题库"""
    
    @staticmethod
    def ten_bar_truss() -> Dict:
        """10杆平面桁架"""
        # 节点坐标 (m)
        nodes = [
            (0, 0, 0), (9.144, 0, 0), (18.288, 0, 0), (27.432, 0, 0),  # 下弦节点
            (9.144, 9.144, 0), (18.288, 9.144, 0), (27.432, 9.144, 0)   # 上弦节点
        ]
        
        # 杆件连接 (节点索引从0开始)
        elements = [
            (0, 1), (1, 2), (2, 3),  # 下弦
            (4, 5), (5, 6),          # 上弦
            (0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6)  # 腹杆
        ]
        
        # 材料属性 (钢材)
        material = MaterialProperties(
            density=7850,  # kg/m³
            young_modulus=2.1e11,  # Pa
            yield_strength=250e6   # Pa
        )
        
        # 载荷工况
        loads = LoadCase(
            forces=[(0, -445410, 0)],  # 向下的集中力
            positions=[(9.144, 9.144, 0)]  # 作用在上弦中间节点
        )
        
        return {
            'nodes': nodes,
            'elements': elements,
            'material': material,
            'loads': loads,
            'bounds': (0.001, 0.02)  # 截面范围 (m²)
        }
    
    @staticmethod
    def twenty_five_bar_truss() -> Dict:
        """25杆空间桁架"""
        # 简化的25杆桁架定义
        nodes = [(i*3, j*3, k*3) for i in range(3) for j in range(3) for k in range(2)]
        
        # 简化的连接方式
        elements = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                # 只连接相邻节点
                dist = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))
                if 2.9 < dist < 4.3:  # 邻接距离阈值
                    elements.append((i, j))
                    
        material = MaterialProperties(
            density=2700,  # 铝材
            young_modulus=69e9,
            yield_strength=276e6
        )
        
        loads = LoadCase(
            forces=[(0, 0, -445410)],
            positions=[(4.5, 4.5, 3)]
        )
        
        return {
            'nodes': nodes,
            'elements': elements[:25],  # 取前25个杆件
            'material': material,
            'loads': loads,
            'bounds': (0.0001, 0.01)
        }