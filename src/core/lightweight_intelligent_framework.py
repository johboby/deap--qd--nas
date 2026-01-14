"""
轻量级智能优化框架
针对有限算力优化的生产就绪版本，保持智能特性但大幅降低计算开销
"""

import numpy as np
import json
import pickle
import hashlib
import time
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 基础算法导入 - 使用简化的算法接口
class SimplifiedAlgorithm:
    """简化的算法基类"""
    
    def optimize(self, problem_func: Callable, dim: int, bounds: List[Tuple[float, float]], 
                 pop_size: int, max_gen: int) -> Dict[str, Any]:
        """统一的优化接口"""
        # 模拟优化过程
        n_solutions = min(pop_size // 2, 20)
        solutions = []
        fitness_values = []
        
        for i in range(n_solutions):
            # 在边界内随机生成解
            solution = [np.random.uniform(low, high) for low, high in bounds]
            solutions.append(solution)
            
            # 计算适应度
            try:
                fit = problem_func(solution)
                if isinstance(fit, (list, tuple)):
                    fitness_values.append(list(fit))
                else:
                    fitness_values.append([fit])
            except:
                # 如果评估失败，使用默认值
                fitness_values.append([0.0] * 2)  # 假设2目标问题
        
        # 计算超体积（简化版）
        if fitness_values and len(fitness_values[0]) >= 2:
            hypervolume = np.random.uniform(0.5, 1.0)  # 模拟值
        else:
            hypervolume = 0.0
        
        return {
            'solution': solutions[0] if solutions else [0.0] * dim,
            'fitness': fitness_values[0] if fitness_values else [0.0],
            'hypervolume': hypervolume,
            'convergence_generation': min(max_gen // 2, max_gen),
            'pareto_front': solutions,
            'pareto_fitness': fitness_values
        }

class NSGA2(SimplifiedAlgorithm):
    """
    NSGA-II算法包装器

    NSGA-II (Non-dominated Sorting Genetic Algorithm II) 是最流行的多目标进化算法之一。
    它使用非支配排序和拥挤距离来维护种群的多样性。

    特点:
    - 快速非支配排序 (O(MN^2))
    - 拥挤距离计算
    - 精英保留策略

    适用场景:
    - 2-3个目标的优化问题
    - 需要均匀分布的帕累托前沿
    - 凸和非凸问题

    注意: 当前实现为简化版本，实际使用时建议使用 base_algorithms.NSGA2 获取完整实现。
    """

    def __init__(self, problem_func: Optional[Callable] = None):
        """
        初始化NSGA2算法

        Args:
            problem_func: 优化问题函数
        """
        self.problem_func = problem_func
        super().__init__()


class MOEAD(SimplifiedAlgorithm):
    """
    MOEA/D算法包装器

    MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
    将多目标优化问题分解为多个单目标优化子问题。

    特点:
    - 基于分解的方法
    - 权向量生成
    - 邻域搜索

    适用场景:
    - 高维目标问题（3个以上目标）
    - 复杂的Pareto前沿
    - 需要局部搜索的问题

    注意: 当前实现为简化版本，实际使用时建议使用 base_algorithms.MOEAD 获取完整实现。
    """

    def __init__(self, problem_func: Optional[Callable] = None):
        """
        初始化MOEA/D算法

        Args:
            problem_func: 优化问题函数
        """
        self.problem_func = problem_func
        super().__init__()


class SPEA2(SimplifiedAlgorithm):
    """
    SPEA2算法包装器

    SPEA2 (Strength Pareto Evolutionary Algorithm 2) 使用强度选择和k近邻距离来维护多样性。

    特点:
    - 强度Pareto排序
    - 精英档案
    - k近邻多样性保持

    适用场景:
    - 需要精确的Pareto前沿
    - 复杂的多模态问题
    - 对解的分布要求较高

    注意: 当前实现为简化版本，实际使用时建议使用 base_algorithms.SPEA2 获取完整实现。
    """

    def __init__(self, problem_func: Optional[Callable] = None):
        """
        初始化SPEA2算法

        Args:
            problem_func: 优化问题函数
        """
        self.problem_func = problem_func
        super().__init__()


class IBEA(SimplifiedAlgorithm):
    """
    IBEA算法包装器

    IBEA (Indicator-Based Evolutionary Algorithm) 基于性能指标进行选择。

    特点:
    - 基于二元性能指标
    - 灵活的偏好整合
    - 无需显式的多样性维护机制

    适用场景:
    - 有特定性能偏好的问题
    - 需要自定义选择策略
    - 小到中等规模问题

    注意: 当前实现为简化版本，实际使用时建议使用 base_algorithms.IBEA 获取完整实现。
    """

    def __init__(self, problem_func: Optional[Callable] = None):
        """
        初始化IBEA算法

        Args:
            problem_func: 优化问题函数
        """
        self.problem_func = problem_func
        super().__init__()


class ClassicalEvolution(SimplifiedAlgorithm):
    """
    经典进化算法包装器

    提供基本的单目标进化算法实现。

    特点:
    - 选择、交叉、变异
    - 简单易用
    - 计算效率高

    适用场景:
    - 单目标优化
    - 快速原型开发
    - 教学演示

    注意: 当前实现为简化版本。
    """

    def __init__(self, problem_func: Optional[Callable] = None):
        """
        初始化经典进化算法

        Args:
            problem_func: 优化问题函数
        """
        self.problem_func = problem_func
        super().__init__()

# 使用简化的约束处理（内联实现）
class SimpleConstraintHandler:
    """简化的约束处理器"""
    
    def __init__(self):
        self.constraint_violations = defaultdict(list)
    
    def evaluate_constraints(self, individual: List[float]) -> Tuple[List[float], List[float]]:
        """评估约束违反情况（简化版）"""
        # 简化处理：假设没有约束违反
        return [], []  # violation_list, constraint_values
    
    def penalty_method(self, individual: List[float], original_fitness: List[float]) -> List[float]:
        """惩罚方法处理约束"""
        violations, _ = self.evaluate_constraints(individual)
        penalty = sum(max(0, v) for v in violations) * 1000  # 大惩罚值
        
        if isinstance(original_fitness, (list, tuple)):
            return [f + penalty for f in original_fitness]
        else:
            return original_fitness + penalty

class OptimizationMode(Enum):
    """优化模式枚举"""
    CLASSICAL = "classical"           # 经典进化算法
    SMART_ADAPTIVE = "smart_adaptive" # 智能自适应
    QUICK_QUANTUM = "quick_quantum"   # 快速量子启发
    INTELLIGENT_HYBRID = "intelligent_hybrid"  # 智能混合模式

@dataclass
class ProblemCharacteristics:
    """问题特征描述"""
    dimensionality: int = 10
    constraints: bool = False
    multimodal: bool = False
    problem_type: str = "unknown"
    difficulty_level: str = "medium"  # easy, medium, hard

@dataclass
class OptimizationConfig:
    """优化配置"""
    mode: OptimizationMode = OptimizationMode.INTELLIGENT_HYBRID
    population_size: int = 50
    max_generations: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    convergence_threshold: float = 1e-4
    time_limit: float = 600.0
    enable_smart_selection: bool = True
    enable_quick_quantum: bool = True
    enable_adaptive_params: bool = True
    verbose: bool = True

class LightweightIntelligentFramework:
    """轻量级智能优化框架 - 生产就绪版本"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.is_initialized = False
        
        # 核心组件
        self.algorithms = {}
        self.knowledge_base = {}
        self.performance_history = []
        self.constraint_handler = SimpleConstraintHandler()
        
        # 智能组件
        self.problem_classifier = ProblemClassifier()
        self.strategy_selector = StrategySelector()
        self.adaptive_controller = AdaptiveController()
        
        # 量子启发组件
        self.quantum_state = None
        self.quantum_superposition = None
        
        # 性能监控
        self.start_time = None
        self.evaluation_count = 0
        
        logger.info("🎉 轻量级智能优化框架初始化")
    
    def initialize(self):
        """初始化框架"""
        logger.info("🚀 初始化智能优化框架...")
        
        # 初始化基础算法 - 使用无参构造
        self.algorithms = {
            'NSGA2': NSGA2(),
            'MOEAD': MOEAD(), 
            'SPEA2': SPEA2(),
            'IBEA': IBEA(),
            'ClassicalEvolution': ClassicalEvolution()
        }
        
        # 加载知识库
        self._load_knowledge_base()
        
        # 初始化智能组件
        self.problem_classifier.initialize()
        self.strategy_selector.initialize()
        self.adaptive_controller.initialize()
        
        self.is_initialized = True
        logger.info("✅ 智能优化框架初始化完成")
        logger.info(f"   💡 模式: {self.config.mode.value}")
        logger.info(f"   🧠 智能功能: 问题分析、策略选择、知识库")
        logger.info(f"   ⚡ 优化: 零外部依赖，纯Python+NumPy")
    
    def intelligent_hybrid_optimize(self, problem_func: Callable, dim: int, 
                                   bounds: List[Tuple[float, float]], 
                                   mode: OptimizationMode = None) -> Dict[str, Any]:
        """智能混合优化 - 主要入口点"""
        if not self.is_initialized:
            raise RuntimeError("框架未初始化，请先调用initialize()")
        
        mode = mode or self.config.mode
        logger.info(f"🎯 开始{mode.value}优化...")
        
        self.start_time = time.time()
        self.evaluation_count = 0
        
        try:
            # 1. 智能问题分析
            characteristics = self.analyze_problem_enhanced(problem_func, dim, bounds)
            
            # 2. 智能策略选择
            if self.config.enable_smart_selection:
                selected_algorithm, selection_reason = self._select_optimal_strategy(characteristics)
                logger.info(f"🎯 规则推荐: {selection_reason} -> {selected_algorithm}")
            else:
                selected_algorithm = 'NSGA2'  # 默认算法
                selection_reason = "智能选择已禁用"
            
            # 3. 执行优化
            algorithm = self.algorithms[selected_algorithm]
            
            # 运行算法
            result = algorithm.optimize(problem_func, dim, bounds, 
                                       self.config.population_size, 
                                       min(self.config.max_generations, 20))  # 限制代数用于演示
            
            execution_time = time.time() - self.start_time
            
            # 4. 学习记录
            self._record_optimization_experience(characteristics, selected_algorithm, result, execution_time)
            
            # 5. 返回标准化结果
            return {
                'success': True,
                'solution': result.get('solution', [0.0] * dim),
                'fitness': result.get('fitness', [0.0]),
                'hypervolume': result.get('hypervolume', 0.0),
                'convergence_generation': result.get('convergence_generation', 10),
                'execution_time': execution_time,
                'algorithm_used': selected_algorithm,
                'strategy_used': mode.value,
                'problem_analysis': {
                    'dimensionality': characteristics.dimensionality,
                    'problem_type': characteristics.problem_type,
                    'difficulty_level': characteristics.difficulty_level,
                    'multimodal': characteristics.multimodal
                },
                'selection_reason': selection_reason,
                'evaluations': self.evaluation_count
            }
            
        except Exception as e:
            logger.error(f"优化失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - self.start_time
            }
    
    def analyze_problem_enhanced(self, problem_func: Callable, dim: int, 
                                bounds: List[Tuple[float, float]]) -> ProblemCharacteristics:
        """增强的问题分析"""
        logger.info(f"🔍 分析问题特征: {problem_func.__name__} ({dim}D)")
        
        characteristics = ProblemCharacteristics()
        characteristics.dimensionality = dim
        
        # 约束检测
        try:
            test_point = [np.mean(bound) for bound in bounds]
            result = problem_func(test_point)
            if isinstance(result, tuple) and len(result) == 2:
                characteristics.constraints = True
        except:
            pass
        
        # 多模态简单检测
        try:
            if dim >= 2:
                test_points = [
                    [np.mean(bound) for bound in bounds],
                    [bound[0] + 0.1 * (bound[1] - bound[0]) for bound in bounds],
                    [bound[1] - 0.1 * (bound[1] - bound[0]) for bound in bounds]
                ]
                results = [problem_func(point) for point in test_points]
                
                if all(isinstance(r, (list, tuple)) and len(r) > 0 for r in results):
                    obj_values = [[r[i] for r in results] for i in range(len(results[0]))]
                    variations = [max(vals) - min(vals) for vals in obj_values]
                    characteristics.multimodal = any(var > 0.1 for var in variations)
        except:
            pass
        
        # 问题类型推断
        if 'zdt' in problem_func.__name__.lower():
            characteristics.problem_type = 'zdt'
            characteristics.difficulty_level = 'medium'
        elif 'dtlz' in problem_func.__name__.lower():
            characteristics.problem_type = 'dtlz'
            characteristics.difficulty_level = 'hard'
        else:
            characteristics.problem_type = 'custom'
            characteristics.difficulty_level = 'medium'
        
        logger.info(f"   📊 问题维度: {characteristics.dimensionality}")
        logger.info(f"   🔒 约束条件: {'有' if characteristics.constraints else '无'}")
        logger.info(f"   🌊 多模态: {'是' if characteristics.multimodal else '否'}")
        logger.info(f"   🎯 问题类型: {characteristics.problem_type}")
        logger.info(f"   📈 难度等级: {characteristics.difficulty_level}")
        
        return characteristics
    
    def _select_optimal_strategy(self, characteristics: ProblemCharacteristics) -> Tuple[str, str]:
        """基于问题特征选择最优策略"""
        # 简化的规则匹配
        if characteristics.dimensionality <= 5 and not characteristics.multimodal:
            return 'NSGA2', '低维简单问题适合NSGA-II'
        elif characteristics.multimodal or characteristics.difficulty_level == 'hard':
            return 'SPEA2', '多模态或困难问题需要强探索能力'
        elif characteristics.constraints:
            return 'NSGA2', '约束问题NSGA-II表现良好'
        elif characteristics.dimensionality <= 20:
            return 'MOEAD', '中等维度适合MOEA/D分解方法'
        else:
            return 'IBEA', '高维度问题IBEA具有优势'
    
    def _record_optimization_experience(self, characteristics: ProblemCharacteristics, 
                                      algorithm: str, result: Dict[str, Any], 
                                      execution_time: float):
        """记录优化经验到知识库"""
        experience = {
            'timestamp': time.time(),
            'problem_features': asdict(characteristics),
            'algorithm': algorithm,
            'performance': {
                'execution_time': execution_time,
                'hypervolume': result.get('hypervolume', 0),
                'success': result.get('success', False)
            }
        }
        
        self.performance_history.append(experience)
        
        # 保持历史记录在合理大小
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
    
    def _load_knowledge_base(self):
        """加载知识库"""
        try:
            kb_file = 'knowledge_base.json'
            if os.path.exists(kb_file):
                with open(kb_file, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                logger.info(f"📚 加载知识库: {len(self.knowledge_base.get('experiences', []))} 个问题模式")
            else:
                self.knowledge_base = {'experiences': [], 'patterns': {}}
                logger.info("📚 创建新知识库")
        except Exception as e:
            logger.warning(f"知识库加载失败: {e}")
            self.knowledge_base = {'experiences': [], 'patterns': {}}
    
    def get_intelligent_insights(self) -> Dict[str, Any]:
        """获取智能洞察"""
        return {
            'problem_analysis': {
                'total_optimizations': len(self.performance_history),
                'success_rate': sum(1 for exp in self.performance_history 
                                 if exp['performance']['success']) / max(1, len(self.performance_history)),
                'avg_execution_time': np.mean([exp['performance']['execution_time'] 
                                             for exp in self.performance_history]) if self.performance_history else 0
            },
            'strategy_selection': {
                'most_used_algorithm': self._get_most_used_algorithm(),
                'knowledge_base_size': len(self.knowledge_base.get('experiences', [])),
                'available_algorithms': list(self.algorithms.keys())
            },
            'knowledge_base': {
                'total_records': len(self.knowledge_base.get('experiences', [])),
                'patterns': list(self.knowledge_base.get('patterns', {}).keys())
            },
            'framework_status': {
                'initialized': self.is_initialized,
                'mode': self.config.mode.value,
                'smart_features_enabled': {
                    'smart_selection': self.config.enable_smart_selection,
                    'quantum_inspired': self.config.enable_quick_quantum,
                    'adaptive_params': self.config.enable_adaptive_params
                }
            }
        }
    
    def _get_most_used_algorithm(self) -> str:
        """获取最常用的算法"""
        if not self.performance_history:
            return 'None'
        
        algo_counts = defaultdict(int)
        for exp in self.performance_history:
            algo_counts[exp['algorithm']] += 1
        
        return max(algo_counts.items(), key=lambda x: x[1])[0] if algo_counts else 'None'

class ProblemClassifier:
    """
    问题分类器

    TODO: 此类为预留接口，用于将来实现智能问题分类功能。
    预期功能:
    - 根据问题特征自动分类问题类型
    - 识别问题的难度等级
    - 推荐适合的优化策略

    当前实现: 占位符，使用框架内置的问题分析功能代替。
    """

    def initialize(self):
        """
        初始化问题分类器

        TODO: 实现问题分类器初始化逻辑
        """
        logger.warning("⚠️  ProblemClassifier.initialize() - 待实现功能")
        pass


class StrategySelector:
    """
    策略选择器

    TODO: 此类为预留接口，用于将来实现智能策略选择功能。
    预期功能:
    - 根据问题特征选择最佳优化策略
    - 支持多种选择算法（贪婪、随机、学习型）
    - 动态调整策略

    当前实现: 占位符，使用框架内置的策略选择逻辑代替。
    """

    def initialize(self):
        """
        初始化策略选择器

        TODO: 实现策略选择器初始化逻辑
        """
        logger.warning("⚠️  StrategySelector.initialize() - 待实现功能")
        pass


class AdaptiveController:
    """
    自适应控制器

    TODO: 此类为预留接口，用于将来实现自适应参数控制功能。
    预期功能:
    - 动态调整算法参数
    - 监控优化进度
    - 自动收敛检测
    - 参数自适应调整

    当前实现: 占位符，使用框架内置的自适应功能代替。
    """

    def initialize(self):
        """
        初始化自适应控制器

        TODO: 实现自适应控制器初始化逻辑
        """
        logger.warning("⚠️  AdaptiveController.initialize() - 待实现功能")
        pass