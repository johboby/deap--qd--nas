"""
轻量级元学习模块
使用启发式规则和经验学习替代深度学习，大幅降低算力需求
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Callable, Any, Optional
from collections import defaultdict, deque
import time

class RuleBasedMetaLearner:
    """基于规则的元学习器 - 无需神经网络"""
    
    def __init__(self):
        # 经验数据库：问题特征 -> 最佳策略
        self.experience_db = defaultdict(list)
        # 策略效果统计
        self.strategy_stats = defaultdict(lambda: {'success': 0, 'total': 0})
        # 最近表现队列（滑动窗口）
        self.recent_performance = deque(maxlen=100)
        
    def extract_problem_features(self, problem_func: Callable, n_dim: int, 
                               problem_name: str = "unknown") -> Dict[str, Any]:
        """提取问题特征 - 轻量级版本"""
        features = {
            'dimensionality': n_dim,
            'problem_type': self._classify_problem_type(problem_name),
            'complexity_estimate': self._estimate_complexity(problem_func, n_dim),
            'constraints': self._detect_constraints(problem_func, n_dim)
        }
        return features
        
    def _classify_problem_type(self, problem_name: str) -> str:
        """分类问题类型"""
        name_lower = problem_name.lower()
        if 'zdt' in name_lower:
            return 'zdt'
        elif 'dtlz' in name_lower:
            return 'dtlz'
        elif 'sphere' in name_lower:
            return 'sphere'
        elif 'rastrigin' in name_lower:
            return 'rastrigin'
        elif 'constrained' in name_lower:
            return 'constrained'
        else:
            return 'unknown'
            
    def _estimate_complexity(self, problem_func: Callable, n_dim: int) -> float:
        """估计问题复杂度 (0-1)"""
        # 基于维度的简单复杂度估计
        dim_complexity = min(n_dim / 50.0, 1.0)  # 维度越高越复杂
        
        # 基于函数调用时间的复杂度估计（简化版）
        try:
            start_time = time.time()
            test_point = [0.5] * n_dim
            problem_func(test_point)
            elapsed = time.time() - start_time
            time_complexity = min(elapsed * 10, 1.0)  # 时间越长越复杂
        except:
            time_complexity = 0.5
            
        return (dim_complexity + time_complexity) / 2
        
    def _detect_constraints(self, problem_func: Callable, n_dim: int) -> bool:
        """检测约束条件"""
        try:
            test_point = [0.5] * n_dim
            result = problem_func(test_point)
            # 如果返回包含约束违反信息，则认为有约束
            if isinstance(result, tuple) and len(result) == 2:
                return True
        except:
            pass
        return False
        
    def recommend_strategy(self, problem_features: Dict[str, Any]) -> Dict[str, Any]:
        """基于经验推荐策略"""
        dim = problem_features['dimensionality']
        p_type = problem_features['problem_type']
        complexity = problem_features['complexity_estimate']
        has_constraints = problem_features['constraints']
        
        # 基于经验数据库的推荐
        similar_problems = self._find_similar_problems(problem_features)
        if similar_problems:
            best_strategy = self._get_best_strategy_from_experience(similar_problems)
            if best_strategy:
                return best_strategy
                
        # 基于规则的推荐
        return self._rule_based_recommendation(dim, p_type, complexity, has_constraints)
        
    def _find_similar_problems(self, features: Dict[str, Any]) -> List[Dict]:
        """找到相似问题"""
        similar = []
        dim = features['dimensionality']
        p_type = features['problem_type']
        
        for exp_features, strategies in self.experience_db.items():
            exp_dim = exp_features.get('dimensionality', 0)
            exp_type = exp_features.get('problem_type', 'unknown')
            
            # 维度相近且类型相同
            if abs(exp_dim - dim) <= 10 and exp_type == p_type:
                similar.extend(strategies)
                
        return similar
        
    def _get_best_strategy_from_experience(self, strategies: List[Dict]) -> Optional[Dict]:
        """从经验中选择最佳策略"""
        if not strategies:
            return None
            
        # 按性能排序
        sorted_strategies = sorted(strategies, 
                                 key=lambda x: x.get('performance', 0), 
                                 reverse=True)
        return sorted_strategies[0] if sorted_strategies else None
        
    def _rule_based_recommendation(self, dim: int, p_type: str, 
                                  complexity: float, has_constraints: bool) -> Dict[str, Any]:
        """基于规则的策略推荐"""
        
        # 基础策略
        if dim <= 10:
            strategy = {'algorithm': 'nsga2', 'pop_size': 30, 'gens': 50}
        elif dim <= 30:
            strategy = {'algorithm': 'adaptive_nsga2', 'pop_size': 50, 'gens': 80}
        else:
            strategy = {'algorithm': 'quick_quantum', 'pop_size': 40, 'gens': 60}
            
        # 问题类型调整
        if p_type == 'zdt':
            strategy.update({'crossover_prob': 0.9, 'mutation_prob': 0.1})
        elif p_type == 'dtlz':
            strategy.update({'crossover_prob': 0.85, 'mutation_prob': 0.15})
        elif p_type == 'constrained':
            strategy.update({'constraint_method': 'penalty', 'penalty_weight': 1e6})
            
        # 复杂度调整
        if complexity > 0.7:
            strategy['gens'] = int(strategy['gens'] * 1.2)  # 增加迭代次数
        elif complexity < 0.3:
            strategy['gens'] = int(strategy['gens'] * 0.8)  # 减少迭代次数
            
        # 约束调整
        if has_constraints:
            strategy['algorithm'] = 'adaptive_nsga2'
            strategy['constraint_method'] = 'penalty'
            
        return strategy
        
    def update_experience(self, problem_features: Dict[str, Any], 
                         strategy: Dict[str, Any], performance: float):
        """更新经验数据库"""
        # 添加新经验
        experience_entry = {
            'features': problem_features.copy(),
            'strategy': strategy.copy(),
            'performance': performance,
            'timestamp': time.time()
        }
        
        key = (problem_features['dimensionality'], problem_features['problem_type'])
        self.experience_db[key].append(experience_entry)
        
        # 更新策略统计
        algo = strategy.get('algorithm', 'unknown')
        self.strategy_stats[algo]['total'] += 1
        if performance > 0.7:  # 假设0.7为良好性能阈值
            self.strategy_stats[algo]['success'] += 1
            
        # 记录总体性能
        self.recent_performance.append(performance)
        
    def get_strategy_stats(self) -> Dict[str, Any]:
        """获取策略统计信息"""
        stats = {}
        for algo, data in self.strategy_stats.items():
            success_rate = data['success'] / max(data['total'], 1)
            stats[algo] = {
                'success_rate': success_rate,
                'total_attempts': data['total']
            }
            
        avg_performance = np.mean(self.recent_performance) if self.recent_performance else 0
        
        return {
            'strategy_stats': stats,
            'average_performance': avg_performance,
            'experience_count': sum(len(v) for v in self.experience_db.values())
        }

class LightweightAutoML:
    """轻量级AutoML - 基于网格搜索和经验规则"""
    
    def __init__(self):
        self.param_grid = {
            'crossover_prob': [0.7, 0.8, 0.9],
            'mutation_prob': [0.05, 0.1, 0.15],
            'population_size': [30, 50, 80]
        }
        self.best_config = None
        self.best_performance = -float('inf')
        
    def quick_parameter_search(self, problem_func: Callable, n_dim: int, 
                             budget: int = 9) -> Dict[str, Any]:
        """快速参数搜索 - 受限预算的网格搜索"""
        print(f"🔍 执行快速参数搜索 (预算: {budget}次试验)...")
        
        # 生成有限的参数组合
        param_combinations = self._generate_param_combinations(budget)
        best_config = None
        best_performance = -float('inf')
        
        for i, params in enumerate(param_combinations):
            print(f"  📊 试验 {i+1}/{len(param_combinations)}: {params}")
            
            # 快速评估（少量迭代）
            performance = self._evaluate_params_quick(problem_func, n_dim, params)
            
            if performance > best_performance:
                best_performance = performance
                best_config = params.copy()
                
        self.best_config = best_config
        self.best_performance = best_performance
        
        print(f"✅ 最佳配置: {best_config}, 性能: {best_performance:.4f}")
        return best_config
        
    def _generate_param_combinations(self, budget: int) -> List[Dict]:
        """生成参数组合"""
        combinations = []
        
        # 根据预算生成组合
        if budget >= 9:
            # 完整3x3网格
            for cp in self.param_grid['crossover_prob']:
                for mp in self.param_grid['mutation_prob']:
                    for ps in self.param_grid['population_size']:
                        combinations.append({
                            'crossover_prob': cp,
                            'mutation_prob': mp,
                            'population_size': ps
                        })
        else:
            # 随机采样
            for _ in range(budget):
                combinations.append({
                    'crossover_prob': random.choice(self.param_grid['crossover_prob']),
                    'mutation_prob': random.choice(self.param_grid['mutation_prob']),
                    'population_size': random.choice(self.param_grid['population_size'])
                })
                
        return combinations[:budget]
        
    def _evaluate_params_quick(self, problem_func: Callable, n_dim: int, 
                              params: Dict) -> float:
        """快速评估参数配置"""
        try:
            # 简化的性能评估（短时间运行）
            from ..algorithms.nsga2 import NSGA2Algorithm
            
            algorithm = NSGA2Algorithm(
                crossover_prob=params['crossover_prob'],
                mutation_prob=params['mutation_prob'],
                population_size=min(params['population_size'], 50),  # 限制大小
                max_generations=20  # 限制代数
            )
            
            result = algorithm.optimize(
                problem_func=problem_func,
                n_dim=n_dim,
                n_gen=20,
                pop_size=min(params['population_size'], 50)
            )
            
            # 简单的性能度量（超体积近似）
            if result['pareto_front']:
                front_values = [ind.fitness.values for ind in result['pareto_front']]
                if front_values:
                    return np.mean([sum(val) for val in front_values])
                    
        except Exception as e:
            print(f"    ❌ 评估失败: {e}")
            
        return 0.0

def get_lightweight_meta_learner():
    """获取轻量级元学习器"""
    return RuleBasedMetaLearner()

def get_lightweight_automl():
    """获取轻量级AutoML"""
    return LightweightAutoML()