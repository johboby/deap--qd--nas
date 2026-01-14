"""
性能分析模块
提供多目标优化结果的深度分析和统计
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import cdist
import warnings

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.results_cache = {}
        
    def analyze_convergence(self, history: List[Dict], 
                           metrics: List[str] = ['hypervolume']) -> Dict[str, Any]:
        """分析收敛性能"""
        if not history:
            return {}
            
        analysis = {}
        
        for metric in metrics:
            if metric not in history[0]:
                continue
                
            values = [h[metric] for h in history]
            generations = [h['generation'] for h in history]
            
            # 收敛速度分析
            convergence_speed = self._analyze_convergence_speed(values, generations)
            
            # 稳定性分析
            stability = self._analyze_stability(values)
            
            # 趋势分析
            trend = self._analyze_trend(values)
            
            analysis[metric] = {
                'convergence_speed': convergence_speed,
                'stability': stability,
                'trend': trend,
                'final_value': values[-1] if values else 0,
                'improvement': values[-1] - values[0] if len(values) > 1 else 0,
                'relative_improvement': ((values[-1] - values[0]) / abs(values[0] + 1e-10)) * 100
            }
            
        return analysis
    
    def _analyze_convergence_speed(self, values: List[float], 
                                  generations: List[int]) -> Dict[str, float]:
        """分析收敛速度"""
        if len(values) < 2:
            return {'speed': 0, 'generations_to_converge': 0}
            
        # 计算变化率
        changes = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
        
        # 找到收敛点（变化小于阈值）
        threshold = np.std(changes) * 0.01  # 使用标准差的1%作为阈值
        converged_idx = len(changes)
        
        for i, change in enumerate(changes):
            if change < threshold:
                converged_idx = i
                break
                
        convergence_generation = generations[min(converged_idx + 1, len(generations)-1)]
        
        # 平均收敛速度
        total_change = abs(values[-1] - values[0])
        speed = total_change / convergence_generation if convergence_generation > 0 else 0
        
        return {
            'speed': speed,
            'generations_to_converge': convergence_generation,
            'convergence_threshold': threshold
        }
    
    def _analyze_stability(self, values: List[float]) -> Dict[str, float]:
        """分析稳定性"""
        if len(values) < 2:
            return {'coefficient_of_variation': 0, 'stability_score': 1}
            
        # 计算变异系数
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / (abs(mean_val) + 1e-10)
        
        # 计算最后10个值的稳定性
        window_size = min(10, len(values))
        recent_values = values[-window_size:]
        recent_cv = np.std(recent_values) / (abs(np.mean(recent_values)) + 1e-10)
        
        # 稳定性评分 (0-1, 1表示最稳定)
        stability_score = max(0, 1 - recent_cv)
        
        return {
            'coefficient_of_variation': cv,
            'recent_stability': recent_cv,
            'stability_score': stability_score
        }
    
    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """分析趋势"""
        if len(values) < 3:
            return {'trend_direction': 'unknown', 'trend_strength': 0}
            
        # 线性回归分析
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # 确定趋势方向
        if slope > 1e-6:
            direction = 'improving'
        elif slope < -1e-6:
            direction = 'degrading'
        else:
            direction = 'stable'
            
        return {
            'trend_direction': direction,
            'trend_slope': slope,
            'correlation_coefficient': r_value,
            'p_value': p_value,
            'trend_strength': abs(r_value)
        }
    
    def compare_algorithms(self, results_dict: Dict[str, Dict], 
                          metric: str = 'hypervolume') -> Dict[str, Any]:
        """比较多算法性能"""
        comparison = {
            'statistical_tests': {},
            'rankings': {},
            'effect_sizes': {}
        }
        
        algorithms = list(results_dict.keys())
        if len(algorithms) < 2:
            return comparison
            
        # 收集各算法的性能指标
        metric_values = {}
        for alg in algorithms:
            if metric in results_dict[alg]:
                metric_values[alg] = results_dict[alg][metric]
                
        if len(metric_values) < 2:
            return comparison
            
        # 统计分析
        alg_list = list(metric_values.keys())
        
        # 成对t检验 (如果有足够样本)
        for i in range(len(alg_list)):
            for j in range(i+1, len(alg_list)):
                alg1, alg2 = alg_list[i], alg_list[j]
                values1 = metric_values[alg1]
                values2 = metric_values[alg2]
                
                if len(values1) > 1 and len(values2) > 1:
                    try:
                        _, p_value = stats.ttest_ind(values1, values2)
                        comparison['statistical_tests'][f'{alg1}_vs_{alg2}'] = {
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except Exception as e:
                        warnings.warn(f"Statistical test failed for {alg1} vs {alg2}: {e}")
        
        # 排名 (基于均值)
        mean_values = {alg: np.mean(values) for alg, values in metric_values.items()}
        ranked_algs = sorted(mean_values.items(), key=lambda x: x[1], reverse=True)
        comparison['rankings'][metric] = [
            {'algorithm': alg, 'mean_value': mean_val, 'rank': idx+1}
            for idx, (alg, mean_val) in enumerate(ranked_algs)
        ]
        
        # 效应量计算 (Cohen's d)
        for i in range(len(alg_list)):
            for j in range(i+1, len(alg_list)):
                alg1, alg2 = alg_list[i], alg_list[j]
                values1 = metric_values[alg1]
                values2 = metric_values[alg2]
                
                pooled_std = np.sqrt(((len(values1)-1)*np.var(values1, ddof=1) + 
                                    (len(values2)-1)*np.var(values2, ddof=1)) / 
                                   (len(values1) + len(values2) - 2))
                
                if pooled_std > 0:
                    cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std
                    comparison['effect_sizes'][f'{alg1}_vs_{alg2}'] = {
                        'cohens_d': cohens_d,
                        'interpretation': self._interpret_cohens_d(cohens_d)
                    }
                    
        return comparison
    
    def _interpret_cohens_d(self, d: float) -> str:
        """解释Cohen's d效应量"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def analyze_diversity(self, pareto_front: List[Tuple]) -> Dict[str, float]:
        """分析Pareto前沿多样性"""
        if not pareto_front or len(pareto_front) < 2:
            return {'diversity_index': 0, 'spread': 0}
            
        front_array = np.array(pareto_front)
        n_points, n_obj = front_array.shape
        
        # 计算点间距离
        distances = []
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.linalg.norm(front_array[i] - front_array[j])
                distances.append(dist)
                
        if not distances:
            return {'diversity_index': 0, 'spread': 0}
            
        # 多样性指标
        diversity_index = np.mean(distances)  # 平均距离
        spread = np.std(distances)  # 距离的离散程度
        
        # 覆盖率分析
        if n_obj == 2:
            # 对于2目标，计算目标空间的覆盖范围
            obj1_range = np.max(front_array[:, 0]) - np.min(front_array[:, 0])
            obj2_range = np.max(front_array[:, 1]) - np.min(front_array[:, 1])
            coverage_area = obj1_range * obj2_range
        else:
            coverage_area = np.prod(np.ptp(front_array, axis=0))  # 广义范围乘积
            
        return {
            'diversity_index': diversity_index,
            'spread': spread,
            'coverage_area': coverage_area,
            'n_points': n_points,
            'avg_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
    
    def statistical_significance_test(self, values1: List[float], 
                                     values2: List[float],
                                     test_type: str = 'ttest') -> Dict[str, Any]:
        """执行统计显著性检验"""
        if len(values1) < 2 or len(values2) < 2:
            return {'error': 'Insufficient data for statistical test'}
            
        results = {}
        
        try:
            if test_type == 'ttest':
                statistic, p_value = stats.ttest_ind(values1, values2)
                results = {
                    'test_type': 'independent_t_test',
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'interpretation': 'significant' if p_value < 0.05 else 'not_significant'
                }
            elif test_type == 'mannwhitney':
                statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                results = {
                    'test_type': 'mann_whitney_u_test',
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'interpretation': 'significant' if p_value < 0.05 else 'not_significant'
                }
        except Exception as e:
            results = {'error': f'Test failed: {str(e)}'}
            
        return results

class ResultAggregator:
    """结果聚合器"""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        
    def aggregate_trial_results(self, trial_results: List[Dict]) -> Dict[str, Any]:
        """聚合多次试验结果"""
        if not trial_results:
            return {}
            
        aggregated = {
            'n_trials': len(trial_results),
            'metrics': {},
            'convergence_analysis': {},
            'diversity_analysis': {},
            'summary_statistics': {}
        }
        
        # 收集各次试验的指标
        all_metrics = set()
        for trial in trial_results:
            if 'metrics' in trial:
                all_metrics.update(trial['metrics'].keys())
                
        # 计算每次试验的指标统计
        for metric in all_metrics:
            values = []
            for trial in trial_results:
                if 'metrics' in trial and metric in trial['metrics']:
                    values.append(trial['metrics'][metric])
                    
            if values:
                aggregated['metrics'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'values': values
                }
        
        # 收敛分析
        if trial_results and 'history' in trial_results[0]:
            first_trial_history = trial_results[0]['history']
            convergence_analysis = self.analyzer.analyze_convergence(first_trial_history)
            aggregated['convergence_analysis'] = convergence_analysis
            
        # 多样性分析
        if trial_results and 'pareto_front' in trial_results[0]:
            first_trial_front = trial_results[0]['pareto_front']
            front_points = [ind.fitness.values for ind in first_trial_front]
            diversity_analysis = self.analyzer.analyze_diversity(front_points)
            aggregated['diversity_analysis'] = diversity_analysis
            
        # 汇总统计
        aggregated['summary_statistics'] = {
            'total_runtime': sum(trial.get('runtime', 0) for trial in trial_results),
            'avg_runtime': np.mean([trial.get('runtime', 0) for trial in trial_results]),
            'success_rate': sum(1 for trial in trial_results if trial.get('success', False)) / len(trial_results)
        }
        
        return aggregated