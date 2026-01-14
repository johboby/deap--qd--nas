"""
机器学习超参数优化
提供基于多目标优化的超参数调优框架
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# 模拟sklearn接口（实际使用时需要安装scikit-learn）
try:
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.datasets import make_classification, make_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Using mock implementations.")

@dataclass
class MLProblemDefinition:
    """机器学习问题定义"""
    model_type: str  # 'classification', 'regression'
    model_name: str  # 'rf', 'svm', 'lr'
    param_space: Dict[str, Tuple[float, float, str]]  # 参数空间 {param: (low, high, type)}
    objective_weights: Tuple[float, float] = (0.5, 0.5)  # (accuracy_weight, complexity_weight)
    cv_folds: int = 5
    
@dataclass
class ModelPerformance:
    """模型性能指标"""
    accuracy: float = 0.0
    f1_score: float = 0.0
    complexity: float = 0.0  # 复杂度度量（如树的数量、特征数量等）
    training_time: float = 0.0
    inference_time: float = 0.0
    
class MockModel:
    """模拟模型（当sklearn不可用时）"""
    
    def __init__(self, **params):
        self.params = params
        
    def fit(self, X, y):
        # 模拟训练时间
        import time
        time.sleep(0.01)
        return self
        
    def predict(self, X):
        # 返回随机预测
        return np.random.randint(0, 2, size=len(X))

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, problem_def: MLProblemDefinition):
        self.problem_def = problem_def
        
    def evaluate_model(self, hyperparameters: Dict[str, Any], 
                      X_train=None, y_train=None) -> ModelPerformance:
        """评估给定超参数的模型性能"""
        # 如果没有提供数据，生成模拟数据
        if X_train is None or y_train is None:
            if self.problem_def.model_type == 'classification':
                X_train, y_train = make_classification(n_samples=1000, n_features=20, 
                                                     n_classes=2, random_state=42)
            else:
                X_train, y_train = make_regression(n_samples=1000, n_features=20, 
                                                  random_state=42)
        
        # 创建模型
        model = self._create_model(hyperparameters)
        
        # 评估性能
        performance = ModelPerformance()
        
        try:
            import time
            
            # 训练时间
            start_time = time.time()
            model.fit(X_train, y_train)
            performance.training_time = time.time() - start_time
            
            # 推理时间
            start_time = time.time()
            predictions = model.predict(X_train)
            performance.inference_time = time.time() - start_time
            
            # 准确率/F1分数
            if self.problem_def.model_type == 'classification':
                from sklearn.metrics import accuracy_score, f1_score
                performance.accuracy = accuracy_score(y_train, predictions)
                performance.f1_score = f1_score(y_train, predictions, average='weighted')
            else:
                from sklearn.metrics import r2_score, mean_squared_error
                performance.accuracy = r2_score(y_train, predictions)
                performance.f1_score = -mean_squared_error(y_train, predictions)  # 负的MSE
            
            # 复杂度度量
            performance.complexity = self._calculate_complexity(model)
            
        except Exception as e:
            warnings.warn(f"Model evaluation failed: {e}")
            # 返回默认值
            performance.accuracy = 0.5
            performance.f1_score = 0.5
            performance.complexity = 1.0
            
        return performance
    
    def _create_model(self, hyperparameters: Dict[str, Any]):
        """根据超参数创建模型"""
        if not SKLEARN_AVAILABLE:
            return MockModel(**hyperparameters)
            
        if self.problem_def.model_type == 'classification':
            if self.problem_def.model_name == 'rf':
                return RandomForestClassifier(
                    n_estimators=int(hyperparameters.get('n_estimators', 100)),
                    max_depth=int(hyperparameters.get('max_depth', 10)),
                    min_samples_split=int(hyperparameters.get('min_samples_split', 2)),
                    random_state=42
                )
            elif self.problem_def.model_name == 'svm':
                return SVC(
                    C=hyperparameters.get('C', 1.0),
                    gamma=hyperparameters.get('gamma', 'scale'),
                    kernel=hyperparameters.get('kernel', 'rbf'),
                    random_state=42
                )
            elif self.problem_def.model_name == 'lr':
                return LogisticRegression(
                    C=hyperparameters.get('C', 1.0),
                    max_iter=int(hyperparameters.get('max_iter', 1000)),
                    random_state=42
                )
        else:  # regression
            if self.problem_def.model_name == 'rf':
                return RandomForestRegressor(
                    n_estimators=int(hyperparameters.get('n_estimators', 100)),
                    max_depth=int(hyperparameters.get('max_depth', 10)),
                    random_state=42
                )
            elif self.problem_def.model_name == 'svm':
                return SVR(
                    C=hyperparameters.get('C', 1.0),
                    gamma=hyperparameters.get('gamma', 'scale'),
                    kernel=hyperparameters.get('kernel', 'rbf')
                )
            elif self.problem_def.model_name == 'lr':
                return LinearRegression()
        
        return MockModel(**hyperparameters)
    
    def _calculate_complexity(self, model) -> float:
        """计算模型复杂度"""
        try:
            if hasattr(model, 'n_estimators'):
                return float(model.n_estimators)
            elif hasattr(model, 'coef_'):
                return float(model.coef_.size)
            elif hasattr(model, 'support_vectors_'):
                return float(len(model.support_vectors_))
            else:
                return 1.0
        except:
            return 1.0

class HyperparameterTuner:
    """超参数优化器"""
    
    def __init__(self, problem_def: MLProblemDefinition):
        self.problem_def = problem_def
        self.evaluator = ModelEvaluator(problem_def)
        self.evaluation_history = []
        
    def multi_objective_function(self, x: List[float]) -> Tuple[float, float]:
        """多目标优化函数"""
        # 将连续变量转换为超参数字典
        hyperparameters = self._decode_hyperparameters(x)
        
        # 评估模型
        performance = self.evaluator.evaluate_model(hyperparameters)
        
        # 记录评估历史
        self.evaluation_history.append({
            'hyperparameters': hyperparameters.copy(),
            'performance': performance,
            'encoded_params': x.copy()
        })
        
        # 标准化目标函数值
        # 目标1: 最大化准确率 (转为最小化负准确率)
        obj1 = 1.0 - performance.accuracy
        
        # 目标2: 最小化复杂度
        obj2 = performance.complexity / 100.0  # 归一化
        
        # 应用权重
        w_acc, w_comp = self.problem_def.objective_weights
        weighted_obj1 = w_acc * obj1
        weighted_obj2 = w_comp * obj2
        
        return weighted_obj1, weighted_obj2
    
    def _decode_hyperparameters(self, x: List[float]) -> Dict[str, Any]:
        """将编码的变量解码为超参数"""
        hyperparameters = {}
        param_names = list(self.problem_def.param_space.keys())
        
        for i, param_name in enumerate(param_names):
            if i < len(x):
                low, high, param_type = self.problem_def.param_space[param_name]
                value = x[i] * (high - low) + low
                
                if param_type == 'int':
                    value = int(round(value))
                elif param_type == 'log':
                    value = 10 ** value  # 对数尺度
                    
                hyperparameters[param_name] = value
                
        return hyperparameters
    
    def _encode_hyperparameters(self, hyperparameters: Dict[str, Any]) -> List[float]:
        """将超参数编码为连续变量"""
        encoded = []
        param_names = list(self.problem_def.param_space.keys())
        
        for param_name in param_names:
            if param_name in hyperparameters:
                low, high, param_type = self.problem_def.param_space[param_name]
                value = hyperparameters[param_name]
                
                if param_type == 'log':
                    value = np.log10(value)
                    
                encoded_value = (value - low) / (high - low)
                encoded.append(encoded_value)
            else:
                encoded.append(0.5)  # 默认值
                
        return encoded
    
    def get_pareto_optimal_configurations(self, population: List, 
                                        fitness_values: List[Tuple]) -> List[Dict]:
        """获取Pareto最优的超参数配置"""
        pareto_configs = []
        
        # 简单的Pareto筛选（非支配排序）
        for i, (ind, fit) in enumerate(zip(population, fitness_values)):
            is_pareto = True
            
            for j, (other_fit) in enumerate(fitness_values):
                if i != j:
                    # 检查是否被其他解支配
                    if (other_fit[0] <= fit[0] and other_fit[1] <= fit[1] and 
                        (other_fit[0] < fit[0] or other_fit[1] < fit[1])):
                        is_pareto = False
                        break
                        
            if is_pareto:
                decoded_params = self._decode_hyperparameters(ind)
                pareto_configs.append({
                    'hyperparameters': decoded_params,
                    'fitness': fit,
                    'index': i
                })
                
        return pareto_configs
    
    def analyze_pareto_front(self, pareto_configs: List[Dict]) -> Dict[str, Any]:
        """分析Pareto前沿"""
        if not pareto_configs:
            return {}
            
        analysis = {
            'n_solutions': len(pareto_configs),
            'hyperparameter_ranges': {},
            'performance_ranges': {},
            'recommendations': []
        }
        
        # 分析超参数范围
        all_params = [config['hyperparameters'] for config in pareto_configs]
        param_names = list(all_params[0].keys())
        
        for param_name in param_names:
            values = [params[param_name] for params in all_params]
            analysis['hyperparameter_ranges'][param_name] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        # 分析性能范围
        accuracies = [1.0 - config['fitness'][0] for config in pareto_configs]
        complexities = [config['fitness'][1] * 100 for config in pareto_configs]
        
        analysis['performance_ranges'] = {
            'accuracy': {
                'min': min(accuracies),
                'max': max(accuracies),
                'mean': np.mean(accuracies)
            },
            'complexity': {
                'min': min(complexities),
                'max': max(complexities),
                'mean': np.mean(complexities)
            }
        }
        
        # 生成推荐
        best_accuracy_idx = np.argmax(accuracies)
        best_complexity_idx = np.argmin(complexities)
        
        analysis['recommendations'] = [
            {
                'type': 'highest_accuracy',
                'config': pareto_configs[best_accuracy_idx]['hyperparameters'],
                'expected_accuracy': accuracies[best_accuracy_idx]
            },
            {
                'type': 'lowest_complexity', 
                'config': pareto_configs[best_complexity_idx]['hyperparameters'],
                'expected_complexity': complexities[best_complexity_idx]
            }
        ]
        
        return analysis

# 预定义的ML优化问题
class PredefinedMLProblems:
    """预定义的机器学习优化问题"""
    
    @staticmethod
    def rf_classifier_tuning() -> MLProblemDefinition:
        """随机森林分类器调优"""
        return MLProblemDefinition(
            model_type='classification',
            model_name='rf',
            param_space={
                'n_estimators': (10, 200, 'int'),      # 树的数量
                'max_depth': (3, 20, 'int'),           # 最大深度
                'min_samples_split': (2, 20, 'int'),   # 最小分割样本数
                'max_features': (0.1, 1.0, 'float')    # 最大特征比例
            },
            objective_weights=(0.7, 0.3),  # 更重视准确率
            cv_folds=5
        )
    
    @staticmethod
    def svm_classifier_tuning() -> MLProblemDefinition:
        """SVM分类器调优"""
        return MLProblemDefinition(
            model_type='classification',
            model_name='svm',
            param_space={
                'C': (-3, 3, 'log'),         # 正则化参数 (10^-3 to 10^3)
                'gamma': (-3, 2, 'log'),      # 核系数 (10^-3 to 10^2)
                'kernel': (0, 2, 'int')       # 核类型编码
            },
            objective_weights=(0.6, 0.4),
            cv_folds=3
        )
    
    @staticmethod
    def lr_regression_tuning() -> MLProblemDefinition:
        """线性回归调优"""
        return MLProblemDefinition(
            model_type='regression',
            model_name='lr',
            param_space={
                'C': (-2, 2, 'log'),         # 正则化强度
                'max_iter': (100, 2000, 'int')  # 最大迭代次数
            },
            objective_weights=(0.8, 0.2),
            cv_folds=5
        )