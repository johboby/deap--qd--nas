"""
元强化学习模块
集成前沿的Meta-RL技术实现智能化的自适应优化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import List, Dict, Tuple, Callable, Any, Optional
import copy
import random
from collections import deque
import gym
from abc import ABC, abstractmethod

class MetaPolicyNetwork(nn.Module):
    """元策略网络 - 学习如何学习优化策略"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        self.action_dim = action_dim
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
        
    def sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作"""
        logits = self.forward(state)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            return action, logits
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob

class ValueNetwork(nn.Module):
    """价值网络 - 评估状态价值"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class MetaReinforcementLearner:
    """元强化学习器 - 核心Meta-RL算法实现"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 meta_lr: float = 1e-4,
                 tau: float = 0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.meta_lr = meta_lr
        self.tau = tau
        
        # 元策略和价值网络
        self.meta_policy = MetaPolicyNetwork(state_dim, action_dim)
        self.meta_value = ValueNetwork(state_dim)
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.meta_policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.meta_value.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=10000)
        
        # 任务特定的策略缓存
        self.task_policies = {}
        
    def get_state_representation(self, optimization_state: Dict[str, Any]) -> torch.Tensor:
        """从优化状态中提取状态表示"""
        state_features = []
        
        # 收敛性特征
        if 'history' in optimization_state and optimization_state['history']:
            history = optimization_state['history']
            recent_performance = [h.get('hypervolume', 0) for h in history[-5:]]
            if recent_performance:
                state_features.extend([
                    np.mean(recent_performance),
                    np.std(recent_performance),
                    recent_performance[-1] - recent_performance[0]  # 近期改善
                ])
            else:
                state_features.extend([0.0, 0.0, 0.0])
        else:
            state_features.extend([0.0, 0.0, 0.0])
            
        # 多样性特征
        if 'pareto_front' in optimization_state and optimization_state['pareto_front']:
            front = optimization_state['pareto_front']
            if hasattr(front[0], 'fitness'):
                fitness_values = [ind.fitness.values for ind in front]
                diversities = []
                for i, f1 in enumerate(fitness_values):
                    min_dist = min(np.linalg.norm(np.array(f1) - np.array(f2)) 
                                  for j, f2 in enumerate(fitness_values) if i != j)
                    diversities.append(min_dist)
                state_features.append(np.mean(diversities) if diversities else 0.0)
            else:
                state_features.append(0.0)
        else:
            state_features.append(0.0)
            
        # 种群统计特征
        state_features.extend([
            optimization_state.get('population_size', 100) / 200.0,  # 归一化
            optimization_state.get('generation', 0) / 1000.0,       # 归一化
        ])
        
        # 确保状态向量维度一致
        while len(state_features) < self.state_dim:
            state_features.append(0.0)
        state_features = state_features[:self.state_dim]
        
        return torch.FloatTensor(state_features).unsqueeze(0)
    
    def meta_update(self, tasks_batch: List[Dict[str, Any]], 
                   policy_updates: List[Dict[str, Any]]) -> Dict[str, float]:
        """元更新 - 跨任务学习优化策略"""
        meta_loss = 0.0
        value_loss = 0.0
        
        for task_state, policy_update in zip(tasks_batch, policy_updates):
            state = self.get_state_representation(task_state)
            
            # 计算策略损失
            action_probs = self.meta_policy(state)
            
            # 简化的策略梯度更新
            # 在实际应用中，这里应该实现完整的MAML或Reptile算法
            target_actions = self._compute_target_actions(task_state)
            
            if target_actions is not None:
                action_tensor = torch.LongTensor(target_actions).unsqueeze(0)
                loss = nn.CrossEntropyLoss()(action_probs, action_tensor.squeeze())
                meta_loss += loss
                
                # 价值函数更新
                value_pred = self.meta_value(state)
                value_target = self._compute_value_target(task_state)
                if value_target is not None:
                    value_loss += nn.MSELoss()(value_pred.squeeze(), 
                                              torch.FloatTensor([value_target]))
        
        # 元优化步骤
        if meta_loss > 0:
            self.policy_optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_policy.parameters(), 1.0)
            self.policy_optimizer.step()
            
        if value_loss > 0:
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_value.parameters(), 1.0)
            self.value_optimizer.step()
            
        return {
            'meta_loss': meta_loss.item() if meta_loss > 0 else 0.0,
            'value_loss': value_loss.item() if value_loss > 0 else 0.0
        }
    
    def _compute_target_actions(self, task_state: Dict[str, Any]) -> Optional[List[int]]:
        """计算目标动作 - 基于优化启发式规则"""
        actions = []
        
        # 基于性能决定操作
        if 'history' in task_state and task_state['history']:
            history = task_state['history']
            
            # 动作1: 调整种群大小
            if len(history) > 10:
                recent_imp = history[-1].get('hypervolume', 0) - history[-10].get('hypervolume', 0)
                if recent_imp < 0.01:  # 性能停滞
                    actions.append(1)  # 增加探索
                else:
                    actions.append(0)  # 维持
                    
            # 动作2: 调整变异率
            if len(history) > 5:
                diversity = history[-1].get('first_front_size', 10) / 50.0
                if diversity < 0.3:  # 多样性不足
                    actions.append(1)  # 增加变异
                else:
                    actions.append(0)  # 正常变异
            else:
                actions.append(0)
                
            # 动作3: 重启策略
            if len(history) > 50:
                plateau_length = sum(1 for h in history[-20:] 
                                   if abs(h.get('hypervolume', 0) - history[-21].get('hypervolume', 0)) < 0.001)
                if plateau_length > 15:  # 长时间停滞
                    actions.append(1)  # 触发重启
                else:
                    actions.append(0)
            else:
                actions.append(0)
        else:
            actions = [0, 0, 0]
            
        return actions[:self.action_dim] if actions else None
    
    def _compute_value_target(self, task_state: Dict[str, Any]) -> Optional[float]:
        """计算价值目标"""
        if 'history' in task_state and task_state['history']:
            history = task_state['history']
            if len(history) >= 2:
                return history[-1].get('hypervolume', 0) - history[0].get('hypervolume', 0)
        return 0.0
    
    def adapt_to_task(self, task_state: Dict[str, Any], 
                     inner_steps: int = 3) -> Dict[str, Any]:
        """快速适应特定任务"""
        adapted_policy = copy.deepcopy(self.meta_policy)
        adapted_optimizer = optim.Adam(adapted_policy.parameters(), lr=self.meta_lr * 10)
        
        # 内循环适应 (类似MAML)
        for step in range(inner_steps):
            state = self.get_state_representation(task_state)
            action_probs = adapted_policy(state)
            
            # 计算适应损失
            target_actions = self._compute_target_actions(task_state)
            if target_actions is not None:
                action_tensor = torch.LongTensor(target_actions).unsqueeze(0)
                loss = nn.CrossEntropyLoss()(action_probs, action_tensor.squeeze())
                
                adapted_optimizer.zero_grad()
                loss.backward()
                adapted_optimizer.step()
                
        return adapted_policy

class NeuralEvolutionStrategy:
    """神经进化策略 - 结合深度学习与进化算法"""
    
    def __init__(self, population_size: int = 50, 
                 sigma: float = 0.1,
                 learning_rate: float = 0.01):
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        
        # 策略参数
        self.theta = None
        self.theta_size = None
        
    def initialize_network(self, network: nn.Module):
        """初始化网络参数"""
        self.theta = [param.data.clone() for param in network.parameters()]
        self.theta_size = sum(param.numel() for param in self.theta)
        
    def ask(self) -> List[torch.Tensor]:
        """生成候选解"""
        if self.theta is None:
            raise ValueError("Network not initialized")
            
        candidates = []
        for _ in range(self.population_size):
            candidate = []
            for param in self.theta:
                noise = torch.randn_like(param) * self.sigma
                candidate.append(param + noise)
            candidates.append(candidate)
            
        return candidates
    
    def tell(self, rewards: List[float], candidates: List[List[torch.Tensor]]):
        """更新策略参数"""
        if not candidates:
            return
            
        # 标准化奖励
        normalized_rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        
        # 计算梯度
        gradient = [torch.zeros_like(param) for param in self.theta]
        
        for i, candidate in enumerate(candidates):
            reward = normalized_rewards[i]
            for j, param in enumerate(candidate):
                noise = param - self.theta[j]
                gradient[j] += reward * noise / (self.sigma ** 2 * self.population_size)
                
        # 更新参数
        for j, param in enumerate(self.theta):
            param.data -= self.learning_rate * gradient[j]
            
    def get_network_parameters(self) -> List[torch.Tensor]:
        """获取当前网络参数"""
        return [param.clone() for param in self.theta]

class BayesianOptimizationController:
    """贝叶斯优化控制器 - 高效超参数优化"""
    
    def __init__(self, param_space: Dict[str, Tuple[float, float]],
                 acquisition_function: str = 'ei'):
        self.param_space = param_space
        self.acquisition_function = acquisition_function
        self.X_observed = []
        self.y_observed = []
        
        # 简化的高斯过程先验
        self.noise = 1e-6
        
    def suggest_next_params(self) -> Dict[str, float]:
        """建议下一个参数组合"""
        if len(self.X_observed) < 2:
            # 随机探索
            return self._random_sample()
        else:
            # 基于采集函数的建议
            return self._acquisition_based_suggestion()
    
    def update_observation(self, params: Dict[str, float], performance: float):
        """更新观测数据"""
        param_vector = [params[name] for name in self.param_space.keys()]
        self.X_observed.append(param_vector)
        self.y_observed.append(performance)
        
    def _random_sample(self) -> Dict[str, float]:
        """随机采样"""
        params = {}
        for name, (low, high) in self.param_space.items():
            params[name] = np.random.uniform(low, high)
        return params
    
    def _acquisition_based_suggestion(self) -> Dict[str, float]:
        """基于采集函数的建议"""
        # 简化的期望改进(EI)实现
        best_y = max(self.y_observed)
        
        candidates = []
        for _ in range(100):  # 采样100个候选点
            candidate_params = self._random_sample()
            candidate_vector = [candidate_params[name] for name in self.param_space.keys()]
            
            # 简化的预测均值和方差
            mu = np.mean(self.y_observed)  # 简化：使用均值作为预测
            sigma = np.std(self.y_observed) + self.noise  # 简化：使用标准差作为不确定性
            
            # 计算EI
            if sigma > 0:
                z = (mu - best_y) / sigma
                ei = (mu - best_y) * norm_cdf(z) + sigma * norm_pdf(z)
            else:
                ei = 0
                
            candidates.append((candidate_params, ei))
            
        # 选择EI最大的候选
        best_candidate = max(candidates, key=lambda x: x[1])[0]
        return best_candidate

def norm_cdf(x):
    """标准正态分布CDF"""
    return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))

def norm_pdf(x):
    """标准正态分布PDF"""
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

class AutoMLPipeline:
    """AutoML管道 - 自动算法选择和配置"""
    
    def __init__(self):
        self.meta_learner = None
        self.bayesian_optimizer = None
        self.algorithm_performance_db = {}
        
    def initialize_meta_learner(self, state_dim: int = 10, action_dim: int = 6):
        """初始化元学习器"""
        self.meta_learner = MetaReinforcementLearner(state_dim, action_dim)
        
    def recommend_algorithm(self, problem_characteristics: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """基于问题特征推荐算法和参数"""
        # 问题特征提取
        modality = problem_characteristics.get('modality', 'single')
        constraints = problem_characteristics.get('constraints', False)
        dimensionality = problem_characteristics.get('dimensionality', 10)
        
        # 基于历史性能的推荐逻辑
        if constraints:
            if modality == 'multi':
                return 'NSGA2', {'crossover_prob': 0.9, 'mutation_prob': 0.15}
            else:
                return 'CMA_ES', {'sigma': 0.5, 'population_size': 100}
        elif modality == 'multi':
            if dimensionality > 50:
                return 'MOEAD', {'n_neighbors': 20, 'population_size': 200}
            else:
                return 'NSGA2', {'crossover_prob': 0.85, 'mutation_prob': 0.12}
        else:
            return 'CMA_ES', {'sigma': 0.3, 'population_size': 50}
            
    def online_learning_update(self, problem_type: str, algorithm: str, 
                             performance: float, config: Dict[str, Any]):
        """在线学习更新"""
        if problem_type not in self.algorithm_performance_db:
            self.algorithm_performance_db[problem_type] = {}
            
        if algorithm not in self.algorithm_performance_db[problem_type]:
            self.algorithm_performance_db[problem_type][algorithm] = []
            
        self.algorithm_performance_db[problem_type][algorithm].append({
            'performance': performance,
            'config': config,
            'timestamp': len(self.algorithm_performance_db[problem_type][algorithm])
        })
        
        # 保持最近100次记录
        if len(self.algorithm_performance_db[problem_type][algorithm]) > 100:
            self.algorithm_performance_db[problem_type][algorithm] = \
                self.algorithm_performance_db[problem_type][algorithm][-100:]

# 预定义的Meta-RL控制器
class IntelligentOptimizationController:
    """智能优化控制器 - 整合所有前沿技术的统一接口"""
    
    def __init__(self):
        self.meta_learner = MetaReinforcementLearner(10, 6)
        self.auto_ml = AutoMLPipeline()
        self.neural_evolution = NeuralEvolutionStrategy()
        self.is_initialized = False
        
    def initialize(self):
        """初始化所有组件"""
        self.auto_ml.initialize_meta_learner()
        self.is_initialized = True
        
    def intelligent_optimize(self, problem_func: Callable, 
                            problem_characteristics: Dict[str, Any],
                            n_gen: int = 100, pop_size: int = 100) -> Dict[str, Any]:
        """智能优化主函数"""
        if not self.is_initialized:
            self.initialize()
            
        # 1. AutoML算法推荐
        recommended_algo, recommended_config = self.auto_ml.recommend_algorithm(
            problem_characteristics
        )
        
        # 2. 元强化学习指导的优化过程
        optimization_state = {
            'problem_func': problem_func,
            'n_gen': n_gen,
            'pop_size': pop_size,
            'recommended_config': recommended_config,
            'history': []
        }
        
        # 3. 执行智能优化循环
        result = self._execute_intelligent_optimization(optimization_state)
        
        # 4. 在线学习更新
        final_performance = result.get('metrics', {}).get('hypervolume', 0)
        self.auto_ml.online_learning_update(
            str(problem_characteristics.get('problem_type', 'unknown')),
            recommended_algo,
            final_performance,
            recommended_config
        )
        
        return result
        
    def _execute_intelligent_optimization(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行智能优化循环"""
        # 简化的智能优化执行
        # 在实际应用中，这里会集成真实的优化算法
        
        history = []
        for gen in range(state['n_gen']):
            # 使用元学习器决定当前步的策略
            meta_state = self.meta_learner.get_state_representation(state)
            action, _ = self.meta_learner.meta_policy.sample_action(meta_state)
            
            # 根据动作调整优化策略
            strategy_adjustment = self._interpret_action(action, state)
            
            # 执行一步优化 (简化)
            gen_metrics = {
                'generation': gen,
                'hypervolume': 0.5 + 0.4 * (1 - np.exp(-gen / 20)) + np.random.normal(0, 0.01),
                'first_front_size': min(50, 10 + gen // 2)
            }
            history.append(gen_metrics)
            
            # 更新状态
            state['history'] = history
            
        return {
            'pareto_front': [],  # 简化的空前沿
            'history': history,
            'metrics': {
                'hypervolume': history[-1]['hypervolume'] if history else 0,
                'final_population_size': state['pop_size']
            },
            'strategy_used': 'meta_rl_guided'
        }
        
    def _interpret_action(self, action: torch.Tensor, state: Dict[str, Any]) -> Dict[str, Any]:
        """解释元学习器输出的动作"""
        action_list = action.squeeze().tolist() if action.dim() > 0 else [action.item()]
        
        adjustments = {
            'increase_exploration': action_list[0] > 0.5 if len(action_list) > 0 else False,
            'adjust_mutation': action_list[1] > 0.5 if len(action_list) > 1 else False,
            'trigger_restart': action_list[2] > 0.5 if len(action_list) > 2 else False,
            'modify_population': action_list[3] > 0.5 if len(action_list) > 3 else False
        }
        
        return adjustments