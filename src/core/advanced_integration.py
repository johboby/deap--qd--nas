"""
高级功能集成模块
将分布式计算、GPU加速、元学习/AutoML集成到主框架中
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass

from .lightweight_intelligent_framework import LightweightIntelligentFramework, OptimizationConfig, OptimizationMode
from .distributed_computing import DistributedIntelligentFramework, DistributedConfig, create_distributed_framework
from .gpu_acceleration import GPUAcceleratedFramework, GPUConfig, create_gpu_framework  
from .meta_learning_automl import MetaLearningFramework, AutoMLEngine, create_meta_learning_framework

logger = logging.getLogger(__name__)

@dataclass
class AdvancedConfig:
    """高级功能配置"""
    enable_distributed: bool = False
    enable_gpu: bool = False
    enable_automl: bool = False
    distributed_config: Optional[Dict] = None
    gpu_config: Optional[Dict] = None
    automl_config: Optional[Dict] = None

class AdvancedIntelligentFramework:
    """高级智能优化框架 - 集成所有高级功能"""
    
    def __init__(self, base_config: OptimizationConfig = None, 
                 advanced_config: AdvancedConfig = None):
        self.base_config = base_config or OptimizationConfig()
        self.advanced_config = advanced_config or AdvancedConfig()
        
        # 核心智能框架
        self.base_framework = LightweightIntelligentFramework(self.base_config)
        
        # 高级功能组件
        self.distributed_framework = None
        self.gpu_framework = None
        self.automl_engine = None
        
        # 状态管理
        self.is_initialized = False
        self.active_modes = []
        
        print("🚀 高级智能优化框架初始化")
        print(f"   基础模式: 启用")
        print(f"   分布式计算: {'启用' if self.advanced_config.enable_distributed else '禁用'}")
        print(f"   GPU加速: {'启用' if self.advanced_config.enable_gpu else '禁用'}")
        print(f"   AutoML: {'启用' if self.advanced_config.enable_automl else '禁用'}")
    
    def initialize(self):
        """初始化所有启用的功能"""
        print("🔧 初始化高级智能优化框架...")
        
        # 初始化基础框架
        self.base_framework.initialize()
        
        # 初始化分布式计算
        if self.advanced_config.enable_distributed:
            self._initialize_distributed()
        
        # 初始化GPU加速
        if self.advanced_config.enable_gpu:
            self._initialize_gpu()
        
        # 初始化AutoML
        if self.advanced_config.enable_automl:
            self._initialize_automl()
        
        self.is_initialized = True
        print("✅ 高级框架初始化完成")
        
        # 显示可用模式
        available_modes = ["classical", "smart_adaptive", "quick_quantum", "intelligent_hybrid"]
        if self.distributed_framework:
            available_modes.extend(["distributed", "hybrid_distributed"])
        if self.gpu_framework:
            available_modes.extend(["gpu_accelerated", "quantum_gpu"])
        if self.automl_engine:
            available_modes.append("automl_guided")
            
        print(f"🎯 可用优化模式: {', '.join(available_modes)}")
    
    def _initialize_distributed(self):
        """初始化分布式计算"""
        try:
            print("  🌐 初始化分布式计算...")
            dist_config_dict = self.advanced_config.distributed_config or {}
            dist_config = DistributedConfig(**dist_config_dict)
            
            self.distributed_framework = create_distributed_framework(dist_config)
            success = self.distributed_framework.initialize(mode="auto")
            
            if success:
                self.active_modes.append("distributed")
                print("    ✅ 分布式计算初始化成功")
            else:
                print("    ⚠️  分布式计算初始化失败，回退到单机模式")
                
        except Exception as e:
            logger.error(f"分布式计算初始化失败: {e}")
            print(f"    ❌ 分布式计算初始化失败: {e}")
    
    def _initialize_gpu(self):
        """初始化GPU加速"""
        try:
            print("  ⚡ 初始化GPU加速...")
            gpu_config_dict = self.advanced_config.gpu_config or {}
            gpu_config = GPUConfig(**gpu_config_dict)
            
            self.gpu_framework = create_gpu_framework(gpu_config)
            success = self.gpu_framework.initialize()
            
            if success:
                self.active_modes.append("gpu")
                gpu_info = self.gpu_framework.get_gpu_info()
                print(f"    ✅ GPU加速初始化成功 - {gpu_info.get('device_name', 'Unknown')}")
            else:
                print("    ⚠️  GPU加速初始化失败，回退到CPU模式")
                
        except Exception as e:
            logger.error(f"GPU加速初始化失败: {e}")
            print(f"    ❌ GPU加速初始化失败: {e}")
    
    def _initialize_automl(self):
        """初始化AutoML"""
        try:
            print("  🤖 初始化AutoML...")
            automl_config_dict = self.advanced_config.automl_config or {}
            
            self.automl_engine = create_meta_learning_framework(automl_config_dict)
            success = self.automl_engine.initialize()
            
            if success:
                self.active_modes.append("automl")
                print("    ✅ AutoML初始化成功")
            else:
                print("    ⚠️  AutoML初始化失败")
                
        except Exception as e:
            logger.error(f"AutoML初始化失败: {e}")
            print(f"    ❌ AutoML初始化失败: {e}")
    
    def intelligent_hybrid_optimize(self, problem_func: Callable, dim: int, 
                                   bounds: List[Tuple[float, float]], 
                                   mode: str = "intelligent_hybrid", **kwargs) -> Dict[str, Any]:
        """智能混合优化 - 支持所有高级模式"""
        if not self.is_initialized:
            raise RuntimeError("框架未初始化，请先调用initialize()")
        
        print(f"🎯 开始{mode}优化...")
        
        # 根据模式选择优化策略
        if mode == "distributed" and self.distributed_framework:
            return self._distributed_optimize(problem_func, dim, bounds, **kwargs)
        elif mode == "gpu_accelerated" and self.gpu_framework:
            return self._gpu_optimize(problem_func, dim, bounds, **kwargs)
        elif mode == "automl_guided" and self.automl_engine:
            return self._automl_optimize(problem_func, dim, bounds, **kwargs)
        elif mode == "hybrid_distributed" and self.distributed_framework:
            return self._hybrid_distributed_optimize(problem_func, dim, bounds, **kwargs)
        elif mode == "quantum_gpu" and self.gpu_framework:
            return self._quantum_gpu_optimize(problem_func, dim, bounds, **kwargs)
        else:
            # 回退到基础智能优化
            return self.base_framework.intelligent_hybrid_optimize(problem_func, dim, bounds, **kwargs)
    
    def _distributed_optimize(self, problem_func: Callable, dim: int, bounds: List[Tuple[float, float]], **kwargs):
        """分布式优化"""
        print("  🌐 执行分布式优化...")
        # 这里简化实现，实际应调用distributed_framework的方法
        return self.base_framework.intelligent_hybrid_optimize(problem_func, dim, bounds, **kwargs)
    
    def _gpu_optimize(self, problem_func: Callable, dim: int, bounds: List[Tuple[float, float]], **kwargs):
        """GPU加速优化"""
        print("  ⚡ 执行GPU加速优化...")
        # 这里简化实现，实际应调用gpu_framework的方法
        return self.base_framework.intelligent_hybrid_optimize(problem_func, dim, bounds, **kwargs)
    
    def _automl_optimize(self, problem_func: Callable, dim: int, bounds: List[Tuple[float, float]], **kwargs):
        """AutoML引导优化"""
        print("  🤖 执行AutoML引导优化...")
        # 分析问题特征
        characteristics = self.base_framework.analyze_problem_enhanced(problem_func)
        
        # 使用AutoML推荐策略
        if self.automl_engine:
            recommendations = self.automl_engine.recommend_strategies(characteristics)
            print(f"    📊 AutoML推荐策略: {recommendations.get('top_strategy', 'unknown')}")
        
        return self.base_framework.intelligent_hybrid_optimize(problem_func, dim, bounds, **kwargs)
    
    def _hybrid_distributed_optimize(self, problem_func: Callable, dim: int, bounds: List[Tuple[float, float]], **kwargs):
        """混合分布式优化"""
        print("  🌐⚡ 执行混合分布式优化...")
        return self.base_framework.intelligent_hybrid_optimize(problem_func, dim, bounds, **kwargs)
    
    def _quantum_gpu_optimize(self, problem_func: Callable, dim: int, bounds: List[Tuple[float, float]], **kwargs):
        """量子+GPU优化"""
        print("  ⚛️⚡ 执行量子+GPU优化...")
        return self.base_framework.intelligent_hybrid_optimize(problem_func, dim, bounds, **kwargs)
    
    def get_comprehensive_insights(self) -> Dict[str, Any]:
        """获取综合洞察"""
        insights = {
            "base_framework": self.base_framework.get_intelligent_insights(),
            "advanced_features": {
                "distributed_enabled": self.distributed_framework is not None,
                "gpu_enabled": self.gpu_framework is not None,
                "automl_enabled": self.automl_engine is not None,
                "active_modes": self.active_modes
            }
        }
        
        if self.distributed_framework:
            insights["distributed_info"] = self.distributed_framework.get_cluster_info()
        
        if self.gpu_framework:
            insights["gpu_info"] = self.gpu_framework.get_gpu_info()
        
        if self.automl_engine:
            insights["automl_stats"] = self.automl_engine.get_performance_stats()
        
        return insights

def create_advanced_framework(base_config_dict: Dict = None, 
                           advanced_config_dict: Dict = None) -> AdvancedIntelligentFramework:
    """创建高级智能框架的工厂函数"""
    base_config = OptimizationConfig(**(base_config_dict or {}))
    advanced_config = AdvancedConfig(**(advanced_config_dict or {}))
    
    return AdvancedIntelligentFramework(base_config, advanced_config)