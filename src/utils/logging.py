"""
日志系统模块
提供结构化的日志记录功能
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import threading

class OptimizationLogger:
    """优化日志记录器"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO",
                 enable_console: bool = True, enable_file: bool = True):
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper())
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.logger = None
        self._lock = threading.Lock()
        
        self._setup_logger()
        
    def _setup_logger(self):
        """设置日志器"""
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger(f"OptimizationLogger_{id(self)}")
        self.logger.setLevel(self.log_level)
        
        # 避免重复添加handler
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # 创建formatter
        formatter = OptimizationFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台handler
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
        # 文件handler
        if self.enable_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.log_dir, f"optimization_{timestamp}.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        self.logger.propagate = False
        
    def info(self, message: str, extra: Optional[Dict] = None):
        """记录信息级别日志"""
        with self._lock:
            if extra:
                self.logger.info(f"{message} | Extra: {json.dumps(extra)}")
            else:
                self.logger.info(message)
                
    def warning(self, message: str, extra: Optional[Dict] = None):
        """记录警告级别日志"""
        with self._lock:
            if extra:
                self.logger.warning(f"{message} | Extra: {json.dumps(extra)}")
            else:
                self.logger.warning(message)
                
    def error(self, message: str, extra: Optional[Dict] = None, exc_info: bool = False):
        """记录错误级别日志"""
        with self._lock:
            if extra:
                self.logger.error(f"{message} | Extra: {json.dumps(extra)}", exc_info=exc_info)
            else:
                self.logger.error(message, exc_info=exc_info)
                
    def debug(self, message: str, extra: Optional[Dict] = None):
        """记录调试级别日志"""
        with self._lock:
            if extra:
                self.logger.debug(f"{message} | Extra: {json.dumps(extra)}")
            else:
                self.logger.debug(message)
                
    def log_experiment_start(self, experiment_name: str, config: Dict[str, Any]):
        """记录实验开始"""
        self.info(f"Starting experiment: {experiment_name}", 
                extra={'event_type': 'experiment_start', 'config': config})
        
    def log_experiment_end(self, experiment_name: str, result_summary: Dict[str, Any]):
        """记录实验结束"""
        self.info(f"Completed experiment: {experiment_name}", 
                extra={'event_type': 'experiment_end', 'result_summary': result_summary})
        
    def log_generation(self, generation: int, population_size: int, 
                      pareto_front_size: int, metrics: Dict[str, float]):
        """记录代数信息"""
        self.debug(f"Generation {generation} completed", 
                 extra={'event_type': 'generation', 
                       'generation': generation,
                       'population_size': population_size,
                       'pareto_front_size': pareto_front_size,
                       'metrics': metrics})
        
    def log_algorithm_event(self, event_type: str, algorithm_name: str, 
                          details: Dict[str, Any]):
        """记录算法事件"""
        self.info(f"Algorithm event: {event_type}", 
                extra={'event_type': 'algorithm_event',
                       'algorithm': algorithm_name,
                       'details': details})

class OptimizationFormatter(logging.Formatter):
    """优化专用的日志格式化器"""
    
    def format(self, record):
        # 添加自定义字段
        if hasattr(record, 'extra'):
            record.message = f"{record.msg} | {record.extra}"
        return super().format(record)

class MetricsLogger:
    """性能指标记录器"""
    
    def __init__(self, logger: OptimizationLogger):
        self.logger = logger
        self.metrics_history = []
        
    def log_metrics(self, generation: int, metrics: Dict[str, float], 
                   population_stats: Optional[Dict] = None):
        """记录性能指标"""
        metric_entry = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'population_stats': population_stats or {}
        }
        
        self.metrics_history.append(metric_entry)
        
        # 记录到日志
        self.logger.log_generation(
            generation=generation,
            population_size=population_stats.get('population_size', 0),
            pareto_front_size=population_stats.get('pareto_front_size', 0),
            metrics=metrics
        )
        
    def save_metrics_history(self, filepath: str):
        """保存指标历史到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics_history': self.metrics_history,
                'summary': self._calculate_summary()
            }, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Metrics history saved to {filepath}")
        
    def _calculate_summary(self) -> Dict[str, Any]:
        """计算指标摘要"""
        if not self.metrics_history:
            return {}
            
        summary = {}
        
        # 收集所有指标名称
        all_metrics = set()
        for entry in self.metrics_history:
            all_metrics.update(entry['metrics'].keys())
            
        # 计算每个指标的统计信息
        for metric in all_metrics:
            values = [entry['metrics'].get(metric, 0) for entry in self.metrics_history 
                     if metric in entry['metrics']]
            
            if values:
                summary[metric] = {
                    'final': values[-1] if values else 0,
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'improvement': values[-1] - values[0] if len(values) > 1 else 0
                }
                
        return summary

class ExperimentTracker:
    """实验跟踪器"""
    
    def __init__(self, logger: OptimizationLogger):
        self.logger = logger
        self.experiments = {}
        
    def start_experiment(self, experiment_id: str, config: Dict[str, Any]):
        """开始跟踪实验"""
        self.experiments[experiment_id] = {
            'start_time': datetime.now(),
            'config': config,
            'status': 'running',
            'events': []
        }
        
        self.logger.log_experiment_start(experiment_id, config)
        
    def add_event(self, experiment_id: str, event_type: str, details: Dict[str, Any]):
        """添加实验事件"""
        if experiment_id in self.experiments:
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': event_type,
                'details': details
            }
            self.experiments[experiment_id]['events'].append(event)
            
    def end_experiment(self, experiment_id: str, result_summary: Dict[str, Any]):
        """结束实验跟踪"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['end_time'] = datetime.now()
            self.experiments[experiment_id]['status'] = 'completed'
            self.experiments[experiment_id]['result_summary'] = result_summary
            
            duration = (self.experiments[experiment_id]['end_time'] - 
                       self.experiments[experiment_id]['start_time']).total_seconds()
            
            result_summary['duration_seconds'] = duration
            
            self.logger.log_experiment_end(experiment_id, result_summary)
            
    def get_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验报告"""
        if experiment_id in self.experiments:
            return self.experiments[experiment_id]
        return {}

# 全局日志器实例
_global_logger = None

def get_logger(log_dir: str = "logs", log_level: str = "INFO",
              enable_console: bool = True, enable_file: bool = True) -> OptimizationLogger:
    """获取全局日志器实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = OptimizationLogger(log_dir, log_level, enable_console, enable_file)
    return _global_logger