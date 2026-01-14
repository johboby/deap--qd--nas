"""
错误处理和恢复机制 (Error Handling and Recovery)
改进的错误处理、恢复机制和异常管理
"""

import logging
import traceback
import time
import os
import json
from typing import List, Dict, Any, Optional, Callable, Type
from dataclasses import dataclass, field
from functools import wraps
from contextlib import contextmanager
from abc import ABC, abstractmethod

from src.core.exceptions import DEAPError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RecoveryAction:
    """
    恢复动作

    Args:
        name: 动作名称
        execute: 执行函数
        priority: 优先级
        max_attempts: 最大尝试次数
    """
    name: str
    execute: Callable[[], bool]
    priority: int = 1
    max_attempts: int = 3


class RecoveryStrategy(ABC):
    """
    恢复策略基类
    """

    @abstractmethod
    def recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        尝试恢复

        Args:
            error: 发生的错误
            context: 上下文信息

        Returns:
            是否恢复成功
        """
        pass


class RetryStrategy(RecoveryStrategy):
    """
    重试策略

    简单的重试恢复策略。
    """

    def __init__(self, max_attempts: int = 3, delay: float = 1.0):
        """
        初始化重试策略

        Args:
            max_attempts: 最大尝试次数
            delay: 重试延迟（秒）
        """
        self.max_attempts = max_attempts
        self.delay = delay
        self.attempts = 0

    def recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        尝试重试恢复

        Args:
            error: 发生的错误
            context: 上下文信息

        Returns:
            是否应该继续重试
        """
        self.attempts += 1

        if self.attempts >= self.max_attempts:
            logger.error(f"❌ 重试策略失败：达到最大尝试次数 {self.max_attempts}")
            return False

        logger.warning(f"⚠️  重试 {self.attempts}/{self.max_attempts}...")
        time.sleep(self.delay)
        return True


class CheckpointRecoveryStrategy(RecoveryStrategy):
    """
    检查点恢复策略

    从检查点恢复。
    """

    def __init__(self, checkpoint_dir: str):
        """
        初始化检查点恢复策略

        Args:
            checkpoint_dir: 检查点目录
        """
        self.checkpoint_dir = checkpoint_dir

    def recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        尝试从检查点恢复

        Args:
            error: 发生的错误
            context: 上下文信息

        Returns:
            是否恢复成功
        """
        logger.info(f"💾 尝试从检查点恢复...")

        # 检查是否有检查点
        checkpoint_files = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.pth') or file.endswith('.pkl'):
                checkpoint_files.append(os.path.join(self.checkpoint_dir, file))

        if not checkpoint_files:
            logger.warning("⚠️  未找到检查点文件")
            return False

        # 加载最新的检查点
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        logger.info(f"📥 加载检查点: {latest_checkpoint}")

        # 这里应该实际加载检查点
        # context['checkpoint'] = load_checkpoint(latest_checkpoint)

        return True


class FallbackStrategy(RecoveryStrategy):
    """
    回退策略

    使用回退方法。
    """

    def __init__(self, fallback_function: Callable):
        """
        初始化回退策略

        Args:
            fallback_function: 回退函数
        """
        self.fallback_function = fallback_function

    def recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        执行回退函数

        Args:
            error: 发生的错误
            context: 上下文信息

        Returns:
            回退是否成功
        """
        logger.info(f"🔄 执行回退函数...")
        try:
            self.fallback_function()
            return True
        except Exception as e:
            logger.error(f"❌ 回退函数失败: {e}")
            return False


class ErrorHandler:
    """
    错误处理器

    统一处理和记录错误。
    """

    def __init__(self,
                 error_log_file: Optional[str] = None,
                 enable_recovery: bool = True):
        """
        初始化错误处理器

        Args:
            error_log_file: 错误日志文件
            enable_recovery: 是否启用恢复
        """
        self.error_log_file = error_log_file
        self.enable_recovery = enable_recovery

        # 错误统计
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []

        # 恢复策略
        self.recovery_strategies: List[RecoveryStrategy] = []

    def register_recovery_strategy(self, strategy: RecoveryStrategy):
        """
        注册恢复策略

        Args:
            strategy: 恢复策略
        """
        self.recovery_strategies.append(strategy)
        logger.info(f"✅ 注册恢复策略: {strategy.__class__.__name__}")

    def handle_error(self,
                   error: Exception,
                   context: Optional[Dict[str, Any]] = None) -> bool:
        """
        处理错误

        Args:
            error: 错误对象
            context: 上下文信息

        Returns:
            是否成功处理
        """
        context = context or {}

        # 记录错误
        self._log_error(error, context)

        # 统计错误
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # 尝试恢复
        if self.enable_recovery:
            for strategy in self.recovery_strategies:
                try:
                    if strategy.recover(error, context):
                        logger.info(f"✅ 恢复策略成功: {strategy.__class__.__name__}")
                        return True
                except Exception as e:
                    logger.error(f"❌ 恢复策略失败: {e}")

        return False

    def _log_error(self, error: Exception, context: Dict[str, Any]):
        """记录错误"""
        error_info = {
            'timestamp': time.time(),
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
        }

        self.error_history.append(error_info)

        # 输出到日志
        logger.error(
            f"❌ 错误: {type(error).__name__}: {error}\n"
            f"Context: {context}"
        )

        # 保存到文件
        if self.error_log_file:
            os.makedirs(os.path.dirname(self.error_log_file), exist_ok=True)
            with open(self.error_log_file, 'a') as f:
                f.write(json.dumps(error_info, indent=2) + '\n')

    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': self.error_counts,
            'recent_errors': self.error_history[-10:],
        }

    def clear_history(self):
        """清除历史"""
        self.error_history = []
        logger.info("✅ 错误历史已清除")


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    重试装饰器

    Args:
        max_attempts: 最大尝试次数
        delay: 重试延迟（秒）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler()
            handler.register_recovery_strategy(RetryStrategy(max_attempts, delay))

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = {
                        'function': func.__name__,
                        'attempt': attempt + 1,
                    }

                    if attempt == max_attempts - 1:
                        # 最后一次尝试失败，抛出异常
                        raise

                    # 尝试恢复
                    if not handler.handle_error(e, context):
                        raise

                    time.sleep(delay)

        return wrapper
    return decorator


def safe_execute(default_value: Any = None):
    """
    安全执行装饰器

    Args:
        default_value: 默认返回值
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"❌ 函数 {func.__name__} 执行失败: {e}")
                return default_value
        return wrapper
    return decorator


@contextmanager
def error_context(name: str, fallback: Optional[Callable] = None):
    """
    错误上下文管理器

    Args:
        name: 上下文名称
        fallback: 回退函数
    """
    handler = ErrorHandler()
    error_occurred = False

    try:
        yield handler
    except Exception as e:
        error_occurred = True
        context = {'context_name': name}

        logger.error(f"❌ 错误上下文 [{name}]: {e}")

        # 尝试处理错误
        if not handler.handle_error(e, context) and fallback is not None:
            logger.info("🔄 执行回退函数...")
            try:
                fallback()
            except Exception as fallback_error:
                logger.error(f"❌ 回退函数失败: {fallback_error}")
                raise
        else:
            raise
    finally:
        if not error_occurred:
            logger.info(f"✅ 上下文 [{name}] 正常完成")


class CircuitBreaker:
    """
    熔断器

    防止级联故障。
    """

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0):
        """
        初始化熔断器

        Args:
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时（秒）
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs):
        """
        通过熔断器调用函数

        Args:
            func: 函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数返回值

        Raises:
            CircuitBreakerOpenError: 熔断器打开时
        """
        # 检查熔断器状态
        if self.state == 'open':
            # 检查是否可以进入半开状态
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
                logger.info("🔄 熔断器进入半开状态")
            else:
                raise Exception("熔断器打开，拒绝请求")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """成功回调"""
        if self.state == 'half-open':
            self.state = 'closed'
            self.failure_count = 0
            logger.info("✅ 熔断器关闭，服务恢复")

    def _on_failure(self):
        """失败回调"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.error(f"🔥 熔断器打开：失败次数 {self.failure_count}")


class ErrorRecoveryManager:
    """
    错误恢复管理器

    统一管理多种恢复策略。
    """

    def __init__(self):
        """初始化错误恢复管理器"""
        self.strategies: Dict[str, RecoveryStrategy] = {}
        self.error_handler = ErrorHandler()

    def register_strategy(self, name: str, strategy: RecoveryStrategy):
        """
        注册恢复策略

        Args:
            name: 策略名称
            strategy: 恢复策略
        """
        self.strategies[name] = strategy
        self.error_handler.register_recovery_strategy(strategy)

    def recover(self,
                error: Exception,
                context: Optional[Dict[str, Any]] = None) -> bool:
        """
        执行恢复

        Args:
            error: 错误对象
            context: 上下文信息

        Returns:
            是否恢复成功
        """
        return self.error_handler.handle_error(error, context)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.error_handler.get_error_statistics()


def handle_errors(error_log_file: str = './error_log.json'):
    """
    错误处理装饰器工厂函数

    Args:
        error_log_file: 错误日志文件

    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler(error_log_file=error_log_file)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {'function': func.__name__}
                if not handler.handle_error(e, context):
                    raise
        return wrapper
    return decorator


__all__ = [
    'RecoveryAction',
    'RecoveryStrategy',
    'RetryStrategy',
    'CheckpointRecoveryStrategy',
    'FallbackStrategy',
    'ErrorHandler',
    'retry',
    'safe_execute',
    'error_context',
    'CircuitBreaker',
    'ErrorRecoveryManager',
    'handle_errors',
]
