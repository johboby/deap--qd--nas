# DEAP框架编码规范和最佳实践

本文档定义了DEAP框架的编码标准和最佳实践。

## 📋 目录

1. [Python编码规范](#python编码规范)
2. [项目结构规范](#项目结构规范)
3. [API设计原则](#api设计原则)
4. [文档编写标准](#文档编写标准)
5. [测试编写指南](#测试编写指南)
6. [性能优化指南](#性能优化指南)
7. [安全最佳实践](#安全最佳实践)

## Python编码规范

### 1. 基础规范 (遵循PEP 8)

#### 缩进和空格
```python
# ✅ 正确：使用4个空格
def function():
    if True:
        pass

# ❌ 错误：混合制表符和空格
def function():
	if True:
        pass
```

#### 行长度
```python
# ✅ 正确：不超过100字符
def complex_function(
    param1: int,
    param2: str,
    param3: Optional[float] = None
) -> Dict[str, Any]:
    """函数文档"""
    pass

# ❌ 错误：过长的行
def complex_function(param1: int, param2: str, param3: Optional[float] = None) -> Dict[str, Any]:
```

#### 命名规范
```python
# 模块和文件名：小写，用下划线分隔
# ✅ my_module.py, utils.py

# 类名：CamelCase
# ✅ class MyAlgorithm, class ArchiveManager

# 函数和变量名：snake_case
# ✅ def get_best_solution(), archive_size = 100

# 常量：大写，用下划线分隔
# ✅ MAX_ITERATIONS = 1000, DEFAULT_MUTATION_RATE = 0.1

# 私有变量/函数：前缀下划线
# ✅ def _initialize_population(), self._cache

# 特殊方法：双下划线
# ✅ def __init__, def __str__, def __eq__
```

### 2. 类型注解

```python
from typing import List, Dict, Tuple, Optional, Callable, Union

# ✅ 完整的类型注解
class Archive:
    def __init__(self, grid_shape: Tuple[int, ...]) -> None:
        self.grid_shape: Tuple[int, ...] = grid_shape
        self.entries: Dict[Tuple, ArchiveEntry] = {}
    
    def add(
        self,
        solution: List[float],
        behavior: List[float],
        fitness: Union[float, List[float]]
    ) -> bool:
        """添加解到档案"""
        ...
    
    def get_best(self) -> Optional[ArchiveEntry]:
        """获取最优解"""
        ...

# ❌ 不完整的类型注解
class Archive:
    def add(self, solution, behavior, fitness):
        """添加解"""
        ...
```

### 3. 导入规范

```python
# ✅ 正确的导入顺序和风格

# 1. 标准库
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional

# 2. 第三方库
import numpy as np
import scipy
from dataclasses import dataclass

# 3. 本地导入
from .base_algorithms import BaseAlgorithm
from ..core.metrics import PerformanceMetrics

# ❌ 错误的导入
from .base_algorithms import *  # 不要使用通配符导入
import numpy, scipy, logging    # 每行一个导入
```

### 4. 异常处理

```python
# ✅ 正确的异常处理
try:
    result = optimize(problem_func, n_iterations=100)
except ValueError as e:
    logger.error(f"Invalid configuration: {e}")
    raise ConfigurationError(f"Invalid parameters: {e}") from e
except OptimizationError as e:
    logger.warning(f"Optimization failed: {e}, trying recovery...")
    return fallback_result()

# ❌ 错误的异常处理
try:
    result = optimize(problem_func)
except:  # 不要捕获所有异常
    pass  # 不要忽略异常
```

## 项目结构规范

### 1. 模块布局

```
src/
├── core/                      # 核心框架
│   ├── __init__.py
│   ├── base_algorithms.py     # 基类
│   ├── framework.py           # 框架
│   ├── metrics.py             # 性能指标
│   ├── test_functions.py      # 测试函数
│   ├── exceptions.py          # 异常定义
│   ├── constants.py           # 常量定义
│   └── utils.py               # 工具函数
│
├── nas/                       # NAS框架
│   ├── __init__.py
│   ├── qd_nas.py              # 主优化器
│   ├── archive.py             # 档案管理
│   ├── map_elites.py          # MAP-Elites算法
│   └── ...
│
├── utils/                     # 工具和辅助
│   ├── __init__.py
│   ├── logging.py             # 日志
│   ├── visualization.py       # 可视化
│   └── analysis.py            # 分析工具
│
└── applications/              # 应用
    ├── engineering/
    └── ml_hpo/
```

### 2. __init__.py规范

```python
# ✅ 清晰的公共API定义

from .base_algorithms import BaseAlgorithm, BaseMultiObjectiveAlgorithm
from .test_functions import TestFunction, TestFunctionLibrary
from .metrics import PerformanceMetrics

# 定义公开API
__all__ = [
    'BaseAlgorithm',
    'BaseMultiObjectiveAlgorithm',
    'TestFunction',
    'TestFunctionLibrary',
    'PerformanceMetrics'
]

# ❌ 不好的做法
# 导入过多不相关的东西
# 没有定义__all__
```

## API设计原则

### 1. 简单易用

```python
# ✅ 简单的API
optimizer = create_default_qd_nas()
result = optimizer.optimize(problem_func, n_iterations=100)

# ❌ 复杂的API
optimizer = QDNASOptimizer(
    search_space=SearchSpace(),
    behavior_space=BehaviorSpace(),
    archive=Archive(),
    map_elites=MAPElites(),
    # ... 很多参数
)
```

### 2. 灵活配置

```python
# ✅ 灵活的配置选项

@dataclass
class OptimizationConfig:
    """优化配置"""
    population_size: int = 100
    n_iterations: int = 100
    mutation_rate: float = 0.1
    
    def validate(self):
        """验证配置"""
        assert 0 < self.population_size
        assert 0 < self.mutation_rate <= 1.0

# 使用
config = OptimizationConfig(
    population_size=200,
    n_iterations=500
)
optimizer = QDNASOptimizer(config=config)

# ❌ 不灵活的API - 只能通过构造函数传参
optimizer = QDNASOptimizer(
    pop_size=200,
    n_iter=500,
    mut_rate=0.1,
    # ... 更多参数
)
```

### 3. 一致的接口

```python
# ✅ 一致的方法签名

class BaseAlgorithm:
    def optimize(
        self,
        problem_func: Callable,
        n_iterations: int,
        pop_size: int,
        verbose: bool = False
    ) -> Tuple[List, List]:
        """优化"""
        pass

class NSGA2(BaseAlgorithm):
    def optimize(
        self,
        problem_func: Callable,
        n_iterations: int,
        pop_size: int,
        verbose: bool = False
    ) -> Tuple[List, List]:
        """NSGA-II优化"""
        pass

# ❌ 不一致的接口
class RandomSearch(BaseAlgorithm):
    def optimize(self, func):  # 不同的参数
        pass
```

### 4. 清晰的返回值

```python
# ✅ 清晰的返回值类型

def optimize(self) -> OptimizationResult:
    """优化并返回结果对象"""
    return OptimizationResult(
        population=self.pop,
        pareto_front=self.pareto,
        archive=self.archive,
        metrics=self.metrics,
        execution_time=elapsed_time
    )

# ❌ 不清晰的返回值
def optimize(self):
    """返回多个值，不清楚什么是什么"""
    return (pop, pareto, archive, metrics, time)
```

## 文档编写标准

### 1. 模块文档

```python
"""
模块名称和简短描述。

更详细的说明，包括模块的目的和主要功能。
可以多行。

主要类和函数：
    - ClassName: 说明
    - function_name: 说明

示例：
    >>> from module import ClassName
    >>> obj = ClassName()
    >>> result = obj.method()
"""
```

### 2. 类文档

```python
class MyAlgorithm(BaseAlgorithm):
    """
    我的算法实现。
    
    这是一个优化算法的实现，继承自BaseAlgorithm。
    
    算法说明：
        1. 初始化随机种群
        2. 迭代进化过程
        3. 返回最优解
    
    主要特性：
        - 特性1说明
        - 特性2说明
    
    使用示例：
        >>> algo = MyAlgorithm(pop_size=100)
        >>> result = algo.optimize(problem_func, n_iterations=100)
    
    参考论文：
        Author, et al. (Year). Title. Journal.
    """
```

### 3. 函数/方法文档

```python
def optimize(
    self,
    problem_func: Callable[[List[float]], Tuple[float, float]],
    n_iterations: int = 100,
    pop_size: int = 100,
    verbose: bool = False
) -> Tuple[List[List[float]], List[Dict[str, float]]]:
    """
    运行优化算法。
    
    对给定的问题进行多代进化优化，返回种群和Pareto前沿。
    
    Args:
        problem_func: 优化问题函数，输入解向量，返回目标函数值。
            签名: Callable[[List[float]], Tuple[float, float]]
        n_iterations: 进化代数，默认100。必须为正整数。
        pop_size: 种群大小，默认100。建议50-500之间。
        verbose: 是否打印进度信息，默认False。
    
    Returns:
        Tuple包含：
            - population: 最终种群，列表of解向量
            - pareto_front: Pareto前沿解，列表of字典，包含'solution'和'fitness'
    
    Raises:
        ValueError: 如果n_iterations或pop_size无效
        OptimizationError: 如果优化过程失败
    
    示例：
        >>> def sphere(x):
        ...     return (sum(xi**2 for xi in x),)
        >>> algo = MyAlgorithm()
        >>> pop, pareto = algo.optimize(sphere, n_iterations=50)
        >>> print(f"最优值: {pareto[0]['fitness']}")
    
    注意：
        - 建议在CPU充足的情况下运行
        - 大的pop_size会消耗更多内存
        - 保存详细日志时性能会降低
    """
    pass
```

## 测试编写指南

### 1. 单元测试结构

```python
import pytest
from src.nas import Archive, ArchiveEntry

class TestArchive:
    """档案管理器的单元测试"""
    
    @pytest.fixture
    def archive(self):
        """创建测试用档案"""
        return Archive(grid_shape=(10, 10))
    
    def test_add_entry(self, archive):
        """测试添加条目"""
        entry = ArchiveEntry(
            solution=[0.5, 0.5],
            behavior=[0.3, 0.7],
            fitness=[0.9, 0.8]
        )
        assert archive.add(entry) is True
        assert len(archive) == 1
    
    def test_add_duplicate(self, archive):
        """测试重复添加"""
        entry = ArchiveEntry([0.5, 0.5], [0.3, 0.7], [0.9, 0.8])
        archive.add(entry)
        
        # 重复添加相同行为的更优解
        better_entry = ArchiveEntry([0.5, 0.5], [0.3, 0.7], [0.95, 0.85])
        assert archive.add(better_entry) is True
    
    def test_get_best(self, archive):
        """测试获取最优解"""
        entry = ArchiveEntry([0.5, 0.5], [0.3, 0.7], [0.9])
        archive.add(entry)
        best = archive.get_best()
        assert best.fitness == [0.9]
    
    @pytest.mark.parametrize("grid_shape", [(5, 5), (10, 10), (20, 20)])
    def test_different_grid_shapes(self, grid_shape):
        """测试不同的网格形状"""
        archive = Archive(grid_shape=grid_shape)
        assert archive.grid_shape == grid_shape
```

### 2. 集成测试

```python
def test_end_to_end_optimization():
    """端到端优化测试"""
    # 1. 设置
    def sphere(x):
        return [sum(xi**2 for xi in x)]
    
    # 2. 运行优化
    optimizer = create_default_qd_nas()
    archive, pareto = optimizer.optimize(
        sphere,
        n_iterations=10,
        batch_size=20
    )
    
    # 3. 验证结果
    assert len(pareto) > 0
    assert archive.size > 0
    
    # 4. 检查质量
    best = pareto[0]
    assert best['fitness'][0] < 10  # 应该接近0
```

### 3. 性能测试

```python
import time

def test_archive_performance():
    """档案查询性能测试"""
    archive = Archive(grid_shape=(100, 100))
    
    # 填充档案
    for i in range(1000):
        entry = ArchiveEntry(
            solution=[np.random.random(10)],
            behavior=[np.random.random(2)],
            fitness=[np.random.random()]
        )
        archive.add(entry)
    
    # 测试查询性能
    start = time.time()
    for _ in range(1000):
        _ = archive.get_best()
    elapsed = time.time() - start
    
    # 应该在100ms以内
    assert elapsed < 0.1, f"查询太慢: {elapsed:.3f}s"
```

## 性能优化指南

### 1. 避免常见的性能陷阱

```python
# ❌ 低效：列表concatenation
result = []
for item in large_list:
    result = result + [process(item)]  # 每次都创建新列表

# ✅ 高效：使用列表append
result = []
for item in large_list:
    result.append(process(item))

# ✅ 更高效：列表推导式或map
result = [process(item) for item in large_list]
```

### 2. 使用NumPy向量化

```python
import numpy as np

# ❌ 低效：Python循环
def compute_distances_slow(points, target):
    distances = []
    for point in points:
        dist = sum((p - t)**2 for p, t in zip(point, target))**0.5
        distances.append(dist)
    return distances

# ✅ 高效：NumPy向量化
def compute_distances_fast(points, target):
    points = np.array(points)
    target = np.array(target)
    return np.linalg.norm(points - target, axis=1)
```

### 3. 内存优化

```python
# ❌ 浪费内存：存储所有中间结果
def process_large_data(data):
    temp1 = [expensive_operation1(x) for x in data]
    temp2 = [expensive_operation2(x) for x in temp1]
    return [expensive_operation3(x) for x in temp2]

# ✅ 节省内存：生成器管道
def process_large_data(data):
    def pipeline():
        for x in data:
            x = expensive_operation1(x)
            x = expensive_operation2(x)
            yield expensive_operation3(x)
    return list(pipeline())
```

### 4. 缓存和记忆化

```python
from functools import lru_cache

class Evaluator:
    def __init__(self, max_cache_size=1000):
        self._cache = {}
        self.max_size = max_cache_size
    
    def evaluate(self, x: tuple, func):
        """评估，使用缓存"""
        if x in self._cache:
            return self._cache[x]
        
        result = func(x)
        
        if len(self._cache) < self.max_size:
            self._cache[x] = result
        
        return result
```

## 安全最佳实践

### 1. 输入验证

```python
# ✅ 正确的输入验证
def optimize(
    self,
    problem_func: Callable,
    n_iterations: int = 100
) -> OptimizationResult:
    """优化"""
    # 验证问题函数
    if not callable(problem_func):
        raise TypeError("problem_func必须是可调用的")
    
    # 验证迭代次数
    if not isinstance(n_iterations, int) or n_iterations <= 0:
        raise ValueError("n_iterations必须是正整数")
    
    # 继续运行
    ...
```

### 2. 资源管理

```python
# ✅ 使用context manager管理资源
class FileResult:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filepath, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
    
    def save_result(self, data):
        self.file.write(data)

# 使用
with FileResult('results.txt') as result:
    result.save_result("优化完成")
```

### 3. 日志和监控

```python
import logging

logger = logging.getLogger(__name__)

def optimize(self, problem_func, n_iterations=100):
    """优化"""
    try:
        logger.info(f"开始优化: n_iterations={n_iterations}")
        
        for gen in range(n_iterations):
            logger.debug(f"第 {gen} 代")
            
            # 优化逻辑
            
            if gen % 10 == 0:
                logger.info(f"进度: {gen}/{n_iterations}")
        
        logger.info("优化完成")
        return result
    
    except Exception as e:
        logger.error(f"优化失败: {e}", exc_info=True)
        raise
```

---

**文档版本**: 1.0  
**最后更新**: 2026年1月14日  
**维护者**: DEAP社区
