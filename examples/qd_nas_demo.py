"""
QD-NAS示例演示
展示如何使用质量-多样性神经架构搜索框架
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nas import (
    QDNASOptimizer, create_default_qd_nas,
    BehaviorSpace, SearchSpace, Architecture,
    StaticCharacterization, Objective, ObjectiveType, Constraint
)
import numpy as np


def example_1_basic_nas():
    """
    示例1: 基础NAS搜索

    使用MAP-Elites算法进行单目标优化
    """
    print("=" * 80)
    print("示例1: 基础NAS搜索（MAP-Elites）")
    print("=" * 80)

    # 创建优化器
    optimizer = create_default_qd_nas(
        optimization_mode='map_elites',
        multi_objective=False,
        population_guided=True
    )

    # 初始化
    print("\n📦 初始化优化器...")
    optimizer.initialize()

    # 优化
    print("\n🔥 开始优化...")
    archive, pareto_front = optimizer.optimize(
        n_iterations=50,  # 使用较少的迭代用于演示
        batch_size=20,
        verbose=True
    )

    # 获取最佳架构
    print("\n📊 优化结果:")
    best_arch = optimizer.get_best_architecture()
    if best_arch:
        print(f"最佳架构:")
        arch_dict = best_arch.to_dict()
        print(f"  Cell数量: {arch_dict['n_cells']}")
        print(f"  节点数/Cell: {arch_dict['n_nodes']}")
        print(f"  初始通道数: {arch_dict['n_channels']}")

    # 获取统计信息
    stats = optimizer.get_statistics()
    print(f"\n归档统计:")
    print(f"  归档大小: {stats['size']}")
    print(f"  行为空间覆盖率: {stats['coverage']:.2%}")
    print(f"  多样性: {stats['diversity']:.4f}")

    print("\n✅ 示例1完成")


def example_2_multi_objective():
    """
    示例2: 多目标多约束优化

    优化精度、延迟和能耗，同时满足约束
    """
    print("\n" + "=" * 80)
    print("示例2: 多目标多约束优化")
    print("=" * 80)

    # 创建优化器
    optimizer = create_default_qd_nas(
        optimization_mode='map_elites',
        multi_objective=True,
        population_guided=True
    )

    # 初始化
    print("\n📦 初始化优化器...")
    optimizer.initialize()

    # 优化
    print("\n🔥 开始多目标优化...")
    archive, pareto_front = optimizer.optimize(
        n_iterations=50,
        batch_size=20,
        verbose=True
    )

    # 获取Pareto前沿
    print("\n📊 Pareto前沿:")
    pareto = optimizer.get_pareto_front()

    print(f"Pareto前沿大小: {len(pareto)}")

    # 显示前5个解
    for i, (arch, metrics) in enumerate(pareto[:5]):
        print(f"\n解 {i+1}:")
        print(f"  精度: {metrics.accuracy:.4f}")
        print(f"  延迟: {metrics.latency:.2f} ms")
        print(f"  能耗: {metrics.energy:.2f} mJ")
        print(f"  参数量: {metrics.parameters:.2f} M")

        arch_dict = arch.to_dict()
        print(f"  Cell数量: {arch_dict['n_cells']}")
        print(f"  初始通道数: {arch_dict['n_channels']}")

    # 获取统计信息
    stats = optimizer.get_statistics()
    print(f"\n归档统计:")
    print(f"  归档大小: {stats['size']}")
    print(f"  Pareto前沿大小: {stats.get('pareto_size', 0)}")

    print("\n✅ 示例2完成")


def example_3_adaptive_search():
    """
    示例3: 自适应混合搜索

    结合多种搜索策略，自动选择最优策略
    """
    print("\n" + "=" * 80)
    print("示例3: 自适应混合搜索")
    print("=" * 80)

    # 创建优化器，使用不同的搜索模式
    optimizer = create_default_qd_nas(
        optimization_mode='random_map_elites',  # 随机搜索增强
        multi_objective=False,
        population_guided=True  # 启用种群引导
    )

    # 初始化
    print("\n📦 初始化优化器...")
    optimizer.initialize()

    # 优化
    print("\n🔥 开始自适应搜索...")
    archive, pareto_front = optimizer.optimize(
        n_iterations=50,
        batch_size=20,
        verbose=True
    )

    # 获取统计信息
    stats = optimizer.get_statistics()

    print(f"\n📊 优化结果:")
    print(f"  归档大小: {stats['size']}")
    print(f"  行为空间覆盖率: {stats['coverage']:.2%}")
    print(f"  多样性: {stats['diversity']:.4f}")

    if 'population_stats' in stats:
        pop_stats = stats['population_stats']
        print(f"\n种群统计:")
        print(f"  平均精度: {pop_stats['mean_accuracy']:.4f}")
        print(f"  精度标准差: {pop_stats['std_accuracy']:.4f}")
        print(f"  种群多样性: {pop_stats['diversity']:.4f}")

    print("\n✅ 示例3完成")


def example_4_custom_objectives():
    """
    示例4: 自定义目标函数

    定义自己的优化目标和约束
    """
    print("\n" + "=" * 80)
    print("示例4: 自定义目标函数")
    print("=" * 80)

    from src.nas import MultiObjectiveNAS, Objective, Constraint

    # 创建搜索空间和特征提取器
    search_space = SearchSpace()
    characterizer = StaticCharacterization()
    behavior_space = BehaviorSpace()

    # 定义优化目标
    objectives = [
        # 精度最大化
        Objective(name='accuracy', type=ObjectiveType.MAXIMIZE, weight=0.7),
        # 延迟最小化
        Objective(name='latency', type=ObjectiveType.MINIMIZE, weight=0.2),
        # 参数量最小化
        Objective(name='params', type=ObjectiveType.MINIMIZE, weight=0.1),
    ]

    # 定义约束
    constraints = [
        # 延迟约束
        Constraint(name='latency', threshold=50.0, type="<="),
        # 参数量约束
        Constraint(name='params', threshold=3.0, type="<="),
    ]

    # 创建多目标NAS优化器
    optimizer = MultiObjectiveNAS(
        behavior_space=behavior_space,
        characterizer=characterizer,
        objectives=objectives,
        constraints=constraints,
    )

    # 优化
    print("\n🔥 开始自定义目标优化...")
    archive, pareto_front = optimizer.evolve(
        generate_function=search_space.random_sample,
        mutate_function=search_space.mutate,
        n_iterations=50,
        batch_size=20,
        verbose=True
    )

    print(f"\n✅ 优化完成，Pareto前沿大小: {len(pareto_front)}")

    print("\n✅ 示例4完成")


def example_5_gradient_guided():
    """
    示例5: 梯度引导搜索

    使用梯度信息引导搜索方向
    """
    print("\n" + "=" * 80)
    print("示例5: 梯度引导搜索")
    print("=" * 80)

    # 创建优化器，使用梯度引导
    optimizer = create_default_qd_nas(
        optimization_mode='gradient_map_elites',
        multi_objective=False,
        population_guided=True
    )

    # 初始化
    print("\n📦 初始化优化器...")
    optimizer.initialize()

    # 优化
    print("\n🔥 开始梯度引导搜索...")
    archive, pareto_front = optimizer.optimize(
        n_iterations=50,
        batch_size=20,
        verbose=True
    )

    # 获取统计信息
    stats = optimizer.get_statistics()
    print(f"\n📊 优化结果:")
    print(f"  归档大小: {stats['size']}")
    print(f"  行为空间覆盖率: {stats['coverage']:.2%}")

    print("\n✅ 示例5完成")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("QD-NAS: 质量-多样性神经架构搜索框架")
    print("示例演示")
    print("=" * 80)

    # 运行所有示例
    try:
        example_1_basic_nas()
        example_2_multi_objective()
        example_3_adaptive_search()
        example_4_custom_objectives()
        example_5_gradient_guided()

        print("\n" + "=" * 80)
        print("所有示例运行完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
