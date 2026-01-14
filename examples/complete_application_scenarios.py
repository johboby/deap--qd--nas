"""
完整应用场景示例 (Complete Application Scenarios)
展示QD-NAS框架的实际应用
"""

import numpy as np
from src.nas import (
    # QD-NAS核心
    QDNASOptimizer, create_default_qd_nas,

    # 搜索空间和特征提取
    Architecture, SearchSpace, StaticCharacterization,

    # 高级QD算法
    create_cvt_map_elites,

    # 分布式计算
    create_evaluator, DistributedNASOptimizer, WorkerConfig,

    # 基准测试
    create_benchmark, BenchmarkRunner,

    # 端到端NAS
    EndToEndNAS, NASConfig,

    # 工具
    LoggerManager, Timer, ProgressBar,
    CheckpointManager, MetricsTracker,
    set_random_seed,

    # 错误处理
    ErrorHandler, retry, safe_execute,
)

import logging


def scenario_1_mobile_nas():
    """
    场景1: 移动端NAS优化

    目标：为移动设备搜索低延迟、低能耗的神经网络架构。
    """
    print("\n" + "="*70)
    print("场景1: 移动端NAS优化")
    print("="*70)

    # 设置日志
    logger_manager = LoggerManager(
        name='mobile_nas',
        level='INFO',
        log_file='./logs/mobile_nas.log'
    )
    logger = logger_manager.get_logger()

    # 设置随机种子
    set_random_seed(42)

    # 创建QD-NAS优化器（多目标：延迟和能耗）
    optimizer = create_default_qd_nas(
        optimization_mode='map_elites',
        multi_objective=True,
        population_guided=True
    )

    # 配置约束（移动设备约束）
    from src.nas import Constraint, ObjectiveType, Objective

    # 定义移动设备的多目标
    objectives = [
        Objective(name='accuracy', type=ObjectiveType.MAXIMIZE, weight=0.5),
        Objective(name='latency', type=ObjectiveType.MINIMIZE, weight=0.3),
        Objective(name='energy', type=ObjectiveType.MINIMIZE, weight=0.2),
    ]

    constraints = [
        Constraint(name='latency', threshold=50.0, type="<="),  # 50ms
        Constraint(name='energy', threshold=500.0, type="<="),  # 500mJ
        Constraint(name='params', threshold=3.0, type="<="),  # 3M参数
    ]

    # 初始化
    optimizer.initialize()

    # 运行优化
    logger.info("🚀 开始移动端NAS优化")
    with Timer('Mobile NAS Optimization') as timer:
        archive, pareto_front = optimizer.optimize(
            n_iterations=200,
            batch_size=50,
            verbose=True
        )

    # 分析结果
    logger.info(f"✅ 优化完成，耗时: {timer.elapsed:.2f}s")

    print("\n" + "-"*70)
    print("移动端架构推荐:")
    print("-"*70)

    # 找到满足约束的最佳架构
    for arch, metrics in pareto_front[:5]:
        latency_ok = metrics.latency <= 50
        energy_ok = metrics.energy <= 500
        params_ok = metrics.parameters / 1e6 <= 3

        print(f"\n架构 {pareto_front.index((arch, metrics)) + 1}:")
        print(f"  Accuracy: {metrics.accuracy:.4f}")
        print(f"  Latency: {metrics.latency:.2f}ms {'✓' if latency_ok else '✗'}")
        print(f"  Energy: {metrics.energy:.2f}mJ {'✓' if energy_ok else '✗'}")
        print(f"  Params: {metrics.parameters/1e6:.2f}M {'✓' if params_ok else '✗'}")

        if latency_ok and energy_ok and params_ok:
            print(f"  ✓✓✓ 满足所有移动设备约束！")


def scenario_2_distributed_nas():
    """
    场景2: 分布式NAS

    目标：使用多进程和GPU加速进行大规模NAS搜索。
    """
    print("\n" + "="*70)
    print("场景2: 分布式NAS")
    print("="*70)

    # 设置日志
    logger_manager = LoggerManager(
        name='distributed_nas',
        level='INFO',
        log_file='./logs/distributed_nas.log'
    )
    logger = logger_manager.get_logger()

    # 创建分布式评估器配置
    worker_config = WorkerConfig(
        n_workers=4,  # 使用4个CPU核心
        use_gpu=False,  # 假设没有GPU
        max_tasks_per_worker=10
    )

    # 创建评估器
    from src.nas import MultiProcessEvaluator
    search_space = SearchSpace()

    # 评估函数
    def evaluate_architecture(arch):
        characterizer = StaticCharacterization()
        return characterizer.characterize(arch)

    evaluator = MultiProcessEvaluator(evaluate_architecture, worker_config)

    # 创建基础优化器
    base_optimizer = create_default_qd_nas(
        optimization_mode='map_elites',
        multi_objective=False,
        population_guided=True
    )

    # 创建分布式NAS优化器
    distributed_optimizer = DistributedNASOptimizer(
        optimizer=base_optimizer,
        evaluator=evaluator,
        batch_size=50
    )

    # 运行分布式优化
    logger.info("🚀 开始分布式NAS优化")

    with Timer('Distributed NAS') as timer:
        archive, pareto_front = distributed_optimizer.optimize_distributed(
            n_iterations=100,
            verbose=True
        )

    logger.info(f"✅ 分布式优化完成，耗时: {timer.elapsed:.2f}s")

    # 输出结果
    print("\n" + "-"*70)
    print("分布式NAS结果:")
    print("-"*70)
    stats = archive.get_statistics()
    print(f"归档大小: {stats['size']}")
    print(f"覆盖率: {stats['coverage']:.2%}")
    print(f"最佳适应度: {stats['best_fitness']:.4f}")


def scenario_3_benchmark_comparison():
    """
    场景3: NAS方法基准比较

    目标：比较不同NAS方法的性能。
    """
    print("\n" + "="*70)
    print("场景3: NAS方法基准比较")
    print("="*70)

    # 设置日志
    logger_manager = LoggerManager(
        name='benchmark_comparison',
        level='INFO',
        log_file='./logs/benchmark_comparison.log'
    )
    logger = logger_manager.get_logger()

    # 创建基准测试
    benchmark = create_benchmark(dataset_name='cifar10')

    # 创建基准测试运行器
    search_space = SearchSpace()
    runner = BenchmarkRunner(benchmark, search_space)

    # 比较不同方法
    methods = [
        'Random Search',
        'MAP-Elites',
        'CVT-MAP-Elites',
        'QD-NAS',
    ]

    results = {}
    for method in methods:
        logger.info(f"🏃 运行基准测试: {method}")

        with Timer(f'{method} Benchmark') as timer:
            stats = runner.run_benchmark(method_name=method, n_samples=5)

        results[method] = stats
        logger.info(f"✅ {method} 完成，耗时: {timer.elapsed:.2f}s")

    # 比较结果
    print("\n" + "-"*70)
    print("基准测试比较:")
    print("-"*70)

    print(f"\n{'方法':<20} {'平均准确率':<15} {'平均参数量':<15}")
    print("-"*70)

    for method, stats in results.items():
        params = stats['mean_parameters'] / 1e6
        print(f"{method:<20} {stats['mean_accuracy']:<15.4f} {params:<15.2f}M")

    # 找到最佳方法
    best_method = max(results.keys(),
                   key=lambda k: results[k]['mean_accuracy'])

    print(f"\n🏆 最佳方法: {best_method}")
    print(f"   准确率: {results[best_method]['mean_accuracy']:.4f}")


def scenario_4_robust_nas():
    """
    场景4: 鲁棒NAS优化

    目标：使用错误处理和恢复机制进行鲁棒的NAS优化。
    """
    print("\n" + "="*70)
    print("场景4: 鲁棒NAS优化")
    print("="*70)

    # 设置日志
    logger_manager = LoggerManager(
        name='robust_nas',
        level='INFO',
        log_file='./logs/robust_nas.log',
        log_file='./logs/robust_nas_errors.log'
    )
    logger = logger_manager.get_logger()

    # 创建错误处理器
    error_handler = ErrorHandler(
        error_log_file='./logs/robust_nas_errors.log',
        enable_recovery=True
    )

    # 注册恢复策略
    from src.nas.error_handling import CheckpointRecoveryStrategy

    checkpoint_strategy = CheckpointRecoveryStrategy(checkpoint_dir='./checkpoints')
    error_handler.register_recovery_strategy(checkpoint_strategy)

    # 创建检查点管理器
    checkpoint_manager = CheckpointManager(
        save_dir='./checkpoints',
        max_checkpoints=5
    )

    # 创建指标跟踪器
    metrics_tracker = MetricsTracker(
        metrics_names=['accuracy', 'latency', 'energy', 'diversity']
    )

    # 创建优化器
    optimizer = create_default_qd_nas(
        optimization_mode='map_elites',
        multi_objective=False,
        population_guided=True
    )

    # 初始化
    optimizer.initialize()

    # 鲁棒优化循环
    logger.info("🚀 开始鲁棒NAS优化")

    iteration = 0
    max_iterations = 100

    with Timer('Robust NAS') as timer:
        progress_bar = ProgressBar(total=max_iterations, desc='优化进度')

        while iteration < max_iterations:
            try:
                # 正常优化步骤
                archive, pareto_front = optimizer.optimize(
                    n_iterations=1,
                    batch_size=20,
                    verbose=False
                )

                # 更新指标
                stats = archive.get_statistics()
                metrics_tracker.update(
                    step=iteration,
                    accuracy=stats['best_fitness'],
                    latency=stats.get('latency', 0),
                    energy=stats.get('energy', 0),
                    diversity=stats['diversity']
                )

                # 保存检查点（每10轮）
                if iteration % 10 == 0:
                    checkpoint_data = {
                        'iteration': iteration,
                        'archive': archive,
                        'metrics': stats,
                    }
                    checkpoint_manager.save_checkpoint(
                        data=checkpoint_data,
                        epoch=iteration
                    )

                iteration += 1
                progress_bar.update()

            except Exception as e:
                logger.error(f"❌ 迭代 {iteration} 失败: {e}")

                # 尝试错误恢复
                context = {
                    'iteration': iteration,
                    'optimizer': 'QD-NAS',
                }

                if error_handler.handle_error(e, context):
                    # 恢复成功，继续
                    logger.info("✅ 错误恢复成功，继续优化")
                else:
                    # 恢复失败，尝试从检查点恢复
                    logger.warning("⚠️  错误恢复失败，尝试从检查点恢复")

                    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
                    if latest_checkpoint:
                        logger.info(f"📥 从检查点恢复: {latest_checkpoint}")
                        # 这里应该实际加载检查点
                        iteration += 1
                    else:
                        logger.error("❌ 没有可用的检查点，停止优化")
                        break

        progress_bar.close()

    logger.info(f"✅ 鲁棒NAS优化完成，耗时: {timer.elapsed:.2f}s")

    # 输出错误统计
    error_stats = error_handler.get_error_statistics()
    print("\n" + "-"*70)
    print("错误统计:")
    print("-"*70)
    print(f"总错误数: {error_stats['total_errors']}")
    print(f"错误类型分布: {error_stats['error_counts']}")

    # 保存指标
    metrics_tracker.save_to_csv('./results/robust_nas_metrics.csv')

    # 输出最终结果
    stats = optimizer.get_statistics()
    print(f"\n最终统计:")
    print(f"归档大小: {stats['size']}")
    print(f"覆盖率: {stats['coverage']:.2%}")
    print(f"最佳适应度: {stats['best_fitness']:.4f}")


def scenario_5_end_to_end_nas():
    """
    场景5: 端到端NAS

    目标：完整的NAS流程，从搜索到部署。
    """
    print("\n" + "="*70)
    print("场景5: 端到端NAS")
    print("="*70)

    # 设置日志
    logger_manager = LoggerManager(
        name='end_to_end_nas',
        level='INFO',
        log_file='./logs/end_to_end_nas.log'
    )
    logger = logger_manager.get_logger()

    # 创建NAS配置
    config = NASConfig(
        name='End-to-End NAS',
        description='完整的端到端神经架构搜索',
        dataset='cifar10',
        optimization_mode='map_elites',
        multi_objective=True,
        population_guided=True,
        n_iterations=100,
        batch_size=50,
        epochs=50,
        early_stopping=True,
        patience=5,
        save_dir='./results/end_to_end',
        device='cpu',
    )

    # 保存配置
    config.save('./results/end_to_end/config.json')

    # 创建端到端NAS
    nas = EndToEndNAS(config)

    # 运行端到端NAS
    logger.info("🚀 开始端到端NAS")

    with Timer('End-to-End NAS') as timer:
        result = nas.run()

    logger.info(f"✅ 端到端NAS完成，耗时: {timer.elapsed:.2f}s")

    # 结果已通过result对象自动保存
    print(f"\n结果已保存至: {config.save_dir}")
    print(f"报告文件: {config.save_dir}/report.txt")


def scenario_6_multi_objective_tradeoff():
    """
    场景6: 多目标权衡分析

    目标：分析Pareto前沿上的不同架构的权衡。
    """
    print("\n" + "="*70)
    print("场景6: 多目标权衡分析")
    print("="*70)

    # 设置日志
    logger_manager = LoggerManager(
        name='multi_objective_tradeoff',
        level='INFO',
        log_file='./logs/multi_objective_tradeoff.log'
    )
    logger = logger_manager.get_logger()

    # 创建多目标优化器
    optimizer = create_default_qd_nas(
        optimization_mode='map_elites',
        multi_objective=True,
        population_guided=True
    )

    # 初始化和优化
    optimizer.initialize()

    logger.info("🚀 开始多目标优化")
    archive, pareto_front = optimizer.optimize(
        n_iterations=100,
        batch_size=50,
        verbose=True
    )

    # 分析Pareto前沿
    print("\n" + "-"*70)
    print("Pareto前沿权衡分析:")
    print("-"*70)

    if not pareto_front:
        print("没有找到Pareto前沿")
        return

    # 分类架构
    high_accuracy = []
    low_latency = []
    low_energy = []

    for arch, metrics in pareto_front:
        if metrics.accuracy > 0.85:
            high_accuracy.append((arch, metrics))
        if metrics.latency < 30:
            low_latency.append((arch, metrics))
        if metrics.energy < 300:
            low_energy.append((arch, metrics))

    print(f"\n高准确率架构 (>85%): {len(high_accuracy)}")
    for i, (arch, metrics) in enumerate(high_accuracy[:3]):
        print(f"  {i+1}. Accuracy={metrics.accuracy:.4f}, "
              f"Latency={metrics.latency:.2f}ms, Energy={metrics.energy:.2f}mJ")

    print(f"\n低延迟架构 (<30ms): {len(low_latency)}")
    for i, (arch, metrics) in enumerate(low_latency[:3]):
        print(f"  {i+1}. Accuracy={metrics.accuracy:.4f}, "
              f"Latency={metrics.latency:.2f}ms, Energy={metrics.energy:.2f}mJ")

    print(f"\n低能耗架构 (<300mJ): {len(low_energy)}")
    for i, (arch, metrics) in enumerate(low_energy[:3]):
        print(f"  {i+1}. Accuracy={metrics.accuracy:.4f}, "
              f"Latency={metrics.latency:.2f}ms, Energy={metrics.energy:.2f}mJ")

    # 找到权衡最优的架构（综合得分）
    best_arch, best_metrics = pareto_front[0]
    best_score = 0

    for arch, metrics in pareto_front:
        # 综合评分（权重可以根据需求调整）
        score = 0.5 * metrics.accuracy + \
                0.25 * (1 - metrics.latency / 100) + \
                0.25 * (1 - metrics.energy / 1000)

        if score > best_score:
            best_score = score
            best_arch = arch
            best_metrics = metrics

    print(f"\n🏆 最优权衡架构:")
    print(f"  Accuracy: {best_metrics.accuracy:.4f}")
    print(f"  Latency: {best_metrics.latency:.2f}ms")
    print(f"  Energy: {best_metrics.energy:.2f}mJ")
    print(f"  Parameters: {best_metrics.parameters/1e6:.2f}M")
    print(f"  综合得分: {best_score:.4f}")


def main():
    """运行所有场景"""
    print("\n" + "="*70)
    print("QD-NAS 完整应用场景示例")
    print("="*70)

    # 运行所有场景
    scenario_1_mobile_nas()
    scenario_2_distributed_nas()
    scenario_3_benchmark_comparison()
    scenario_4_robust_nas()
    scenario_5_end_to_end_nas()
    scenario_6_multi_objective_tradeoff()

    print("\n" + "="*70)
    print("所有场景运行完成！")
    print("="*70)


if __name__ == "__main__":
    main()
