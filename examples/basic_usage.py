"""
基础使用示例
展示如何使用DEAP框架进行多目标优化
"""

import sys
sys.path.insert(0, 'src')

from main_clean import run_simple_nsga2, demo_basic_usage

if __name__ == "__main__":
    print("运行基础示例...")
    
    # 运行演示
    demo_basic_usage()
    
    # 或运行自定义优化
    # result = run_simple_nsga2(n_dim=10, pop_size=100, n_gen=100)
    # print(f"找到 {len(result['pareto_front'])} 个Pareto最优解")
