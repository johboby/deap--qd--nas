"""
可视化工具模块
提供多目标优化的各种可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
import matplotlib.colors as mcolors

class ParetoVisualizer:
    """Pareto前沿可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
        plt.style.use('default')
        
    def plot_pareto_2d(self, pareto_front: List[Tuple], 
                      reference_point: Optional[Tuple] = None,
                      title: str = "Pareto Front",
                      save_path: Optional[str] = None):
        """绘制2D Pareto前沿"""
        if not pareto_front:
            print("Warning: Empty Pareto front")
            return
            
        pareto_array = np.array(pareto_front)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制Pareto前沿
        scatter = ax.scatter(pareto_array[:, 0], pareto_array[:, 1], 
                           c='red', s=50, alpha=0.7, label='Pareto Front')
        
        # 绘制理想点和最低点
        ideal_point = np.min(pareto_array, axis=0)
        nadir_point = np.max(pareto_array, axis=0)
        
        ax.scatter(*ideal_point, c='green', s=100, marker='*', 
                  label='Ideal Point', edgecolors='black', linewidth=1)
        ax.scatter(*nadir_point, c='blue', s=100, marker='s', 
                  label='Nadir Point', edgecolors='black', linewidth=1)
        
        # 绘制参考点
        if reference_point:
            ax.scatter(*reference_point, c='orange', s=100, marker='x', 
                      label='Reference Point', linewidth=3)
            
            # 绘制超体积区域
            self._plot_hypervolume_2d(ax, pareto_front, reference_point)
        
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置相等的纵横比
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pareto_3d(self, pareto_front: List[Tuple],
                      title: str = "3D Pareto Front",
                      save_path: Optional[str] = None):
        """绘制3D Pareto前沿"""
        if not pareto_front or len(pareto_front[0]) < 3:
            print("Warning: Need at least 3 objectives for 3D plot")
            return
            
        pareto_array = np.array(pareto_front)
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制3D散点图
        scatter = ax.scatter(pareto_array[:, 0], pareto_array[:, 1], pareto_array[:, 2],
                           c=pareto_array[:, 0], cmap='viridis', s=50, alpha=0.7)
        
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        ax.set_title(title)
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='Objective 1 Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_convergence(self, history: List[Dict], 
                        metrics: List[str] = ['hypervolume'],
                        title: str = "Convergence History",
                        save_path: Optional[str] = None):
        """绘制收敛历史"""
        if not history:
            print("Warning: Empty history")
            return
            
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
            
        generations = [h['generation'] for h in history]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            if metric in history[0]:
                values = [h[metric] for h in history]
                ax.plot(generations, values, 'b-', linewidth=2, label=metric)
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            if idx == len(metrics) - 1:
                ax.set_xlabel('Generation')
                
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_population_distribution(self, populations: List[List[Tuple]], 
                                    generation_gap: int = 10,
                                    title: str = "Population Distribution Evolution"):
        """绘制种群分布演化"""
        if not populations:
            print("Warning: Empty populations")
            return
            
        n_plots = len(populations)
        cols = min(4, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(-1)
            
        for idx, (gen, pop) in enumerate(enumerate(populations)):
            if idx % generation_gap == 0 or idx == len(populations) - 1:
                row = idx // cols
                col = idx % cols
                
                if rows > 1 and cols > 1:
                    ax = axes[row, col]
                else:
                    ax = axes[idx]
                
                if pop and len(pop[0]) >= 2:
                    pop_array = np.array(pop)
                    ax.scatter(pop_array[:, 0], pop_array[:, 1], 
                             alpha=0.6, s=20)
                    ax.set_title(f'Generation {gen}')
                    ax.set_xlabel('Objective 1')
                    ax.set_ylabel('Objective 2')
                    ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(populations), rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1 and cols > 1:
                axes[row, col].set_visible(False)
            elif rows == 1 or cols == 1:
                axes[idx].set_visible(False)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _plot_hypervolume_2d(self, ax, pareto_front: List[Tuple], 
                           reference_point: Tuple):
        """绘制2D超体积区域（简化版）"""
        try:
            from scipy.spatial import ConvexHull
            
            pareto_array = np.array(pareto_front)
            
            if len(pareto_array) >= 3:
                # 计算凸包
                hull = ConvexHull(pareto_array)
                
                # 填充凸包区域
                for simplex in hull.simplices:
                    ax.fill(pareto_array[simplex, 0], pareto_array[simplex, 1],
                           alpha=0.2, color='red', label='_nolegend_')
        except ImportError:
            # 如果没有scipy，使用简单的矩形近似
            min_obj1 = np.min(pareto_array[:, 0])
            min_obj2 = np.min(pareto_array[:, 1])
            
            width = reference_point[0] - min_obj1
            height = reference_point[1] - min_obj2
            
            rect = plt.Rectangle((min_obj1, min_obj2), width, height, 
                               alpha=0.2, color='red', label='Hypervolume Area')
            ax.add_patch(rect)

class PerformancePlotter:
    """性能分析绘图器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_algorithm_comparison(self, results: Dict[str, Dict], 
                                 metrics: List[str] = ['hypervolume'],
                                 title: str = "Algorithm Comparison"):
        """比较多算法的性能"""
        fig, axes = plt.subplots(len(metrics), 1, figsize=self.figsize)
        if len(metrics) == 1:
            axes = [axes]
            
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            algorithms = list(results.keys())
            means = [results[alg].get('mean', {}).get(metric, 0) for alg in algorithms]
            stds = [results[alg].get('std', {}).get(metric, 0) for alg in algorithms]
            
            x_pos = np.arange(len(algorithms))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                         alpha=0.7, color=plt.cm.Set3(np.linspace(0, 1, len(algorithms))))
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(algorithms, rotation=45)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, mean_val in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{mean_val:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

# 预设的颜色方案
COLOR_SCHEMES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'pastel': ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'],
    'vibrant': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
}