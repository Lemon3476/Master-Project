import sys
sys.path.append(".")

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import pandas as pd

def visualize_results_from_excel(excel_file, output_dir, config_info=None, compare_models=False, create_improvement_charts=False, num_bins=5):
    """从已有的Excel文件读取数据并生成多种可视化图表，支持模型比较"""
    print(f"Reading data from {excel_file}")
    
    # 读取每个工作表的数据
    transition_dfs = {}
    with pd.ExcelFile(excel_file) as xls:
        sheets = [sheet for sheet in xls.sheet_names if sheet.startswith('Transition_')]
        
        for sheet in sheets:
            trans_len = int(sheet.split('_')[1])
            transition_dfs[trans_len] = pd.read_excel(xls, sheet_name=sheet)
    
    # 提取所有过渡长度
    transition_lengths = sorted(transition_dfs.keys())
    print(f"Found transition lengths: {transition_lengths}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查数据格式以确定是否包含两个模型的数据
    first_df = next(iter(transition_dfs.values()))
    has_model_comparison = 'Baseline_L2P' in first_df.columns and 'Proposed_L2P' in first_df.columns
    
    if compare_models:
        if not has_model_comparison:
            print("Warning: --compare_models flag was set, but the Excel file does not contain model comparison data.")
            print("Falling back to single-model visualization.")
            compare_models = False
    
    if compare_models and has_model_comparison:
        print("Using model comparison visualization mode.")
        
        # 创建模型比较的4×4网格图表
        create_model_comparison_grid(transition_dfs, transition_lengths, os.path.join(output_dir, "model_comparison_grid.png"), config_info)
        
        # 创建模型比较的综合图表
        create_model_comparison_combined(transition_dfs, transition_lengths, os.path.join(output_dir, "model_comparison_combined.png"), config_info)
        
        # 已经在Excel中计算了性能提升，可以直接使用
        if create_improvement_charts:
            # 合并所有数据
            all_data = pd.concat([df.assign(Transition_Length=t) for t, df in transition_dfs.items()], ignore_index=True)
            
            # 创建性能提升散点图
            create_improvement_scatter_plots(all_data, transition_lengths, output_dir, config_info)
            
            # 创建分桶柱状图
            create_binned_bar_charts(all_data, transition_lengths, output_dir, config_info, num_bins=num_bins)
    else:
        # 兼容旧版的单模型可视化
        print("Using single-model visualization mode.")
        
        # 创建一个4×4的网格图表
        create_grid_chart(transition_dfs, transition_lengths, os.path.join(output_dir, "all_metrics_grid.png"), config_info)
        
        # 额外创建一个综合图表（原来的2×2布局，每个子图显示所有过渡长度）
        create_combined_chart(transition_dfs, transition_lengths, os.path.join(output_dir, "combined_chart.png"), config_info)
        
        # 创建差异分析图表（如果需要）
        if create_improvement_charts:
            # 合并所有数据
            all_data = pd.concat([df.assign(Transition_Length=t) for t, df in transition_dfs.items()], ignore_index=True)
            
            # 检查数据中是否已经包含了性能提升信息
            if 'L2P_Improvement' not in all_data.columns:
                # 计算基线值（使用所有数据的平均值）
                baseline_values = {
                    'L2P': all_data['L2P'].mean(),
                    'L2Q': all_data['L2Q'].mean(),
                    'NPSS': all_data['NPSS'].mean(),
                    'Foot_Skate': all_data['Foot_Skate'].mean()
                }
                
                # 计算性能提升
                all_data = calculate_improvements(all_data, baseline_values)
            
            # 创建差异散点图
            create_improvement_scatter_plots(all_data, transition_lengths, output_dir, config_info)
            
            # 创建分桶柱状图
            create_binned_bar_charts(all_data, transition_lengths, output_dir, config_info, num_bins=num_bins)
    
    print(f"All visualizations have been generated in {output_dir}")

def create_grid_chart(transition_dfs, transition_lengths, output_file, config_info=None):
    """创建一个4×4网格的大图表，包含所有过渡长度和指标的组合"""
    print("Generating 4×4 grid chart with all metrics and transition lengths...")
    
    # 定义指标
    metrics = ['L2P', 'L2Q', 'NPSS', 'Foot_Skate']
    
    # 创建4×4网格图表
    fig, axs = plt.subplots(4, 4, figsize=(24, 20))
    
    # 为每个指标（行）和过渡长度（列）创建子图
    for i, metric in enumerate(metrics):
        for j, trans_len in enumerate(transition_lengths):
            if j >= len(transition_lengths):
                continue  # 防止过渡长度超出列数
                
            ax = axs[i, j]
            df = transition_dfs[trans_len].sort_values('Height_Range')
            
            # 绘制散点图
            ax.scatter(
                df['Height_Range'], 
                df[metric], 
                color='blue', 
                alpha=0.7, 
                s=30, 
                edgecolors='k', 
                linewidths=0.5
            )
            
            # 添加趋势线
            if len(df) > 1:  # 确保有足够的数据点拟合
                z = np.polyfit(df['Height_Range'], df[metric], 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(df['Height_Range']), max(df['Height_Range']), 100)
                ax.plot(x_range, p(x_range), color='red', linestyle='--', alpha=0.7, linewidth=2)
                
                # 计算相关性和斜率
                corr = df['Height_Range'].corr(df[metric])
                
                # 在图上添加相关系数
                ax.text(
                    0.05, 0.95, 
                    f'r = {corr:.3f}', 
                    transform=ax.transAxes,
                    verticalalignment='top', 
                    horizontalalignment='left',
                    fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
            
            # 设置子图标题
            ax.set_title(f"{metric} (T={trans_len})", fontsize=14)
            
            # 仅在最下面一行设置x轴标签
            if i == 3:
                ax.set_xlabel('Height Range (m)', fontsize=12)
            
            # 仅在最左边一列设置y轴标签
            if j == 0:
                ax.set_ylabel(f'{metric} Score', fontsize=12)
                
            # 添加网格线
            ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加标题
    if config_info:
        plt.suptitle(f"Motion Complexity vs Performance Metrics: {config_info}", fontsize=18, y=0.98)
    else:
        plt.suptitle(f"Motion Complexity vs Performance Metrics", fontsize=18, y=0.98)
    
    # 添加样本数量说明
    sample_counts = {t: len(df) for t, df in transition_dfs.items()}
    sample_text = ", ".join([f"T={t}: {n} samples" for t, n in sample_counts.items()])
    plt.figtext(0.5, 0.01, f"Analysis includes {sample_text}", 
                ha='center', fontsize=12, style='italic')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Grid chart saved to {output_file}")
    plt.close()

def create_combined_chart(transition_dfs, transition_lengths, output_file, config_info=None):
    """创建一个包含所有过渡长度数据的综合图表"""
    print("Generating combined chart with all transition lengths...")
    
    # 定义颜色和标记
    colors = ['blue', 'green', 'orange', 'red']
    markers = ['o', 's', '^', 'd']
    
    # 创建2x2图表布局
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # 定义指标和子图位置
    metrics = ['L2P', 'L2Q', 'NPSS', 'Foot_Skate']
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # 创建每个子图
    for (metric, pos) in zip(metrics, subplot_positions):
        ax = axs[pos]
        
        # 为每个过渡长度绘制数据
        for i, trans_len in enumerate(transition_lengths):
            if i >= len(colors):  # 防止过渡长度超过颜色数量
                i = i % len(colors)
            
            df = transition_dfs[trans_len]
            df = df.sort_values('Height_Range')
            
            # 绘制散点图
            ax.scatter(
                df['Height_Range'], 
                df[metric], 
                color=colors[i], 
                marker=markers[i % len(markers)],
                alpha=0.7, 
                s=30, 
                edgecolors='k', 
                linewidths=0.5,
                label=f'T={trans_len}'
            )
            
            # 添加趋势线
            if len(df) > 1:
                z = np.polyfit(df['Height_Range'], df[metric], 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(df['Height_Range']), max(df['Height_Range']), 100)
                ax.plot(x_range, p(x_range), color=colors[i], linestyle='--', alpha=0.7, linewidth=1.5)
                
                # 计算相关性
                corr = df['Height_Range'].corr(df[metric])
                
                # 在趋势线附近添加相关系数
                text_x = max(df['Height_Range']) * 0.9
                text_y = p(text_x)
                ax.text(text_x, text_y, f'r={corr:.3f}', color=colors[i], fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # 设置子图标题和标签
        ax.set_title(f'{metric} vs Height Range', fontsize=14, fontweight='bold')
        ax.set_xlabel('Height Range (m)', fontsize=12)
        ax.set_ylabel(f'{metric} Score', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()
    
    # 添加标题
    if config_info:
        plt.suptitle(f"Height Range Analysis: {config_info}", fontsize=16, y=0.98)
    else:
        plt.suptitle("Height Range vs Performance Analysis", fontsize=16, y=0.98)
    
    # 添加样本数量说明
    sample_counts = {t: len(df) for t, df in transition_dfs.items()}
    sample_text = ", ".join([f"T={t}: {n} samples" for t, n in sample_counts.items()])
    plt.figtext(0.5, 0.01, f"Analysis includes {sample_text}", 
                ha='center', fontsize=10, style='italic')
    
    # 保存图表
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined chart saved to {output_file}")
    plt.close()

def create_model_comparison_grid(transition_dfs, transition_lengths, output_file, config_info=None):
    """创建一个4×4网格图表，比较基准模型和提议模型的性能"""
    print("Generating 4×4 model comparison grid chart...")
    
    # 定义指标
    metrics = ['L2P', 'L2Q', 'NPSS', 'Foot_Skate']
    
    # 创建4×4网格图表
    fig, axs = plt.subplots(4, 4, figsize=(24, 20))
    
    # 为每个指标（行）和过渡长度（列）创建子图
    for i, metric in enumerate(metrics):
        for j, trans_len in enumerate(transition_lengths):
            if j >= len(transition_lengths):
                continue  # 防止过渡长度超出列数
                
            ax = axs[i, j]
            df = transition_dfs[trans_len].sort_values('Height_Range')
            
            baseline_col = f'Baseline_{metric}'
            proposed_col = f'Proposed_{metric}'
            
            # 绘制基准模型的散点图
            ax.scatter(
                df['Height_Range'], 
                df[baseline_col], 
                color='blue', 
                alpha=0.7, 
                s=30, 
                edgecolors='k', 
                linewidths=0.5,
                label='Baseline'
            )
            
            # 绘制提议模型的散点图
            ax.scatter(
                df['Height_Range'], 
                df[proposed_col], 
                color='red', 
                alpha=0.7, 
                s=30, 
                edgecolors='k', 
                linewidths=0.5,
                label='Proposed'
            )
            
            # 添加基准模型的趋势线
            if len(df) > 1:  # 确保有足够的数据点拟合
                z = np.polyfit(df['Height_Range'], df[baseline_col], 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(df['Height_Range']), max(df['Height_Range']), 100)
                ax.plot(x_range, p(x_range), color='blue', linestyle='--', alpha=0.7, linewidth=2)
                
                # 计算相关性
                corr = df['Height_Range'].corr(df[baseline_col])
                
                # 在图上添加相关系数
                ax.text(
                    0.05, 0.95, 
                    f'Baseline r = {corr:.3f}', 
                    transform=ax.transAxes,
                    verticalalignment='top', 
                    horizontalalignment='left',
                    fontsize=10,
                    color='blue',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
            
            # 添加提议模型的趋势线
            if len(df) > 1:  # 确保有足够的数据点拟合
                z = np.polyfit(df['Height_Range'], df[proposed_col], 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(df['Height_Range']), max(df['Height_Range']), 100)
                ax.plot(x_range, p(x_range), color='red', linestyle='--', alpha=0.7, linewidth=2)
                
                # 计算相关性
                corr = df['Height_Range'].corr(df[proposed_col])
                
                # 在图上添加相关系数
                ax.text(
                    0.05, 0.85, 
                    f'Proposed r = {corr:.3f}', 
                    transform=ax.transAxes,
                    verticalalignment='top', 
                    horizontalalignment='left',
                    fontsize=10,
                    color='red',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
            
            # 设置子图标题
            ax.set_title(f"{metric} (T={trans_len})", fontsize=14)
            
            # 仅在最下面一行设置x轴标签
            if i == 3:
                ax.set_xlabel('Height Range (m)', fontsize=12)
            
            # 仅在最左边一列设置y轴标签
            if j == 0:
                ax.set_ylabel(f'{metric} Score', fontsize=12)
                
            # 添加网格线
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend()
    
    # 添加标题
    if config_info:
        plt.suptitle(f"Model Comparison: Motion Complexity vs Performance Metrics: {config_info}", fontsize=18, y=0.98)
    else:
        plt.suptitle(f"Model Comparison: Motion Complexity vs Performance Metrics", fontsize=18, y=0.98)
    
    # 添加样本数量说明
    sample_counts = {t: len(df) for t, df in transition_dfs.items()}
    sample_text = ", ".join([f"T={t}: {n} samples" for t, n in sample_counts.items()])
    plt.figtext(0.5, 0.01, f"Analysis includes {sample_text}", 
                ha='center', fontsize=12, style='italic')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Model comparison grid chart saved to {output_file}")
    plt.close()

def create_model_comparison_combined(transition_dfs, transition_lengths, output_file, config_info=None):
    """创建一个包含所有过渡长度的模型比较综合图表"""
    print("Generating combined model comparison chart...")
    
    # 定义颜色和标记
    baseline_colors = ['darkblue', 'darkgreen', 'darkorange', 'darkred']
    proposed_colors = ['royalblue', 'limegreen', 'orange', 'tomato']
    markers = ['o', 's', '^', 'd']
    
    # 创建2x2图表布局
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # 定义指标和子图位置
    metrics = ['L2P', 'L2Q', 'NPSS', 'Foot_Skate']
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # 创建每个子图
    for (metric, pos) in zip(metrics, subplot_positions):
        ax = axs[pos]
        baseline_col = f'Baseline_{metric}'
        proposed_col = f'Proposed_{metric}'
        
        # 为每个过渡长度绘制数据
        for i, trans_len in enumerate(transition_lengths):
            if i >= len(baseline_colors):  # 防止过渡长度超过颜色数量
                i = i % len(baseline_colors)
            
            df = transition_dfs[trans_len]
            df = df.sort_values('Height_Range')
            
            # 绘制基准模型的散点图
            ax.scatter(
                df['Height_Range'], 
                df[baseline_col], 
                color=baseline_colors[i], 
                marker=markers[i % len(markers)],
                alpha=0.5, 
                s=30, 
                edgecolors='k', 
                linewidths=0.5,
                label=f'Baseline T={trans_len}'
            )
            
            # 绘制提议模型的散点图
            ax.scatter(
                df['Height_Range'], 
                df[proposed_col], 
                color=proposed_colors[i], 
                marker=markers[i % len(markers)],
                alpha=0.5, 
                s=30, 
                edgecolors='k', 
                linewidths=0.5,
                label=f'Proposed T={trans_len}'
            )
            
            # 添加基准模型的趋势线
            if len(df) > 1:
                z = np.polyfit(df['Height_Range'], df[baseline_col], 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(df['Height_Range']), max(df['Height_Range']), 100)
                ax.plot(x_range, p(x_range), color=baseline_colors[i], linestyle='--', alpha=0.7, linewidth=1.5)
                
                # 计算相关性
                corr = df['Height_Range'].corr(df[baseline_col])
                
                # 在趋势线附近添加相关系数
                text_x = max(df['Height_Range']) * 0.9
                text_y = p(text_x)
                ax.text(text_x, text_y, f'B r={corr:.2f}', color=baseline_colors[i], fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # 添加提议模型的趋势线
            if len(df) > 1:
                z = np.polyfit(df['Height_Range'], df[proposed_col], 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(df['Height_Range']), max(df['Height_Range']), 100)
                ax.plot(x_range, p(x_range), color=proposed_colors[i], linestyle='--', alpha=0.7, linewidth=1.5)
                
                # 计算相关性
                corr = df['Height_Range'].corr(df[proposed_col])
                
                # 在趋势线附近添加相关系数
                text_x = max(df['Height_Range']) * 0.9
                text_y = p(text_x)
                ax.text(text_x, text_y, f'P r={corr:.2f}', color=proposed_colors[i], fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # 设置子图标题和标签
        ax.set_title(f'{metric} vs Height Range', fontsize=14, fontweight='bold')
        ax.set_xlabel('Height Range (m)', fontsize=12)
        ax.set_ylabel(f'{metric} Score', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加图例，并将其放在图表外面
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 fancybox=True, shadow=True, ncol=4, fontsize=8)
    
    # 添加标题
    if config_info:
        plt.suptitle(f"Model Comparison: {config_info}", fontsize=16, y=0.98)
    else:
        plt.suptitle("Model Comparison: Baseline vs Proposed", fontsize=16, y=0.98)
    
    # 添加样本数量说明
    sample_counts = {t: len(df) for t, df in transition_dfs.items()}
    sample_text = ", ".join([f"T={t}: {n} samples" for t, n in sample_counts.items()])
    plt.figtext(0.5, 0.01, f"Analysis includes {sample_text}", 
                ha='center', fontsize=10, style='italic')
    
    # 保存图表
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Model comparison combined chart saved to {output_file}")
    plt.close()

def calculate_improvements(df, baseline_values):
    """计算相对于基线的性能提升"""
    print("Calculating performance improvements...")
    
    # 对于每种指标计算改进值（对于所有指标，值越低越好，所以改进 = 基线 - 实际）
    df = df.copy()  # 避免修改原始数据
    df['L2P_Improvement'] = baseline_values['L2P'] - df['L2P']
    df['L2Q_Improvement'] = baseline_values['L2Q'] - df['L2Q']
    df['NPSS_Improvement'] = baseline_values['NPSS'] - df['NPSS']
    df['Foot_Skate_Improvement'] = baseline_values['Foot_Skate'] - df['Foot_Skate']
    
    # 计算相对改进（百分比）
    if baseline_values['L2P'] != 0: df['L2P_Rel_Improvement'] = df['L2P_Improvement'] / baseline_values['L2P'] * 100
    if baseline_values['L2Q'] != 0: df['L2Q_Rel_Improvement'] = df['L2Q_Improvement'] / baseline_values['L2Q'] * 100
    if baseline_values['NPSS'] != 0: df['NPSS_Rel_Improvement'] = df['NPSS_Improvement'] / baseline_values['NPSS'] * 100
    if baseline_values['Foot_Skate'] != 0: df['Foot_Skate_Rel_Improvement'] = df['Foot_Skate_Improvement'] / baseline_values['Foot_Skate'] * 100
    
    print(f"Improvements calculated relative to baseline: {baseline_values}")
    return df

def create_improvement_scatter_plots(df, transition_lengths, output_dir, config_info=None):
    """为每种过渡长度创建差异散点图（展示性能提升与动作复杂度的关系）"""
    print("Creating improvement scatter plots...")
    
    # 定义指标
    metrics = ['L2P', 'L2Q', 'NPSS', 'Foot_Skate']
    improvement_cols = [f'{m}_Improvement' for m in metrics]
    rel_improvement_cols = [f'{m}_Rel_Improvement' for m in metrics]
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # 检查是否有相对提升列
    has_rel_improvement = all(col in df.columns for col in rel_improvement_cols)
    
    # 为每个过渡长度创建一个图表
    for trans_len in transition_lengths:
        print(f"  Creating scatter plot for T={trans_len}...")
        
        # 过滤当前过渡长度的数据
        trans_df = df[df['Transition_Length'] == trans_len]
        
        # 创建2x2的子图布局
        fig, axs = plt.subplots(2, 2, figsize=(16, 14))
        
        # 为每个指标创建一个子图
        for idx, (metric, improvement_col, pos) in enumerate(zip(metrics, improvement_cols, subplot_positions)):
            ax = axs[pos]
            
            # 确定散点图的颜色，根据改进值的正负
            colors = ['green' if x > 0 else 'red' for x in trans_df[improvement_col]]
            
            # 绘制散点图
            scatter = ax.scatter(
                trans_df['Height_Range'], 
                trans_df[improvement_col], 
                c=colors,
                alpha=0.7, 
                s=30, 
                edgecolors='k', 
                linewidths=0.5
            )
            
            # 添加趋势线
            if len(trans_df) > 1:
                z = np.polyfit(trans_df['Height_Range'], trans_df[improvement_col], 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(trans_df['Height_Range']), max(trans_df['Height_Range']), 100)
                ax.plot(x_range, p(x_range), color='blue', linestyle='--', alpha=0.7, linewidth=2)
                
                # 计算相关性和斜率
                corr = trans_df['Height_Range'].corr(trans_df[improvement_col])
                slope = z[0]
                
                # 计算正向和负向改进的数量
                positive_count = sum(1 for x in trans_df[improvement_col] if x > 0)
                negative_count = sum(1 for x in trans_df[improvement_col] if x <= 0)
                total_count = len(trans_df[improvement_col])
                positive_percentage = (positive_count / total_count) * 100 if total_count > 0 else 0
                
                # 添加统计信息
                stats_text = (
                    f'Correlation: {corr:.3f}\n'
                    f'Slope: {slope:.4f}\n'
                    f'Mean Improvement: {trans_df[improvement_col].mean():.4f}\n'
                    f'Positive Improvements: {positive_count}/{total_count} ({positive_percentage:.1f}%)'
                )
                
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 设置图表标题和标签
            ax.set_title(f"{metric} Improvement vs. Complexity", fontsize=14, fontweight='bold')
            ax.set_xlabel('Height Range (m)', fontsize=12)
            ax.set_ylabel(f'{metric} Improvement', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 添加零线（无改进的基准线）
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加总标题
        if config_info:
            plt.suptitle(f"Performance Improvements vs. Complexity (T={trans_len})\n{config_info}", fontsize=16, y=0.98)
        else:
            plt.suptitle(f"Performance Improvements vs. Complexity (T={trans_len})", fontsize=16, y=0.98)
        
        # 添加样本数量说明
        improvement_mean_text = ", ".join([f"{m}: {trans_df[imp_col].mean():.4f}" for m, imp_col in zip(metrics, improvement_cols)])
        plt.figtext(0.5, 0.01, 
                    f"Analysis includes {len(trans_df)} samples with T={trans_len}\nMean improvements: {improvement_mean_text}", 
                    ha='center', fontsize=10, style='italic', wrap=True)
        
        # 保存图表
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_file = os.path.join(output_dir, f"scatter_improvement_T{trans_len}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved to {output_file}")
        
        # 如果有相对提升数据，也创建相对提升的散点图
        if has_rel_improvement:
            fig, axs = plt.subplots(2, 2, figsize=(16, 14))
            
            for idx, (metric, rel_improvement_col, pos) in enumerate(zip(metrics, rel_improvement_cols, subplot_positions)):
                ax = axs[pos]
                
                # 确定散点图的颜色，根据改进值的正负
                colors = ['green' if x > 0 else 'red' for x in trans_df[rel_improvement_col]]
                
                # 绘制散点图
                scatter = ax.scatter(
                    trans_df['Height_Range'], 
                    trans_df[rel_improvement_col], 
                    c=colors,
                    alpha=0.7, 
                    s=30, 
                    edgecolors='k', 
                    linewidths=0.5
                )
                
                # 添加趋势线
                if len(trans_df) > 1:
                    z = np.polyfit(trans_df['Height_Range'], trans_df[rel_improvement_col], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(trans_df['Height_Range']), max(trans_df['Height_Range']), 100)
                    ax.plot(x_range, p(x_range), color='blue', linestyle='--', alpha=0.7, linewidth=2)
                    
                    # 计算相关性和斜率
                    corr = trans_df['Height_Range'].corr(trans_df[rel_improvement_col])
                    slope = z[0]
                    
                    # 添加统计信息
                    stats_text = (
                        f'Correlation: {corr:.3f}\n'
                        f'Slope: {slope:.4f}\n'
                        f'Mean Rel. Improvement: {trans_df[rel_improvement_col].mean():.2f}%'
                    )
                    
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # 设置图表标题和标签
                ax.set_title(f"{metric} Relative Improvement vs. Complexity", fontsize=14, fontweight='bold')
                ax.set_xlabel('Height Range (m)', fontsize=12)
                ax.set_ylabel(f'{metric} Relative Improvement (%)', fontsize=12)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # 添加零线（无改进的基准线）
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 添加总标题
            if config_info:
                plt.suptitle(f"Relative Performance Improvements vs. Complexity (T={trans_len})\n{config_info}", 
                             fontsize=16, y=0.98)
            else:
                plt.suptitle(f"Relative Performance Improvements vs. Complexity (T={trans_len})", 
                             fontsize=16, y=0.98)
            
            # 添加样本数量说明
            rel_improvement_mean_text = ", ".join([f"{m}: {trans_df[rel_imp_col].mean():.2f}%" 
                                                for m, rel_imp_col in zip(metrics, rel_improvement_cols)])
            plt.figtext(0.5, 0.01, 
                        f"Analysis includes {len(trans_df)} samples with T={trans_len}\nMean relative improvements: {rel_improvement_mean_text}", 
                        ha='center', fontsize=10, style='italic', wrap=True)
            
            # 保存图表
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            rel_output_file = os.path.join(output_dir, f"scatter_rel_improvement_T{trans_len}.png")
            plt.savefig(rel_output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved relative improvement plot to {rel_output_file}")

def create_binned_bar_charts(df, transition_lengths, output_dir, config_info=None, num_bins=5):
    """为每种过渡长度创建分桶柱状图，显示不同复杂度级别下的平均性能提升"""
    print(f"Creating binned bar charts with {num_bins} complexity levels...")
    
    # 定义指标
    metrics = ['L2P', 'L2Q', 'NPSS', 'Foot_Skate']
    improvement_cols = [f'{m}_Improvement' for m in metrics]
    rel_improvement_cols = [f'{m}_Rel_Improvement' for m in metrics]
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # 检查是否有相对提升列
    has_rel_improvement = all(col in df.columns for col in rel_improvement_cols)
    
    # 为每个过渡长度创建一个图表
    for trans_len in transition_lengths:
        print(f"  Creating binned bar chart for T={trans_len}...")
        
        # 过滤当前过渡长度的数据
        trans_df = df[df['Transition_Length'] == trans_len].copy()
        
        # 根据高度范围分桶
        try:
            trans_df['Complexity_Bin'] = pd.qcut(trans_df['Height_Range'], num_bins, 
                                               labels=[f'Level {i+1}' for i in range(num_bins)])
        except ValueError:
            # 如果唯一值太少，无法分成指定数量的桶，尝试减少桶数
            actual_bins = min(num_bins, len(trans_df['Height_Range'].unique()))
            print(f"    Warning: Could not create {num_bins} bins, using {actual_bins} instead.")
            
            if actual_bins <= 1:
                print(f"    ERROR: Not enough unique height range values for T={trans_len}")
                continue
                
            trans_df['Complexity_Bin'] = pd.qcut(trans_df['Height_Range'], actual_bins, 
                                               labels=[f'Level {i+1}' for i in range(actual_bins)])
        
        # 创建绝对改进值的柱状图
        create_single_binned_chart(
            trans_df, metrics, improvement_cols, subplot_positions, trans_len, output_dir,
            title_prefix="Average Performance Improvements",
            y_label_prefix="Average",
            y_label_suffix="Improvement",
            output_suffix="improvement",
            config_info=config_info,
            value_format=".2f"
        )
        
        # 如果有相对改进值，创建相对改进的柱状图
        if has_rel_improvement:
            create_single_binned_chart(
                trans_df, metrics, rel_improvement_cols, subplot_positions, trans_len, output_dir,
                title_prefix="Average Relative Performance Improvements",
                y_label_prefix="Average",
                y_label_suffix="Rel. Improvement (%)",
                output_suffix="rel_improvement",
                config_info=config_info,
                value_format=".1f"
            )
            
        # 检查是否有原始模型数据，创建模型比较柱状图
        if ('Baseline_L2P' in trans_df.columns and 'Proposed_L2P' in trans_df.columns):
            # 为每个指标创建模型得分对比图
            for metric in metrics:
                baseline_col = f'Baseline_{metric}'
                proposed_col = f'Proposed_{metric}'
                
                # 计算每个复杂度桶的平均得分
                bin_baseline_means = trans_df.groupby('Complexity_Bin')[baseline_col].mean()
                bin_proposed_means = trans_df.groupby('Complexity_Bin')[proposed_col].mean()
                bin_baseline_stds = trans_df.groupby('Complexity_Bin')[baseline_col].std()
                bin_proposed_stds = trans_df.groupby('Complexity_Bin')[proposed_col].std()
                
                # 创建图表
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 获取桶标签
                bin_labels = bin_baseline_means.index
                
                # 设置柱状图的位置
                x = np.arange(len(bin_labels))
                width = 0.35
                
                # 绘制基准模型和提议模型的柱状图
                rects1 = ax.bar(x - width/2, bin_baseline_means, width, color='royalblue', 
                                yerr=bin_baseline_stds, label='Baseline', 
                                capsize=5, error_kw={'ecolor':'darkgray', 'capthick':1.5})
                
                rects2 = ax.bar(x + width/2, bin_proposed_means, width, color='tomato',
                                yerr=bin_proposed_stds, label='Proposed', 
                                capsize=5, error_kw={'ecolor':'darkgray', 'capthick':1.5})
                
                # 添加数值标签
                for rect in rects1:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3点垂直偏移
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
                
                for rect in rects2:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3点垂直偏移
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
                
                # 设置图表标题和标签
                ax.set_title(f"{metric} Performance Comparison by Complexity Level (T={trans_len})", 
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('Complexity Level', fontsize=12)
                ax.set_ylabel(f'{metric} Score (Lower is better)', fontsize=12)
                ax.set_xticks(x)
                ax.set_xticklabels(bin_labels)
                ax.legend()
                ax.grid(True, alpha=0.3, linestyle='--', axis='y')
                
                # 添加样本数量说明
                bin_counts = trans_df.groupby('Complexity_Bin').size()
                bin_ranges = {}
                
                # 获取每个分桶的范围
                for bin_name in bin_counts.index:
                    bin_data = trans_df[trans_df['Complexity_Bin'] == bin_name]
                    bin_ranges[bin_name] = (bin_data['Height_Range'].min(), bin_data['Height_Range'].max())
                
                # 构建注释文本
                range_text = "Complexity Levels (Height Range in meters):\n"
                for bin_name in sorted(bin_ranges.keys()):
                    min_val, max_val = bin_ranges[bin_name]
                    count = bin_counts[bin_name]
                    range_text += f"{bin_name}: {min_val:.2f}-{max_val:.2f} m ({count} samples)  "
                
                # 添加标题和注释
                if config_info:
                    plt.title(f"{metric} Performance Comparison (T={trans_len})\n{config_info}", 
                             fontsize=14, pad=20)
                
                plt.figtext(0.5, 0.01, range_text, ha='center', fontsize=8, style='italic', wrap=True)
                
                # 保存图表
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                comp_output_file = os.path.join(output_dir, f"model_comparison_{metric}_T{trans_len}.png")
                plt.savefig(comp_output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"    Saved model comparison for {metric} to {comp_output_file}")

def create_single_binned_chart(df, metrics, value_cols, subplot_positions, trans_len, output_dir, 
                              title_prefix="Performance", y_label_prefix="", y_label_suffix="",
                              output_suffix="chart", config_info=None, value_format=".2f"):
    """创建单个分桶柱状图"""
    # 创建2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    
    # 为每个指标创建一个子图
    for (metric, value_col, pos) in zip(metrics, value_cols, subplot_positions):
        ax = axs[pos]
        
        # 计算每个桶的平均值
        bin_means = df.groupby('Complexity_Bin')[value_col].mean()
        bin_stds = df.groupby('Complexity_Bin')[value_col].std()
        
        # 获取桶标签和对应的值
        bin_labels = bin_means.index
        bin_values = bin_means.values
        error_values = bin_stds.values if len(bin_stds) > 0 else None
        
        # 设置柱状图的颜色
        bar_colors = ['lightblue'] * len(bin_labels)
        # 标记负值为淡红色
        for i, val in enumerate(bin_values):
            if val < 0:
                bar_colors[i] = 'lightcoral'
        
        # 绘制柱状图
        bars = ax.bar(bin_labels, bin_values, yerr=error_values, 
                 color=bar_colors, edgecolor='black', linewidth=1, 
                 capsize=5, error_kw={'ecolor':'darkgray', 'capthick':1.5})
        
        # 在柱子上方添加具体数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height * 1.02 if height >= 0 else height * 0.9,
                    f'{height:{value_format}}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10)
        
        # 设置图表标题和标签
        ax.set_title(f"{metric} {y_label_suffix} vs. Complexity", fontsize=14, fontweight='bold')
        ax.set_xlabel('Complexity Level', fontsize=12)
        ax.set_ylabel(f'{y_label_prefix} {metric} {y_label_suffix}', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # 添加零线（无改进的基准线）
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 尝试设置y轴范围，确保零线在中间
        try:
            y_max = max(abs(min(bin_values)), abs(max(bin_values))) * 1.2
            ax.set_ylim(-y_max if min(bin_values) < 0 else 0, y_max)
        except:
            pass  # 如果出错，保持默认范围
    
    # 添加总标题
    if config_info:
        plt.suptitle(f"{title_prefix} by Complexity Level (T={trans_len})\n{config_info}", 
                     fontsize=16, y=0.98)
    else:
        plt.suptitle(f"{title_prefix} by Complexity Level (T={trans_len})", 
                     fontsize=16, y=0.98)
    
    # 添加数据说明
    bin_counts = df.groupby('Complexity_Bin').size()
    bin_ranges = {}
    
    # 获取每个分桶的范围
    for bin_name in bin_counts.index:
        bin_data = df[df['Complexity_Bin'] == bin_name]
        bin_ranges[bin_name] = (bin_data['Height_Range'].min(), bin_data['Height_Range'].max())
    
    # 构建注释文本
    range_text = "Complexity Levels (Height Range in meters):\n"
    for bin_name in sorted(bin_ranges.keys()):
        min_val, max_val = bin_ranges[bin_name]
        count = bin_counts[bin_name]
        range_text += f"{bin_name}: {min_val:.2f}-{max_val:.2f} m ({count} samples)\n"
    
    # 添加注释文本
    plt.figtext(0.5, 0.01, range_text, ha='center', fontsize=10, style='italic')
    
    # 保存图表
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_file = os.path.join(output_dir, f"barchart_{output_suffix}_T{trans_len}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved {output_suffix} bar chart to {output_file}")

def calculate_improvements(df, baseline_values):
    """计算相对于基线的性能提升"""
    print("Calculating performance improvements...")
    
    # 对于每种指标计算改进值（对于所有指标，值越低越好，所以改进 = 基线 - 实际）
    df = df.copy()  # 避免修改原始数据
    df['L2P_Improvement'] = baseline_values['L2P'] - df['L2P']
    df['L2Q_Improvement'] = baseline_values['L2Q'] - df['L2Q']
    df['NPSS_Improvement'] = baseline_values['NPSS'] - df['NPSS']
    df['Foot_Skate_Improvement'] = baseline_values['Foot_Skate'] - df['Foot_Skate']
    
    # 计算相对改进（百分比）
    if baseline_values['L2P'] != 0: df['L2P_Rel_Improvement'] = df['L2P_Improvement'] / baseline_values['L2P'] * 100
    if baseline_values['L2Q'] != 0: df['L2Q_Rel_Improvement'] = df['L2Q_Improvement'] / baseline_values['L2Q'] * 100
    if baseline_values['NPSS'] != 0: df['NPSS_Rel_Improvement'] = df['NPSS_Improvement'] / baseline_values['NPSS'] * 100
    if baseline_values['Foot_Skate'] != 0: df['Foot_Skate_Rel_Improvement'] = df['Foot_Skate_Improvement'] / baseline_values['Foot_Skate'] * 100
    
    print(f"Improvements calculated relative to baseline: {baseline_values}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize complexity vs performance results from Excel file')
    parser.add_argument('--excel_file', type=str, required=True,
                        help='Path to the Excel file containing analysis results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for image files (default: same directory as Excel file)')
    parser.add_argument('--config_info', type=str, default=None,
                        help='Configuration information to display in the plot title')
    parser.add_argument('--compare_models', action='store_true',
                        help='Compare baseline and proposed models (requires Excel file with model comparison data)')
    parser.add_argument('--improvement_charts', action='store_true',
                        help='Generate additional improvement analysis charts')
    parser.add_argument('--num_bins', type=int, default=5,
                        help='Number of complexity bins for bar charts (default: 5)')
    
    args = parser.parse_args()
    
    # 如果未指定输出目录，则使用Excel文件所在的目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.excel_file)
    
    print("=" * 80)
    print("Visualizing Complexity vs Performance Results")
    print("=" * 80)
    print(f"Excel file: {args.excel_file}")
    print(f"Output directory: {args.output_dir}")
    if args.config_info:
        print(f"Config info: {args.config_info}")
    print(f"Compare models: {args.compare_models}")
    print(f"Generate improvement charts: {args.improvement_charts}")
    if args.improvement_charts:
        print(f"Number of complexity bins: {args.num_bins}")
    print("-" * 80)
    
    # 生成可视化图表
    visualize_results_from_excel(args.excel_file, args.output_dir, args.config_info, 
                                compare_models=args.compare_models, 
                                create_improvement_charts=args.improvement_charts,
                                num_bins=args.num_bins)