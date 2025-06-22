#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轨迹曲率与模型性能关系分析

此脚本实现轨迹曲率与模型性能之间关系的深度分析，针对不同过渡长度(15,30,60,90)
分别生成散点图、提升分析图和分桶柱状图。

用法:
    python eval/analyze_curvature.py --eval_file evaluation_results.xlsx

输出:
    在output/curvature_analysis/目录下生成各类分析图表和合并后的数据Excel文件
"""

import os
import sys
sys.path.append(".")  # 添加项目根目录到路径
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt错误
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import time

import torch
from aPyOpenGL import transforms as trf
from utils.dataset import MotionDataset
from utils import utils

# 从analyze_advantage_features.py中提取的轨迹曲率计算函数
def calculate_trajectory_curvature(positions):
    """
    计算轨迹的平均曲率。
    
    Args:
        positions (np.ndarray): 位置数据，形状为[T, 3]，代表xyz坐标
        
    Returns:
        float: 平均轨迹曲率
    """
    # 需要至少3个点才能计算曲率
    if positions.shape[0] < 3:
        return 0.0
    
    try:
        # 计算速度向量
        velocities = np.diff(positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # 检查速度是否接近零
        if np.all(speeds < 1e-6):
            return 0.0
        
        # 避免除以零
        non_zero_speeds = np.where(speeds > 1e-6, speeds, 1e-6)
        
        # 归一化速度向量
        normalized_velocities = velocities / non_zero_speeds[:, np.newaxis]
        
        # 计算连续速度向量之间的夹角
        dot_products = np.sum(normalized_velocities[:-1] * normalized_velocities[1:], axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)  # 限制在有效范围内
        angles = np.arccos(dot_products)
        
        # 计算曲率为单位距离角度变化
        valid_speeds = speeds[1:] > 1e-6
        if not np.any(valid_speeds):
            return 0.0
            
        curvatures = np.zeros_like(angles)
        curvatures[valid_speeds] = angles[valid_speeds] / speeds[1:][valid_speeds]
        
        # 过滤无效值（NaN或Inf）
        valid_curvatures = curvatures[~np.isnan(curvatures) & ~np.isinf(curvatures)]
        return np.mean(valid_curvatures) if len(valid_curvatures) > 0 else 0.0
    
    except Exception as e:
        print(f"警告: 计算轨迹曲率时出错: {str(e)}。返回0.")
        return 0.0

def compute_trajectory_curvature_for_batch(motion_batch, skeleton):
    """
    计算批次中每个样本的轨迹曲率。
    
    Args:
        motion_batch (torch.Tensor): 运动数据，形状为[B, T, D]
        skeleton: 用于前向运动学的骨架结构
        
    Returns:
        np.ndarray: 每个样本的轨迹曲率，形状为[B]
    """
    B, T, D = motion_batch.shape
    curvatures = []
    
    # 分批处理每个样本
    for i in range(B):
        # 提取单个样本
        motion = motion_batch[i]  # [T, D]
        
        # 分离旋转和位置
        rot, pos = torch.split(motion, [D-3, 3], dim=-1)
        
        # 重塑旋转数据以匹配fk函数要求
        rot = rot.reshape(1, T, -1, 6)  # [1, T, J, 6]
        pos = pos.reshape(1, T, 3)  # [1, T, 3]
        
        # 使用前向运动学计算关节全局位置
        _, joints_positions = trf.t_ortho6d.fk(rot, pos, skeleton)  # [1, T, J, 3]
        
        # 转换为NumPy数组并移除批次维度
        joints_positions = joints_positions.cpu().numpy()[0]  # [T, J, 3]
        
        # 获取根关节(髋关节)位置
        root_positions = joints_positions[:, 0]  # [T, 3]
        
        # 计算轨迹曲率
        curvature = calculate_trajectory_curvature(root_positions)
        curvatures.append(curvature)
    
    return np.array(curvatures)

def analyze_excel_structure(eval_file, baseline_model=None, proposed_model=None):
    """
    分析评估Excel文件的结构，检查内容并返回关键信息。
    
    Args:
        eval_file (str): Excel评估文件路径
        baseline_model (str, optional): 基准模型的标签名称
        proposed_model (str, optional): 提议模型的标签名称
        
    Returns:
        tuple: (sheet_names, transitions, methods)
            - sheet_names: 所有工作表名称
            - transitions: 所有过渡长度
            - methods: 不同的方法名称
    """
    print(f"分析评估文件: {eval_file}")
    
    # 加载Excel文件
    excel_data = pd.read_excel(eval_file, sheet_name=None)
    sheet_names = list(excel_data.keys())
    
    # 提取过渡长度
    transitions = []
    for sheet_name in sheet_names:
        if sheet_name.startswith("Transition_"):
            try:
                trans = int(sheet_name.split("_")[1])
                transitions.append(trans)
            except:
                pass
    
    # 获取第一个工作表的方法列
    first_sheet = excel_data[sheet_names[0]]
    all_methods = first_sheet["Method"].unique().tolist()
    
    # 如果指定了基线和提议模型，验证它们是否存在
    if baseline_model is not None and proposed_model is not None:
        if baseline_model not in all_methods:
            print(f"警告: 指定的基准模型 '{baseline_model}' 不在评估结果中。可用模型: {all_methods}")
            print(f"将尝试查找包含该名称的模型...")
            # 尝试模糊匹配
            matching_methods = [m for m in all_methods if baseline_model in m]
            if matching_methods:
                baseline_model = matching_methods[0]
                print(f"找到匹配的基准模型: {baseline_model}")
            else:
                print(f"无法找到匹配的基准模型，将使用第一个模型作为基准")
                baseline_model = all_methods[0]
                
        if proposed_model not in all_methods:
            print(f"警告: 指定的提议模型 '{proposed_model}' 不在评估结果中。可用模型: {all_methods}")
            print(f"将尝试查找包含该名称的模型...")
            # 尝试模糊匹配
            matching_methods = [m for m in all_methods if proposed_model in m]
            if matching_methods:
                proposed_model = matching_methods[0]
                print(f"找到匹配的提议模型: {proposed_model}")
            else:
                print(f"无法找到匹配的提议模型，将使用最后一个模型作为提议模型")
                proposed_model = all_methods[-1]
        
        # 只保留这两个模型
        methods = [baseline_model, proposed_model]
    else:
        methods = all_methods
    
    print(f"找到 {len(sheet_names)} 个工作表: {sheet_names}")
    print(f"找到 {len(transitions)} 个过渡长度: {transitions}")
    print(f"分析将使用的模型: {methods}")
    
    return sheet_names, transitions, methods, baseline_model, proposed_model

def compute_curvatures_for_dataset(dataset_name="lafan1", config_name="default.yaml"):
    """
    为整个数据集计算轨迹曲率。
    
    Args:
        dataset_name (str): 数据集名称
        config_name (str): 配置文件名称
        
    Returns:
        pd.DataFrame: 包含每个样本(batch和sequence)的轨迹曲率的DataFrame
    """
    print(f"为数据集 {dataset_name} 计算轨迹曲率...")
    
    # 加载配置
    config = utils.load_config(f"config/{dataset_name}/{config_name}")
    
    # 加载测试数据集
    dataset = MotionDataset(train=False, config=config, verbose=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    # 收集结果
    all_curvatures = []
    batch_indices = []
    sequence_indices = []
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 遍历数据集计算曲率
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="计算曲率")):
        motion = batch["motion"].to(device)
        
        # 计算这个批次中所有样本的轨迹曲率
        curvatures = compute_trajectory_curvature_for_batch(motion, dataset.skeleton)
        
        # 收集信息
        all_curvatures.extend(curvatures.tolist())
        batch_indices.extend([batch_idx] * len(curvatures))
        sequence_indices.extend(list(range(len(curvatures))))
    
    # 创建DataFrame
    df = pd.DataFrame({
        "Batch": batch_indices,
        "Sequence": sequence_indices,
        "Trajectory_Curvature": all_curvatures
    })
    
    print(f"完成计算。共 {len(df)} 个样本的轨迹曲率。")
    return df

def merge_evaluation_with_curvatures(eval_file, curvature_df, baseline_model=None, proposed_model=None):
    """
    合并评估结果和曲率数据，创建宽格式的综合数据集。
    
    Args:
        eval_file (str): 评估Excel文件路径
        curvature_df (pd.DataFrame): 包含曲率数据的DataFrame
        baseline_model (str, optional): 基准模型的标签名称
        proposed_model (str, optional): 提议模型的标签名称
        
    Returns:
        dict: 每个过渡长度的综合DataFrame字典
    """
    print("合并评估结果和曲率数据...")
    
    # 加载Excel评估结果
    excel_data = pd.read_excel(eval_file, sheet_name=None)
    
    # 识别所有方法
    sheet_names = list(excel_data.keys())
    first_sheet = excel_data[sheet_names[0]]
    all_methods = sorted(first_sheet["Method"].unique().tolist())
    
    # 确定使用哪些方法
    methods_to_use = all_methods
    if baseline_model is not None and proposed_model is not None:
        methods_to_use = [baseline_model, proposed_model]
        print(f"仅使用指定的两个模型: {baseline_model}(基准) 和 {proposed_model}(提议)")
    
    # 用于保存每个过渡长度的合并结果
    merged_data = {}
    
    # 处理每个过渡长度的工作表
    for sheet_name, df in excel_data.items():
        if sheet_name.startswith("Transition_"):
            trans = int(sheet_name.split("_")[1])
            print(f"处理过渡长度 {trans}...")
            
            # 将评估数据透视成宽格式
            # 每个样本一行，每个方法的指标为列
            pivot_data = []
            
            # 获取唯一的(Batch, Sequence)对
            unique_samples = df[["Batch", "Sequence"]].drop_duplicates()
            
            for _, row in tqdm(unique_samples.iterrows(), total=len(unique_samples), desc=f"处理样本"):
                batch, seq = row["Batch"], row["Sequence"]
                
                # 创建一条新记录
                new_record = {
                    "Batch": batch,
                    "Sequence": seq
                }
                
                # 检查是否有需要的两个模型的数据
                required_methods_data = True
                
                # 为指定的方法添加指标
                for method in methods_to_use:
                    method_data = df[(df["Batch"] == batch) & 
                                    (df["Sequence"] == seq) & 
                                    (df["Method"] == method)]
                    
                    if method_data.empty:
                        required_methods_data = False
                        break
                    
                    # 检查评估Excel文件中"Foot Skate"列的确切名称
                    foot_skate_col = None
                    for col in method_data.columns:
                        if col.lower().replace(" ", "_") == "foot_skate":
                            foot_skate_col = col
                            break
                    
                    # 收集所有度量指标
                    for metric_name, excel_col in [
                        ("L2P", "L2P"),
                        ("L2Q", "L2Q"),
                        ("NPSS", "NPSS"),
                        ("Foot_Skate", foot_skate_col or "Foot Skate")  # 使用找到的列名或默认名
                    ]:
                        if excel_col in method_data.columns:
                            new_record[f"{method}_{metric_name}"] = method_data[excel_col].values[0]
                        else:
                            print(f"警告: 在数据中找不到列 '{excel_col}'。可用列: {method_data.columns.tolist()}")
                            required_methods_data = False
                
                # 只有当所有必需的数据都存在时，才添加该记录
                if required_methods_data:
                    pivot_data.append(new_record)
                
            # 创建透视后的DataFrame
            pivot_df = pd.DataFrame(pivot_data)
            
            if len(pivot_df) == 0:
                print(f"警告: 过渡长度 {trans} 没有有效数据，跳过。")
                continue
            
            # 与曲率数据合并
            merged_df = pd.merge(
                pivot_df, 
                curvature_df, 
                on=["Batch", "Sequence"],
                how="left"
            )
            
            # 计算方法间的性能提升
            if baseline_model is not None and proposed_model is not None:
                # 使用指定的基准和提议模型
                for metric in ["L2P", "L2Q", "NPSS", "Foot_Skate"]:
                    # 检查列是否存在
                    baseline_col = f"{baseline_model}_{metric}"
                    proposed_col = f"{proposed_model}_{metric}"
                    
                    if baseline_col in merged_df.columns and proposed_col in merged_df.columns:
                        # 指标值越低越好，所以基线减去提议方法的值为"改进"
                        merged_df[f"{metric}_Improvement"] = merged_df[baseline_col] - merged_df[proposed_col]
                        
                        # 计算相对改进百分比
                        valid_mask = merged_df[baseline_col] != 0  # 避免除以零
                        merged_df[f"{metric}_Rel_Improvement"] = 0.0  # 初始化为零
                        if valid_mask.any():
                            merged_df.loc[valid_mask, f"{metric}_Rel_Improvement"] = (
                                merged_df.loc[valid_mask, f"{metric}_Improvement"] / 
                                merged_df.loc[valid_mask, baseline_col]
                            ) * 100
                    else:
                        if baseline_col not in merged_df.columns:
                            print(f"警告: 找不到列 '{baseline_col}'")
                        if proposed_col not in merged_df.columns:
                            print(f"警告: 找不到列 '{proposed_col}'")
            
            # 存储结果
            merged_data[trans] = merged_df
            
            print(f"过渡长度 {trans}: {len(merged_df)} 行, {len(merged_df.columns)} 列")
    
    return merged_data

def create_performance_vs_curvature_plot(merged_df, transition, output_dir, metric="L2P", 
                                 baseline_model=None, proposed_model=None):
    """
    Create performance vs. curvature scatter plots, comparing baseline and proposed methods.
    
    Args:
        merged_df (pd.DataFrame): Merged data
        transition (int): Transition length
        output_dir (str): Output directory
        metric (str): Metric to analyze
        baseline_model (str, optional): Baseline model label name
        proposed_model (str, optional): Proposed model label name
    """
    plt.figure(figsize=(10, 6))
    
    # Use specified methods or default
    baseline_method = baseline_model if baseline_model is not None else "TS-Trans"
    proposed_method = proposed_model if proposed_model is not None else "Ours"
    
    # Check if column names exist
    baseline_col = f"{baseline_method}_{metric}"
    proposed_col = f"{proposed_method}_{metric}"
    
    if baseline_col not in merged_df.columns or proposed_col not in merged_df.columns:
        print(f"Error: Failed to create {metric} chart for T{transition} - columns {baseline_col} or {proposed_col} not found")
        return
    
    # Create scatter plots
    sns.scatterplot(x="Trajectory_Curvature", 
                    y=baseline_col, 
                    data=merged_df, 
                    alpha=0.6, 
                    color="blue",
                    label=baseline_method)
    
    sns.scatterplot(x="Trajectory_Curvature", 
                    y=proposed_col, 
                    data=merged_df, 
                    alpha=0.6, 
                    color="red",
                    label=proposed_method)
    
    # Fit trend lines
    try:
        # Baseline trend
        baseline_mask = ~np.isnan(merged_df[baseline_col])
        x_baseline = merged_df["Trajectory_Curvature"][baseline_mask]
        y_baseline = merged_df[baseline_col][baseline_mask]
        
        slope_base, intercept_base, _, p_value_base, _ = stats.linregress(x_baseline, y_baseline)
        sns.regplot(x="Trajectory_Curvature", 
                   y=baseline_col, 
                   data=merged_df,
                   scatter=False,
                   line_kws={"color": "blue", "alpha": 0.8})
        
        # Proposed method trend
        proposed_mask = ~np.isnan(merged_df[proposed_col])
        x_proposed = merged_df["Trajectory_Curvature"][proposed_mask]
        y_proposed = merged_df[proposed_col][proposed_mask]
        
        slope_prop, intercept_prop, _, p_value_prop, _ = stats.linregress(x_proposed, y_proposed)
        sns.regplot(x="Trajectory_Curvature", 
                   y=proposed_col, 
                   data=merged_df,
                   scatter=False,
                   line_kws={"color": "red", "alpha": 0.8})
        
        # Calculate growth rate difference (slope difference)
        slope_diff = slope_base - slope_prop
        slope_rel_diff = (slope_diff / abs(slope_base)) * 100 if slope_base != 0 else float('inf')
        
        plt.text(0.05, 0.95, 
                f"{baseline_method}: y = {slope_base:.4f}x + {intercept_base:.4f} (p={p_value_base:.4f})\n"
                f"{proposed_method}: y = {slope_prop:.4f}x + {intercept_prop:.4f} (p={p_value_prop:.4f})\n"
                f"Slope diff: {slope_diff:.4f} ({slope_rel_diff:.1f}%)",
                transform=plt.gca().transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    except Exception as e:
        print(f"Warning: Failed to fit trend lines for {metric} at T{transition}, possibly insufficient data: {str(e)}")
        
    # Set title and labels
    plt.title(f"Transition Length T={transition}: {metric} vs Trajectory Curvature")
    plt.xlabel("Trajectory Curvature")
    plt.ylabel(f"{metric} Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save chart
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"T{transition}_{metric}_vs_curvature.png"), dpi=300, bbox_inches="tight")
    plt.close()

def create_improvement_vs_curvature_plot(merged_df, transition, output_dir, metric="L2P",
                                 baseline_model=None, proposed_model=None):
    """
    Create performance improvement vs. curvature scatter plots.
    
    Args:
        merged_df (pd.DataFrame): Merged data
        transition (int): Transition length
        output_dir (str): Output directory
        metric (str): Metric to analyze
        baseline_model (str, optional): Baseline model label name
        proposed_model (str, optional): Proposed model label name
    """
    plt.figure(figsize=(10, 6))
    
    # Metric improvement column
    improvement_col = f"{metric}_Improvement"
    
    if improvement_col not in merged_df.columns:
        print(f"Error: Failed to create {metric} improvement chart for T{transition} - column {improvement_col} not found")
        return
    
    # Create scatter plot
    sns.scatterplot(x="Trajectory_Curvature", 
                    y=improvement_col, 
                    data=merged_df, 
                    alpha=0.6)
    
    # Fit trend line
    try:
        mask = ~np.isnan(merged_df[improvement_col])
        x = merged_df["Trajectory_Curvature"][mask]
        y = merged_df[improvement_col][mask]
        
        if len(x) < 2:
            raise ValueError(f"At least 2 valid data points needed to fit trend line, but only found {len(x)}")
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        sns.regplot(x="Trajectory_Curvature", 
                   y=improvement_col, 
                   data=merged_df,
                   scatter=False,
                   color="green")
        
        # Add trend line equation and statistics
        plt.text(0.05, 0.95, 
                f"y = {slope:.4f}x + {intercept:.4f}\n"
                f"R² = {r_value**2:.4f}, p = {p_value:.4f}",
                transform=plt.gca().transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                
        # Add average improvement info
        mean_improvement = np.mean(y)
        plt.text(0.05, 0.80, 
                f"Mean improvement: {mean_improvement:.4f}",
                transform=plt.gca().transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    except Exception as e:
        print(f"Warning: Failed to fit improvement trend line for {metric} at T{transition}: {str(e)}")
        
    # Set title and labels
    model_labels = ""
    if baseline_model is not None and proposed_model is not None:
        model_labels = f" ({baseline_model} vs {proposed_model})"
    
    plt.title(f"Transition Length T={transition}: {metric} Improvement vs Trajectory Curvature{model_labels}")
    plt.xlabel("Trajectory Curvature")
    plt.ylabel(f"{metric} Improvement")
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Save chart
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"T{transition}_{metric}_improvement_vs_curvature.png"), dpi=300, bbox_inches="tight")
    plt.close()

def create_binned_bar_chart(merged_df, transition, output_dir, metric="L2P", num_bins=3,
                        baseline_model=None, proposed_model=None):
    """
    Create binned bar charts showing average performance improvement by trajectory curvature groups.
    
    Args:
        merged_df (pd.DataFrame): Merged data
        transition (int): Transition length
        output_dir (str): Output directory
        metric (str): Metric to analyze
        num_bins (int): Number of bins
        baseline_model (str, optional): Baseline model label name
        proposed_model (str, optional): Proposed model label name
    """
    plt.figure(figsize=(8, 6))
    
    # Metric improvement column
    improvement_col = f"{metric}_Improvement"
    
    if improvement_col not in merged_df.columns:
        print(f"Error: Failed to create {metric} binned chart for T{transition} - column {improvement_col} not found")
        return
    
    # Group by curvature
    # First determine quantiles
    curvature_min = merged_df["Trajectory_Curvature"].min()
    curvature_max = merged_df["Trajectory_Curvature"].max()
    
    if num_bins == 3:
        # Tertiles
        bins = [
            curvature_min,
            merged_df["Trajectory_Curvature"].quantile(0.33),
            merged_df["Trajectory_Curvature"].quantile(0.66),
            curvature_max
        ]
        labels = ["Low Curvature", "Medium Curvature", "High Curvature"]
    elif num_bins == 5:
        # Quintiles
        bins = [
            curvature_min,
            merged_df["Trajectory_Curvature"].quantile(0.2),
            merged_df["Trajectory_Curvature"].quantile(0.4),
            merged_df["Trajectory_Curvature"].quantile(0.6),
            merged_df["Trajectory_Curvature"].quantile(0.8),
            curvature_max
        ]
        labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    else:
        # Default to tertiles
        bins = [
            curvature_min,
            merged_df["Trajectory_Curvature"].quantile(0.33),
            merged_df["Trajectory_Curvature"].quantile(0.66),
            curvature_max
        ]
        labels = ["Low Curvature", "Medium Curvature", "High Curvature"]
    
    # Add curvature group column
    merged_df["Curvature_Group"] = pd.cut(
        merged_df["Trajectory_Curvature"], 
        bins=bins, 
        labels=labels, 
        include_lowest=True
    )
    
    # Calculate average improvement for each group
    grouped_data = merged_df.groupby("Curvature_Group")[improvement_col].mean().reset_index()
    
    # Calculate sample count for each group
    group_counts = merged_df.groupby("Curvature_Group").size().reset_index(name="Count")
    grouped_data = pd.merge(grouped_data, group_counts, on="Curvature_Group")
    
    # Add standard deviation for each group
    grouped_std = merged_df.groupby("Curvature_Group")[improvement_col].std().reset_index()
    grouped_std = grouped_std.rename(columns={improvement_col: f"{improvement_col}_std"})
    grouped_data = pd.merge(grouped_data, grouped_std, on="Curvature_Group")
    
    # Add sample count and mean labels to bar chart
    def add_labels(x, y, count, std=None):
        for i, (xi, yi, ci) in enumerate(zip(x, y, count)):
            # Calculate label position (adjust based on value sign)
            if yi >= 0:
                va = 'bottom'
                offset = 0.01
            else:
                va = 'top'
                offset = -0.1
                
            # Add sample count label
            plt.text(xi, yi + offset, 
                     f"n={ci}", 
                     ha='center', va=va, 
                     fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # If std available, add mean±std label
            if std is not None and i < len(std):
                plt.text(xi, yi - offset*2 if yi >= 0 else yi + offset*2, 
                         f"μ={yi:.3f}±{std[i]:.3f}", 
                         ha='center', va='top' if yi >= 0 else 'bottom', 
                         fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
    # Draw bar chart
    ax = sns.barplot(x="Curvature_Group", y=improvement_col, data=grouped_data)
    
    # Add sample count and mean labels
    add_labels(
        ax.get_xticks(), 
        grouped_data[improvement_col], 
        grouped_data["Count"],
        grouped_data[f"{improvement_col}_std"] if f"{improvement_col}_std" in grouped_data.columns else None
    )
    
    # Add horizontal reference line
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Set title and labels
    model_labels = ""
    if baseline_model is not None and proposed_model is not None:
        model_labels = f" ({baseline_model} vs {proposed_model})"
        
    plt.title(f"Transition Length T={transition}: Average {metric} Improvement by Curvature Group{model_labels}")
    plt.xlabel("Trajectory Curvature Group")
    plt.ylabel(f"Average {metric} Improvement")
    plt.grid(True, alpha=0.3)
    
    # Save chart
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"T{transition}_{metric}_binned_improvement.png"), dpi=300, bbox_inches="tight")
    plt.close()

def create_combined_chart(merged_data, output_dir, metric="L2P", baseline_model=None, proposed_model=None):
    """
    Create combined charts for all transition lengths, showing the relationship between curvature and performance.
    
    Args:
        merged_data (dict): Dictionary containing merged data for each transition length
        output_dir (str): Output directory
        metric (str): Metric to analyze
        baseline_model (str): Baseline model name
        proposed_model (str): Proposed model name
    """
    # Create a 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Map transition lengths to subplot positions
    positions = {
        15: (0, 0),
        30: (0, 1),
        60: (1, 0),
        90: (1, 1)
    }
    
    # Define methods
    baseline_method = baseline_model
    proposed_method = proposed_model
    
    # Create scatter plots for each transition length
    for transition, df in merged_data.items():
        if transition in positions:
            row, col = positions[transition]
            ax = axs[row, col]
            
            # Draw scatter plots
            sns.scatterplot(x="Trajectory_Curvature", 
                           y=f"{baseline_method}_{metric}", 
                           data=df, 
                           alpha=0.6, 
                           color="blue", 
                           label=baseline_method,
                           ax=ax)
            
            sns.scatterplot(x="Trajectory_Curvature", 
                           y=f"{proposed_method}_{metric}", 
                           data=df, 
                           alpha=0.6, 
                           color="red", 
                           label=proposed_method,
                           ax=ax)
            
            # Fit trend lines
            try:
                baseline_mask = ~np.isnan(df[f"{baseline_method}_{metric}"])
                x_baseline = df["Trajectory_Curvature"][baseline_mask]
                y_baseline = df[f"{baseline_method}_{metric}"][baseline_mask]
                
                slope_base, intercept_base, _, p_value_base, _ = stats.linregress(x_baseline, y_baseline)
                sns.regplot(x="Trajectory_Curvature", 
                           y=f"{baseline_method}_{metric}", 
                           data=df,
                           scatter=False,
                           line_kws={"color": "blue", "alpha": 0.8},
                           ax=ax)
                
                proposed_mask = ~np.isnan(df[f"{proposed_method}_{metric}"])
                x_proposed = df["Trajectory_Curvature"][proposed_mask]
                y_proposed = df[f"{proposed_method}_{metric}"][proposed_mask]
                
                slope_prop, intercept_prop, _, p_value_prop, _ = stats.linregress(x_proposed, y_proposed)
                sns.regplot(x="Trajectory_Curvature", 
                           y=f"{proposed_method}_{metric}", 
                           data=df,
                           scatter=False,
                           line_kws={"color": "red", "alpha": 0.8},
                           ax=ax)
                
                # Add trend line information
                ax.text(0.05, 0.95,
                        f"{baseline_method}: y = {slope_base:.4f}x + {intercept_base:.4f} (p={p_value_base:.4f})\n"
                        f"{proposed_method}: y = {slope_prop:.4f}x + {intercept_prop:.4f} (p={p_value_prop:.4f})",
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            except Exception as e:
                print(f"Warning: Failed to fit trend lines for T{transition} in combined chart: {str(e)}")
            
            # Set subplot titles and labels
            ax.set_title(f"Transition Length T={transition}")
            ax.set_xlabel("Trajectory Curvature")
            ax.set_ylabel(f"{metric} Value")
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Set overall title
    fig.suptitle(f"Relationship between {metric} and Trajectory Curvature across Transition Lengths", fontsize=16)
    
    # Save chart
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"combined_{metric}_vs_curvature.png"), dpi=300, bbox_inches="tight")
    plt.close()

def create_multi_metric_grid(merged_data, output_dir, baseline_model=None, proposed_model=None):
    """
    Create a grid of charts for all metrics, for comprehensive analysis.
    
    Args:
        merged_data (dict): Dictionary containing merged data for each transition length
        output_dir (str): Output directory
        baseline_model (str): Baseline model name
        proposed_model (str): Proposed model name
    """
    # Define metrics list
    metrics = ["L2P", "L2Q", "NPSS", "Foot_Skate"]
    
    # Create a large 4x4 grid (4 metrics x 4 transition lengths)
    fig, axs = plt.subplots(4, 4, figsize=(20, 16))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Transition length list
    transitions = [15, 30, 60, 90]
    
    # Create subplots for each metric and transition length
    for i, metric in enumerate(metrics):
        for j, transition in enumerate(transitions):
            if transition in merged_data:
                df = merged_data[transition]
                ax = axs[i, j]
                
                # Create improvement vs curvature scatter plot
                improvement_col = f"{metric}_Improvement"
                
                # Use alpha to reflect data density
                sns.scatterplot(x="Trajectory_Curvature", 
                                y=improvement_col, 
                                data=df, 
                                alpha=0.6,
                                ax=ax)
                
                # Fit trend line
                try:
                    mask = ~np.isnan(df[improvement_col])
                    x = df["Trajectory_Curvature"][mask]
                    y = df[improvement_col][mask]
                    
                    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
                    sns.regplot(x="Trajectory_Curvature", 
                               y=improvement_col, 
                               data=df,
                               scatter=False,
                               line_kws={"color": "green", "alpha": 0.8},
                               ax=ax)
                    
                    # Add trend line equation
                    ax.text(0.05, 0.95, 
                            f"y = {slope:.4f}x + {intercept:.4f}\n"
                            f"R² = {r_value**2:.4f}, p = {p_value:.4f}",
                            transform=ax.transAxes,
                            fontsize=8,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                except Exception as e:
                    print(f"Warning: Failed to fit trend line for T{transition} {metric}: {e}")
                
                # Add horizontal reference line
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                
                # Set title and labels
                model_info = ""
                if baseline_model and proposed_model:
                    model_info = f" ({baseline_model} vs {proposed_model})"
                
                ax.set_title(f"T={transition}: {metric} Improvement{model_info}")
                ax.set_xlabel("Trajectory Curvature" if i == 3 else "")  # Only show X labels on bottom row
                ax.set_ylabel(f"{metric} Improvement" if j == 0 else "")  # Only show Y labels on leftmost column
                ax.grid(True, alpha=0.3)
    
    # Set overall title
    fig.suptitle("Relationship between Trajectory Curvature and Metric Improvements", fontsize=20)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save chart
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "all_metrics_grid.png"), dpi=300, bbox_inches="tight")
    plt.close()

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="轨迹曲率与模型性能关系分析")
    parser.add_argument("--eval_file", type=str, default="evaluation_results.xlsx",
                        help="评估结果Excel文件路径")
    parser.add_argument("--dataset", type=str, default="lafan1",
                        help="数据集名称")
    parser.add_argument("--config", type=str, default="default.yaml",
                        help="配置文件名称")
    parser.add_argument("--output_dir", type=str, default="output/curvature_analysis",
                        help="输出目录")
    parser.add_argument("--compute_curvature", action="store_true",
                        help="重新计算所有样本的曲率")
    parser.add_argument("--curvature_file", type=str, default="trajectory_curvatures.csv",
                        help="保存/加载曲率数据的CSV文件名")
    parser.add_argument("--baseline_model", type=str, default=None, 
                        help="基准模型的标签名称，例如 'Ours-0'")
    parser.add_argument("--proposed_model", type=str, default=None,
                        help="提议模型的标签名称，例如 'Ours-2'")
    parser.add_argument("--num_bins", type=int, default=3,
                        help="分桶柱状图中的分桶数量 (3或5)")
    
    args = parser.parse_args()
    
    # 验证必要的参数
    if args.baseline_model is None or args.proposed_model is None:
        print("警告: 未指定 --baseline_model 或 --proposed_model，将尝试使用默认值")
    else:
        print(f"使用指定的基准模型: {args.baseline_model}")
        print(f"使用指定的提议模型: {args.proposed_model}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 分析评估文件结构
    sheet_names, transitions, methods, baseline_model, proposed_model = analyze_excel_structure(
        args.eval_file, 
        baseline_model=args.baseline_model, 
        proposed_model=args.proposed_model
    )
    
    # 如果函数根据文件内容调整了模型标签，更新args中的值
    if args.baseline_model != baseline_model or args.proposed_model != proposed_model:
        print(f"注意: 模型标签已根据文件内容调整为: {baseline_model}(基准) 和 {proposed_model}(提议)")
        args.baseline_model = baseline_model
        args.proposed_model = proposed_model
    
    # 计算或加载曲率数据
    curvature_file = os.path.join(args.output_dir, args.curvature_file)
    
    if args.compute_curvature or not os.path.exists(curvature_file):
        print(f"计算数据集的轨迹曲率...")
        curvature_df = compute_curvatures_for_dataset(args.dataset, args.config)
        curvature_df.to_csv(curvature_file, index=False)
        print(f"曲率数据已保存到: {curvature_file}")
    else:
        print(f"从文件加载曲率数据: {curvature_file}")
        curvature_df = pd.read_csv(curvature_file)
    
    # 合并评估结果和曲率数据
    merged_data = merge_evaluation_with_curvatures(
        args.eval_file, 
        curvature_df,
        baseline_model=args.baseline_model,
        proposed_model=args.proposed_model
    )
    
    # 如果没有合并数据，退出
    if not merged_data:
        print("错误: 未能生成合并的数据。请检查输入文件和模型名称。")
        return
    
    # 保存合并后的数据
    model_suffix = f"_{args.baseline_model}_vs_{args.proposed_model}" if args.baseline_model and args.proposed_model else ""
    merged_file = os.path.join(args.output_dir, f"merged_evaluation_with_curvature{model_suffix}.xlsx")
    
    # 检查是否已存在相同的文件
    if os.path.exists(merged_file):
        print(f"发现已存在的轨迹曲率表格: {merged_file}")
        try:
            # 尝试读取已有文件
            existing_data = pd.read_excel(merged_file, sheet_name=None)
            sheet_names = list(existing_data.keys())
            
            # 检查是否包含所有必要的工作表
            all_transitions_present = True
            for trans in merged_data.keys():
                if f"T{trans}" not in sheet_names:
                    all_transitions_present = False
                    break
            
            if all_transitions_present:
                print(f"已有的表格包含所有必要的数据，跳过生成表格步骤")
                # 从现有Excel加载数据到merged_data
                for trans in merged_data.keys():
                    merged_data[trans] = existing_data[f"T{trans}"]
            else:
                print(f"已有的表格不完整，将重新生成")
                with pd.ExcelWriter(merged_file) as writer:
                    for trans, df in merged_data.items():
                        df.to_excel(writer, sheet_name=f"T{trans}", index=False)
                print(f"合并数据已保存到: {merged_file}")
        except Exception as e:
            print(f"读取已有表格时出错: {str(e)}，将重新生成")
            with pd.ExcelWriter(merged_file) as writer:
                for trans, df in merged_data.items():
                    df.to_excel(writer, sheet_name=f"T{trans}", index=False)
            print(f"合并数据已保存到: {merged_file}")
    else:
        # 文件不存在，生成新文件
        with pd.ExcelWriter(merged_file) as writer:
            for trans, df in merged_data.items():
                df.to_excel(writer, sheet_name=f"T{trans}", index=False)
        print(f"合并数据已保存到: {merged_file}")
    
    # 为每个过渡长度创建分析图表
    metrics = ["L2P", "L2Q", "NPSS", "Foot_Skate"]
    
    # 创建图表子目录
    figures_dir = os.path.join(args.output_dir, f"figures{model_suffix}")
    os.makedirs(figures_dir, exist_ok=True)
    
    for metric in metrics:
        print(f"创建 {metric} 指标的分析图表...")
        
        for transition, df in merged_data.items():
            print(f"  - 过渡长度 T{transition}...")
            # 性能 vs. 曲率散点图
            create_performance_vs_curvature_plot(
                df, transition, figures_dir, metric,
                baseline_model=args.baseline_model,
                proposed_model=args.proposed_model
            )
            
            # 性能提升 vs. 曲率散点图
            create_improvement_vs_curvature_plot(
                df, transition, figures_dir, metric,
                baseline_model=args.baseline_model,
                proposed_model=args.proposed_model
            )
            
            # 分桶柱状图
            create_binned_bar_chart(
                df, transition, figures_dir, metric, num_bins=args.num_bins,
                baseline_model=args.baseline_model,
                proposed_model=args.proposed_model
            )
    
    # Create combined charts
    print("Creating combined charts...")
    for metric in metrics:
        create_combined_chart(merged_data, figures_dir, metric, 
                             baseline_model=args.baseline_model,
                             proposed_model=args.proposed_model)
    
    # Create multi-metric grid
    print("Creating multi-metric grid...")
    create_multi_metric_grid(merged_data, figures_dir,
                            baseline_model=args.baseline_model,
                            proposed_model=args.proposed_model)
    
    print(f"所有分析完成。结果保存在: {figures_dir}")
    
    # 返回一个简单的分析总结
    print("\n=== 分析总结 ===")
    print(f"比较的模型: {args.baseline_model}(基准) vs {args.proposed_model}(提议)")
    print(f"过渡长度: {sorted(merged_data.keys())}")
    
    # 计算每个过渡长度的平均改进
    for trans, df in merged_data.items():
        print(f"\n过渡长度 T={trans} 的平均改进:")
        for metric in metrics:
            improvement_col = f"{metric}_Improvement"
            if improvement_col in df.columns:
                mean_improvement = df[improvement_col].mean()
                std_improvement = df[improvement_col].std()
                
                # 计算改进占比
                positive_count = (df[improvement_col] > 0).sum()
                total_count = len(df)
                positive_percentage = (positive_count / total_count) * 100 if total_count > 0 else 0
                
                print(f"  {metric}: {mean_improvement:.4f}±{std_improvement:.4f} (正改进样本比例: {positive_percentage:.1f}%)")
                
                # 计算相关性
                correlation = df[[improvement_col, "Trajectory_Curvature"]].corr().iloc[0, 1]
                print(f"  与曲率相关性: {correlation:.4f}")
    
    print("\n图表已生成在目录: " + figures_dir)

if __name__ == "__main__":
    main()