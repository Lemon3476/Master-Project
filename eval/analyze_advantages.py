#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析优势样本脚本 (Advantage Sample Analysis) - 按过渡长度分离版
该脚本用于读取evaluation_results.xlsx文件，按照不同的过渡长度独立计算并筛选出模型性能提升最显著的动作序列。
"""

import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats


def load_excel_data(excel_file):
    """
    加载Excel文件中的所有工作表
    
    Args:
        excel_file (str): Excel文件路径
        
    Returns:
        dict: 包含所有工作表数据的字典，键为工作表名，值为DataFrame
    """
    print(f"正在加载Excel文件: {excel_file}")
    
    # 读取所有工作表
    excel_data = pd.read_excel(excel_file, sheet_name=None)
    
    # 存储有效的过渡长度工作表
    transition_sheets = {}
    
    # 遍历所有工作表，筛选出有效的过渡长度工作表
    for sheet_name, df in excel_data.items():
        # 检查工作表名称是否符合格式 "Transition_XX"
        if sheet_name.startswith("Transition_"):
            try:
                # 提取过渡长度
                transition_length = int(sheet_name.split("_")[1])
                print(f"  找到有效工作表: {sheet_name}, 过渡长度: {transition_length}")
                
                # 将DataFrame添加到字典中
                transition_sheets[sheet_name] = df
            except ValueError:
                print(f"  警告: 无法从工作表名称 '{sheet_name}' 中提取过渡长度，跳过该工作表")
        else:
            print(f"  跳过非过渡长度工作表: {sheet_name}")
    
    if not transition_sheets:
        raise ValueError("未找到有效的过渡长度工作表，请检查Excel文件格式")
    
    print(f"成功加载 {len(transition_sheets)} 个过渡长度工作表")
    
    return transition_sheets


def pivot_data(df, baseline_method="Ours-0", proposed_method="Ours-2", transition_length=None):
    """
    将长格式数据转换为宽格式，便于直接比较两种方法
    
    Args:
        df (pd.DataFrame): 单个过渡长度的DataFrame
        baseline_method (str): 基线方法名称
        proposed_method (str): 提议方法名称
        transition_length (int, optional): 过渡长度，仅用于日志输出
        
    Returns:
        pd.DataFrame: 透视后的DataFrame
    """
    transition_info = f"过渡长度 {transition_length}" if transition_length is not None else ""
    print(f"执行数据透视 {transition_info}，基线方法: {baseline_method}，提议方法: {proposed_method}")
    
    # 检查方法列是否存在
    if "Method" not in df.columns:
        raise ValueError("DataFrame中缺少'Method'列")
    
    # 筛选出基线和提议方法的数据
    methods_df = df[df["Method"].isin([baseline_method, proposed_method])].copy()
    
    if methods_df.empty:
        raise ValueError(f"找不到方法 '{baseline_method}' 或 '{proposed_method}' 的数据")
    
    # 检查所需的评估指标列是否存在
    metrics = ["L2P", "L2Q", "NPSS", "Foot_Skate"]
    for metric in metrics:
        if metric not in df.columns:
            raise ValueError(f"DataFrame中缺少'{metric}'列")
    
    # 确定索引列 - 不再包含Transition_Length，因为每个过渡长度单独处理
    index_cols = ["Batch", "Sequence"]
    for col in index_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame中缺少'{col}'列")
    
    # 执行数据透视
    pivoted_df = pd.pivot_table(
        methods_df,
        index=index_cols,
        columns="Method",
        values=metrics
    )
    
    # 重置MultiIndex，使结果更易于处理
    pivoted_df = pivoted_df.reset_index()
    
    # 将列名从MultiIndex转换为单层索引，格式为 "metric_method"
    new_columns = []
    for col in pivoted_df.columns:
        if isinstance(col, tuple) and len(col) > 1:
            # 指标列，格式为 "metric_method"
            new_columns.append(f"{col[0]}_{col[1]}")
        elif col == "Batch" or col == "Sequence":
            # 确保索引列没有下划线后缀
            new_columns.append(col)
        else:
            # 其他任何列，保持原样
            new_columns.append(col)
    
    pivoted_df.columns = new_columns
    
    # 额外的检查，确保没有"Batch_"或"Sequence_"这样带下划线后缀的列
    rename_dict = {}
    for col in pivoted_df.columns:
        if col in ["Batch_", "Sequence_"]:
            rename_dict[col] = col.rstrip("_")
    
    # 应用重命名（如果需要）
    if rename_dict:
        print(f"  注意: 修正列名 {', '.join(rename_dict.keys())} -> {', '.join(rename_dict.values())}")
        pivoted_df = pivoted_df.rename(columns=rename_dict)
    
    print(f"  数据透视完成，共 {len(pivoted_df)} 行")
    
    return pivoted_df


def calculate_advantage_score(df, baseline_method="Ours-0", proposed_method="Ours-2", transition_length=None, l2p_only=False):
    """
    计算综合提升分数 (Advantage Score)
    
    Args:
        df (pd.DataFrame): 单个过渡长度的透视后DataFrame
        baseline_method (str): 基线方法名称
        proposed_method (str): 提议方法名称
        transition_length (int, optional): 过渡长度，仅用于日志输出
        l2p_only (bool, optional): 是否只考虑L2P指标进行排名，默认为False
        
    Returns:
        pd.DataFrame: 添加了提升分数的DataFrame
    """
    transition_info = f"过渡长度 {transition_length}" if transition_length is not None else ""
    
    if l2p_only:
        print(f"计算L2P单指标提升分数 {transition_info}...")
    else:
        print(f"计算综合提升分数 {transition_info}...")
    
    # 确保输入数据是副本，避免警告
    result_df = df.copy()
    
    # 指标列表
    metrics = ["L2P", "L2Q", "NPSS", "Foot_Skate"]
    
    # 如果只考虑L2P指标，则只使用L2P
    if l2p_only:
        metrics = ["L2P"]
    
    # 为每个指标计算提升率
    for metric in metrics:
        baseline_col = f"{metric}_{baseline_method}"
        proposed_col = f"{metric}_{proposed_method}"
        
        # 确保列存在
        if baseline_col not in result_df.columns or proposed_col not in result_df.columns:
            raise ValueError(f"缺少列 '{baseline_col}' 或 '{proposed_col}'")
        
        # 计算绝对提升值
        impr_col = f"{metric}_Impr"
        result_df[impr_col] = result_df[baseline_col] - result_df[proposed_col]
        
        # 计算相对提升率，处理分母为零的情况
        impr_ratio_col = f"{metric}_Impr_Ratio"
        result_df[impr_ratio_col] = np.where(
            result_df[baseline_col] != 0,
            result_df[impr_col] / result_df[baseline_col],
            np.where(result_df[impr_col] > 0, 1.0, 0.0)  # 如果基线为0，根据提升方向设置为1或0
        )
        
        # 标准化提升率 (Z-Score) - 在当前过渡长度的数据子集上独立进行
        z_score_col = f"{metric}_Z_Score"
        # 只对当前过渡长度的数据计算Z-Score
        result_df[z_score_col] = stats.zscore(result_df[impr_ratio_col], nan_policy='omit')
        
        # 处理可能的NaN值
        result_df[z_score_col] = result_df[z_score_col].fillna(0)
    
    # 计算提升分数
    if l2p_only:
        # 如果只考虑L2P，则直接使用L2P的Z-Score作为优势分数
        result_df["Advantage_Score"] = result_df["L2P_Z_Score"]
        print(f"  {transition_info} L2P单指标提升分数计算完成")
    else:
        # 计算综合提升分数 (四个指标的Z-Score的等权重平均)
        z_score_cols = [f"{metric}_Z_Score" for metric in metrics]
        result_df["Advantage_Score"] = result_df[z_score_cols].mean(axis=1)
        print(f"  {transition_info} 综合提升分数计算完成")
    
    return result_df


def sort_results(df):
    """
    对DataFrame按综合提升分数排序
    
    Args:
        df (pd.DataFrame): 包含提升分数的DataFrame
        
    Returns:
        pd.DataFrame: 排序后的DataFrame
    """
    # 根据综合提升分数降序排序
    return df.sort_values("Advantage_Score", ascending=False).reset_index(drop=True)


def print_top_samples(sorted_df, transition_length, top_n=20):
    """
    打印某个过渡长度下优势最明显的Top-N样本
    
    Args:
        sorted_df (pd.DataFrame): 排序后的DataFrame
        transition_length (int): 过渡长度
        top_n (int): 要打印的样本数量
    """
    sample_count = min(top_n, len(sorted_df))
    print(f"\nTransition Length {transition_length} - Most Advantageous Samples (Top {sample_count}):")
    print("=" * 100)
    
    # 定义要显示的列
    display_cols = [
        "Batch", "Sequence", "Advantage_Score",
        "L2P_Impr", "L2Q_Impr", "NPSS_Impr", "Foot_Skate_Impr",
        "L2P_Impr_Ratio", "L2Q_Impr_Ratio", "NPSS_Impr_Ratio", "Foot_Skate_Impr_Ratio"
    ]
    
    # 只显示存在的列
    display_cols = [col for col in display_cols if col in sorted_df.columns]
    
    # 输出表头
    header = " | ".join([f"{col:15}" for col in display_cols])
    print(header)
    print("-" * 100)
    
    # 输出Top-N样本
    for i in range(min(top_n, len(sorted_df))):
        row = sorted_df.iloc[i]
        
        # 构建行字符串
        row_values = []
        for col in display_cols:
            if col in ["Batch", "Sequence"]:
                row_values.append(f"{row[col]:15.0f}")
            else:
                row_values.append(f"{row[col]:15.4f}")
        
        print(" | ".join(row_values))
    
    print("=" * 100)


def output_results_by_transition(results_dict, output_file, top_n=None, l2p_only=False):
    """
    将不同过渡长度的结果保存到Excel文件的不同工作表
    
    Args:
        results_dict (dict): 包含每个过渡长度的排序后DataFrame的字典
        output_file (str): 输出Excel文件路径
        top_n (int, optional): 要保留的每个过渡长度的优势样本数量。如果为None，则保留所有样本。
        l2p_only (bool, optional): 是否只考虑L2P指标进行排名，默认为False
    """
    print(f"\nSaving sorted results to: {output_file}")
    
    # 检查是否使用L2P单指标排名
    l2p_only = "_l2p_ranking" in output_file
    
    # 使用ExcelWriter一次性创建所有工作表
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 为每个过渡长度创建一个工作表
        for transition_length, df in sorted(results_dict.items()):
            # 如果指定了top_n，则仅保留Top-N行数据；否则保留所有数据
            if top_n is not None:
                output_df = df.iloc[:top_n] if len(df) >= top_n else df
                sample_info = f"Top-{len(output_df)}"
            else:
                output_df = df
                sample_info = "All"
            
            # 根据是否只考虑L2P调整工作表名称
            if l2p_only:
                sheet_name = f"L2P_Advantage_Rank_T{transition_length}"
            else:
                sheet_name = f"Advantage_Rank_T{transition_length}"
                
            output_df.to_excel(writer, sheet_name=sheet_name, index=False, float_format="%.4f")
            print(f"  - Created worksheet '{sheet_name}', containing {sample_info} samples ({len(output_df)} out of {len(df)} total)")
    
    # 控制台打印样本，最多显示20个，避免输出过多
    display_count = 20 if top_n is None else min(top_n, 20)
    for transition_length, df in sorted(results_dict.items()):
        print_top_samples(df, transition_length, display_count)
    
    print(f"\nDetailed results saved to: {output_file}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="分析并筛选模型性能提升最显著的动作序列 - 按过渡长度分离版")
    
    parser.add_argument("--input", type=str, default="evaluation_results.xlsx",
                        help="输入Excel文件路径 (默认: evaluation_results.xlsx)")
    
    parser.add_argument("--output", type=str, default="advantage_ranking_by_transition.xlsx",
                        help="输出Excel文件路径 (默认: advantage_ranking_by_transition.xlsx)")
    
    parser.add_argument("--baseline", type=str, default="Ours-0",
                        help="基线方法名称 (默认: Ours-0)")
    
    parser.add_argument("--proposed", type=str, default="Ours-2",
                        help="提议方法名称 (默认: Ours-2)")
    
    parser.add_argument("--top", type=int, default=None,
                        help="每个过渡长度保留的优势样本数量 (默认: 所有样本)")
    
    parser.add_argument("--l2p_ranking", action="store_true",
                        help="只使用L2P指标进行排名 (默认: 使用所有指标)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    if args.l2p_ranking:
        print("优势样本分析工具 - 按过渡长度分离版 [L2P单指标排名]")
    else:
        print("优势样本分析工具 - 按过渡长度分离版")
    print("=" * 80)
    
    try:
        # 根据是否只考虑L2P调整输出文件名
        output_file = args.output
        if args.l2p_ranking:
            # 在文件扩展名前添加"_l2p_ranking"后缀
            base, ext = os.path.splitext(output_file)
            output_file = f"{base}_l2p_ranking{ext}"
            print(f"使用L2P单指标排名，输出文件: {output_file}")
        else:
            print(f"使用综合指标排名，输出文件: {output_file}")
            
        # 加载Excel文件中的所有工作表
        transition_sheets = load_excel_data(args.input)
        
        # 存储每个过渡长度的处理结果
        results_by_transition = {}
        
        # 处理每个过渡长度的工作表
        for sheet_name, df in transition_sheets.items():
            # 从工作表名称中提取过渡长度
            transition_length = int(sheet_name.split("_")[1])
            print(f"\n处理过渡长度 {transition_length} 的数据...")
            
            # 数据透视 - 针对当前过渡长度独立处理
            pivoted_df = pivot_data(df, args.baseline, args.proposed, transition_length)
            
            # 计算提升分数 - 根据l2p_ranking参数决定是否只考虑L2P
            scored_df = calculate_advantage_score(
                pivoted_df, 
                args.baseline, 
                args.proposed, 
                transition_length,
                l2p_only=args.l2p_ranking
            )
            
            # 对结果排序
            sorted_df = sort_results(scored_df)
            
            # 添加到结果字典中
            results_by_transition[transition_length] = sorted_df
            
            print(f"过渡长度 {transition_length} 的数据处理完成")
        
        # 输出所有过渡长度的结果
        output_results_by_transition(results_by_transition, output_file, args.top, args.l2p_ranking)
        
        print("\n分析完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()