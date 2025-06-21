import sys
sys.path.append(".")

import os
import argparse
import numpy as np
# 设置matplotlib使用Agg后端，这是一个非交互式后端，不依赖Qt
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader

from utils import eval_backup, utils
from utils.dataset import MotionDataset
from model.twostage import ContextTransformer, DetailTransformer

# 分析结果的数据结构
class AnalysisResults:
    def __init__(self):
        self.transition_lengths = []
        self.batch_indices = []
        self.sequence_indices = []
        self.height_ranges = []
        
        # 基准模型的评估指标
        self.baseline_l2p_scores = []
        self.baseline_l2q_scores = []
        self.baseline_npss_scores = []
        self.baseline_foot_skate_scores = []
        
        # 提议模型的评估指标
        self.proposed_l2p_scores = []
        self.proposed_l2q_scores = []
        self.proposed_npss_scores = []
        self.proposed_foot_skate_scores = []
        
    def add_result(self, transition_length, batch_idx, seq_idx, height_range, 
                   baseline_l2p, baseline_l2q, baseline_npss, baseline_foot_skate,
                   proposed_l2p, proposed_l2q, proposed_npss, proposed_foot_skate):
        self.transition_lengths.append(transition_length)
        self.batch_indices.append(batch_idx)
        self.sequence_indices.append(seq_idx)
        self.height_ranges.append(height_range)
        
        # 基准模型的评估指标
        self.baseline_l2p_scores.append(baseline_l2p)
        self.baseline_l2q_scores.append(baseline_l2q)
        self.baseline_npss_scores.append(baseline_npss)
        self.baseline_foot_skate_scores.append(baseline_foot_skate)
        
        # 提议模型的评估指标
        self.proposed_l2p_scores.append(proposed_l2p)
        self.proposed_l2q_scores.append(proposed_l2q)
        self.proposed_npss_scores.append(proposed_npss)
        self.proposed_foot_skate_scores.append(proposed_foot_skate)
        
    def save_to_file(self, filename):
        """将结果保存到Excel文件，每种过渡长度使用单独的工作表，包括两个模型的结果比较"""
        df = pd.DataFrame({
            'Transition_Length': self.transition_lengths,
            'Batch': self.batch_indices,
            'Sequence': self.sequence_indices,
            'Height_Range': self.height_ranges,
            
            # 基准模型的指标
            'Baseline_L2P': self.baseline_l2p_scores,
            'Baseline_L2Q': self.baseline_l2q_scores,
            'Baseline_NPSS': self.baseline_npss_scores,
            'Baseline_Foot_Skate': self.baseline_foot_skate_scores,
            
            # 提议模型的指标
            'Proposed_L2P': self.proposed_l2p_scores,
            'Proposed_L2Q': self.proposed_l2q_scores,
            'Proposed_NPSS': self.proposed_npss_scores,
            'Proposed_Foot_Skate': self.proposed_foot_skate_scores,
            
            # 计算性能提升（基准模型 - 提议模型，因为对所有指标，值越低越好）
            'L2P_Improvement': [b - p for b, p in zip(self.baseline_l2p_scores, self.proposed_l2p_scores)],
            'L2Q_Improvement': [b - p for b, p in zip(self.baseline_l2q_scores, self.proposed_l2q_scores)],
            'NPSS_Improvement': [b - p for b, p in zip(self.baseline_npss_scores, self.proposed_npss_scores)],
            'Foot_Skate_Improvement': [b - p for b, p in zip(self.baseline_foot_skate_scores, self.proposed_foot_skate_scores)]
        })
        
        # 计算相对性能提升（百分比）
        for metric in ['L2P', 'L2Q', 'NPSS', 'Foot_Skate']:
            baseline_col = f'Baseline_{metric}'
            rel_improvement_col = f'{metric}_Rel_Improvement'
            improvement_col = f'{metric}_Improvement'
            
            # 避免除以零
            df[rel_improvement_col] = df.apply(
                lambda row: (row[improvement_col] / row[baseline_col] * 100) if row[baseline_col] != 0 else 0, 
                axis=1
            )
        
        # 保存为Excel文件，每种过渡长度使用单独的工作表
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 为每个过渡长度创建单独的工作表
            unique_transitions = sorted(df['Transition_Length'].unique())
            for trans_len in unique_transitions:
                trans_data = df[df['Transition_Length'] == trans_len].copy()
                
                # 按高度范围排序
                trans_data = trans_data.sort_values('Height_Range')
                
                # 去掉现在多余的过渡长度列
                trans_data = trans_data.drop('Transition_Length', axis=1)
                
                sheet_name = f'Transition_{trans_len}'
                trans_data.to_excel(writer, sheet_name=sheet_name, index=False, float_format="%.4f")
                print(f"  - Sheet '{sheet_name}' created with {len(trans_data)} rows")
            
            # 添加统计数据到新的工作表
            summary_stats = []
            for trans_len in unique_transitions:
                trans_data = df[df['Transition_Length'] == trans_len]
                
                # 计算高度范围与每个指标的相关性
                # 基准模型
                corr_baseline_l2p = trans_data['Height_Range'].corr(trans_data['Baseline_L2P'])
                corr_baseline_l2q = trans_data['Height_Range'].corr(trans_data['Baseline_L2Q'])
                corr_baseline_npss = trans_data['Height_Range'].corr(trans_data['Baseline_NPSS'])
                corr_baseline_fs = trans_data['Height_Range'].corr(trans_data['Baseline_Foot_Skate'])
                
                # 提议模型
                corr_proposed_l2p = trans_data['Height_Range'].corr(trans_data['Proposed_L2P'])
                corr_proposed_l2q = trans_data['Height_Range'].corr(trans_data['Proposed_L2Q'])
                corr_proposed_npss = trans_data['Height_Range'].corr(trans_data['Proposed_NPSS'])
                corr_proposed_fs = trans_data['Height_Range'].corr(trans_data['Proposed_Foot_Skate'])
                
                # 性能提升
                corr_l2p_improvement = trans_data['Height_Range'].corr(trans_data['L2P_Improvement'])
                corr_l2q_improvement = trans_data['Height_Range'].corr(trans_data['L2Q_Improvement'])
                corr_npss_improvement = trans_data['Height_Range'].corr(trans_data['NPSS_Improvement'])
                corr_fs_improvement = trans_data['Height_Range'].corr(trans_data['Foot_Skate_Improvement'])
                
                # 计算每个指标的平均值和标准差
                stats = {
                    'Transition_Length': trans_len,
                    'Sample_Count': len(trans_data),
                    'Height_Range_Mean': trans_data['Height_Range'].mean(),
                    'Height_Range_Std': trans_data['Height_Range'].std(),
                    
                    # 基准模型统计数据
                    'Baseline_L2P_Mean': trans_data['Baseline_L2P'].mean(),
                    'Baseline_L2P_Std': trans_data['Baseline_L2P'].std(),
                    'Baseline_L2P_Correlation': corr_baseline_l2p,
                    'Baseline_L2Q_Mean': trans_data['Baseline_L2Q'].mean(),
                    'Baseline_L2Q_Std': trans_data['Baseline_L2Q'].std(),
                    'Baseline_L2Q_Correlation': corr_baseline_l2q,
                    'Baseline_NPSS_Mean': trans_data['Baseline_NPSS'].mean(),
                    'Baseline_NPSS_Std': trans_data['Baseline_NPSS'].std(),
                    'Baseline_NPSS_Correlation': corr_baseline_npss,
                    'Baseline_Foot_Skate_Mean': trans_data['Baseline_Foot_Skate'].mean(),
                    'Baseline_Foot_Skate_Std': trans_data['Baseline_Foot_Skate'].std(),
                    'Baseline_Foot_Skate_Correlation': corr_baseline_fs,
                    
                    # 提议模型统计数据
                    'Proposed_L2P_Mean': trans_data['Proposed_L2P'].mean(),
                    'Proposed_L2P_Std': trans_data['Proposed_L2P'].std(),
                    'Proposed_L2P_Correlation': corr_proposed_l2p,
                    'Proposed_L2Q_Mean': trans_data['Proposed_L2Q'].mean(),
                    'Proposed_L2Q_Std': trans_data['Proposed_L2Q'].std(),
                    'Proposed_L2Q_Correlation': corr_proposed_l2q,
                    'Proposed_NPSS_Mean': trans_data['Proposed_NPSS'].mean(),
                    'Proposed_NPSS_Std': trans_data['Proposed_NPSS'].std(),
                    'Proposed_NPSS_Correlation': corr_proposed_npss,
                    'Proposed_Foot_Skate_Mean': trans_data['Proposed_Foot_Skate'].mean(),
                    'Proposed_Foot_Skate_Std': trans_data['Proposed_Foot_Skate'].std(),
                    'Proposed_Foot_Skate_Correlation': corr_proposed_fs,
                    
                    # 性能提升统计数据
                    'L2P_Improvement_Mean': trans_data['L2P_Improvement'].mean(),
                    'L2P_Improvement_Std': trans_data['L2P_Improvement'].std(),
                    'L2P_Improvement_Correlation': corr_l2p_improvement,
                    'L2Q_Improvement_Mean': trans_data['L2Q_Improvement'].mean(),
                    'L2Q_Improvement_Std': trans_data['L2Q_Improvement'].std(),
                    'L2Q_Improvement_Correlation': corr_l2q_improvement,
                    'NPSS_Improvement_Mean': trans_data['NPSS_Improvement'].mean(),
                    'NPSS_Improvement_Std': trans_data['NPSS_Improvement'].std(),
                    'NPSS_Improvement_Correlation': corr_npss_improvement,
                    'Foot_Skate_Improvement_Mean': trans_data['Foot_Skate_Improvement'].mean(),
                    'Foot_Skate_Improvement_Std': trans_data['Foot_Skate_Improvement'].std(),
                    'Foot_Skate_Improvement_Correlation': corr_fs_improvement
                }
                summary_stats.append(stats)
            
            # 创建汇总统计表
            if summary_stats:
                summary_df = pd.DataFrame(summary_stats)
                summary_df.to_excel(writer, sheet_name='Summary_Stats', index=False, float_format="%.4f")
                print(f"  - Sheet 'Summary_Stats' created")
        
        print(f"Results saved to {filename}")

def calculate_height_range(motion_tensor):
    """计算运动序列的高度范围（最大高度减去最小高度）"""
    # 提取根节点位置（最后3个维度）
    B, T, D = motion_tensor.shape
    local_rot, root_pos = torch.split(motion_tensor, [D-3, 3], dim=-1)
    
    # 提取Y轴分量（索引为1）
    root_pos_y = root_pos[:, :, 1]  # Shape: [B, T]
    
    # 计算每个序列的高度范围
    height_range = torch.max(root_pos_y, dim=1)[0] - torch.min(root_pos_y, dim=1)[0]  # Shape: [B]
    
    return height_range

def visualize_results(results, output_file, config_info=None):
    """将分析结果可视化为2x2布局的散点图，针对不同过渡长度使用不同颜色"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # 获取所有不同的过渡长度
    transition_lengths = sorted(set(results.transition_lengths))
    print(f"Visualizing for transition lengths: {transition_lengths}")
    
    # 为不同的过渡长度定义不同的颜色和标记
    colors = ['blue', 'green', 'orange', 'red']
    markers = ['o', 's', '^', 'd']
    
    # 定义指标名称和子图位置
    metric_names = ['L2P', 'L2Q', 'NPSS', 'Foot Skate']
    data_lists = [results.l2p_scores, results.l2q_scores, results.npss_scores, results.foot_skate_scores]
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # 创建每个子图
    for (metric_name, data_y, pos) in zip(metric_names, data_lists, subplot_positions):
        ax = axs[pos]
        
        # 为每个过渡长度绘制数据
        for i, trans_len in enumerate(transition_lengths):
            if i >= len(colors):  # 防止过渡长度超过颜色数量
                i = i % len(colors)
                
            # 筛选该过渡长度的数据
            indices = [j for j, t in enumerate(results.transition_lengths) if t == trans_len]
            if not indices:  # 如果没有该过渡长度的数据，跳过
                continue
                
            filtered_x = [results.height_ranges[j] for j in indices]
            filtered_y = [data_y[j] for j in indices]
            
            # 绘制散点图
            ax.scatter(
                filtered_x, 
                filtered_y, 
                color=colors[i], 
                marker=markers[i % len(markers)],
                alpha=0.7, 
                s=30, 
                edgecolors='k', 
                linewidths=0.5,
                label=f'T={trans_len}'
            )
            
            # 添加趋势线
            if filtered_x and filtered_y:  # 确保有数据可以拟合
                z = np.polyfit(filtered_x, filtered_y, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(filtered_x), max(filtered_x), 100)
                ax.plot(x_range, p(x_range), color=colors[i], linestyle='--', alpha=0.7, linewidth=1.5)
                
                # 计算相关性
                corr = np.corrcoef(filtered_x, filtered_y)[0, 1]
                
                # 在趋势线附近添加相关系数
                text_x = max(filtered_x) * 0.9
                text_y = p(text_x)
                ax.text(text_x, text_y, f'r={corr:.3f}', color=colors[i], fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # 设置子图标题和标签
        ax.set_title(f'{metric_name} vs Height Range', fontsize=14, fontweight='bold')
        ax.set_xlabel('Height Range (m)', fontsize=12)
        ax.set_ylabel(f'{metric_name} Score', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()
    
    # 添加标题
    if config_info:
        plt.suptitle(f"Height Range Analysis: {config_info}", fontsize=16, y=0.98)
    else:
        plt.suptitle("Height Range vs Performance Analysis", fontsize=16, y=0.98)
    
    # 添加样本数量说明
    sample_counts = {}
    for trans_len in transition_lengths:
        count = sum(1 for t in results.transition_lengths if t == trans_len)
        sample_counts[trans_len] = count
    
    sample_text = ", ".join([f"T={t}: {n} samples" for t, n in sample_counts.items()])
    plt.figtext(0.5, 0.01, f"Analysis includes {sample_text}", 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()

def run_height_range_analysis(args):
    """主分析函数，执行高度范围分析，支持两个模型的比较"""
    print(f"Starting height range analysis for dataset: {args.dataset}")
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载配置
    config = utils.load_config(f"config/{args.dataset}/{args.config}")
    
    # 加载基准模型的配置
    baseline_kf_config = utils.load_config(f"config/{args.dataset}/{args.baseline_kf_config}")
    baseline_ref_config = utils.load_config(f"config/{args.dataset}/{args.baseline_ref_config}")
    
    # 加载提议模型的配置
    proposed_kf_config = utils.load_config(f"config/{args.dataset}/{args.proposed_kf_config}")
    proposed_ref_config = utils.load_config(f"config/{args.dataset}/{args.proposed_ref_config}")
    
    # 加载测试数据集
    test_dataset = MotionDataset(train=False, config=config)
    
    # 获取统计数据
    mean, std = test_dataset.motion_statistics(device)
    traj_mean, traj_std = test_dataset.traj_statistics(device)
    l2p_mean, l2p_std = test_dataset.l2p_statistics(device)
    skeleton = test_dataset.skeleton
    
    # 设置contact joints
    contact_idx = []
    for joint in config.contact_joints:
        contact_idx.append(skeleton.idx_by_name[joint])
    
    # 加载基准模型
    baseline_kf_model = ContextTransformer(baseline_kf_config, test_dataset).to(device)
    baseline_ref_model = DetailTransformer(baseline_ref_config, test_dataset).to(device)
    
    utils.load_model(baseline_kf_model, baseline_kf_config)
    utils.load_model(baseline_ref_model, baseline_ref_config)
    
    baseline_kf_model.eval()
    baseline_ref_model.eval()
    
    print(f"Baseline models loaded: KeyframeNet={args.baseline_kf_config}, RefineNet={args.baseline_ref_config}")
    
    # 加载提议模型
    proposed_kf_model = ContextTransformer(proposed_kf_config, test_dataset).to(device)
    proposed_ref_model = DetailTransformer(proposed_ref_config, test_dataset).to(device)
    
    utils.load_model(proposed_kf_model, proposed_kf_config)
    utils.load_model(proposed_ref_model, proposed_ref_config)
    
    proposed_kf_model.eval()
    proposed_ref_model.eval()
    
    print(f"Proposed models loaded: KeyframeNet={args.proposed_kf_config}, RefineNet={args.proposed_ref_config}")
    
    # 初始化结果存储
    results = AnalysisResults()
    
    # 设置转换长度
    if args.transition_lengths:
        transitions = args.transition_lengths
    elif args.transition_length:
        transitions = [args.transition_length]
    else:
        transitions = [15, 30, 60, 90]  # 默认使用所有四种过渡长度
    
    print(f"Using transition lengths: {transitions}")
    
    # 设置keyframe采样方法
    if args.kf_param is not None:
        kf_sampling = [args.kf_sampling, args.kf_param]
    else:
        kf_sampling = [args.kf_sampling]
    print(f"Using keyframe sampling method: {kf_sampling}")
    
    # 遍历数据集
    with torch.no_grad():
        # 为了追踪序列ID，我们需要一个全局计数器
        sequence_id = 0
        
        for transition in transitions:
            print(f"\nProcessing transition length: {transition}")
            
            # 为当前过渡长度创建数据加载器
            test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
            
            # 重置序列ID计数器，确保每个过渡长度都从0开始
            sequence_id = 0
            
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc=f"Processing batches for T={transition}")):
                # 如果设置了最大批次数，超过则跳出循环
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    print(f"Reached maximum number of batches ({args.max_batches}), stopping.")
                    break
                
                # 获取数据
                GT_motion = batch["motion"].to(device)
                GT_phase = batch["phase"].to(device) if "phase" in batch else None
                GT_traj = batch["traj"].to(device) if "traj" in batch else None
                GT_score = batch["score"].to(device) if "score" in batch else None
                
                # 限制序列长度
                num_frames = config.context_frames + transition + 1
                GT_motion = GT_motion[:, :num_frames]
                if GT_phase is not None:
                    GT_phase = GT_phase[:, :num_frames]
                if GT_traj is not None:
                    GT_traj = GT_traj[:, :num_frames]
                if GT_score is not None:
                    GT_score = GT_score[:, :num_frames]
                
                # 计算GT的高度范围
                height_range = calculate_height_range(GT_motion)
                
                # 计算GT的接触数据
                GT_contact = eval_backup._get_foot_velocity(GT_motion, skeleton, contact_idx)
                GT_contact = (GT_contact < config.contact_threshold).float()
                
                # 生成基准模型的预测结果
                baseline_out = eval_backup._ours_transition(
                    baseline_ref_config, baseline_kf_model, baseline_ref_model, 
                    GT_motion, mean, std, GT_contact, GT_phase, GT_traj, GT_score, 
                    traj_mean, traj_std, kf_sampling
                )
                baseline_pred_motion = baseline_out["motion"]
                baseline_pred_contact = baseline_out["contact"]
                
                # 生成提议模型的预测结果
                proposed_out = eval_backup._ours_transition(
                    proposed_ref_config, proposed_kf_model, proposed_ref_model, 
                    GT_motion, mean, std, GT_contact, GT_phase, GT_traj, GT_score, 
                    traj_mean, traj_std, kf_sampling
                )
                proposed_pred_motion = proposed_out["motion"]
                proposed_pred_contact = proposed_out["contact"]
                
                # 为每个序列计算单独的评估指标，而不是批次平均值
                for i in range(GT_motion.shape[0]):
                    # 提取单个序列
                    gt_motion_single = GT_motion[i:i+1]
                    
                    # 基准模型的预测和评估
                    baseline_pred_motion_single = baseline_pred_motion[i:i+1]
                    baseline_pred_contact_single = baseline_pred_contact[i:i+1]
                    
                    baseline_l2p = eval_backup.l2p(gt_motion_single, baseline_pred_motion_single, skeleton, l2p_mean, l2p_std, config.context_frames)
                    baseline_l2q = eval_backup.l2q(gt_motion_single, baseline_pred_motion_single, config.context_frames)
                    baseline_npss = eval_backup.npss(gt_motion_single, baseline_pred_motion_single, config.context_frames)
                    baseline_foot_skate = eval_backup.foot_skate(baseline_pred_motion_single, baseline_pred_contact_single, skeleton, contact_idx, ctx_frames=config.context_frames)
                    
                    # 提议模型的预测和评估
                    proposed_pred_motion_single = proposed_pred_motion[i:i+1]
                    proposed_pred_contact_single = proposed_pred_contact[i:i+1]
                    
                    proposed_l2p = eval_backup.l2p(gt_motion_single, proposed_pred_motion_single, skeleton, l2p_mean, l2p_std, config.context_frames)
                    proposed_l2q = eval_backup.l2q(gt_motion_single, proposed_pred_motion_single, config.context_frames)
                    proposed_npss = eval_backup.npss(gt_motion_single, proposed_pred_motion_single, config.context_frames)
                    proposed_foot_skate = eval_backup.foot_skate(proposed_pred_motion_single, proposed_pred_contact_single, skeleton, contact_idx, ctx_frames=config.context_frames)
                    
                    # 存储两个模型的结果，包括过渡长度信息
                    results.add_result(
                        transition,
                        batch_idx,
                        sequence_id,
                        height_range[i].item(),
                        baseline_l2p,
                        baseline_l2q,
                        baseline_npss,
                        baseline_foot_skate * 10,  # 放大foot_skate值以便于可视化
                        proposed_l2p,
                        proposed_l2q,
                        proposed_npss,
                        proposed_foot_skate * 10  # 放大foot_skate值以便于可视化
                    )
                    
                    # 递增序列ID
                    sequence_id += 1
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建结果文件名，包含基准模型和提议模型的信息
    baseline_suffix = f"{args.baseline_kf_config.replace('.yaml', '')}-{args.baseline_ref_config.replace('.yaml', '')}"
    proposed_suffix = f"{args.proposed_kf_config.replace('.yaml', '')}-{args.proposed_ref_config.replace('.yaml', '')}"
    result_prefix = f"{args.dataset}_{baseline_suffix}_vs_{proposed_suffix}"
    
    # 保存Excel文件
    excel_file = os.path.join(args.output_dir, f"complexity_vs_performance_{result_prefix}.xlsx")
    results.save_to_file(excel_file)
    
    print("\nHeight range vs performance analysis completed!")
    print(f"Results saved to {excel_file}")
    
    # 如果不需要继续生成可视化，可以使用以下命令：
    print(f"\n要生成基本可视化图表，请运行以下命令：")
    print(f"python eval/visualize_complexity_results.py --excel_file {excel_file} --output_dir {args.output_dir}/figures " + 
          f"--config_info \"{args.dataset}, Baseline: {baseline_suffix}, Proposed: {proposed_suffix}\" --compare_models")
    
    print(f"\n要生成包含性能提升分析的高级可视化图表，请运行以下命令：")
    print(f"python eval/visualize_complexity_results.py --excel_file {excel_file} --output_dir {args.output_dir}/figures " +
          f"--config_info \"{args.dataset}, Baseline: {baseline_suffix}, Proposed: {proposed_suffix}\" --compare_models " + 
          f"--improvement_charts --num_bins 5")
    
    # 也可以直接调用可视化脚本
    if args.generate_viz:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import eval.visualize_complexity_results as viz
            
            print("\nGenerating visualizations...")
            config_info = f"{args.dataset}, Baseline: {baseline_suffix}, Proposed: {proposed_suffix}"
            
            # 创建专门的图表目录
            figures_dir = os.path.join(args.output_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            
            # 生成所有类型的可视化图表，包括模型比较和改进分析
            viz.visualize_results_from_excel(excel_file, figures_dir, config_info, compare_models=True, create_improvement_charts=True)
            print("Visualizations completed!")
        except ImportError as e:
            print(f"Visualization failed due to missing dependencies: {e}")
            print("Please install required packages and run the visualization script separately.")
            print("Required packages: numpy, matplotlib, pandas, openpyxl")
    else:
        print("\nSkipping visualization generation as requested.")
        print("You can generate visualizations later using the commands above.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Complexity vs Performance Analysis for Motion In-betweening Models')
    parser.add_argument('--dataset', type=str, default='lafan1', help='Dataset name')
    parser.add_argument('--config', type=str, default='default.yaml', help='Configuration file')
    
    # 基准模型配置
    parser.add_argument('--baseline_kf_config', type=str, default='keyframe.yaml', 
                        help='Baseline keyframe model configuration file')
    parser.add_argument('--baseline_ref_config', type=str, default='refine.yaml', 
                        help='Baseline refine model configuration file')
    
    # 提议模型配置（默认使用支持轨迹建模的版本）
    parser.add_argument('--proposed_kf_config', type=str, default='keyframe-traj-enc.yaml', 
                        help='Proposed keyframe model configuration file')
    parser.add_argument('--proposed_ref_config', type=str, default='refine-traj-enc.yaml', 
                        help='Proposed refine model configuration file')
    
    # 过渡长度参数
    transition_group = parser.add_mutually_exclusive_group()
    transition_group.add_argument('--transition_length', type=int, default=None, 
                                 help='Single transition length to evaluate')
    transition_group.add_argument('--transition_lengths', type=int, nargs='+', default=None, 
                                 help='Multiple transition lengths to evaluate, e.g. --transition_lengths 15 30 60 90')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='eval/complexity_analysis', 
                        help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU device ID')
    parser.add_argument('--max_batches', type=int, default=None, 
                        help='Maximum number of batches to process (default: all)')
    parser.add_argument('--kf_sampling', type=str, default="score", 
                       help='Keyframe sampling method: score, threshold, topk, random, uniform')
    parser.add_argument('--kf_param', type=float, default=None, 
                        help='Parameter for keyframe sampling method')
    parser.add_argument('--no_viz', action='store_true', 
                        help='Skip generating visualizations after analysis')
    
    args = parser.parse_args()
    
    # 默认生成可视化，除非明确使用--no_viz参数
    args.generate_viz = not args.no_viz
    
    print("=" * 80)
    print("Height Range vs Performance Analysis for Motion In-betweening")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Baseline models: KeyframeNet={args.baseline_kf_config}, RefineNet={args.baseline_ref_config}")
    print(f"Proposed models: KeyframeNet={args.proposed_kf_config}, RefineNet={args.proposed_ref_config}")
    
    if args.transition_lengths:
        print(f"Transition lengths: {args.transition_lengths}")
    elif args.transition_length:
        print(f"Transition length: {args.transition_length}")
    else:
        print("Transition lengths: [15, 30, 60, 90] (default)")
        
    print(f"Output directory: {args.output_dir}")
    print(f"Generate visualizations: {args.generate_viz} (use --no_viz to skip)")
    print("-" * 80)
    
    run_height_range_analysis(args)