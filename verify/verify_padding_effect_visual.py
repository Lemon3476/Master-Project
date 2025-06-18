#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

import argparse
import torch
import numpy as np
from aPyOpenGL import agl
from vis.motionapp import MotionApp

from utils.dataset import MotionDataset
from utils import utils
from model.twostage import DetailTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="验证填充与掩码对RefineNet推理结果的影响")
    parser.add_argument("--dataset", type=str, default="lafan1", help="数据集名称")
    parser.add_argument("--config", type=str, default="refine.yaml", help="RefineNet的配置文件")
    parser.add_argument("--seq_length", type=int, default=61, help="短序列长度")
    parser.add_argument("--pad_length", type=int, default=101, help="填充后序列长度")
    parser.add_argument("--batch_idx", type=int, default=0, help="选择批次索引")
    parser.add_argument("--seq_idx", type=int, default=0, help="选择序列索引")
    parser.add_argument("--atol", type=float, default=1e-6, help="比较时的绝对容差")
    return parser.parse_args()

def main():
    args = parse_args()

    # 步骤1: 加载配置、数据集和模型
    print("=== 初始化 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = utils.load_config(f"config/{args.dataset}/{args.config}")
    dataset = MotionDataset(train=False, config=config)
    
    # 加载预训练的RefineNet模型
    model = DetailTransformer(config, dataset).to(device)
    utils.load_model(model, config)
    model.eval()
    
    # 步骤2: 准备数据 - 获取一个完整样本并截取序列
    print("\n=== 准备数据 ===")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, batch in enumerate(dataloader):
        if i == args.batch_idx:
            # 提取单个序列
            motion = batch["motion"].to(device)  # [1, B, T, D]
            if "phase" in batch:
                phase = batch["phase"].to(device)
            else:
                phase = None
            if "traj" in batch:
                traj = batch["traj"].to(device)
            else:
                traj = None
            break
    
    # 打印调试信息
    print(f"加载的数据形状:")
    print(f"  - motion: {motion.shape}")
    if phase is not None:
        print(f"  - phase: {phase.shape}")
    if traj is not None:
        print(f"  - traj: {traj.shape}")
        
    # 从批次中选择一个序列并截取指定长度
    # 基于打印的形状，数据形状为 [1, T, D]，不需要额外选择序列
    input_short = motion[:, :args.seq_length].clone()
    if phase is not None:
        phase_short = phase[:, :args.seq_length].clone()
    else:
        phase_short = None
    if traj is not None:
        traj_short = traj[:, :args.seq_length].clone()
    else:
        traj_short = None
    
    print(f"短序列形状: {input_short.shape}")
    
    # 创建填充序列 (用零填充至args.pad_length长度)
    motion_dim = motion.shape[2]  # 获取特征维度
    input_long = torch.zeros((1, args.pad_length, motion_dim), device=device)
    input_long[:, :args.seq_length] = input_short
    
    if phase is not None:
        phase_dim = phase.shape[2]
        phase_long = torch.zeros((1, args.pad_length, phase_dim), device=device)
        phase_long[:, :args.seq_length] = phase_short
    else:
        phase_long = None
        
    if traj is not None:
        traj_dim = traj.shape[2]
        traj_long = torch.zeros((1, args.pad_length, traj_dim), device=device)
        traj_long[:, :args.seq_length] = traj_short
    else:
        traj_long = None
    
    # 创建标准注意力掩码 (掩蔽填充部分)
    # 注意: 填充部分为True表示这些位置应被掩蔽
    attention_mask = torch.zeros((1, 1, args.pad_length, args.pad_length), dtype=torch.bool, device=device)
    attention_mask[:, :, :, args.seq_length:] = True  # 掩蔽填充的列
    attention_mask[:, :, args.seq_length:, :] = True  # 掩蔽填充的行
    print("注意力掩码: 填充部分完全屏蔽")
    
    # 打印掩码模式进行调试
    if attention_mask is not None:
        print("注意力掩码示例 (T表示被掩蔽):")
        sample_mask = attention_mask[0, 0]
        for i in range(min(5, args.seq_length)):  # 真实序列前5行
            row = ''.join(['T' if m else 'F' for m in sample_mask[i, :10]])  # 前10列
            print(f"真实行 {i}: {row}...")
        for i in range(args.seq_length, min(args.seq_length+5, args.pad_length)):  # 填充序列前5行
            row = ''.join(['T' if m else 'F' for m in sample_mask[i, :10]])  # 前10列
            print(f"填充行 {i}: {row}...")
    
    print(f"短序列形状: {input_short.shape}")
    print(f"填充序列形状: {input_long.shape}")
    print(f"注意力掩码形状: {attention_mask.shape}")
    
    # 步骤3: 执行推理
    print("\n=== 执行推理 ===")
    
    # 设置原始目标帧索引 (最后一帧)
    original_target_idx = args.seq_length - 1
    print(f"将原始目标帧 (索引 {original_target_idx}) 作为已知约束")
    
    with torch.no_grad():
        # 场景A: 直接推理
        print("场景A: 对短序列直接推理...")
        output_short = model(input_short, midway_targets=[], phase=phase_short, traj=traj_short, attention_mask=None)
        
        # 场景B: 填充后推理 - 关键是将原始目标帧作为已知约束
        print("场景B: 对填充序列使用掩码推理，并将原始目标帧作为已知约束...")
        output_long = model(input_long, 
                           midway_targets=[original_target_idx],  # 关键修正：将原始目标帧作为已知约束
                           phase=phase_long, 
                           traj=traj_long, 
                           attention_mask=attention_mask)
        
        # 提取有效部分进行比较
        output_long_truncated = output_long["motion"][:, :args.seq_length]
        
    # 步骤4: 详细调试和数值验证
    print("\n=== 详细调试 ===")
    
    # 检查NaN值
    has_nan_short = torch.isnan(output_short["motion"]).any()
    has_nan_long = torch.isnan(output_long_truncated).any()
    print(f"短序列输出中有NaN? {has_nan_short}")
    print(f"填充序列输出中有NaN? {has_nan_long}")
    
    # 如果有NaN，打印出更详细的信息
    if has_nan_short or has_nan_long:
        print("\n错误: 输出中包含NaN值，实验无法继续。以下是详细诊断信息:")
        
        if has_nan_short:
            nan_count_short = torch.isnan(output_short["motion"]).sum().item()
            nan_percentage_short = nan_count_short / output_short["motion"].numel() * 100
            print(f"  - 短序列输出中有 {nan_count_short} 个NaN值 ({nan_percentage_short:.2f}%)")
            
            # 检查第一个NaN的位置
            nan_indices = torch.where(torch.isnan(output_short["motion"]))
            if len(nan_indices[0]) > 0:
                print(f"  - 第一个NaN位置: batch={nan_indices[0][0]}, frame={nan_indices[1][0]}, feature={nan_indices[2][0]}")
        
        if has_nan_long:
            nan_count_long = torch.isnan(output_long_truncated).sum().item()
            nan_percentage_long = nan_count_long / output_long_truncated.numel() * 100
            print(f"  - 填充序列输出中有 {nan_count_long} 个NaN值 ({nan_percentage_long:.2f}%)")
            
            # 检查第一个NaN的位置
            nan_indices = torch.where(torch.isnan(output_long_truncated))
            if len(nan_indices[0]) > 0:
                print(f"  - 第一个NaN位置: batch={nan_indices[0][0]}, frame={nan_indices[1][0]}, feature={nan_indices[2][0]}")
            
            # 检查完整输出中是否只有填充部分有NaN
            nan_in_padded_only = torch.isnan(output_long["motion"][:, args.seq_length:]).any() and not torch.isnan(output_long["motion"][:, :args.seq_length]).any()
            print(f"  - NaN值仅在填充部分? {nan_in_padded_only}")
        
        print("\n原因分析:")
        print("  1. 注意力掩码可能导致梯度消失或爆炸")
        print("  2. 模型权重对填充序列输入可能过于敏感")
        print("  3. 在没有掩码的情况下训练的模型对掩码处理不当")
        
        print("\n可能的解决方案:")
        print("  1. 修改掩码应用方式，例如只在特定层应用")
        print("  2. 使用更小的填充长度")
        print("  3. 在训练时就引入掩码机制")
        
        print("\n实验结论:")
        print("  当前的RefineNet模型在无填充条件下训练，无法通过简单添加注意力掩码来处理填充序列。")
        print("  需要在训练阶段就引入填充和掩码机制，以使模型适应这种处理方式。")
        
        # 终止程序，因为结果不可靠
        return
    
    # 如果没有NaN，继续进行数值验证
    print("\n=== 数值验证结果 ===")
    are_identical = torch.allclose(output_short["motion"], output_long_truncated, atol=args.atol)
    max_difference = torch.max(torch.abs(output_short["motion"] - output_long_truncated)).item()
    
    # 计算平均差异
    mean_difference = torch.mean(torch.abs(output_short["motion"] - output_long_truncated)).item()
    
    print(f"输出在数值上是否几乎完全相同? -> {are_identical}")
    print(f"最大绝对差值: {max_difference}")
    print(f"平均绝对差值: {mean_difference}")
    
    if not are_identical:
        print(f"\n不完全相同的原因分析:")
        print(f"  - 可能是浮点舍入误差导致的微小差异")
        print(f"  - 可能是掩码计算路径不同导致的累积误差")
        print(f"  - 如果差异很大，说明模型对掩码敏感")
    
    # 只要没有NaN值就继续可视化，即使结果不完全相同
    if not (has_nan_short or has_nan_long):
        # 步骤5: 启动三段式可视化对比
        print("\n=== 启动三段式视觉验证 ===")
        print("Stage 1: 播放填充序列处理结果 (标签显示为'Intra-Batch')")
        print("Stage 2: 播放直接处理结果 (标签显示为'Multi-IB')")
        print("Stage 3: 同时播放两者，观察是否完美重叠")
        
        if are_identical:
            print("注意：两个输出在数值上几乎完全相同，应该会完美重叠。")
        else:
            print(f"注意：两个输出在数值上有差异（最大差值: {max_difference}），观察是否有明显的视觉差异。")
        
        # 打印最终形状
        print(f"可视化数据形状:")
        print(f"  - output_short[\"motion\"]: {output_short['motion'].shape}")
        print(f"  - output_long_truncated: {output_long_truncated.shape}")
        
        # 准备MotionApp所需的数据 - 使用特定标签触发三段式播放
        tags_for_visualize = ["Intra-Batch", "Multi-IB"]  # 这些标签会触发三段式播放逻辑
        
        # 启动可视化App
        # 调整数据格式以适应MotionApp的需求
        # 注意：MotionApp需要正确的shape和帧数才能触发三段式播放
        # 确保帧数足够但不要太多，以便能够完成一个完整的播放周期
        
        # 确保每个动作只有一个完整序列，不重复播放
        # MotionApp会将数据解释为 [batch_size, frames_per_batch] 的形状
        # 将batch_size设为1，frames_per_batch设为实际帧数
        
        # 检查形状并调整
        output_frames = output_long_truncated.shape[1]
        print(f"原始帧数: {output_frames}")
        
        # 我们不需要扩展帧数，保持原始长度
        output_long_exp = output_long_truncated
        output_short_exp = output_short["motion"]
        
        # 在启动前添加一些提示
        print("\n提示: 按照最终实验方案进行三段式播放")
        print("第一阶段: 仅播放填充序列处理的结果（一个角色）")
        print("第二阶段: 仅播放直接处理的结果（一个角色）")
        print("第三阶段: 同时播放两个结果（两个角色应完美重叠）")
        print("按空格键可暂停观察，按'M'键可将角色分开观察")
        
        app = MotionApp(
            motions=[output_long_exp, output_short_exp],
            tags=["Intra-Batch", "Multi-IB"],  # 这些标签对应三段式播放的阶段
            skeleton=dataset.skeleton,
            dataset="lafan1",
            paused=False,          # 自动开始播放
            compare_mode=True,     # 激活三段式播放逻辑
            show_keyframes=False,  # 在此实验中不关心关键帧
            record_sequence_idx=0  # 指定序列索引，确保三段式播放正常工作
        )
        
        # 手动设置帧数，确保只播放一次
        app.frame_per_batch = output_frames
        agl.AppManager.start(app)
    else:
        print("\n可视化阶段已跳过，因为输出中包含NaN值。")

if __name__ == "__main__":
    main()