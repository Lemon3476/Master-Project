#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最终版修正方案：验证 RefineNet 在高质量、完整输入下的填充推理一致性
Final Corrected Plan: Verifying Inference Consistency of RefineNet on High-Quality, Complete Inputs with Padding
"""

import sys
sys.path.append(".")

import os
import argparse
import torch
import numpy as np
import random
from aPyOpenGL import agl
from vis.motionapp import MotionApp

from verify.custom_ops import get_keyframes_evenly_distributed
from utils.dataset import MotionDataset
from utils import utils, ops
from model.twostage import DetailTransformer, ContextTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="验证 RefineNet 在高质量、完整输入下的填充推理一致性")
    parser.add_argument("--dataset", type=str, default="lafan1", help="数据集名称")
    parser.add_argument("--refine_config", type=str, default="refine.yaml", help="RefineNet的配置文件")
    parser.add_argument("--keyframe_config", type=str, default="keyframe.yaml", help="KeyframeNet的配置文件")
    parser.add_argument("--seq_length", type=int, default=61, help="短序列长度")
    parser.add_argument("--pad_length", type=int, default=101, help="填充后序列长度")
    parser.add_argument("--batch_idx", type=int, default=0, help="选择批次索引")
    parser.add_argument("--seq_idx", type=int, default=0, help="选择序列索引")
    parser.add_argument("--target_idx", type=int, default=None, help="目标帧索引，默认为seq_length-1")
    parser.add_argument("--context_frames", type=int, default=None, help="上下文帧数量，默认从配置文件读取")
    parser.add_argument("--topk", type=int, default=3, help="从 KeyframeNet 输出中选取的关键帧数量")
    parser.add_argument("--atol", type=float, default=1e-6, help="比较时的绝对容差")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备选择: 'cuda' 或 'cpu'")
    parser.add_argument("--no_vis", action="store_true", help="禁用可视化，仅执行数值比较")
    parser.add_argument("--save_output", action="store_true", help="保存模型输出到文件")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出文件保存目录")
    parser.add_argument("--verbose", action="store_true", help="显示详细调试信息")
    return parser.parse_args()

def main():
    args = parse_args()

    # 设置随机种子以确保结果可重现
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 步骤1: 加载配置、数据集和模型
    print("=== 初始化 ===")
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"使用设备: {device}")
    
    # 加载 RefineNet 和 KeyframeNet 配置
    refine_config = utils.load_config(f"config/{args.dataset}/{args.refine_config}")
    keyframe_config = utils.load_config(f"config/{args.dataset}/{args.keyframe_config}")
    dataset = MotionDataset(train=False, config=refine_config)
    
    # 加载预训练的 RefineNet 模型
    refine_model = DetailTransformer(refine_config, dataset).to(device)
    utils.load_model(refine_model, refine_config)
    refine_model.eval()
    print(f"RefineNet 模型加载成功")
    
    # 加载预训练的 KeyframeNet 模型
    keyframe_model = ContextTransformer(keyframe_config, dataset).to(device)
    utils.load_model(keyframe_model, keyframe_config)
    keyframe_model.eval()
    print(f"KeyframeNet 模型加载成功")
    
    # 如果需要输出目录，创建目录
    if args.save_output:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 步骤2: 准备数据 - 从数据集中获取样本
    print("\n=== 准备数据 ===")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, batch in enumerate(dataloader):
        if i == args.batch_idx:
            # 提取单个序列
            motion = batch["motion"].to(device)
            if "phase" in batch:
                phase = batch["phase"].to(device)
            else:
                phase = None
            if "traj" in batch:
                traj = batch["traj"].to(device)
            else:
                traj = None
            break
    
    # 打印原始数据形状
    print(f"加载的原始数据:")
    print(f"  - motion: {motion.shape}")
    if phase is not None:
        print(f"  - phase: {phase.shape}")
    if traj is not None:
        print(f"  - traj: {traj.shape}")
    
    # 加载归一化统计量
    print("\n=== 加载归一化统计量 ===")
    motion_mean, motion_std = dataset.motion_statistics(device)
    if traj is not None:
        traj_mean, traj_std = dataset.traj_statistics(device)
    else:
        traj_mean, traj_std = None, None
    
    print(f"Motion 统计量已加载 - mean: {motion_mean.shape}, std: {motion_std.shape}")
    if traj_mean is not None:
        print(f"Trajectory 统计量已加载 - mean: {traj_mean.shape}, std: {traj_std.shape}")
    
    # 定义关键帧索引
    context_frames = args.context_frames if args.context_frames is not None else refine_config.context_frames  # 默认从配置读取，通常是10
    target_frame_idx = args.target_idx if args.target_idx is not None else (args.seq_length - 1)  # 默认为最后一帧
    
    # 初始约束关键帧 (仅包含起始和结束帧)
    init_keyframes = [context_frames - 1, target_frame_idx]
    
    print(f"\n=== 关键帧信息 ====")
    print(f"初始约束关键帧: {init_keyframes}")
    print(f"  - 上下文帧结束索引: {context_frames - 1}")
    print(f"  - 目标帧索引: {target_frame_idx}")
    
    # 步骤3: 准备仅包含约束的序列
    print("\n=== 准备约束序列 ===")
    
    # 构造仅包含约束的短序列 (将非约束帧设为零)
    constrained_short = torch.zeros((1, args.seq_length, motion.shape[2]), device=device)
    constrained_short[:, :context_frames] = motion[:, :context_frames]  # 复制上下文帧
    constrained_short[:, target_frame_idx:target_frame_idx+1] = motion[:, target_frame_idx:target_frame_idx+1]  # 复制目标帧
    
    # 构造仅包含约束的长序列(带填充)
    constrained_long = torch.zeros((1, args.pad_length, motion.shape[2]), device=device)
    constrained_long[:, :context_frames] = motion[:, :context_frames]  # 复制上下文帧
    constrained_long[:, target_frame_idx:target_frame_idx+1] = motion[:, target_frame_idx:target_frame_idx+1]  # 复制目标帧
    
    print(f"创建了仅包含约束的短序列(形状: {constrained_short.shape})和长序列(形状: {constrained_long.shape})")
    print(f"所有非约束帧被设为零，约束帧保持原始值")
    
    # 处理phase和traj的约束
    if phase is not None:
        phase_short = torch.zeros((1, args.seq_length, phase.shape[2]), device=device)
        phase_short[:, :context_frames] = phase[:, :context_frames]
        phase_short[:, target_frame_idx:target_frame_idx+1] = phase[:, target_frame_idx:target_frame_idx+1]
        
        phase_long = torch.zeros((1, args.pad_length, phase.shape[2]), device=device)
        phase_long[:, :context_frames] = phase[:, :context_frames]
        phase_long[:, target_frame_idx:target_frame_idx+1] = phase[:, target_frame_idx:target_frame_idx+1]
        
        print(f"  创建了仅包含约束的phase短序列(形状: {phase_short.shape})和长序列(形状: {phase_long.shape})")
    else:
        phase_short = None
        phase_long = None
        print("  未找到phase数据")
        
    if traj is not None:
        traj_short = torch.zeros((1, args.seq_length, traj.shape[2]), device=device)
        traj_short[:, :context_frames] = traj[:, :context_frames]
        traj_short[:, target_frame_idx:target_frame_idx+1] = traj[:, target_frame_idx:target_frame_idx+1]
        
        traj_long = torch.zeros((1, args.pad_length, traj.shape[2]), device=device)
        traj_long[:, :context_frames] = traj[:, :context_frames]
        traj_long[:, target_frame_idx:target_frame_idx+1] = traj[:, target_frame_idx:target_frame_idx+1]
        
        print(f"  创建了仅包含约束的traj短序列(形状: {traj_short.shape})和长序列(形状: {traj_long.shape})")
    else:
        traj_short = None
        traj_long = None
        print("  未找到traj数据")
    
    # 显示 KeyframeNet 生成的关键帧位置
    print("\n=== KeyframeNet 关键帧位置信息 ====")
    print(f"短序列长度: {args.seq_length}, 填充序列长度: {args.pad_length}")
    # 使用 KeyframeNet 预测关键帧
    with torch.no_grad():
        keyframe_output, _ = keyframe_model(
            (constrained_short - motion_mean) / motion_std, 
            phase=phase_short, 
            traj=traj_short if traj_short is not None else None,
            train=False
        )
    
    # 提取预测的分数
    if "score" in keyframe_output:
        pred_score = keyframe_output["score"]
        print(f"KeyframeNet 成功预测了关键帧得分, 形状: {pred_score.shape}")
        
        # 使用自定义的均匀分布关键帧方法
        print(f"使用均匀分布关键帧方法选择关键帧...")
        keyframes_list = get_keyframes_evenly_distributed(keyframe_config, pred_score, context_frames, target_frame_idx, args.topk + 2)
        keyframes = keyframes_list[0]  # 取第一个批次的关键帧
        
        # 确保关键帧包含了初始约束
        for kf in init_keyframes:
            if kf not in keyframes:
                keyframes.append(kf)
        
        # 排序关键帧
        keyframes = sorted(keyframes)
        print(f"KeyframeNet 生成的关键帧: {keyframes}")
    else:
        # 如果没有预测分数，则使用初始约束关键帧
        keyframes = init_keyframes
        print(f"KeyframeNet 未预测分数，使用初始约束关键帧: {keyframes}")
    print(f"  - 第一个关键帧(上下文结束): {keyframes[0]} / {args.seq_length-1} (序列百分比: {keyframes[0]/(args.seq_length-1)*100:.1f}%)")
    print(f"  - 第二个关键帧(目标帧): {keyframes[1]} / {args.seq_length-1} (序列百分比: {keyframes[1]/(args.seq_length-1)*100:.1f}%)")
    
    # 关键帧可视化
    seq_vis = ['|'] * args.seq_length
    for i, kf in enumerate(keyframes):
        seq_vis[kf] = f'{i}'  # 使用关键帧索引标记
    
    print("\n序列可视化 (0,1=关键帧索引, |=普通帧):")
    print(f"{''.join(seq_vis[:60])}")
    if args.seq_length > 60:
        print(f"{''.join(seq_vis[60:120])}")
    if args.seq_length > 120:
        print(f"{''.join(seq_vis[120:])}")
        
    # 显示关键帧之间的间隔
    gaps = []
    for i in range(1, len(keyframes)):
        gaps.append(keyframes[i] - keyframes[i-1])
    
    print("\n关键帧间隔:")
    for i, gap in enumerate(gaps):
        print(f"  - 关键帧 {i} 到 {i+1} 间隔: {gap} 帧 (百分比: {gap/(args.seq_length-1)*100:.1f}%)")
    
    # 如果是填充序列，显示有效部分和填充部分
    if args.pad_length > args.seq_length:
        pad_vis = ['-'] * args.pad_length
        # 标记有效部分
        for i in range(args.seq_length):
            pad_vis[i] = '|'  # |表示有效帧
        # 标记关键帧
        for i, kf in enumerate(keyframes):
            if kf < args.pad_length:
                pad_vis[kf] = f'{i}'
        
        print("\n填充序列可视化 (|=有效帧, -=填充帧, 0,1,..=关键帧):")
        print(f"{''.join(pad_vis[:60])}")
        if args.pad_length > 60:
            print(f"{''.join(pad_vis[60:120])}")
        if args.pad_length > 120:
            print(f"{''.join(pad_vis[120:])}")
            
        print(f"\n有效序列长度: {args.seq_length}, 填充序列长度: {args.pad_length}")
        print(f"填充比例: {(args.pad_length - args.seq_length) / args.pad_length * 100:.1f}%")
    
    # 验证参数是否有效
    if context_frames >= args.seq_length or target_frame_idx >= args.seq_length:
        raise ValueError(f"上下文帧数({context_frames})或目标帧索引({target_frame_idx})大于等于序列长度({args.seq_length})")
    if context_frames >= target_frame_idx:
        raise ValueError(f"上下文帧数({context_frames})大于等于目标帧索引({target_frame_idx})")
    if args.pad_length <= args.seq_length:
        raise ValueError(f"填充长度({args.pad_length})应大于原序列长度({args.seq_length})")
    
    print(f"  - 短序列长度: {args.seq_length}")
    print(f"  - 填充序列长度: {args.pad_length}")
    
    # 步骤4: 使用 KeyframeNet 生成高质量关键帧
    print("\n=== 使用 KeyframeNet 预测关键帧 ===")
    print("将已网络归一化的原始序列输入到 KeyframeNet 中预测关键帧...")
    
    # 归一化原始序列用于 KeyframeNet 预测
    motion_norm = (motion - motion_mean) / motion_std
    
    # 使用 KeyframeNet 预测分数和关键帧
    with torch.no_grad():
        keyframe_output, _ = keyframe_model(
            motion_norm, 
            phase=phase, 
            traj=traj if traj is not None else None,
            train=False
        )
    
    # 提取预测的分数
    if "score" in keyframe_output:
        pred_score = keyframe_output["score"]
        print(f"KeyframeNet 成功预测了关键帧得分, 形状: {pred_score.shape}")
        
        # 使用自定义的均匀分布关键帧方法
        print(f"使用均匀分布关键帧方法选择关键帧...")
        keyframes_list = get_keyframes_evenly_distributed(keyframe_config, pred_score, context_frames, target_frame_idx, args.topk + 2)
        keyframes = keyframes_list[0]  # 取第一个批次的关键帧
        
        # 确保关键帧包含了初始约束
        for kf in init_keyframes:
            if kf not in keyframes:
                keyframes.append(kf)
        
        # 排序关键帧
        keyframes = sorted(keyframes)
        print(f"KeyframeNet 生成的丰富关键帧: {keyframes}")
    else:
        # 如果没有预测分数，则使用初始约束关键帧
        keyframes = init_keyframes
        print(f"KeyframeNet 未预测分数，使用初始约束关键帧: {keyframes}")
    
    # 步骤5: 创建包含所有关键帧的稀疏约束序列
    print("\n=== 准备包含所有关键帧的稀疏约束序列 ===")
    
    # 构造短序列
    constrained_short = torch.zeros((1, args.seq_length, motion.shape[2]), device=device)
    # 关键：遍历所有找到的关键帧（包括上下文和中间点），用真实的姿态数据填充
    for kf in keyframes:
        if kf < args.seq_length:
            # 从原始的、未修改的 motion 张量中复制真实姿态
            constrained_short[:, kf:kf+1] = motion[:, kf:kf+1]
    
    # 构造长序列（执行相同逻辑）
    constrained_long = torch.zeros((1, args.pad_length, motion.shape[2]), device=device)
    for kf in keyframes:
        if kf < args.pad_length:
            constrained_long[:, kf:kf+1] = motion[:, kf:kf+1]
    
    # 为了确保上下文的完整性，再次复制所有上下文帧
    constrained_short[:, :context_frames] = motion[:, :context_frames]
    constrained_long[:, :context_frames] = motion[:, :context_frames]
    
    print(f"已创建在所有 {len(keyframes)} 个关键帧位置上都包含真实数据的稀疏序列")
    print(f"  - 短序列形状: {constrained_short.shape}")
    print(f"  - 长序列形状: {constrained_long.shape}")
    
    # 处理phase和traj的关键帧数据
    if phase is not None:
        phase_short = torch.zeros((1, args.seq_length, phase.shape[2]), device=device)
        phase_long = torch.zeros((1, args.pad_length, phase.shape[2]), device=device)
        
        # 复制所有关键帧的相位数据
        for kf in keyframes:
            if kf < args.seq_length:
                phase_short[:, kf:kf+1] = phase[:, kf:kf+1]
            if kf < args.pad_length:
                phase_long[:, kf:kf+1] = phase[:, kf:kf+1]
        
        # 确保上下文帧的完整性
        phase_short[:, :context_frames] = phase[:, :context_frames]
        phase_long[:, :context_frames] = phase[:, :context_frames]
        
        print(f"  创建了在所有关键帧位置上包含真实数据的phase稀疏序列")
    else:
        phase_short = None
        phase_long = None
        print("  未找到phase数据")
        
    if traj is not None:
        traj_short = torch.zeros((1, args.seq_length, traj.shape[2]), device=device)
        traj_long = torch.zeros((1, args.pad_length, traj.shape[2]), device=device)
        
        # 复制所有关键帧的轨迹数据
        for kf in keyframes:
            if kf < args.seq_length:
                traj_short[:, kf:kf+1] = traj[:, kf:kf+1]
            if kf < args.pad_length:
                traj_long[:, kf:kf+1] = traj[:, kf:kf+1]
        
        # 确保上下文帧的完整性
        traj_short[:, :context_frames] = traj[:, :context_frames]
        traj_long[:, :context_frames] = traj[:, :context_frames]
        
        print(f"  创建了在所有关键帧位置上包含真实数据的traj稀疏序列")
    else:
        traj_short = None
        traj_long = None
        print("  未找到traj数据")
    
    # 步骤6: 基于高质量约束进行插值
    print("\n=== 基于高质量约束进行插值 ===")
    
    # 过滤出短序列有效的关键帧
    short_keyframes = [kf for kf in keyframes if kf < args.seq_length]
    print(f"短序列使用 {len(short_keyframes)} 个有效关键帧: {short_keyframes}")
    
    # 过滤出长序列有效的关键帧
    long_keyframes = [kf for kf in keyframes if kf < args.pad_length]
    print(f"长序列使用 {len(long_keyframes)} 个有效关键帧: {long_keyframes}")
    
    # 对短序列进行插值，生成RefineNet真正需要的输入
    input_short_raw = ops.interpolate_motion_by_keyframes(constrained_short, short_keyframes)
    
    # 对长序列也进行插值，模拟经过线性插值填充的输入
    input_long_raw = ops.interpolate_motion_by_keyframes(constrained_long, long_keyframes)
    
    # 检查插值结果
    print(f"插值后的短序列形状: {input_short_raw.shape}")
    print(f"插值后的长序列形状: {input_long_raw.shape}")
    
    # 对 phase 和 traj 也进行插值，确保所有特征连贯
    print("\n=== 插值辅助特征 ===")
    
    # 线性插值函数
    def interpolate_feature(feature_tensor, keyframes, context_frames):
        if feature_tensor is None:
            return None
            
        # 创建插值后的结果容器
        result = feature_tensor.clone()
        
        # 对每对相邻关键帧进行插值
        for i in range(len(keyframes) - 1):
            start_idx = keyframes[i]
            end_idx = keyframes[i + 1]
            
            if end_idx <= start_idx + 1:  # 相邻帧不需要插值
                continue
                
            # 提取起始和结束帧的值
            start_val = feature_tensor[:, start_idx]
            end_val = feature_tensor[:, end_idx]
            
            # 生成插值权重
            steps = end_idx - start_idx
            for j in range(1, steps):
                curr_idx = start_idx + j
                t = j / steps  # 插值权重 0.0 -> 1.0
                
                # 线性插值
                interp_val = start_val * (1 - t) + end_val * t
                result[:, curr_idx] = interp_val
                
        return result
    
    # 插值 phase
    if phase_short is not None and phase_long is not None:
        # 使用已经包含真实关键帧数据的稀疏序列进行插值
        phase_short_interp = interpolate_feature(phase_short, short_keyframes, context_frames)
        phase_long_interp = interpolate_feature(phase_long, long_keyframes, context_frames)
        print(f"Phase 特征插值完成 - 短序列: {phase_short_interp.shape}, 长序列: {phase_long_interp.shape}")
    else:
        phase_short_interp = None
        phase_long_interp = None
        print("Phase 特征不存在，跳过插值")
    
    # 插值 trajectory
    if traj_short is not None and traj_long is not None:
        # 使用已经包含真实关键帧数据的稀疏序列进行插值
        traj_short_interp = interpolate_feature(traj_short, short_keyframes, context_frames)
        traj_long_interp = interpolate_feature(traj_long, long_keyframes, context_frames)
        print(f"Trajectory 特征插值完成 - 短序列: {traj_short_interp.shape}, 长序列: {traj_long_interp.shape}")
    else:
        traj_short_interp = None
        traj_long_interp = None
        print("Trajectory 特征不存在，跳过插值")
    
    # 检查每个关键帧的插值情况
    print("\n=== KeyframeNet 插值效果分析 ====")
    # 检查每个关键帧的插值前后差异
    frames_to_check = short_keyframes.copy()
    frames_names = [f"关键帧 {i}" for i in range(len(short_keyframes))]
    
    # 添加一些非关键帧作为对比
    # 选取关键帧之间的点
    if len(short_keyframes) > 1:
        for i in range(len(short_keyframes)-1):
            mid_point = (short_keyframes[i] + short_keyframes[i+1]) // 2
            if mid_point not in frames_to_check:
                frames_to_check.append(mid_point)
                frames_names.append(f"非关键帧(关键帧{i}-{i+1}中间)")
    
    # 按帧索引排序
    sorted_indices = sorted(range(len(frames_to_check)), key=lambda i: frames_to_check[i])
    frames_to_check = [frames_to_check[i] for i in sorted_indices]
    frames_names = [frames_names[i] for i in sorted_indices]
    
    print("\n短序列中 KeyframeNet 关键帧和插值帧分析:")
    for i, (frame, name) in enumerate(zip(frames_to_check, frames_names)):
        if frame < constrained_short.shape[1] and frame < input_short_raw.shape[1]:
            # 计算插值前后的平均绝对差异
            if torch.sum(constrained_short[0, frame]) > 0:  # 避免除数为0
                diff = torch.abs(constrained_short[0, frame] - input_short_raw[0, frame])
                avg_diff = torch.mean(diff).item()
                is_keyframe = "[是关键帧]" if frame in short_keyframes else "[插值帧]"
                kf_idx = short_keyframes.index(frame) if frame in short_keyframes else ""
                kf_info = f"(KeyframeNet 关键帧{kf_idx})" if frame in short_keyframes else ""
                print(f"  - 帧 {frame} {name} {is_keyframe} {kf_info} 插值前后差异: {avg_diff:.6f}")
            else:
                print(f"  - 帧 {frame} {name}: 原始数据为空")
    
    print("\n长序列中 KeyframeNet 关键帧和插值帧分析:")
    for i, (frame, name) in enumerate(zip(frames_to_check, frames_names)):
        if frame < constrained_long.shape[1] and frame < input_long_raw.shape[1]:
            # 计算插值前后的平均绝对差异
            if torch.sum(constrained_long[0, frame]) > 0:  # 避免除数为0
                diff = torch.abs(constrained_long[0, frame] - input_long_raw[0, frame])
                avg_diff = torch.mean(diff).item()
                is_keyframe = "[是关键帧]" if frame in long_keyframes else "[插值帧]"
                kf_idx = long_keyframes.index(frame) if frame in long_keyframes else ""
                kf_info = f"(KeyframeNet 关键帧{kf_idx})" if frame in long_keyframes else ""
                print(f"  - 帧 {frame} {name} {is_keyframe} {kf_info} 插值前后差异: {avg_diff:.6f}")
            else:
                print(f"  - 帧 {frame} {name}: 原始数据为空")
    
    # 步骤4: 创建注意力掩码
    print("\n=== 创建注意力掩码 ===")
    # 注意: 填充部分为True表示这些位置应被掩蔽
    attention_mask = torch.zeros((1, 1, args.pad_length, args.pad_length), dtype=torch.bool, device=device)
    attention_mask[:, :, :, args.seq_length:] = True  # 掩蔽填充列
    attention_mask[:, :, args.seq_length:, :] = True  # 掩蔽填充行
    
    # 打印掩码模式进行调试
    print("注意力掩码示例 (T表示被掩蔽):")
    sample_mask = attention_mask[0, 0]
    for i in range(min(5, args.seq_length)):  # 真实序列前5行
        row = ''.join(['T' if m else 'F' for m in sample_mask[i, :10]])  # 前10列
        print(f"真实行 {i}: {row}...")
    for i in range(args.seq_length, min(args.seq_length+5, args.pad_length)):  # 填充序列前5行
        row = ''.join(['T' if m else 'F' for m in sample_mask[i, :10]])  # 前10列
        print(f"填充行 {i}: {row}...")
        
    # 计算掩码统计信息
    total_cells = args.pad_length * args.pad_length
    masked_cells = torch.sum(sample_mask).item()
    masked_percentage = (masked_cells / total_cells) * 100
    
    print(f"\n掩码统计信息:")
    print(f"  - 总元素数: {total_cells}")
    print(f"  - 被掩蒙元素数: {masked_cells}")
    print(f"  - 被掩蒙百分比: {masked_percentage:.2f}%")
    
    # 计算有效序列部分的掩码统计
    valid_mask = sample_mask[:args.seq_length, :args.seq_length]
    valid_cells = args.seq_length * args.seq_length
    valid_masked_cells = torch.sum(valid_mask).item()
    valid_masked_percentage = (valid_masked_cells / valid_cells) * 100
    
    print(f"  - 有效部分被掩蒙元素数: {valid_masked_cells}")
    print(f"  - 有效部分被掩蒙百分比: {valid_masked_percentage:.2f}%")
    
    # 注意力掩码中 KeyframeNet 关键帧位置详情
    print("\nKeyframeNet 关键帧在注意力掩码中的状态:")
    frames_to_analyze = []
    for i, kf in enumerate(long_keyframes):
        frames_to_analyze.append((f"KeyframeNet 关键帧 {i}", kf))
    
    # 添加中间插值帧作为参考
    if len(long_keyframes) > 1:
        middle_frame = (long_keyframes[0] + long_keyframes[1]) // 2
        frames_to_analyze.append(("插值中间帧", middle_frame))
    
    for frame_name, frame_idx in frames_to_analyze:
        if frame_idx < args.pad_length:
            # 检查该帧与其他帧的注意力连接
            # 1. 该帧可以关注哪些帧
            can_attend_to = ~sample_mask[frame_idx, :args.seq_length]
            can_attend_to_count = torch.sum(can_attend_to).item()
            
            # 2. 哪些帧可以关注该帧
            attended_by = ~sample_mask[:args.seq_length, frame_idx]
            attended_by_count = torch.sum(attended_by).item()
            
            print(f"  - {frame_name} (索引 {frame_idx})")
            print(f"    * 该帧可以关注 {can_attend_to_count} 个帧 (有效序列中 {can_attend_to_count/args.seq_length*100:.1f}%)")
            print(f"    * 该帧被 {attended_by_count} 个帧关注 (有效序列中 {attended_by_count/args.seq_length*100:.1f}%)")
    
    # KeyframeNet 关键帧之间的注意力连接
    print("\nKeyframeNet 关键帧之间的注意力连接:")
    
    # 分析所有关键帧之间的关系
    for i, kf_i in enumerate(long_keyframes):
        for j, kf_j in enumerate(long_keyframes):
            if i != j:  # 不分析自身
                can_attend = "可以" if not sample_mask[kf_i, kf_j].item() else "不可以"
                print(f"  - 关键帧 {i} (索引 {kf_i}) {can_attend} 关注 关键帧 {j} (索引 {kf_j})")
    
    # 如果有中间帧，分析中间帧与关键帧的关系
    if len(long_keyframes) > 1:
        print("\n插值中间帧与关键帧的注意力连接:")
        for i, kf in enumerate(long_keyframes):
            mid_to_kf = "可以" if not sample_mask[middle_frame, kf].item() else "不可以"
            kf_to_mid = "可以" if not sample_mask[kf, middle_frame].item() else "不可以"
            print(f"  - 插值中间帧 {mid_to_kf} 关注 关键帧 {i} (索引 {kf})")
            print(f"  - 关键帧 {i} (索引 {kf}) {kf_to_mid} 关注 插值中间帧")
    
    # 创建一个直观的可视化表示
    print("\n注意力掩码可视化 (T=被掩蒙, F=可见):")
    print("    " + "".join([f"{i:3d}" for i in range(min(20, args.pad_length))]))
    print("    " + "-" * (min(20, args.pad_length) * 3))
    
    for i in range(min(20, args.pad_length)):  # 显示前20行
        row = ''.join(['T  ' if m else 'F  ' for m in sample_mask[i, :min(20, args.pad_length)]])  # 前20列
        print(f"{i:2d} | {row}")
        
        # 在关键帧位置添加标记
        if i == long_keyframes[0]:
            print("    " + "^" * (min(20, args.pad_length) * 3) + " (上下文帧结束)")
        elif i == long_keyframes[1]:
            print("    " + "^" * (min(20, args.pad_length) * 3) + " (目标帧)")
        elif i == middle_frame:
            print("    " + "^" * (min(20, args.pad_length) * 3) + " (中间帧)")
    
    # 这一部分不需要了，因为我们之前已经加载了归一化统计量
    
    # 步骤5: 执行推理
    print("\n=== 执行推理 ===")
    
    # --- START: 最终修正的完整数据准备与推理流程 ---

    # 1. 使用已插值好的序列作为模型输入
    input_motion_short_raw = input_short_raw
    input_motion_long_raw = input_long_raw

    # 2. 使用已插值的phase和traj
    phase_short_interp_used = phase_short_interp if phase_short_interp is not None else None
    phase_long_interp_used = phase_long_interp if phase_long_interp is not None else None
    traj_short_interp_used = traj_short_interp if traj_short_interp is not None else None
    traj_long_interp_used = traj_long_interp if traj_long_interp is not None else None

    # 3. 在送入模型前，对所有数据进行归一化
    input_motion_short_norm = (input_motion_short_raw - motion_mean) / motion_std
    input_motion_long_norm = (input_motion_long_raw - motion_mean) / motion_std
    # 如果 traj 有统计量，也需要归一化
    if traj_short_interp_used is not None and traj_mean is not None and traj_std is not None:
        traj_short_norm = (traj_short_interp_used - traj_mean) / traj_std
        traj_long_norm = (traj_long_interp_used - traj_mean) / traj_std
    else:
        traj_short_norm = None
        traj_long_norm = None

    # 4. 执行推理
    with torch.no_grad():
        # 场景A: 对短序列直接推理
        print("场景A: 对插值短序列直接推理...")
        output_short_norm = refine_model(
            input_motion_short_norm, 
            midway_targets=[], 
            phase=phase_short_interp_used, 
            traj=traj_short_norm,  # 使用插值并归一化后的 traj
            attention_mask=None
        )

        # 场景B: 对长序列使用掩码推理
        print("场景B: 对插值长序列使用掩码推理，并指定目标帧约束...")
        output_long_norm = refine_model(
            input_motion_long_norm,
            midway_targets=[target_frame_idx],  # 指定目标帧为已知约束
            phase=phase_long_interp_used,
            traj=traj_long_norm,   # 使用插值并归一化后的 traj
            attention_mask=attention_mask
        )

    # 5. 对模型的输出进行反归一化，以得到真实世界坐标的动作
    output_short = {}
    output_short["motion"] = output_short_norm["motion"] * motion_std + motion_mean
    if "contact" in output_short_norm:
        output_short["contact"] = output_short_norm["contact"]
    if "phase" in output_short_norm:
        output_short["phase"] = output_short_norm["phase"]
        
    output_long = {}
    output_long["motion"] = output_long_norm["motion"] * motion_std + motion_mean
    if "contact" in output_long_norm:
        output_long["contact"] = output_long_norm["contact"]
    if "phase" in output_long_norm:
        output_long["phase"] = output_long_norm["phase"]
        
    # 提取有效部分进行比较
    output_long_truncated = output_long["motion"][:, :args.seq_length]
    # --- END: 修正流程 ---
    
    # 步骤6: 数值验证
    print("\n=== 数值验证结果 ===")
    # 检查NaN值
    has_nan_short = torch.isnan(output_short["motion"]).any()
    has_nan_long = torch.isnan(output_long_truncated).any()
    print(f"短序列输出中有NaN? {has_nan_short}")
    print(f"填充序列输出中有NaN? {has_nan_long}")
    
    if has_nan_short or has_nan_long:
        print("\n错误: 输出中包含NaN值，实验无法继续。")
        return
    
    # 数值对比
    are_identical = torch.allclose(output_short["motion"], output_long_truncated, atol=args.atol)
    max_difference = torch.max(torch.abs(output_short["motion"] - output_long_truncated)).item()
    mean_difference = torch.mean(torch.abs(output_short["motion"] - output_long_truncated)).item()
    
    print(f"输出在数值上是否几乎完全相同? -> {are_identical}")
    print(f"最大绝对差值: {max_difference}")
    print(f"平均绝对差值: {mean_difference}")
    
    # 模型输出关键帧差异分析
    print("\n=== 模型输出关键帧差异分析 ====")
    
    # 计算中间帧位置
    middle_frame = (short_keyframes[0] + short_keyframes[1]) // 2 if len(short_keyframes) > 1 else short_keyframes[0]
    
    # 计算并显示关键帧差异
    context_diff = torch.abs(output_short["motion"][0, short_keyframes[0]] - output_long_truncated[0, short_keyframes[0]]).mean().item()
    target_diff = torch.abs(output_short["motion"][0, short_keyframes[1]] - output_long_truncated[0, short_keyframes[1]]).mean().item()
    middle_diff = torch.abs(output_short["motion"][0, middle_frame] - output_long_truncated[0, middle_frame]).mean().item()
    
    # 计算平均差异和最大差异的帧
    all_diffs = torch.abs(output_short["motion"] - output_long_truncated).mean(dim=-1).squeeze(0)
    max_diff_frame = torch.argmax(all_diffs).item()
    max_diff_value = all_diffs[max_diff_frame].item()
    
    print(f"关键帧差异:")
    print(f"  - 上下文帧结束差异: {context_diff:.6f}")
    print(f"  - 目标帧差异: {target_diff:.6f}")
    print(f"  - 中间帧差异: {middle_diff:.6f}")
    print(f"  - 最大差异帧: {max_diff_frame} (差异值: {max_diff_value:.6f})")
    
    # 帧差异统计
    diff_arr = all_diffs.cpu().numpy()
    diff_percentiles = np.percentile(diff_arr, [0, 25, 50, 75, 100])
    print(f"\n差异统计信息:")
    print(f"  - 最小差异: {diff_percentiles[0]:.6f}")
    print(f"  - 25%分位数: {diff_percentiles[1]:.6f}")
    print(f"  - 中位数: {diff_percentiles[2]:.6f}")
    print(f"  - 75%分位数: {diff_percentiles[3]:.6f}")
    print(f"  - 最大差异: {diff_percentiles[4]:.6f}")
    
    # 步骤7: 保存输出结果(如果指定)
    if args.save_output:
        output_path = os.path.join(args.output_dir, f"seq_{args.seq_idx}_batch_{args.batch_idx}.pt")
        torch.save({
            "short_motion": output_short["motion"].cpu(),
            "long_motion": output_long_truncated.cpu(),
            "args": vars(args),
            "keyframes": keyframes,
            "max_difference": max_difference,
            "mean_difference": mean_difference,
            "are_identical": are_identical
        }, output_path)
        print(f"\n=== 已保存输出到 {output_path} ===")
        
    # 步骤8: 如果启用可视化，启动三段式可视化对比
    if not args.no_vis:
        print("\n=== 启动三段式视觉验证 ===")
        print("第一阶段: 播放填充序列处理结果")
        print("第二阶段: 播放直接处理结果")
        print("第三阶段: 同时播放两者，观察是否完美重叠")
        
        if are_identical:
            print("注意：两个输出在数值上几乎完全相同，应该会完美重叠。")
        else:
            print(f"注意：两个输出在数值上有差异（最大差值: {max_difference}），观察是否有明显的视觉差异。")
        
        # 打印最终形状
        print(f"可视化数据形状:")
        print(f"  - output_short[\"motion\"]: {output_short['motion'].shape}")
        print(f"  - output_long_truncated: {output_long_truncated.shape}")
        
        # 准备可视化所需的数据
        # 确保每个动作只有一个完整序列
        output_frames = output_long_truncated.shape[1]
        print(f"动画帧数: {output_frames}")
        
        # 为三段式播放提供提示
        print("\n提示: 按照最终实验方案进行三段式播放")
        print("第一阶段: 仅播放填充序列处理的结果（一个角色）")
        print("第二阶段: 仅播放直接处理的结果（一个角色）")
        print("第三阶段: 同时播放两个结果（两个角色应完美重叠）")
        print("按空格键可暂停观察，按'M'键可将角色分开观察")
        
        # 启动可视化App
        app = MotionApp(
            motions=[output_long_truncated, output_short["motion"]],
            tags=["Intra-Batch", "Multi-IB"],  # 使用MotionApp预设标签以触发三段式逻辑
            skeleton=dataset.skeleton,
            dataset=args.dataset,
            paused=False,          # 自动开始播放
            compare_mode=True,     # 激活三段式播放逻辑
            show_keyframes=False,  # 不显示关键帧
            record_sequence_idx=args.seq_idx  # 指定序列索引，确保三段式播放正常工作
        )
        
        # 手动设置帧数，确保只播放一次
        app.frame_per_batch = output_frames
        agl.AppManager.start(app)
    else:
        print("\n=== 可视化已禁用 ===")
        print(f"数值验证结果: {'相同' if are_identical else '不同'}")
        print(f"  - 最大差异: {max_difference}")
        print(f"  - 平均差异: {mean_difference}")

if __name__ == "__main__":
    main()