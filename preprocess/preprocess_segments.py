import sys
sys.path.append(".")

import os
import argparse
import numpy as np
import random
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from aPyOpenGL import transforms as trf
from utils import utils, ops
from utils.dataset import MotionDataset
from model.twostage import ContextTransformer

def prepare_segment_data(keyframes, GT_motion, GT_phase, GT_traj, GT_score, GT_contact,
                        mean, std, traj_mean, traj_std, config, device):
    """
    预处理段数据：为每对相邻关键帧提取训练用的片段。
    """
    B, T, M = GT_motion.shape
    segments = []
    
    # 归一化运动和轨迹
    GT_motion_norm = (GT_motion - mean) / std
    GT_traj_norm = (GT_traj - traj_mean) / traj_std if GT_traj is not None else None
    
    for b in range(B):
        batch_keyframes = keyframes[b]
        
        # 遍历关键帧对
        for i in range(len(batch_keyframes) - 1):
            kf_start, kf_end = batch_keyframes[i], batch_keyframes[i+1]
            
            # 计算重叠起始位置
            overlap_start = max(0, kf_start - config.overlap_frames + 1)
            
            # 提取段数据
            segment_range = slice(overlap_start, kf_end + 1)
            segment_motion = GT_motion_norm[b:b+1, segment_range].clone()
            segment_phase = GT_phase[b:b+1, segment_range].clone() if GT_phase is not None else None
            segment_traj = GT_traj_norm[b:b+1, segment_range].clone() if GT_traj_norm is not None else None
            segment_contact = GT_contact[b:b+1, segment_range].clone() if GT_contact is not None else None
            segment_score = GT_score[b:b+1, segment_range].clone() if GT_score is not None else None
            
            # 调整段长度为固定大小
            curr_len = segment_motion.shape[1]
            
            if curr_len > config.max_segment_length:
                # 截断段到最大长度，同时保留重叠和末尾帧
                preserve_start = min(config.overlap_frames, curr_len)
                preserve_end = min(5, curr_len - preserve_start)  # 保留末尾最多5帧
                
                # 保留起始和末尾部分
                start_portion = segment_motion[:, :preserve_start]
                end_portion = segment_motion[:, -preserve_end:]
                
                # 计算需要保留的中间帧数量
                middle_frames = config.max_segment_length - preserve_start - preserve_end
                
                if middle_frames > 0:
                    # 从中间部分选择均匀分布的帧
                    middle_start = preserve_start
                    middle_end = curr_len - preserve_end
                    middle_indices = torch.linspace(middle_start, middle_end - 1, middle_frames, dtype=torch.long)
                    middle_portion = segment_motion[:, middle_indices]
                    
                    # 组合所有部分
                    segment_motion = torch.cat([start_portion, middle_portion, end_portion], dim=1)
                    
                    if segment_phase is not None:
                        segment_phase = torch.cat([
                            segment_phase[:, :preserve_start],
                            segment_phase[:, middle_indices],
                            segment_phase[:, -preserve_end:]
                        ], dim=1)
                    
                    if segment_traj is not None:
                        segment_traj = torch.cat([
                            segment_traj[:, :preserve_start],
                            segment_traj[:, middle_indices],
                            segment_traj[:, -preserve_end:]
                        ], dim=1)
                        
                    if segment_contact is not None:
                        segment_contact = torch.cat([
                            segment_contact[:, :preserve_start],
                            segment_contact[:, middle_indices],
                            segment_contact[:, -preserve_end:]
                        ], dim=1)
                        
                    if segment_score is not None:
                        segment_score = torch.cat([
                            segment_score[:, :preserve_start],
                            segment_score[:, middle_indices],
                            segment_score[:, -preserve_end:]
                        ], dim=1)
                else:
                    # 如果middle_frames <= 0，只连接起始和末尾部分
                    segment_motion = torch.cat([start_portion, end_portion], dim=1)
                    
                    if segment_phase is not None:
                        segment_phase = torch.cat([
                            segment_phase[:, :preserve_start],
                            segment_phase[:, -preserve_end:]
                        ], dim=1)
                    
                    if segment_traj is not None:
                        segment_traj = torch.cat([
                            segment_traj[:, :preserve_start],
                            segment_traj[:, -preserve_end:]
                        ], dim=1)
                        
                    if segment_contact is not None:
                        segment_contact = torch.cat([
                            segment_contact[:, :preserve_start],
                            segment_contact[:, -preserve_end:]
                        ], dim=1)
                        
                    if segment_score is not None:
                        segment_score = torch.cat([
                            segment_score[:, :preserve_start],
                            segment_score[:, -preserve_end:]
                        ], dim=1)
            
            elif curr_len < config.max_segment_length:
                # 填充段到最大长度
                pad_len = config.max_segment_length - curr_len
                
                # 通过重复最后一帧创建填充
                motion_padding = segment_motion[:, -1:].expand(-1, pad_len, -1)
                segment_motion = torch.cat([segment_motion, motion_padding], dim=1)
                
                if segment_phase is not None:
                    phase_padding = segment_phase[:, -1:].expand(-1, pad_len, -1)
                    segment_phase = torch.cat([segment_phase, phase_padding], dim=1)
                
                if segment_traj is not None:
                    traj_padding = segment_traj[:, -1:].expand(-1, pad_len, -1)
                    segment_traj = torch.cat([segment_traj, traj_padding], dim=1)
                    
                if segment_contact is not None:
                    contact_padding = segment_contact[:, -1:].expand(-1, pad_len, -1)
                    segment_contact = torch.cat([segment_contact, contact_padding], dim=1)
                    
                if segment_score is not None:
                    score_padding = segment_score[:, -1:].expand(-1, pad_len, -1)
                    segment_score = torch.cat([segment_score, score_padding], dim=1)
            
            # 通过线性插值在归一化空间中创建草稿
            segment_keyframes = [0, segment_motion.shape[1] - 1]  # 第一帧和最后一帧
            draft_motion = ops.interpolate_motion_by_keyframes(segment_motion, segment_keyframes)
            
            # 去除批次维度，以便于保存
            segment_motion = segment_motion.squeeze(0)
            draft_motion = draft_motion.squeeze(0)
            
            if segment_phase is not None:
                segment_phase = segment_phase.squeeze(0)
            if segment_traj is not None:
                segment_traj = segment_traj.squeeze(0)
            if segment_contact is not None:
                segment_contact = segment_contact.squeeze(0)
            if segment_score is not None:
                segment_score = segment_score.squeeze(0)
            
            # 存储段数据
            segment_data = {
                "gt_motion": segment_motion.cpu().numpy(),
                "draft_motion": draft_motion.cpu().numpy(),
                "overlap_frames": config.overlap_frames
            }
            
            if segment_phase is not None:
                segment_data["phase"] = segment_phase.cpu().numpy()
            if segment_traj is not None:
                segment_data["traj"] = segment_traj.cpu().numpy()
            if segment_contact is not None:
                segment_data["contact"] = segment_contact.cpu().numpy()
            if segment_score is not None:
                segment_data["score"] = segment_score.cpu().numpy()
            
            segments.append(segment_data)
    
    return segments

def process_dataset(dataset, kf_model, config, args, device):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 获取统计数据用于归一化
    mean, std = dataset.motion_statistics()
    mean, std = mean.to(device), std.to(device)
    
    traj_mean, traj_std = dataset.traj_statistics()
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)
    
    all_segments = []
    
    # 处理每个批次
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="处理批次")):
        # 获取GT数据
        GT_motion = batch["motion"].to(device)
        GT_phase = batch["phase"].to(device) if config.use_phase else None
        GT_traj = batch["traj"].to(device) if config.use_traj else None
        GT_score = batch["score"].to(device) if config.use_score else None
        
        # 计算接触信息
        B, T, M = GT_motion.shape
        GT_local_ortho6ds, GT_root_pos = torch.split(GT_motion, [M-3, 3], dim=-1)
        GT_local_ortho6ds = GT_local_ortho6ds.reshape(B, T, dataset.skeleton.num_joints, 6)
        _, GT_global_positions = trf.t_ortho6d.fk(GT_local_ortho6ds, GT_root_pos, dataset.skeleton)
        
        contact_idx = []
        for joint in config.contact_joints:
            contact_idx.append(dataset.skeleton.idx_by_name[joint])
        
        GT_foot_vel = GT_global_positions[:, 1:, contact_idx] - GT_global_positions[:, :-1, contact_idx]
        GT_foot_vel = torch.sum(GT_foot_vel ** 2, dim=-1)  # (B, t-1, 4)
        GT_foot_vel = torch.cat([GT_foot_vel[:, 0:1], GT_foot_vel], dim=1)  # (B, t, 4)
        GT_contact = (GT_foot_vel < config.contact_threshold).float()  # (B, t, 4)
        
        # 归一化运动和轨迹用于KeyframeNet
        GT_motion_norm = (GT_motion - mean) / std
        GT_traj_norm = (GT_traj - traj_mean) / traj_std if GT_traj is not None else None
        
        # 获取关键帧
        with torch.no_grad():
            kf_out, _ = kf_model.forward(GT_motion_norm, phase=GT_phase, traj=GT_traj_norm, train=False)
            
            # 根据分数提取关键帧
            if config.use_score:
                pred_score = kf_out["score"]
                keyframes = ops.get_keyframes_by_score(config, pred_score)
            else:
                # 如果没有分数可用，使用随机关键帧
                keyframes = []
                for b in range(B):
                    kfs = ops.get_random_keyframe(config, T)
                    keyframes.append(kfs)
        
        # 准备段数据
        batch_segments = prepare_segment_data(
            keyframes, GT_motion, GT_phase, GT_traj, GT_score, GT_contact,
            mean, std, traj_mean, traj_std, config, device
        )
        
        all_segments.extend(batch_segments)
    
    return all_segments

def collect_segment_data(segments):
    """
    将片段数据收集到一个统一的字典中，准备保存为NPZ文件。
    """
    # 初始化收集所有数据的字典
    collected_data = {
        "gt_motion": [],
        "draft_motion": [],
        "overlap_frames": []
    }
    
    # 检查第一个段以确定哪些字段可用
    first_segment = segments[0]
    if "phase" in first_segment:
        collected_data["phase"] = []
    if "traj" in first_segment:
        collected_data["traj"] = []
    if "contact" in first_segment:
        collected_data["contact"] = []
    if "score" in first_segment:
        collected_data["score"] = []
    
    # 收集所有片段的数据
    for segment in segments:
        collected_data["gt_motion"].append(segment["gt_motion"])
        collected_data["draft_motion"].append(segment["draft_motion"])
        collected_data["overlap_frames"].append(segment["overlap_frames"])
        
        if "phase" in collected_data and "phase" in segment:
            collected_data["phase"].append(segment["phase"])
        if "traj" in collected_data and "traj" in segment:
            collected_data["traj"].append(segment["traj"])
        if "contact" in collected_data and "contact" in segment:
            collected_data["contact"].append(segment["contact"])
        if "score" in collected_data and "score" in segment:
            collected_data["score"].append(segment["score"])
    
    # 将列表转换为NumPy数组
    for key in collected_data:
        if key == "overlap_frames":
            collected_data[key] = np.array(collected_data[key])
        else:
            collected_data[key] = np.stack(collected_data[key], axis=0)
    
    return collected_data

if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default="segment.yaml")
    parser.add_argument("--kf_config", type=str, default="keyframe.yaml")
    parser.add_argument("--output_dir", type=str, default="dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载配置
    config = utils.load_config(f"config/{args.dataset}/{args.config}")
    kf_config = utils.load_config(f"config/{args.dataset}/{args.kf_config}")
    
    # 设置随机种子以确保可重复性
    utils.seed(1234)
    
    # 加载训练集
    train_dataset = MotionDataset(train=True, config=config)
    print(f"训练集大小: {len(train_dataset)}")
    
    # 加载验证集
    val_dataset = MotionDataset(train=False, config=config)
    print(f"验证集大小: {len(val_dataset)}")
    
    # 加载KeyframeNet模型
    kf_model = ContextTransformer(kf_config, train_dataset).to(device)
    load_result = utils.load_model(kf_model, kf_config)
    
    if load_result == -1:
        print("无法加载KeyframeNet模型，请确保模型已正确训练")
        sys.exit(1)
    
    kf_model.eval()
    print("KeyframeNet模型加载成功")
    
    # 处理训练集
    print("开始处理训练集...")
    train_segments = process_dataset(train_dataset, kf_model, config, args, device)
    print(f"训练集生成了 {len(train_segments)} 个段")
    
    # 处理验证集
    print("开始处理验证集...")
    val_segments = process_dataset(val_dataset, kf_model, config, args, device)
    print(f"验证集生成了 {len(val_segments)} 个段")
    
    # 收集并保存数据
    print("收集训练集数据...")
    train_data = collect_segment_data(train_segments)
    
    print("收集验证集数据...")
    val_data = collect_segment_data(val_segments)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存为NPZ文件
    train_output_path = os.path.join(args.output_dir, f"{args.dataset}-segment-train.npz")
    val_output_path = os.path.join(args.output_dir, f"{args.dataset}-segment-val.npz")
    
    print(f"保存训练集到 {train_output_path}...")
    np.savez_compressed(train_output_path, **train_data)
    
    print(f"保存验证集到 {val_output_path}...")
    np.savez_compressed(val_output_path, **val_data)
    
    print("预处理完成！")
    
    # 输出一些统计信息
    print("\n统计信息:")
    print(f"训练集段数: {train_data['gt_motion'].shape[0]}")
    print(f"验证集段数: {val_data['gt_motion'].shape[0]}")
    print(f"段长度: {train_data['gt_motion'].shape[1]}")
    print(f"特征维度: {train_data['gt_motion'].shape[2]}")
    
    total_size_mb = (train_data['gt_motion'].nbytes + train_data['draft_motion'].nbytes) / (1024 * 1024)
    print(f"训练集文件大小（估计）: {total_size_mb:.2f} MB")
    
    val_size_mb = (val_data['gt_motion'].nbytes + val_data['draft_motion'].nbytes) / (1024 * 1024)
    print(f"验证集文件大小（估计）: {val_size_mb:.2f} MB")