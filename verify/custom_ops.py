#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def get_keyframes_evenly_distributed(config, score, context_frames, target_frame, num_keyframes=5):
    """
    选择均匀分布的关键帧，确保覆盖整个动画序列
    
    Args:
        config: 配置对象
        score: 关键帧分数张量
        context_frames: 上下文帧数量
        target_frame: 目标帧索引
        num_keyframes: 要选择的关键帧总数 (包括起始和结束帧)
    
    Returns:
        list: 批次关键帧列表
    """
    B, T, _ = score.shape
    keyframes_list = []
    
    for b in range(B):
        # 起始和结束关键帧
        start_frame = context_frames - 1  # 上下文帧结束
        end_frame = target_frame  # 目标帧
        
        # 计算关键帧间隔
        if num_keyframes > 2:
            interval = (end_frame - start_frame) / (num_keyframes - 1)
        else:
            interval = end_frame - start_frame
        
        # 初始化关键帧列表
        batch_keyframes = [start_frame]  # 起始关键帧
        
        # 在中间添加均匀分布的关键帧
        for i in range(1, num_keyframes - 1):
            # 计算当前帧位置
            frame_idx = int(start_frame + i * interval)
            batch_keyframes.append(frame_idx)
        
        batch_keyframes.append(end_frame)  # 结束关键帧
        
        # 确保关键帧列表是唯一的且有序的
        batch_keyframes = sorted(list(set(batch_keyframes)))
        keyframes_list.append(batch_keyframes)
    
    return keyframes_list