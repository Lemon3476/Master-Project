import torch
import torch.nn as nn
import torch.nn.functional as F

def rot_loss(pred, gt, context_frames):
    loss = F.l1_loss(pred[:, context_frames:-1], gt[:, context_frames:-1])
    return loss

def pos_loss(pred, gt, context_frames):
    loss = F.l1_loss(pred[:, context_frames:-1], gt[:, context_frames:-1])
    return loss

def smooth_loss(pred, context_frames):
    loss = F.l1_loss(pred[:, context_frames:] - pred[:, context_frames-1:-1],
                     torch.zeros_like(pred[:, context_frames:]))
    return loss

def contact_loss(pred, gt, context_frames):
    loss = F.l1_loss(pred[:, context_frames:-1], gt[:, context_frames:-1])
    return loss

def foot_loss(contact, vel, context_frames):
    loss = F.l1_loss(contact[:, context_frames:-1].detach() * vel[:, context_frames:-1],
                     torch.zeros_like(vel[:, context_frames:-1]))
    return loss

def phase_loss(pred_phase, gt_phase, context_frames):
    loss_phase = F.l1_loss(pred_phase[:, context_frames:-1], gt_phase[:, context_frames:-1])
    return loss_phase

def traj_loss(pred, gt, context_frames):
    pred_pos, pred_dir = torch.split(pred, [2, 2], dim=-1)
    gt_pos, gt_dir = torch.split(gt, [2, 2], dim=-1)
    loss_pos = F.l1_loss(pred_pos[:, context_frames:-1], gt_pos[:, context_frames:-1])
    loss_dir = F.l1_loss(1 - torch.sum(pred_dir[:, context_frames:-1] * gt_dir[:, context_frames:-1], dim=-1),
                            torch.zeros_like(pred_dir[:, context_frames:-1, 0]))
    return loss_pos + loss_dir

def score_loss(pred, gt, context_frames):
    loss = F.l1_loss(pred[:, context_frames:-1], gt[:, context_frames:-1])
    return loss

def foot_consistency_loss(foot_positions, contact_probs):
    """Calculate foot consistency loss to enforce stability during contact
    
    Args:
        foot_positions: Foot joint positions [B, T, J, 3]
        contact_probs: Contact probabilities [B, T, J]
    
    Returns:
        Loss value penalizing foot sliding during stable contacts
    """
    # Create mask for stable contact states (contact in both frames)
    contact_maintained_mask = (contact_probs[:, 1:] > 0.8) & (contact_probs[:, :-1] > 0.8)
    
    # Calculate foot displacement between consecutive frames
    displacement = torch.norm(foot_positions[:, 1:] - foot_positions[:, :-1], p=2, dim=-1)
    
    # Only penalize displacement for feet that should be in stable contact
    masked_displacement = displacement * contact_maintained_mask
    
    # Average over all samples and time steps with valid errors
    loss = torch.sum(masked_displacement) / (torch.sum(contact_maintained_mask) + 1e-6)
    
    return loss

# 带掩码的损失函数 - 用于稀疏关键帧模式
def masked_rot_loss(pred, gt, mask):
    """计算带掩码的旋转损失
    
    Args:
        pred: 预测的旋转 [B, T, J, 6]
        gt: 真实的旋转 [B, T, J, 6]
        mask: 掩码 [B, T, J, 6]，为True的位置计算损失
    
    Returns:
        损失值
    """
    # 将掩码应用于预测和真实值
    masked_pred = pred * mask
    masked_gt = gt * mask
    
    # 计算L1损失
    if mask.sum() > 0:
        loss = F.l1_loss(masked_pred, masked_gt, reduction='sum') / (mask.sum() + 1e-8)
    else:
        loss = torch.tensor(0.0, device=pred.device)
    
    return loss

def masked_pos_loss(pred, gt, mask):
    """计算带掩码的位置损失
    
    Args:
        pred: 预测的位置 [B, T, J, 3]
        gt: 真实的位置 [B, T, J, 3]
        mask: 掩码 [B, T, J, 3]，为True的位置计算损失
    
    Returns:
        损失值
    """
    # 将掩码应用于预测和真实值
    masked_pred = pred * mask
    masked_gt = gt * mask
    
    # 计算L1损失
    if mask.sum() > 0:
        loss = F.l1_loss(masked_pred, masked_gt, reduction='sum') / (mask.sum() + 1e-8)
    else:
        loss = torch.tensor(0.0, device=pred.device)
    
    return loss

def masked_contact_loss(pred, gt, mask):
    """计算带掩码的接触损失
    
    Args:
        pred: 预测的接触 [B, T, 4]
        gt: 真实的接触 [B, T, 4]
        mask: 掩码 [B, T, 4]，为True的位置计算损失
    
    Returns:
        损失值
    """
    # 将掩码应用于预测和真实值
    masked_pred = pred * mask
    masked_gt = gt * mask
    
    # 计算L1损失
    if mask.sum() > 0:
        loss = F.l1_loss(masked_pred, masked_gt, reduction='sum') / (mask.sum() + 1e-8)
    else:
        loss = torch.tensor(0.0, device=pred.device)
    
    return loss

def masked_phase_loss(pred, gt, mask):
    """计算带掩码的相位损失
    
    Args:
        pred: 预测的相位 [B, T, P]
        gt: 真实的相位 [B, T, P]
        mask: 掩码 [B, T, P]，为True的位置计算损失
    
    Returns:
        损失值
    """
    # 将掩码应用于预测和真实值
    masked_pred = pred * mask
    masked_gt = gt * mask
    
    # 计算L1损失
    if mask.sum() > 0:
        loss = F.l1_loss(masked_pred, masked_gt, reduction='sum') / (mask.sum() + 1e-8)
    else:
        loss = torch.tensor(0.0, device=pred.device)
    
    return loss

def masked_traj_loss(pred, gt, mask):
    """计算带掩码的轨迹损失
    
    Args:
        pred: 预测的轨迹 [B, T, 4]
        gt: 真实的轨迹 [B, T, 4]
        mask: 掩码 [B, T] 布尔值掩码，为True的帧位置计算损失
    
    Returns:
        损失值
    """
    # 分离位置和方向
    pred_pos, pred_dir = torch.split(pred, [2, 2], dim=-1)
    gt_pos, gt_dir = torch.split(gt, [2, 2], dim=-1)
    
    # 将掩码扩展为正确的维度
    pos_mask = mask.unsqueeze(-1).expand(-1, -1, 2)
    dir_mask = mask.unsqueeze(-1).expand(-1, -1, 2)
    
    # 应用掩码
    masked_pred_pos = pred_pos * pos_mask
    masked_gt_pos = gt_pos * pos_mask
    
    # 计算位置L1损失
    if pos_mask.sum() > 0:
        loss_pos = F.l1_loss(masked_pred_pos, masked_gt_pos, reduction='sum') / (pos_mask.sum() + 1e-8)
    else:
        loss_pos = torch.tensor(0.0, device=pred.device)
    
    # 计算方向损失
    dir_dot_product = torch.sum(pred_dir * gt_dir, dim=-1)
    dir_dot_product = torch.clamp(dir_dot_product, -1.0, 1.0)  # 确保值在[-1, 1]范围内
    dir_loss = 1.0 - dir_dot_product
    
    # 应用掩码到方向损失
    masked_dir_loss = dir_loss * mask
    
    if mask.sum() > 0:
        loss_dir = masked_dir_loss.sum() / (mask.sum() + 1e-8)
    else:
        loss_dir = torch.tensor(0.0, device=pred.device)
    
    return loss_pos + loss_dir


"""
Loss functions for RMI
"""
def disc_loss(real_score, fake_score):
    real = torch.mean((real_score - 1) ** 2)
    fake = torch.mean(fake_score ** 2)
    return 0.5 * (real + fake)

def gen_loss(fake_score):
    loss = torch.mean((fake_score - 1) ** 2)
    return 0.5 * loss