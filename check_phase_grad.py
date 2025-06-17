import sys
sys.path.append(".")

import os
import torch
from torch.utils.data import DataLoader

from utils import utils
from utils.dataset import MotionDataset
from model.twostage import ContextTransformer
from utils import loss

def check_phase_gradients():
    # 设置GPU设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载配置文件
    config = utils.load_config("config/lafan1/keyframe-d-enc-dec.yaml")
    
    # 检查配置
    print(f"配置信息:")
    print(f"- weight_phase: {config.weight_phase}")
    print(f"- decoupled_traj_encoder: {config.get('decoupled_traj_encoder', False)}")
    print(f"- decoupled_phase_encoder: {config.get('decoupled_phase_encoder', False)}")
    print(f"- decoupled_traj_decoder: {config.get('decoupled_traj_decoder', False)}")
    
    # 加载数据集
    dataset = MotionDataset(train=True, config=config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 加载统计数据
    mean, std = dataset.motion_statistics(device)
    traj_mean, traj_std = dataset.traj_statistics(device)
    
    # 创建模型
    model = ContextTransformer(config, dataset).to(device)
    
    # 打印模型中 phase 相关组件
    print("\n相位相关组件:")
    if hasattr(model, 'phase_encoder'):
        print(f"- phase_encoder 存在")
        for name, param in model.phase_encoder.named_parameters():
            print(f"  - {name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
    print(f"- phase_decoder 存在")
    for name, param in model.phase_decoder.named_parameters():
        print(f"  - {name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
    # 前向传播测试
    batch = next(iter(dataloader))
    motion = batch["motion"].to(device)
    phase = batch["phase"].to(device)
    traj = batch["traj"].to(device) if config.use_traj else None
    
    # 归一化
    motion = (motion - mean) / std
    if config.use_traj:
        traj = (traj - traj_mean) / traj_std
    
    # 前向传播
    outputs, midway_targets = model.forward(motion, phase=phase, traj=traj, train=True)
    
    # 检查输出
    print("\n输出形状:")
    for key, value in outputs.items():
        print(f"- {key}: {value.shape}")
    
    # 提取预测和真实相位
    pred_phase = outputs["phase"]
    GT_phase = phase
    
    # 检查相位值
    print("\n相位值检查:")
    print(f"- 真实相位范围: min={GT_phase.min().item():.4f}, max={GT_phase.max().item():.4f}")
    print(f"- 预测相位范围: min={pred_phase.min().item():.4f}, max={pred_phase.max().item():.4f}")
    
    # 计算相位损失
    phase_loss_val = loss.phase_loss(pred_phase, GT_phase, config.context_frames)
    print(f"- 相位损失值: {phase_loss_val.item():.6f}")
    
    # 计算总损失
    total_loss = config.weight_phase * phase_loss_val
    
    # 检查所有相关组件的梯度是否有效
    print("\n进行反向传播...")
    total_loss.backward()
    
    print("\n相位编码器梯度检查:")
    if hasattr(model, 'phase_encoder'):
        for name, param in model.phase_encoder.named_parameters():
            if param.grad is not None:
                grad_info = f"min={param.grad.min().item():.8f}, max={param.grad.max().item():.8f}, mean={param.grad.abs().mean().item():.8f}"
                print(f"- {name} 梯度: {grad_info}")
            else:
                print(f"- {name} 梯度: None")
    
    print("\n相位解码器梯度检查:")
    for name, param in model.phase_decoder.named_parameters():
        if param.grad is not None:
            grad_info = f"min={param.grad.min().item():.8f}, max={param.grad.max().item():.8f}, mean={param.grad.abs().mean().item():.8f}"
            print(f"- {name} 梯度: {grad_info}")
        else:
            print(f"- {name} 梯度: None")
    
    # 检查转换器(Transformer)的最后一层梯度
    print("\nTransformer最后一层梯度检查:")
    last_layer = model.pff_layers[-1]
    for name, param in last_layer.named_parameters():
        if param.grad is not None:
            grad_info = f"min={param.grad.min().item():.8f}, max={param.grad.max().item():.8f}, mean={param.grad.abs().mean().item():.8f}"
            print(f"- {name} 梯度: {grad_info}")
        else:
            print(f"- {name} 梯度: None")
    
    print("\n检查完成!")

if __name__ == "__main__":
    check_phase_gradients()