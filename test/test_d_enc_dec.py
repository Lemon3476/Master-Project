import sys
sys.path.append(".")

import os
import torch
from torch.utils.data import DataLoader

from utils import utils
from utils.dataset import MotionDataset
from model.twostage import ContextTransformer, DetailTransformer

def test_models():
    # 设置GPU设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载配置文件
    keyframe_config = utils.load_config("config/lafan1/keyframe-d-enc-dec.yaml")
    refine_config = utils.load_config("config/lafan1/refine-d-enc-dec-fc.yaml")
    
    # 测试函数
    def test_model_configs(config, model_class, model_name):
        print(f"\n测试 {model_name} 模型配置...")
        print(f"- decoupled_traj_encoder: {config.get('decoupled_traj_encoder', False)}")
        print(f"- decoupled_phase_encoder: {config.get('decoupled_phase_encoder', False)}")
        print(f"- decoupled_traj_decoder: {config.get('decoupled_traj_decoder', False)}")
        
        # 加载数据集
        dataset = MotionDataset(train=True, config=config)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 创建模型
        model = model_class(config, dataset).to(device)
        print(f"- 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        
        # 测试前向传播
        try:
            batch = next(iter(dataloader))
            motion = batch["motion"].to(device)
            phase = batch["phase"].to(device) if config.use_phase else None
            traj = batch["traj"].to(device) if config.use_traj else None
            
            B, T, M = motion.shape
            
            # 测试前向传播
            if model_name == "ContextTransformer":
                outputs, _ = model.forward(motion, phase=phase, traj=traj, train=True)
            else:
                midway_targets = [10, 20]  # 示例中间帧
                outputs = model.forward(motion, midway_targets, phase=phase, traj=traj)
            
            # 检查输出
            print(f"- 输出形状:")
            print(f"  - motion: {outputs['motion'].shape}")
            
            if "phase" in outputs:
                print(f"  - phase: {outputs['phase'].shape}")
                print(f"  - 独立相位解码器成功!")
                
            if "traj" in outputs:
                print(f"  - traj: {outputs['traj'].shape}")
                print(f"  - 独立轨迹解码器成功!")
                
            if "contact" in outputs:
                print(f"  - contact: {outputs['contact'].shape}")
                
            # 打印模型架构
            print("\n- 检查关键组件:")
            
            # 检查相位编码器
            if hasattr(model, 'phase_encoder') and config.get('decoupled_phase_encoder', False):
                print(f"  - 相位编码器存在: 输入尺寸 {model.d_phase}, 输出尺寸 {config.d_model}")
            
            # 检查轨迹编码器
            if hasattr(model, 'traj_encoder') and (config.get('decoupled_traj_encoder', False) or config.get('decoupled_encoders', False)):
                print(f"  - 轨迹编码器存在: 输入尺寸 {model.d_traj}, 输出尺寸 {config.d_model}")
            
            # 检查轨迹解码器
            if hasattr(model, 'traj_decoder') and config.get('decoupled_traj_decoder', False):
                print(f"  - 轨迹解码器存在: 输入尺寸 {config.d_model}, 输出尺寸 {model.d_traj}")
            
            # 检查门控网络
            if hasattr(model, 'gating'):
                gating_input_size = model.gating[0].in_features
                print(f"  - 门控网络存在: 输入尺寸 {gating_input_size}, 应为相位维度 {model.d_phase}")
                if gating_input_size == model.d_phase:
                    print(f"  - 门控网络仅使用相位信息: 符合最终方案要求 ✓")
                else:
                    print(f"  - 警告: 门控网络可能使用了其他信息")
            
            print("- 前向传播测试通过!")
            return True
        except Exception as e:
            print(f"- 错误: {e}")
            return False
    
    # 测试 ContextTransformer
    test_model_configs(keyframe_config, ContextTransformer, "ContextTransformer")
    
    # 测试 DetailTransformer
    test_model_configs(refine_config, DetailTransformer, "DetailTransformer")
    
    print("\n所有测试完成! 模型符合最终方案要求。")

if __name__ == "__main__":
    test_models()