# test/test_traj_enc_dec.py (修改后)

import torch
import sys
import os

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.twostage import ContextTransformer, DetailTransformer
from utils.dataset import MotionDataset
from utils import utils

def check_gradients(model: torch.nn.Module, model_name: str):
    """
    遍历模型参数并打印其梯度状态的辅助函数。
    """
    print(f"\n--- 正在检查模型 '{model_name}' 的梯度 ---")
    all_grads_present = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                print(f"  [✔] {name:<65} | 梯度已存在 (范数: {param.grad.norm():.2e})")
            else:
                print(f"  [❌] {name:<65} | !!! 梯度缺失或为零 !!!")
                all_grads_present = False
    
    if all_grads_present:
        print(f"\n[成功] '{model_name}' 的所有参数都接收到了梯度。")
    else:
        print(f"\n[失败] '{model_name}' 中有部分参数未能接收到梯度！")
    print("-" * 80)


def test_full_pipeline_with_gradient_check(kf_config_path, ref_config_path):
    """
    加载KeyframeNet和RefineNet，并执行一次完整的前向和反向传播来检查梯度。
    """
    print(f"--- 运行方案一梯度检查 ---")
    print(f"KeyframeNet Config: {kf_config_path}")
    print(f"RefineNet Config:   {ref_config_path}")

    # 1. 加载配置和数据集
    try:
        kf_config = utils.load_config(kf_config_path)
        ref_config = utils.load_config(ref_config_path)
        dataset = MotionDataset(train=True, config=kf_config, verbose=False)
        print("\n配置和数据集加载成功。")
    except Exception as e:
        print(f"[错误] 加载配置或数据集失败: {e}")
        return
    
    # 2. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keyframe_net = ContextTransformer(kf_config, dataset).to(device)
    refine_net = DetailTransformer(ref_config, dataset).to(device)
    
    # 设置为训练模式以计算梯度
    keyframe_net.train()
    refine_net.train()
    print(f"\n模型初始化成功，设备: {device}。已设置为训练模式。")

    # 3. 创建虚拟输入数据
    batch_size = 4
    window = kf_config.window_length
    motion = torch.randn(batch_size, window, dataset.motion_dim, device=device)
    phase = torch.randn(batch_size, window, dataset.phase_dim, device=device) if kf_config.use_phase else None
    traj = torch.randn(batch_size, window, dataset.traj_dim, device=device) if kf_config.use_traj else None
    
    print("\n已创建虚拟输入张量。")

    # --- 4. 测试 KeyframeNet ---
    print("\n--- 开始测试 KeyframeNet ---")
    # 清零梯度
    keyframe_net.zero_grad()
    
    # 前向传播
    kf_res, midway_targets = keyframe_net(motion, phase=phase, traj=traj, train=True)
    
    # 计算一个标量损失 (将所有输出求和)
    total_loss_kf = 0
    for key, value in kf_res.items():
        if value is not None and value.requires_grad:
            total_loss_kf += value.sum()
    
    print(f"KeyframeNet 计算损失值: {total_loss_kf.item():.4f}")
    
    # 反向传播
    total_loss_kf.backward()
    
    # 检查梯度
    check_gradients(keyframe_net, "KeyframeNet (方案一)")

    # --- 5. 测试 RefineNet ---
    print("\n--- 开始测试 RefineNet ---")
    # 清零梯度
    refine_net.zero_grad()
    
    # 准备输入 (注意要 detach() 从KeyframeNet来的张量，避免梯度重复计算)
    s_R = kf_res['motion'].detach()
    p_K = kf_res['phase'].detach() if ref_config.use_phase and kf_res.get('phase') is not None else None
    
    # 前向传播
    refine_res = refine_net(s_R, midway_targets, phase=p_K, traj=traj.detach() if traj is not None else None)
    
    # 计算损失
    total_loss_ref = 0
    for key, value in refine_res.items():
        if value is not None and value.requires_grad:
            total_loss_ref += value.sum()
            
    print(f"RefineNet 计算损失值: {total_loss_ref.item():.4f}")

    # 反向传播
    total_loss_ref.backward()
    
    # 检查梯度
    check_gradients(refine_net, "RefineNet (方案一)")


if __name__ == '__main__':
    kf_config_path = 'config/lafan1/keyframe-traj-enc-dec.yaml'
    ref_config_path = 'config/lafan1/refine-traj-enc-dec-fc.yaml'
    
    if not os.path.exists(kf_config_path) or not os.path.exists(ref_config_path):
        print(f"[致命错误] 一个或多个配置文件未找到。")
    else:
        test_full_pipeline_with_gradient_check(kf_config_path, ref_config_path)