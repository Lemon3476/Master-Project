import torch
import sys
import os

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.twostage import ContextTransformer, DetailTransformer
from utils.dataset import MotionDataset
from utils import utils

# The test function is identical to the one in the other script
def test_full_pipeline(kf_config_path, ref_config_path):
    """
    加载KeyframeNet和RefineNet的配置，并串联执行一次前向传播测试。
    """
    print(f"--- Running Pipeline Test ---")
    print(f"KeyframeNet Config: {kf_config_path}")
    print(f"RefineNet Config:   {ref_config_path}")

    # 1. 加载两个模型的配置
    try:
        kf_config = utils.load_config(kf_config_path)
        ref_config = utils.load_config(ref_config_path)
        print("\nConfigs loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load config files: {e}")
        return

    # 2. 准备数据集实例
    try:
        dataset = MotionDataset(train=True, config=kf_config, verbose=False)
        print("Dataset instance created successfully for model initialization.")
    except Exception as e:
        print(f"[ERROR] Failed to create dataset instance: {e}")
        return
    
    # 3. 初始化两个模型
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keyframe_net = ContextTransformer(kf_config, dataset).to(device)
        refine_net = DetailTransformer(ref_config, dataset).to(device)
        keyframe_net.eval()
        refine_net.eval()
        print(f"\nModels initialized successfully on device: {device}.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize models: {e}")
        return

    # 4. 创建虚拟输入数据
    batch_size = 4
    window = kf_config.window_length
    motion = torch.randn(batch_size, window, dataset.motion_dim).to(device)
    phase = torch.randn(batch_size, window, dataset.phase_dim).to(device) if kf_config.use_phase else None
    traj = torch.randn(batch_size, window, dataset.traj_dim).to(device) if kf_config.use_traj else None
    
    print("\nDummy input tensors created.")

    # 5. 执行 KeyframeNet 的前向传播
    try:
        with torch.no_grad():
            kf_res, midway_targets = keyframe_net(motion, phase=phase, traj=traj, train=False)
        print("\n--- KeyframeNet Forward Pass Successful ---")
        print("Output shapes:")
        for key, value in kf_res.items():
            if value is not None: print(f"  - {key}: {value.shape}")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during the KeyframeNet forward pass: {e}")
        return
        
    # 6. 准备 RefineNet 的输入
    s_R = kf_res['motion'] 
    p_K = kf_res['phase'] if ref_config.use_phase else None
    
    # 7. 执行 RefineNet 的前向传播
    try:
        with torch.no_grad():
            refine_res = refine_net(s_R, midway_targets, phase=p_K, traj=traj)
        print("\n--- RefineNet Forward Pass Successful ---")
        print("Output shapes:")
        for key, value in refine_res.items():
            if value is not None: print(f"  - {key}: {value.shape}")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] An error occurred during the RefineNet forward pass:")
        traceback.print_exc()
        return

    print(f"\n--- Pipeline Test PASSED ---")

if __name__ == '__main__':
    # **唯一的不同**: 加载了不同的配置文件
    kf_config_path = 'config/lafan1/keyframe-d-enc-dec.yaml'
    ref_config_path = 'config/lafan1/refine-d-enc-dec-fc.yaml' # 假设RefineNet也使用对应的全解耦配置
    
    if not os.path.exists(kf_config_path) or not os.path.exists(ref_config_path):
        print(f"[FATAL] One or both config files not found.")
    else:
        test_full_pipeline(kf_config_path, ref_config_path)