import sys
sys.path.append(".")

import os
import time
import random
from tqdm import tqdm
import argparse

from aPyOpenGL import transforms as trf

import torch
# 启用TF32高精度以提高训练速度（仅在支持的GPU如A6000上生效）
torch.set_float32_matmul_precision('high')

# 启用CUDA性能优化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import utils, loss, ops
from utils.dataset import MotionDataset
from model.twostage import ContextTransformer, DetailTransformer
from model.scheduler import NoamScheduler

def train_model(args):
    # 设置设备和配置
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    config = utils.load_config(f"config/{args.dataset}/{args.config}")
    utils.seed()

    # 打印GPU信息和当前配置
    print(f"使用设备: {device}")
    print(f"GPU信息: {torch.cuda.get_device_name(device)}")
    print(f"最大内存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    
    print(f"配置: {args.config}")
    if hasattr(config, "decoupled_traj_encoder"):
        print(f"使用分离的轨迹编码器: {config.decoupled_traj_encoder}")
    if hasattr(config, "decoupled_phase_encoder"):
        print(f"使用分离的相位编码器: {config.decoupled_phase_encoder}")
    
    # 加载数据集
    print("加载训练数据集...")
    dataset = MotionDataset(train=True, config=config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                           shuffle=True, num_workers=8, pin_memory=True,
                           persistent_workers=True)
    
    # 加载模型
    print("初始化模型...")
    model = DetailTransformer(config, dataset).to(device)

    # 尝试使用torch.compile编译模型（如果PyTorch版本支持）
    if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
        try:
            print("尝试使用torch.compile优化模型...")
            model = torch.compile(model, mode='reduce-overhead')
            print("模型已使用torch.compile优化")
        except Exception as e:
            print(f"torch.compile失败: {e}")
            print("继续使用未编译的模型")
    
    # 优化器
    optimizer = Adam(model.parameters(), lr=0)  # lr由调度器设置
    scheduler = NoamScheduler(optimizer, config.d_model, config.warmup_iters)
    
    # 训练循环
    print("开始训练...")
    num_iterations = 100  # 只训练几个迭代进行测试
    
    # 模拟训练
    model.train()
    total_time = 0
    batch_times = []
    
    for i in range(num_iterations):
        # 从数据加载器获取一个批次
        for batch in dataloader:
            # 测量时间
            start_time = time.time()
            
            # 模拟forward pass
            motion = batch["motion"].to(device)
            phase = batch["phase"].to(device) if config.use_phase else None
            traj = batch["traj"].to(device) if config.use_traj else None
            
            # 随机生成中间帧索引
            midway_targets = [random.randint(config.context_frames, motion.shape[1]-2)]
            
            # 前向传播
            outputs = model.forward(motion, midway_targets, phase=phase, traj=traj)
            
            # 计算损失
            loss_val = torch.mean(outputs["motion"])
            
            # 反向传播
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            scheduler.step()
            
            # 计算批次时间
            batch_time = time.time() - start_time
            total_time += batch_time
            batch_times.append(batch_time)
            
            # 打印进度
            if i % 10 == 0:
                print(f"Iteration {i}/{num_iterations}, Batch time: {batch_time:.4f}s, "
                     f"GPU Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB, "
                     f"GPU Utilization: {torch.cuda.utilization(device)}%")
            
            break  # 只使用每个epoch的第一个批次
    
    # 打印统计信息
    avg_time = total_time / num_iterations
    print(f"\n性能统计:")
    print(f"总运行时间: {total_time:.2f}s")
    print(f"平均批次时间: {avg_time:.4f}s")
    print(f"吞吐量: {config.batch_size / avg_time:.2f} 样本/秒")
    
    # 返回性能统计
    return {
        "avg_batch_time": avg_time,
        "throughput": config.batch_size / avg_time,
        "batch_times": batch_times
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lafan1")
    parser.add_argument("--config", type=str, default="refine-traj-phase-encoder-foot-consistency.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    # 运行测试
    stats = train_model(args)