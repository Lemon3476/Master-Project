import os
import torch
import numpy as np
import random
from tqdm import tqdm
from omegaconf import OmegaConf

def load_config(filepath):
    cfg = OmegaConf.load(filepath)
    if cfg.get("context_frames", None) is None:
        cfg.npz_path = f"length{cfg.window_length}-offset{cfg.window_offset}.npz"
    else:
        cfg.npz_path = f"length{cfg.window_length}-offset{cfg.window_offset}-context{cfg.context_frames}.npz"
    cfg.skeleton_path = os.path.join(cfg.dataset_dir, "skeleton.pkl")
    return cfg

def write_config(config):
    with open(os.path.join(config.save_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)

def seed(x=1234):  # Default seed set to 1234
    torch.manual_seed(x)
    torch.cuda.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(x)
    random.seed(x)

def write_log(writer, loss_dict, interval, iter, elapsed=None, train=True):
    msg = f"{'Train' if train else 'Val'} at {iter}: "
    for key, value in loss_dict.items():
        writer.add_scalar(f"{'train' if train else 'val'}/{key}", value / interval, iter)
        msg += f"{key}: {value / interval:.4f} | "
    if elapsed is not None:
        msg += f"Time: {(elapsed / 60):.2f} min"
    tqdm.write(msg)

def reset_log(loss_dict):
    for key in loss_dict.keys():
        loss_dict[key] = 0

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from state_dict keys if they exist.
    This allows using a model trained with DDP in non-DDP mode.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Modified state dictionary with 'module.' prefix removed
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix (length=7)
        else:
            new_state_dict[k] = v
    return new_state_dict

def load_model(model, config, epoch=None):
    try:
        ckpt_list = os.listdir(config.save_dir)
        if len(ckpt_list) > 0:
            ckpt_list = [ckpt for ckpt in ckpt_list if ckpt.endswith(".pth")]
            if len(ckpt_list) > 0:
                ckpt_list = sorted(ckpt_list)
                if epoch is None:
                    ckpt_path = os.path.join(config.save_dir, ckpt_list[-1])
                    ckpt = torch.load(ckpt_path, map_location="cuda:0")
                    state_dict = remove_module_prefix(ckpt["model"])
                    
                    # Handle the case where current model is wrapped in DDP but saved model is not
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        model.module.load_state_dict(state_dict)
                    else:
                        model.load_state_dict(state_dict)
                        
                    print(f"> Loaded checkpoint: {ckpt_path}")
                    return 0  # Successfully loaded
                else:
                    ckpt_path = os.path.join(config.save_dir, f"ckpt_{epoch:04d}.pth")
                    ckpt = torch.load(ckpt_path, map_location="cuda:0")
                    state_dict = remove_module_prefix(ckpt["model"])
                    
                    # Handle the case where current model is wrapped in DDP but saved model is not
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        model.module.load_state_dict(state_dict)
                    else:
                        model.load_state_dict(state_dict)
                        
                    print(f"> Loaded checkpoint: {ckpt_path}")
                    return 0  # Successfully loaded
            else:
                print(f"WARNING: No checkpoint (.pth) files found in {config.save_dir}")
                return -1  # No checkpoint found
        else:
            print(f"WARNING: No files found in {config.save_dir}")
            return -1  # No checkpoint found
    except Exception as e:
        print(f"WARNING: Error loading checkpoint: {e}")
        return -1  # Error loading checkpoint
    
def load_latest_ckpt(model, optim, config, scheduler=None, rank=0):
    """
    加载最新的检查点，自动处理DDP状态字典，适配不同进程的GPU设备。
    
    Args:
        model: 模型实例
        optim: 优化器实例
        config: 配置对象
        scheduler: 可选的学习率调度器
        rank: 当前进程的rank，用于确定加载到哪个GPU上
        
    Returns:
        int: 加载的检查点epoch，或者0表示没有找到检查点
    """
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
        
    ckpt_list = os.listdir(config.save_dir)
    ckpt_list = [f for f in ckpt_list if f.endswith(".pth")]
    ckpt_list = sorted(ckpt_list)
    if len(ckpt_list) > 0:
        ckpt_path = os.path.join(config.save_dir, ckpt_list[-1])
        
        # 动态指定map_location，确保张量加载到正确的GPU上
        map_location = f'cuda:{rank}'
        ckpt = torch.load(ckpt_path, map_location=map_location)
        
        # 处理模块前缀
        state_dict = remove_module_prefix(ckpt["model"])
        
        # 处理DDP模型
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
            
        optim.load_state_dict(ckpt["optim"])
        epoch = ckpt["epoch"]
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"> Rank {rank}: 检查点已加载: {ckpt_path}, epoch: {epoch}")
    else:
        epoch = 0
        print(f"> Rank {rank}: 未找到检查点，从头开始训练。")

    return epoch

def save_ckpt(model, optim, epoch, config, scheduler=None):
    ckpt_path = os.path.join(config.save_dir, f"ckpt_{epoch:04d}.pth")
    ckpt = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    
    torch.save(ckpt, ckpt_path)
    print(f"> Saved checkpoint at epoch {epoch}: {ckpt_path}")