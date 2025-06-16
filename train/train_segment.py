import sys
sys.path.append(".")

import os
import time
import random
from tqdm import tqdm
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from aPyOpenGL import transforms as trf
from utils import utils, loss, ops
from utils.dataset import SegmentDataset, MotionDataset
from model.segment_net import SegmentNet
# 移除 NoamScheduler 导入

def train_iter(batch, model, skeleton, config, mean, std, traj_mean, traj_std, loss_dict, device, train=True):
    # 1. 直接从dataloader获取一个批次的、已经处理好的、形状统一的片段数据
    draft_motion = batch["draft_motion"].to(device)
    gt_motion_raw = batch["gt_motion"].to(device) # GT Motion in raw space
    overlap_frames = batch["overlap_frames"] # (B,) tensor on CPU

    # 获取可选数据
    phase = batch.get("phase", None)
    traj = batch.get("traj", None)
    gt_contact = batch.get("contact", None)
    if phase is not None: phase = phase.to(device)
    if traj is not None: traj = traj.to(device)
    if gt_contact is not None: gt_contact = gt_contact.to(device)

    B, T, M = draft_motion.shape
    
    # 2. 对整个批次进行一次归一化
    draft_motion_norm = (draft_motion - mean) / std
    
    if config.use_traj and traj is not None and traj_mean is not None and traj_std is not None:
        traj_norm = (traj - traj_mean) / traj_std
    else:
        traj_norm = None

    # 3. 准备模型输入
    midway_targets = [overlap_frames[0].item() - 1]
    
    # 4. 前向传播
    # 注意：model 可能已经是DDP封装后的模型
    det_out = model(draft_motion_norm, midway_targets, phase=phase, traj=traj_norm)

    pred_motion_norm = det_out["motion"]
    pred_contact = det_out["contact"]
    if config.use_phase:
        pred_phase = det_out.get("phase")

    # 5. 创建损失掩码
    loss_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    for b in range(B):
        loss_mask[b, overlap_frames[b].item():-1] = True
    
    rot_mask = loss_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, skeleton.num_joints, 6)
    pos_mask = loss_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, skeleton.num_joints, 3)
    
    # 6. 为损失计算进行反归一化
    pred_motion_denorm = pred_motion_norm * std + mean
    pred_local_ortho6ds, pred_root_pos = torch.split(pred_motion_denorm, [M-3, 3], dim=-1)
    pred_local_ortho6ds = pred_local_ortho6ds.reshape(B, T, skeleton.num_joints, 6)
    _, pred_global_positions = trf.t_ortho6d.fk(pred_local_ortho6ds, pred_root_pos, skeleton)
    
    gt_local_ortho6ds, gt_root_pos = torch.split(gt_motion_raw, [M-3, 3], dim=-1)
    gt_local_ortho6ds = gt_local_ortho6ds.reshape(B, T, skeleton.num_joints, 6)
    _, gt_global_positions = trf.t_ortho6d.fk(gt_local_ortho6ds, gt_root_pos, skeleton)

    # 7. 计算损失
    loss_rot = loss.masked_rot_loss(pred_local_ortho6ds, gt_local_ortho6ds, rot_mask)
    loss_pos = loss.masked_pos_loss(pred_global_positions, gt_global_positions, pos_mask)
    total_loss = config.weight_rot * loss_rot + config.weight_pos * loss_pos
    
    if gt_contact is not None:
        contact_mask = loss_mask.unsqueeze(-1).expand(-1, -1, pred_contact.shape[-1])
        loss_contact = loss.masked_contact_loss(pred_contact, gt_contact, contact_mask)
        total_loss += config.weight_contact * loss_contact
        loss_dict["contact"] += loss_contact.item()

    if config.use_phase and phase is not None and pred_phase is not None:
        phase_mask = loss_mask.unsqueeze(-1).expand(-1, -1, phase.shape[-1])
        loss_phase = loss.masked_phase_loss(pred_phase, phase, phase_mask)
        total_loss += config.weight_phase * loss_phase
        loss_dict["phase"] += loss_phase.item()

    if config.use_traj and traj is not None:
        traj_denorm = traj * traj_std + traj_mean if traj_std is not None else traj
        pred_traj = ops.motion_to_traj(pred_motion_denorm)
        loss_traj = loss.masked_traj_loss(pred_traj, traj_denorm, loss_mask)
        total_loss += config.weight_traj * loss_traj
        loss_dict["traj"] += loss_traj.item()
    
    loss_dict["rot"] += loss_rot.item()
    loss_dict["pos"] += loss_pos.item()
    loss_dict["total"] += total_loss.item()
    
    return total_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default="segment.yaml")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser.add_argument("--gpu", type=int, default=0, help="单卡模式下使用的GPU ID")
    return parser.parse_args()

def main_worker(rank, world_size, args, mean, std, traj_mean, traj_std):
    # 1. DDP 初始化与设备设置
    is_ddp = world_size > 1
    if is_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ==================== 调试点 1 ====================
    if is_ddp:
        print(f"Rank {rank}: 已完成DDP初始化，准备进入第一个屏障。")
        dist.barrier()
        print(f"Rank {rank}: 已通过第一个屏障。")
    # ===============================================

    # 将统计数据移动到当前GPU
    mean, std = mean.to(device), std.to(device)
    if traj_mean is not None and traj_std is not None:
        traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)
    
    config = utils.load_config(f"config/{args.dataset}/{args.config}")
    utils.seed(args.seed + rank)

    if rank == 0:
        os.makedirs(config.save_dir, exist_ok=True)
        utils.write_config(config)
        writer = SummaryWriter(config.save_dir)
    else:
        writer = None

    # 2. 创建数据集、DataLoader
    dataset = SegmentDataset(train=True, config=config, verbose=(rank == 0))
    sampler = DistributedSampler(dataset) if is_ddp else None
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=(sampler is None), num_workers=0, pin_memory=True, sampler=sampler)  # 保持 num_workers=0
    
    val_dataset = SegmentDataset(train=False, config=config, verbose=(rank == 0))
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=val_sampler)  # 保持 num_workers=0
    
    skeleton = dataset.skeleton
    
    # ==================== 调试点 2 ====================
    if is_ddp:
        print(f"Rank {rank}: 已完成模型创建并移至设备，准备进入第二个屏障。")
        dist.barrier()
        print(f"Rank {rank}: 已通过第二个屏障。")
    # ===============================================
    
    # 3. 创建模型、优化器（使用固定学习率）
    model = SegmentNet(config, dataset).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[rank])

    # 根据是否为DDP环境调整学习率
    effective_lr = config.lr * world_size if is_ddp else config.lr
    optimizer = Adam(model.parameters(), lr=effective_lr)
    init_epoch = 0 # 明确从0开始
    
    # ==================== 调试点 3 (最关键) ====================
    if is_ddp:
        print(f"Rank {rank}: 已完成所有准备工作，准备进入最终屏障。")
        dist.barrier()
        if rank == 0:
            print("所有进程已同步，即将开始训练循环...")
    # =======================================================

    # 4. 训练循环
    for epoch in range(init_epoch + 1, config.epochs + 1):
        if is_ddp:
            sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        model.train()
        loss_dict = { "rot": 0.0, "pos": 0.0, "contact": 0.0, "total": 0.0, "phase": 0.0, "traj": 0.0 }
        
        train_iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}", disable=(rank != 0))
        
        if is_ddp and epoch == 1:
            print(f"Rank {rank}: 即将开始处理第一个批次数据")
        
        for i, batch in enumerate(train_iterator):
            if is_ddp and epoch == 1 and i == 0:
                print(f"Rank {rank}: 成功获取到第一个批次，开始处理")
            total_loss = train_iter(batch, model, skeleton, config, mean, std, traj_mean, traj_std, loss_dict, device, train=True)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # 移除 scheduler.step()
        
        # ==========================================================
        # 关键修正：修正日志聚合与验证逻辑
        # ==========================================================
        
        # 1. 首先，所有进程都必须参与损失的汇总 (all_reduce)
        #    为了确保所有进程都完成了训练迭代，在这里加一个屏障
        if is_ddp:
            dist.barrier()
            for key in loss_dict.keys():
                loss_tensor = torch.tensor(loss_dict[key], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                # 注意：所有进程都会得到总和，但我们只在rank 0使用它
                if rank == 0:
                    loss_dict[key] = loss_tensor.item()
        
        # 2. 然后，只有主进程 (rank 0) 负责计算平均值和写入日志
        if rank == 0:
            num_batches_total = len(dataloader) * world_size if is_ddp else len(dataloader)
            for key in loss_dict.keys():
                # 平均每个batch的损失
                loss_dict[key] /= num_batches_total

            utils.write_log(writer, loss_dict, 1, epoch, train=True)

        # --- 验证 (所有进程都需要参与数据加载和模型前向传播) ---
        if epoch % config.val_interval == 0:
            model.eval()
            # 重置验证集的损失字典
            val_loss_dict = { "rot": 0.0, "pos": 0.0, "contact": 0.0, "total": 0.0, "phase": 0.0, "traj": 0.0 }
            
            with torch.no_grad():
                val_iterator = tqdm(val_dataloader, desc=f"Validation", disable=(rank != 0))
                for batch in val_iterator:
                    # 所有进程都执行验证迭代
                    train_iter(batch, model, skeleton, config, mean, std, traj_mean, traj_std, val_loss_dict, device, train=False)

            # 验证结束后，再次进行同步和日志记录
            if is_ddp:
                dist.barrier()
                for key in val_loss_dict.keys():
                    loss_tensor = torch.tensor(val_loss_dict[key], device=device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    if rank == 0:
                        val_loss_dict[key] = loss_tensor.item()

            if rank == 0:
                num_val_batches_total = len(val_dataloader) * world_size if is_ddp else len(val_dataloader)
                for key in val_loss_dict.keys():
                    val_loss_dict[key] /= num_val_batches_total
                utils.write_log(writer, val_loss_dict, 1, epoch, train=False)

            # --- 保存检查点 (只在主进程) ---
            if rank == 0 and epoch % config.save_interval == 0:
                model_to_save = model.module if is_ddp else model
                utils.save_ckpt(model_to_save, optimizer, epoch, config)
                
            # 验证结束后的同步
            if is_ddp:
                dist.barrier()

    # 保存最终模型
    if rank == 0:
        model_to_save = model.module if is_ddp else model
        utils.save_ckpt(model_to_save, optimizer, config.epochs, config)
    
    # ==========================================================
    # 关键修正：在训练完全结束前，确保所有进程同步
    # ==========================================================
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()

def main():
    args = parse_args()
    
    # 自动检测DDP环境
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        # 由 torchrun 启动
        # 在主进程中预加载统计数据
        print("在主进程中预加载统计数据...")
        temp_device = torch.device("cpu")
        config_for_stats = utils.load_config(f"config/{args.dataset}/{args.config}")
        base_dataset_for_stats = MotionDataset(train=True, config=config_for_stats, verbose=False)
        mean, std = base_dataset_for_stats.motion_statistics(temp_device)
        traj_mean, traj_std = base_dataset_for_stats.traj_statistics(temp_device)
        print("统计数据预加载完成。")
        
        # torchrun 已经创建了进程，我们直接调用 main_worker
        main_worker(int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), args, mean, std, traj_mean, traj_std)
    else:
        # 由 python 直接启动，进入单卡模式
        print("未检测到DDP环境，以单卡模式运行...")
        # 预加载统计数据
        temp_device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        config_for_stats = utils.load_config(f"config/{args.dataset}/{args.config}")
        base_dataset_for_stats = MotionDataset(train=True, config=config_for_stats, verbose=True)
        mean, std = base_dataset_for_stats.motion_statistics(temp_device)
        traj_mean, traj_std = base_dataset_for_stats.traj_statistics(temp_device)
        
        main_worker(0, 1, args, mean, std, traj_mean, traj_std)

if __name__ == "__main__":
    main()