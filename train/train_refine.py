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

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import utils, loss, ops
from utils.dataset import MotionDataset
from model.twostage import ContextTransformer, DetailTransformer
from model.scheduler import NoamScheduler

if __name__ =="__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default="refine.yaml")
    parser.add_argument("--ctx_config", type=str, default="keyframe.yaml")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    config = utils.load_config(f"config/{args.dataset}/{args.config}")
    utils.seed()

    # dataset
    dataset = MotionDataset(train=True, config=config)
    # dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    skeleton = dataset.skeleton

    mean, std = dataset.motion_statistics()
    mean, std = mean.to(device), std.to(device)

    traj_mean, traj_std = dataset.traj_statistics()
    traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)

    val_dataset = MotionDataset(train=False, config=config)
    # val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)


    contact_idx = []
    for joint in config.contact_joints:
        contact_idx.append(skeleton.idx_by_name[joint])

    # load trained ContextTransformer
    ctx_config = utils.load_config(f"config/{args.dataset}/{args.ctx_config}")
    ctx_model = ContextTransformer(ctx_config, dataset).to(device)
    utils.load_model(ctx_model, ctx_config)
    ctx_model.eval()


    # model, optimizer, scheduler
    model = DetailTransformer(config, dataset).to(device)
    optimizer = Adam(model.parameters(), lr=0) # lr is set by scheduler
    scheduler = NoamScheduler(optimizer, config.d_model, config.warmup_iters)
    init_epoch = utils.load_latest_ckpt(model, optimizer, config, scheduler=scheduler)
    epoch = init_epoch  # Initialize epoch variable


    # save and log
    os.makedirs(config.save_dir, exist_ok=True)
    utils.write_config(config)
    writer = SummaryWriter(config.save_dir)
    loss_dict = {
        "rot": 0.0,
        "pos": 0.0,
        "contact": 0.0,
        "total": 0.0,
    }
    if config.use_phase:
        loss_dict["phase"] = 0.0
    if config.use_traj:
        loss_dict["traj"] = 0.0
    if config.get("use_foot_consistency", False):
        loss_dict["foot_consistency"] = 0.0
    if config.get("use_dynamic_loss_weighting", False):
        loss_dict["pose_weight"] = 0.0
        loss_dict["traj_weight"] = 0.0

    # function for each iteration
    def train_iter(batch, train=True):
        # transitiion length
        trans_len = random.randint(config.min_trans, config.max_trans) if train else config.max_trans
        target_idx = config.context_frames + trans_len
        
        # GT data
        GT_motion = batch["motion"].to(device)
        GT_phase  = batch["phase"].to(device) if config.use_phase else None
        GT_traj   = batch["traj"].to(device)  if config.use_traj  else None
        GT_score  = batch["score"].to(device) if config.use_score else None

        GT_motion = GT_motion[:, :target_idx+1].to(device)
        if config.use_phase:
            GT_phase = GT_phase[:, :target_idx+1].to(device)
        if config.use_traj:
            GT_traj = GT_traj[:, :target_idx+1].to(device)
        if config.use_score:
            GT_score = GT_score[:, :target_idx+1].to(device)

        B, T, M = GT_motion.shape
        
        # 分割和处理输入数据
        GT_local_ortho6ds, GT_root_pos = torch.split(GT_motion, [M-3, 3], dim=-1)
        GT_local_ortho6ds = GT_local_ortho6ds.reshape(B, T, skeleton.num_joints, 6)
        _, GT_global_positions = trf.t_ortho6d.fk(GT_local_ortho6ds, GT_root_pos, skeleton)
        
        GT_foot_vel = GT_global_positions[:, 1:, contact_idx] - GT_global_positions[:, :-1, contact_idx]
        GT_foot_vel = torch.sum(GT_foot_vel ** 2, dim=-1) # (B, t-1, 4)
        GT_foot_vel = torch.cat([GT_foot_vel[:, 0:1], GT_foot_vel], dim=1) # (B, t, 4)
        GT_contact  = (GT_foot_vel < config.contact_threshold).float() # (B, t, 4)

        # forward ContextTransformer
        GT_motion = (GT_motion - mean) / std
        if config.use_traj:
            GT_traj = (GT_traj - traj_mean) / traj_std
        with torch.no_grad():
            ctx_out, midway_targets = ctx_model.forward(GT_motion, phase=GT_phase, traj=GT_traj, train=train)

            # output of ContextTransformer
            ctx_motion = ctx_out["motion"]
            if config.use_phase:
                ctx_phase = ctx_out["phase"]
            if config.use_score:
                ctx_score = ctx_out["score"]
            
            # restore constrained frames
            pred_ctx_motion = GT_motion.clone().detach()
            pred_ctx_motion[:, config.context_frames:-1] = ctx_motion[:, config.context_frames:-1]
            pred_ctx_motion[:, midway_targets] = GT_motion[:, midway_targets]

            if config.use_phase:
                pred_ctx_phase = GT_phase.clone().detach()
                pred_ctx_phase[:, config.context_frames:-1] = ctx_phase[:, config.context_frames:-1]
                pred_ctx_phase[:, midway_targets] = GT_phase[:, midway_targets]
            else:
                pred_ctx_phase = None

            if config.use_score:
                pred_score = GT_score.clone().detach()
                pred_score[:, config.context_frames:-1] = ctx_score[:, config.context_frames:-1]

                pred_ctx_motion = pred_ctx_motion * std + mean
                
                # 为每个批次生成关键帧
                keyframes_list = []
                for b in range(B):
                    kfs = ops.get_random_keyframe(config, T)
                    keyframes_list.append(kfs)
                
                # 检查是否跳过线性插值
                if hasattr(config, 'skip_interpolation') and config.skip_interpolation:
                    # 跳过插值模式：保留关键帧，其他帧保持原样（将用于训练模型直接从稀疏关键帧生成）
                    if i == 0 and epoch % 10 == 0:
                        print(f"Training with sparse keyframes (skipping interpolation), epoch {epoch}")
                    
                    # 创建关键帧掩码，稍后用于在损失计算中防止修改关键帧
                    keyframe_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
                    for b in range(B):
                        for kf in keyframes_list[b]:
                            if kf < T:
                                keyframe_mask[b, kf] = True
                    
                    # 保存掩码供后续使用
                    batch["keyframe_mask"] = keyframe_mask
                    
                    # 不做插值，直接将具有稀疏关键帧的序列传给模型
                else:
                    # 正常模式：应用线性插值
                    # 对每个批次进行插值
                    for b in range(B):
                        pred_ctx_motion[b:b+1] = ops.interpolate_motion_by_keyframes(pred_ctx_motion[b:b+1], keyframes_list[b])
                    
                    # 创建一个空掩码（不使用）
                    batch["keyframe_mask"] = None
                
                pred_ctx_motion = (pred_ctx_motion - mean) / std

        # forward DetailTransformer
        det_out = model.forward(pred_ctx_motion, midway_targets, phase=pred_ctx_phase, traj=GT_traj)
        
        det_motion = det_out["motion"]
        det_contact = det_out["contact"]
        if config.use_phase:
            det_phase = det_out["phase"]
        
        # restore constrained frames
        pred_det_motion = GT_motion.clone().detach()
        
        # 正常模式或跳过插值但不使用掩码的情况
        if not hasattr(config, 'skip_interpolation') or not config.skip_interpolation or batch.get("keyframe_mask") is None:
            # 常规处理：只保持上下文帧和结尾帧不变
            pred_det_motion[:, config.context_frames:-1] = det_motion[:, config.context_frames:-1]
        else:
            # 跳过插值模式：保留关键帧不变，其他帧使用模型输出
            keyframe_mask = batch["keyframe_mask"]
            
            # 首先复制所有非约束帧
            pred_det_motion[:, config.context_frames:-1] = det_motion[:, config.context_frames:-1]
            
            # 然后恢复关键帧（除了上下文帧和结尾帧，它们已经被保留）
            for b in range(B):
                # 只处理非约束帧中的关键帧
                frame_mask = keyframe_mask[b, config.context_frames:-1]
                if frame_mask.any():
                    # 恢复关键帧的姿态（使用原始GT数据）
                    pred_det_motion[b, config.context_frames:-1][frame_mask] = GT_motion[b, config.context_frames:-1][frame_mask]

        pred_det_contact = GT_contact.clone().detach()
        pred_det_contact[:, config.context_frames:-1] = det_contact[:, config.context_frames:-1]

        if config.use_phase:
            pred_det_phase = GT_phase.clone().detach()
            pred_det_phase[:, config.context_frames:-1] = det_phase[:, config.context_frames:-1]

        # denormalize
        pred_det_motion = pred_det_motion * std + mean

        # predicted motion data
        pred_local_ortho6ds, pred_root_pos = torch.split(pred_det_motion, [M-3, 3], dim=-1)
        pred_local_ortho6ds = pred_local_ortho6ds.reshape(B, -1, skeleton.num_joints, 6) # (B, t, J, 6)
        _, pred_global_positions = trf.t_ortho6d.fk(pred_local_ortho6ds, pred_root_pos, skeleton) # (B, t, J, 3)

        # loss
        # 在跳过插值模式下创建mask，排除关键帧在损失计算中的影响（因为它们已经是GT值）
        skip_interpolation_mode = hasattr(config, 'skip_interpolation') and config.skip_interpolation and batch.get("keyframe_mask") is not None
        
        if skip_interpolation_mode:
            # 创建mask，关键帧位置为0，其他位置为1（即只计算非关键帧的损失）
            keyframe_mask = batch["keyframe_mask"]
            # 反转mask: True变为False，False变为True
            non_keyframe_mask = ~keyframe_mask
            # 约束帧和末尾帧都不参与损失计算
            non_keyframe_mask[:, :config.context_frames] = False
            non_keyframe_mask[:, -1:] = False
            
            # 将mask扩展到与关节形状匹配
            rot_mask = non_keyframe_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, skeleton.num_joints, 6)
            pos_mask = non_keyframe_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, skeleton.num_joints, 3)
            contact_mask = non_keyframe_mask.unsqueeze(-1).expand(-1, -1, 4)
            
            if i == 0 and epoch % 10 == 0:
                kf_count = keyframe_mask.sum().item()
                non_kf_count = non_keyframe_mask.sum().item()
                print(f"训练中的帧统计: 关键帧 {kf_count}, 非关键帧 {non_kf_count}, 关键帧比例: {kf_count/(kf_count+non_kf_count+1e-8):.2%}")
            
            # 使用掩码计算损失（仅计算非关键帧的损失）
            loss_rot = loss.masked_rot_loss(pred_local_ortho6ds, GT_local_ortho6ds, rot_mask)
            loss_pos = loss.masked_pos_loss(pred_global_positions, GT_global_positions, pos_mask)
            loss_contact = loss.masked_contact_loss(pred_det_contact, GT_contact, contact_mask)
        else:
            # 正常模式：计算所有非约束帧的损失
            loss_rot = loss.rot_loss(pred_local_ortho6ds, GT_local_ortho6ds, config.context_frames)
            loss_pos = loss.pos_loss(pred_global_positions, GT_global_positions, config.context_frames)
            loss_contact = loss.contact_loss(pred_det_contact, GT_contact, config.context_frames)

        loss_dict["rot"] += loss_rot.item()
        loss_dict["pos"] += loss_pos.item()
        loss_dict["contact"] += loss_contact.item()

        # Phase loss
        if config.use_phase:
            loss_phase = loss.phase_loss(pred_det_phase, GT_phase, config.context_frames)
            loss_dict["phase"] += loss_phase.item()
        else:
            loss_phase = torch.tensor(0.0, device=device)

        # Trajectory loss
        if config.use_traj:
            GT_traj = GT_traj * traj_std + traj_mean
            
            # Handle trajectory loss based on decoder configuration
            if config.get("decoupled_traj_decoder", False) and "traj" in det_out:
                # New behavior: Use direct trajectory output from decoder
                # Denormalize the predicted trajectory and apply appropriate masking
                pred_traj_from_decoder = det_out["traj"].clone().detach()
                pred_traj_from_decoder = pred_traj_from_decoder * traj_std + traj_mean
                
                # Restore constrained frames like we do for motion
                pred_det_traj = GT_traj.clone().detach()
                pred_det_traj[:, config.context_frames:-1] = pred_traj_from_decoder[:, config.context_frames:-1]
                
                # Calculate loss on the reconstructed trajectory
                loss_traj = loss.traj_loss(pred_det_traj, GT_traj, config.context_frames)
                loss_dict["traj"] += loss_traj.item()
            else:
                # Original behavior: Derive trajectory from motion
                pred_det_traj = ops.motion_to_traj(pred_det_motion)
                loss_traj = loss.traj_loss(pred_det_traj, GT_traj, config.context_frames)
                loss_dict["traj"] += loss_traj.item()
        else:
            loss_traj = torch.tensor(0.0, device=device)

        # Foot consistency loss
        loss_foot_consistency = torch.tensor(0.0, device=device)
        if config.get("use_foot_consistency", False):
            # Calculate foot positions
            foot_positions = pred_global_positions[:, :, contact_idx, :]
            
            # Get contact probabilities
            contact_probs = pred_det_contact
            
            # Create mask for stable contact states (contact in both frames)
            contact_maintained_mask = (contact_probs[:, 1:] > 0.8) & (contact_probs[:, :-1] > 0.8)
            
            # Calculate foot displacement between consecutive frames
            displacement = torch.norm(foot_positions[:, 1:] - foot_positions[:, :-1], p=2, dim=-1)
            
            # Only penalize displacement for feet that should be in stable contact
            masked_displacement = displacement * contact_maintained_mask
            
            # Average over all samples and time steps with valid errors
            loss_foot_consistency = torch.sum(masked_displacement) / (torch.sum(contact_maintained_mask) + 1e-6)
            
            loss_dict["foot_consistency"] += loss_foot_consistency.item()

        # Calculate total loss with either dynamic or static weighting
        if config.get("use_dynamic_loss_weighting", False) and config.use_traj:
            # Group losses into pose-related and trajectory-related
            loss_pose_group = (config.weight_rot * loss_rot + 
                            config.weight_pos * loss_pos + 
                            config.weight_contact * loss_contact +
                            (config.weight_phase * loss_phase if config.use_phase else 0) +
                            (config.weight_foot_consistency * loss_foot_consistency if config.get("use_foot_consistency", False) else 0))
            
            loss_traj_group = config.weight_traj * loss_traj
            
            # Compute dynamic weighting factor
            epsilon = 1e-8
            alpha = loss_traj_group.detach() / (loss_traj_group.detach() + loss_pose_group.detach() + epsilon)
            
            # Apply dynamic weighting
            loss_total = (1 - alpha) * loss_pose_group + alpha * loss_traj_group
            
            # Log weights for monitoring
            loss_dict["pose_weight"] += (1 - alpha).item()
            loss_dict["traj_weight"] += alpha.item()
        else:
            # Traditional static weighting
            loss_total = (config.weight_rot * loss_rot + 
                        config.weight_pos * loss_pos +
                        config.weight_contact * loss_contact)
            
            if config.use_phase:
                loss_total += config.weight_phase * loss_phase
                
            if config.use_traj:
                loss_total += config.weight_traj * loss_traj
                
            if config.get("use_foot_consistency", False):
                loss_total += config.weight_foot_consistency * loss_foot_consistency

        loss_dict["total"] += loss_total.item()
        
        # backward
        if train:
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

    # train
    start_time = time.perf_counter()
    for epoch in range(init_epoch+1, config.epochs+1):
        # train
        model.train()
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False)):
            train_iter(batch, train=True)

        # log training
        elapsed = time.perf_counter() - start_time
        utils.write_log(writer, loss_dict, len(dataloader), epoch, elapsed=elapsed, train=True)
        utils.reset_log(loss_dict)

        # validation
        if epoch % config.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_dataloader, desc=f"Validation", leave=False)):
                    train_iter(batch, train=False)

                # log validation
                utils.write_log(writer, loss_dict, len(val_dataloader), epoch, train=False)
                utils.reset_log(loss_dict)

        # save checkpoint - every 10 epochs
        if epoch % config.save_interval == 0:
            utils.save_ckpt(model, optimizer, epoch, config, scheduler=scheduler)

    # save checkpoint - last epoch
    utils.save_ckpt(model, optimizer, epoch, config, scheduler=scheduler)
    print(f"Training finished in {(time.perf_counter() - start_time) / 60:.2f} min")