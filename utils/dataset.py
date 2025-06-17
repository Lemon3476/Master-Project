import os
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from aPyOpenGL import agl, transforms as trf

class MotionDataset(Dataset):
    def __init__(self, train, config, verbose=True):
        self.train = train
        self.config = config

        # ========== 在这里加入调试代码（部分1） ==========
        # 动态构建文件路径，并打印出来
        npz_full_path = os.path.join(config.dataset_dir, "MIB", f"{'train' if train else 'test'}-{config.npz_path}")
        print("-" * 60)
        print(f"[DEBUG-INFO] Attempting to load NPZ file:")
        print(f"[DEBUG-INFO] -> {npz_full_path}")
        print("-" * 60)
        # ===============================================   

        # load features
        features = np.load(os.path.join(config.dataset_dir, "MIB", f"{'train' if train else 'test'}-{config.npz_path}"))

        self.motion = torch.from_numpy(features["motion"]).float() # (B, T, 6J+3)
        self.phase = torch.from_numpy(features["phase"]).float()   # (B, T, 2P) where P is the number of phase channels
        self.traj = torch.from_numpy(features["traj"]).float()     # (B, T, 4)
        self.score = torch.from_numpy(features["scores"]).float()  # (B, T, 1)

        # ========== 在这里加入调试代码（部分2） ==========
        print(f"[DEBUG-INFO] NPZ file loaded successfully.")
        print(f"[DEBUG-INFO] Raw data shape from NPZ (Samples, Frames, Features): {self.motion.shape}")
        print(f"[DEBUG-INFO] -> Total Samples loaded: {self.motion.shape[0]}")
        print(f"[DEBUG-INFO] -> Frames per Sample: {self.motion.shape[1]}")
        print("-" * 60)
        # ===============================================

        # if "human36m" in self.config.dataset_dir:
        #     if self.config.window_offset != 1:
        #         raise ValueError("window_offset must be 1 for human36m dataset")
        #     self.motion = self.motion[::2]
        #     self.phase = self.phase[::2]
        #     self.traj = self.traj[::2]
        #     self.score = self.score[::2]

        if verbose:
            print("Shapes:")
            print(f"\t- motion.shape: {self.motion.shape}")
            print(f"\t- phase.shape: {self.phase.shape}")
            print(f"\t- traj.shape: {self.traj.shape}")
            print(f"\t- score.shape: {self.score.shape}")

        # dimensions
        self.num_frames = self.motion.shape[1]
        self.motion_dim = self.motion.shape[2]
        self.phase_dim = self.phase.shape[2]
        self.traj_dim = self.traj.shape[2]
        self.score_dim = self.score.shape[2]

        # load skeletons
        self.skeleton = self.load_skeleton(os.path.join(config.dataset_dir, "skeleton.pkl"))
    
    def load_skeleton(self, path) -> agl.Skeleton:
        if not os.path.exists(path):
            print(f"Cannot find skeleton from {path}")
            return None
        
        with open(path, "rb") as f:
            skeleton = pickle.load(f)
            
        return skeleton
    
    def __len__(self):
        return len(self.motion)
    
    def __getitem__(self, idx):
        res = {
            "motion": self.motion[idx],
            "phase": self.phase[idx],
            "traj": self.traj[idx],
            "score": self.score[idx],
        }
        return res
    
    def motion_statistics(self, device="cuda"):
        if os.path.exists(os.path.join(self.config.dataset_dir, "MIB", "motion_statistics.pth")):
            mean = torch.load(os.path.join(self.config.dataset_dir, "MIB", "motion_statistics.pth"))["mean"].to(device)
            std = torch.load(os.path.join(self.config.dataset_dir, "MIB", "motion_statistics.pth"))["std"].to(device)
        else:
            if not self.train:
                motion = MotionDataset(train=True, config=self.config, verbose=False).motion.to(device)
            else:
                motion = self.motion.to(device)
            mean = torch.mean(motion, dim=(0, 1))
            std = torch.std(motion, dim=(0, 1)) + 1e-8
            torch.save({"mean": mean.cpu(), "std": std.cpu()}, os.path.join(self.config.dataset_dir, "MIB", "motion_statistics.pth"))
        return mean, std

    def traj_statistics(self, device="cuda"):
        if os.path.exists(os.path.join(self.config.dataset_dir, "MIB", "traj_statistics.pth")):
            mean = torch.load(os.path.join(self.config.dataset_dir, "MIB", "traj_statistics.pth"))["mean"].to(device)
            std = torch.load(os.path.join(self.config.dataset_dir, "MIB", "traj_statistics.pth"))["std"].to(device)
        else:
            if not self.train:
                traj = MotionDataset(train=True, config=self.config, verbose=False).traj.to(device)
            else:
                traj = self.traj.to(device)
            mean = torch.mean(traj, dim=(0, 1))
            std = torch.std(traj, dim=(0, 1)) + 1e-8
            torch.save({"mean": mean.cpu(), "std": std.cpu()}, os.path.join(self.config.dataset_dir, "MIB", "traj_statistics.pth"))
        return mean, std
    
    def l2p_statistics(self, device="cuda"):
        if os.path.exists(os.path.join(self.config.dataset_dir, "MIB", "l2p_statistics.pth")):
            mean = torch.load(os.path.join(self.config.dataset_dir, "MIB", "l2p_statistics.pth"))["mean"].to(device)
            std = torch.load(os.path.join(self.config.dataset_dir, "MIB", "l2p_statistics.pth"))["std"].to(device)
        else:
            if not self.train:
                motion = MotionDataset(train=True, config=self.config, verbose=False).motion
            else:
                motion = self.motion
            dloader = DataLoader(motion, batch_size=self.config.batch_size, shuffle=False)
            global_pos = []
            for batch in dloader:
                batch = batch.to(device)
                B, T, D = batch.shape
                local_ortho6ds, root_pos = torch.split(batch, [D-3, 3], dim=-1)
                local_ortho6ds = local_ortho6ds.reshape(B, T, -1, 6)
                _, gp = trf.t_ortho6d.fk(local_ortho6ds, root_pos, self.skeleton)
                global_pos.append(gp)

            global_pos = torch.cat(global_pos, dim=0) # (B, T, J, 3)
            mean = torch.mean(global_pos, dim=(0, 1))
            std = torch.std(global_pos, dim=(0, 1)) + 1e-8
            torch.save({"mean": mean.cpu(), "std": std.cpu()}, os.path.join(self.config.dataset_dir, "MIB", "l2p_statistics.pth"))
        return mean, std
    
class PAEDataset(Dataset):
    def __init__(self, train, config):
        self.train = train
        self.config = config

        # load features
        print(os.path.join(config.dataset_dir, "PAE", f"{'train' if train else 'test'}-{config.npz_path}"))
        features = np.load(os.path.join(config.dataset_dir, "PAE", f"{'train' if train else 'test'}-{config.npz_path}"))
        self.features = torch.from_numpy(features["motion"]).float() # (B, T, 3J)

        # if "human36m" in self.config.dataset_dir:
        #     if self.config.window_offset != 1:
        #         raise ValueError("window_offset must be 1 for human36m dataset")
        #     self.features = self.features[::2]

        # dimensions
        self.num_frames = self.features.shape[1]
        self.motion_dim = self.features.shape[2]
        print(f"Shapes:")
        print(f"\t- features.shape: {self.features.shape}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    def motion_statistics(self, device="cuda"):
        if os.path.exists(os.path.join(self.config.dataset_dir, "PAE", "motion_statistics.pth")):
            mean = torch.load(os.path.join(self.config.dataset_dir, "PAE", "motion_statistics.pth"))["mean"].to(device)
            std = torch.load(os.path.join(self.config.dataset_dir, "PAE", "motion_statistics.pth"))["std"].to(device)
        else:
            if not self.train:
                feat = PAEDataset(train=True, config=self.config).features
            else:
                feat = self.features
            mean = torch.mean(feat, dim=(0, 1))
            std = torch.std(feat, dim=(0, 1)) + 1e-8
            torch.save({"mean": mean, "std": std}, os.path.join(self.config.dataset_dir, "PAE", "motion_statistics.pth"))
        return mean.to(device), std.to(device)


class SegmentDataset(Dataset):
    """
    用于SegmentNet训练的数据集类，加载预处理的段数据。
    
    预处理的数据包括：
    - gt_motion: 真实运动段 (N, max_segment_length, motion_dim)
    - draft_motion: 插值后的草稿段 (N, max_segment_length, motion_dim)
    - phase: 相位数据 (可选)
    - traj: 轨迹数据 (可选)
    - contact: 接触数据 (可选)
    - score: 分数数据 (可选)
    - overlap_frames: 每个段的重叠帧数
    """
    def __init__(self, train, config, verbose=True, preloaded_stats=None):
        self.train = train
        self.config = config
        self.preloaded_stats = preloaded_stats
        
        # 确定数据文件路径
        filename = f"{config.dataset_dir.split('/')[1]}-segment-{'train' if train else 'val'}.npz"
        data_path = os.path.join(config.dataset_dir, filename)
        
        # 加载预处理的段数据
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"预处理的段数据文件 {data_path} 不存在。请先运行 preprocess/preprocess_segments.py 生成数据。")
        
        # data = np.load(data_path)
        data = np.load(data_path, mmap_mode='r')
        
        # 加载必要的数据
        self.gt_motion = torch.from_numpy(data["gt_motion"]).float()
        self.draft_motion = torch.from_numpy(data["draft_motion"]).float()
        self.overlap_frames = torch.from_numpy(data["overlap_frames"]).long()
        
        # 加载可选数据
        self.has_phase = "phase" in data
        self.has_traj = "traj" in data
        self.has_contact = "contact" in data
        self.has_score = "score" in data
        
        if self.has_phase:
            self.phase = torch.from_numpy(data["phase"]).float()
        if self.has_traj:
            self.traj = torch.from_numpy(data["traj"]).float()
        if self.has_contact:
            self.contact = torch.from_numpy(data["contact"]).float()
        if self.has_score:
            self.score = torch.from_numpy(data["score"]).float()
        
        # 打印数据信息
        if verbose:
            print("Segment Dataset Shapes:")
            print(f"\t- gt_motion.shape: {self.gt_motion.shape}")
            print(f"\t- draft_motion.shape: {self.draft_motion.shape}")
            if self.has_phase:
                print(f"\t- phase.shape: {self.phase.shape}")
            if self.has_traj:
                print(f"\t- traj.shape: {self.traj.shape}")
            if self.has_contact:
                print(f"\t- contact.shape: {self.contact.shape}")
            if self.has_score:
                print(f"\t- score.shape: {self.score.shape}")
        
        # 数据维度
        self.num_segments = self.gt_motion.shape[0]
        self.segment_length = self.gt_motion.shape[1]
        self.motion_dim = self.gt_motion.shape[2]
        
        if self.has_phase:
            self.phase_dim = self.phase.shape[2]
        else:
            self.phase_dim = 0
            
        if self.has_traj:
            self.traj_dim = self.traj.shape[2]
        else:
            self.traj_dim = 0
            
        if self.has_score:
            self.score_dim = self.score.shape[2]
        else:
            self.score_dim = 0
        
        # 加载骨架
        self.skeleton = self.load_skeleton(os.path.join(config.dataset_dir, "skeleton.pkl"))
    
    def load_skeleton(self, path) -> agl.Skeleton:
        if not os.path.exists(path):
            print(f"Cannot find skeleton from {path}")
            skeleton = agl.Skeleton()
        else:
            with open(path, "rb") as f:
                skeleton = pickle.load(f)
        return skeleton
    
    def __len__(self):
        return self.num_segments
    
    def __getitem__(self, idx):
        # 构建返回的字典
        item = {
            "gt_motion": self.gt_motion[idx],
            "draft_motion": self.draft_motion[idx],
            "overlap_frames": self.overlap_frames[idx]
        }
        
        # 添加可选数据
        if self.has_phase:
            item["phase"] = self.phase[idx]
        if self.has_traj:
            item["traj"] = self.traj[idx]
        if self.has_contact:
            item["contact"] = self.contact[idx]
        if self.has_score:
            item["score"] = self.score[idx]
        
        return item
    
    def motion_statistics(self, device="cuda"):
        # 如果有预加载的统计数据，直接使用
        if self.preloaded_stats is not None:
            mean, std = self.preloaded_stats[0], self.preloaded_stats[1]
            return mean.to(device), std.to(device)
            
        # 否则，使用与MotionDataset相同的统计数据
        # 这确保了段数据和原始数据使用相同的归一化参数
        stats_path = os.path.join(self.config.dataset_dir, "train-motion-statistics.pth")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path)
            mean = stats["mean"].to(device)
            std = stats["std"].to(device)
        else:
            # 如果找不到预计算的统计数据，使用当前数据计算
            mean = torch.mean(self.gt_motion, dim=(0, 1))
            std = torch.std(self.gt_motion, dim=(0, 1)) + 1e-8
        
        return mean.to(device), std.to(device)
    
    def traj_statistics(self, device="cuda"):
        # 如果有预加载的统计数据，直接使用
        if self.preloaded_stats is not None and len(self.preloaded_stats) >= 4:
            traj_mean, traj_std = self.preloaded_stats[2], self.preloaded_stats[3]
            if traj_mean is not None:
                return traj_mean.to(device), traj_std.to(device)
                
        # 如果没有轨迹数据，返回None
        if not self.has_traj:
            return None, None
            
        # 否则，使用与MotionDataset相同的统计数据
        stats_path = os.path.join(self.config.dataset_dir, "train-traj-statistics.pth")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path)
            mean = stats["mean"].to(device)
            std = stats["std"].to(device)
        else:
            # 如果找不到预计算的统计数据，使用当前数据计算
            mean = torch.mean(self.traj, dim=(0, 1))
            std = torch.std(self.traj, dim=(0, 1)) + 1e-8
        
        return mean.to(device), std.to(device)