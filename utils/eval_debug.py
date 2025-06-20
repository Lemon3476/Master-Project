import torch
from torch.utils.data import DataLoader
from aPyOpenGL import transforms as trf

from model.twostage import ContextTransformer, DetailTransformer
from model.rmi import RmiGenerator
from utils.dataset import MotionDataset
from utils import ops, utils

def debug_print(message):
    """调试信息打印函数"""
    print(f"[DEBUG-EVAL] {message}")

class EvaluatorDebug:
    def __init__(self, args):
        # arguments
        self.args = args
        self.config = utils.load_config(f"config/{args.dataset}/{args.config}")
        self.ts_configs = [utils.load_config(f"config/{args.dataset}/{config}") for config in args.ts_configs]
        self.rmi_configs = [utils.load_config(f"config/{args.dataset}/{config}") for config in args.rmi_configs]
        self.ours_configs = [utils.load_config(f"config/{args.dataset}/{config}") for config in args.ours_configs]
        
        # SegmentNet configs for replace_target_with_segment mode
        self.seg_configs = []
        self.seg_models = []
        
        if hasattr(args, 'seg_config') and args.seg_config is not None:
            self.seg_configs = [utils.load_config(f"config/{args.dataset}/{args.seg_config}")]
            print(f"Loaded SegmentNet config: {args.seg_config}")
            
        # Set replace_target_with_segment mode flag
        self.replace_target_with_segment = args.replace_target_with_segment if hasattr(args, 'replace_target_with_segment') else False
        
        # Mode flags
        self.replace_target_with_refine = args.replace_target if hasattr(args, 'replace_target') else False

        # dataset
        # Set seed before creating dataloader to ensure consistent shuffling
        utils.seed(1234)  # Always use the same seed 1234 for data loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = MotionDataset(train=False, config=self.config)
        
        # Always shuffle with fixed seed for consistent ordering across all modes
        print(f"Data loader using fixed seed 1234 for consistent sequence shuffling")
        
        # Create PyTorch random generator with fixed seed for DataLoader
        g = torch.Generator()
        g.manual_seed(1234)
        
        # Use the generator for shuffling to ensure consistent order
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,  # Always shuffle
            generator=g    # With fixed generator for consistency
        )
        self.skeleton = dataset.skeleton

        # statistics
        self.mean, self.std = dataset.motion_statistics(self.device)
        self.traj_mean, self.traj_std = dataset.traj_statistics(self.device)
        self.l2p_mean, self.l2p_std = dataset.l2p_statistics(self.device)

        self.contact_idx = []
        for joint in self.config.contact_joints:
            self.contact_idx.append(self.skeleton.idx_by_name[joint])
            
        # Now load SegmentNet model if in segment mode (after device is initialized)
        if self.replace_target_with_segment and len(self.seg_configs) > 0:
            from model.segment_net import SegmentNet
            seg_dataset = MotionDataset(train=False, config=self.seg_configs[0])
            seg_model = SegmentNet(self.seg_configs[0], seg_dataset).to(self.device)
            utils.load_model(seg_model, self.seg_configs[0])
            seg_model.eval()
            self.seg_models.append(seg_model)
            print(f"Loaded SegmentNet model successfully")

        # load trained models
        self.ctx_models, self.det_models = [], []
        for ctx_config, det_config in zip(self.ts_configs[::2], self.ts_configs[1::2]):
            ctx_model = ContextTransformer(ctx_config, dataset).to(self.device)
            det_model = DetailTransformer(det_config, dataset).to(self.device)
            utils.load_model(ctx_model, ctx_config)
            utils.load_model(det_model, det_config)
            ctx_model.eval()
            det_model.eval()
            self.ctx_models.append(ctx_model)
            self.det_models.append(det_model)
        
        self.rmi_models = []
        for rmi_config in self.rmi_configs:
            rmi_model = RmiGenerator(rmi_config, self.skeleton.num_joints).to(self.device)
            utils.load_model(rmi_model, rmi_config)
            rmi_model.eval()
            self.rmi_models.append(rmi_model)

        self.kf_models, self.ref_models = [], []
        self.ref_configs = []
        
        # Only load the necessary models based on the mode
        if self.replace_target_with_segment:
            # For segment mode, only load KeyframeNet
            for kf_config in self.ours_configs[::2]:
                kf_model = ContextTransformer(kf_config, dataset).to(self.device)
                utils.load_model(kf_model, kf_config)
                kf_model.eval()
                self.kf_models.append(kf_model)
                # Add dummy placeholder for ref_models to maintain index alignment
                self.ref_models.append(None)
                self.ref_configs.append(None)
        else:
            # For standard or refine modes, load both KeyframeNet and RefineNet
            for kf_config, ref_config in zip(self.ours_configs[::2], self.ours_configs[1::2]):
                kf_model = ContextTransformer(kf_config, dataset).to(self.device)
                ref_model = DetailTransformer(ref_config, dataset).to(self.device)
                utils.load_model(kf_model, kf_config)
                utils.load_model(ref_model, ref_config)
                kf_model.eval()
                ref_model.eval()
                self.kf_models.append(kf_model)
                self.ref_models.append(ref_model)
                self.ref_configs.append(ref_config)

    @torch.no_grad()
    def eval_direct(self, batch_id=None, num_frames=None, kf_sampling=["score"], traj_option=None):
        """直接加载指定batch_id的数据"""
        debug_print(f"Directly loading batch {batch_id}")
        
        for i, batch in enumerate(self.dataloader):
            if batch_id is not None and i != batch_id:
                debug_print(f"Skipping batch {i}...")
                continue
                
            debug_print(f"Processing batch {i}")
            
            res = {
                "motions": [],
                "tags": [],
                "trajs": [],
                "contacts": [],
                "keyframes": [],
                "skeleton": self.skeleton,
            }

            # GT data
            GT_motion  = batch["motion"].to(self.device)
            GT_phase   = batch["phase"].to(self.device)
            GT_traj    = batch["traj"].to(self.device)
            GT_score   = batch["score"].to(self.device)
            GT_contact = ops.get_contact(GT_motion, self.skeleton, self.contact_idx, self.config.contact_threshold)

            debug_print(f"GT_motion shape: {GT_motion.shape}")
            debug_print(f"GT_phase shape: {GT_phase.shape}")
            debug_print(f"GT_traj shape: {GT_traj.shape}")
            debug_print(f"GT_score shape: {GT_score.shape}")
            debug_print(f"GT_contact shape: {GT_contact.shape}")

            if num_frames is not None:
                GT_motion = GT_motion[:, :num_frames]
                GT_phase = GT_phase[:, :num_frames]
                GT_traj = GT_traj[:, :num_frames]
                GT_score = GT_score[:, :num_frames]
                GT_contact = GT_contact[:, :num_frames]

            res["motions"].append(GT_motion.clone())
            res["contacts"].append(GT_contact.clone())
            res["keyframes"].append(None)
            res["tags"].append("GT")
            res["trajs"].append(GT_traj.clone())

            # interpolate motion
            B, T, D = GT_motion.shape
            if self.args.interp:
                keyframes = [self.config.context_frames-1, T-1]
                interp_motion = ops.interpolate_motion_by_keyframes(GT_motion, keyframes)
                res["motions"].append(interp_motion)
                res["contacts"].append(GT_contact)
                res["keyframes"].append(None)
                res["tags"].append("Interp")
                res["trajs"].append(GT_traj)

            # forward rmi model
            for idx, rmi_model in enumerate(self.rmi_models):
                rmi = _rmi_transition(self.config, rmi_model, GT_motion, self.mean, self.std, GT_contact)
                res["motions"].append(rmi["motion"])
                res["contacts"].append(rmi["contact"])
                res["keyframes"].append(None)
                res["trajs"].append(GT_traj)
                if len(self.rmi_models) > 1:
                    res["tags"].append(f"ERD-QV-{idx}")
                else:
                    res["tags"].append("ERD-QV")

            # forward two-stage model
            for idx, (ctx_model, det_model) in enumerate(zip(self.ctx_models, self.det_models)):
                twostage = _twostage_transition(self.config, ctx_model, det_model, GT_motion, self.mean, self.std, GT_contact)

                res["motions"].append(twostage["motion"])
                res["contacts"].append(twostage["contact"])
                res["keyframes"].append(None)
                res["trajs"].append(GT_traj)
                if len(self.ctx_models) > 1:
                    res["tags"].append(f"TS-Trans-{idx}")
                else:
                    res["tags"].append("TS-Trans")

            # forward our model
            for idx, (kf_model, ref_model) in enumerate(zip(self.kf_models, self.ref_models)):
                ctx_frames = self.config.context_frames
                if traj_option == "interp":
                    t = torch.linspace(0, 1, T-ctx_frames+1).to(self.device)
                    traj_pos = GT_traj[..., 0:2]
                    traj_dir = GT_traj[..., 2:4] # (sin, cos)
                    traj_ang = torch.atan2(traj_dir[..., 0:1], traj_dir[..., 1:2])
                    
                    traj_pos_from = GT_traj[:, ctx_frames-1:ctx_frames, 0:2]
                    traj_pos_to = GT_traj[:, -1:, 0:2]

                    traj_ang_from = traj_ang[:, ctx_frames-1:ctx_frames]
                    traj_ang_to = traj_ang[:, -1:]

                    traj_pos = traj_pos_from + (traj_pos_to - traj_pos_from) * t[None, :, None]
                    traj_ang = traj_ang_from + (traj_ang_to - traj_ang_from) * t[None, :, None]
                    traj_dir = torch.cat([torch.sin(traj_ang), torch.cos(traj_ang)], dim=-1)

                    GT_traj[:, ctx_frames-1:, 0:2] = traj_pos
                    GT_traj[:, ctx_frames-1:, 2:4] = traj_dir

                elif traj_option == "scale":
                    traj_pos = GT_traj[..., 0:2]
                    traj_vel = traj_pos[:, 1:] - traj_pos[:, :-1]
                    traj_vel = torch.cat([torch.zeros_like(traj_vel[:, :1]), traj_vel], dim=1)
                    traj_vel *= 1.2 # velocity scale
                    traj_pos = torch.cumsum(traj_vel, dim=1)
                    traj_pos = traj_pos - traj_pos[:, ctx_frames-1:ctx_frames]
                    GT_traj[:, ctx_frames-1:, 0:2] = traj_pos[:, ctx_frames-1:]
                    GT_motion[:, ctx_frames-1:, (-3, -1)] = traj_pos[:, ctx_frames-1:]
                
                elif traj_option in ["replace", "random"]:
                    batch_idx = torch.arange(B)
                    shuffle_idx = torch.randperm(B)
                    GT_traj[batch_idx, ctx_frames:] = GT_traj[shuffle_idx, ctx_frames:]

                    fwd_from = torch.matmul(trf.t_ortho6d.to_rotmat(GT_motion[batch_idx, -1, 0:6]), torch.tensor([0, 0, 1.0]).to(self.device))
                    fwd_to   = torch.matmul(trf.t_ortho6d.to_rotmat(GT_motion[shuffle_idx, -1, 0:6]), torch.tensor([0, 0, 1.0]).to(self.device))

                    up_axis  = torch.tensor([0, 1.0, 0]).to(self.device)
                    signed_angles = ops.get_signed_angle_torch(fwd_from, fwd_to, up_axis)
                    delta_R = trf.t_rotmat.from_aaxis(up_axis * signed_angles)

                    GT_root_rotmat = trf.t_ortho6d.to_rotmat(GT_motion[:, -1, 0:6])
                    new_root_rotmat = torch.matmul(delta_R, GT_root_rotmat)
                    new_rot6d = trf.t_ortho6d.from_rotmat(new_root_rotmat)

                    GT_motion[batch_idx, -1, 0:6] = new_rot6d
                    GT_motion[:, -1, (-3, -1)] = GT_traj[:, -1, 0:2]

                elif traj_option is not None:
                    raise NotImplementedError(f"Invalid traj_option: {traj_option}")
                
                # Get flags for different modes
                replace_target_with_segment_flag = hasattr(self, 'replace_target_with_segment') and self.replace_target_with_segment
                replace_target_with_refine_flag = hasattr(self, 'replace_target_with_refine') and self.replace_target_with_refine
                
                debug_print(f"Using KF model: {kf_model}")
                debug_print(f"Using Refine model: {ref_model}")
                
                # Choose which function to call based on flags
                if replace_target_with_segment_flag:
                    # Use the replace_target_with_segment mode
                    print(f"Using replace_target_with_segment mode with SegmentNet")
                    
                    # Use the pre-loaded SegmentNet model
                    seg_model_idx = 0 if idx >= len(self.seg_models) else idx
                    seg_model = self.seg_models[seg_model_idx]
                    seg_config = self.seg_configs[0]
                    
                    ours = _segment_net_transition(
                        seg_config, 
                        kf_model, 
                        seg_model, 
                        GT_motion, 
                        self.mean, 
                        self.std, 
                        GT_contact, 
                        GT_phase, 
                        GT_traj, 
                        GT_score, 
                        self.traj_mean, 
                        self.traj_std, 
                        kf_sampling
                    )
                elif replace_target_with_refine_flag:
                    # Use the replace_target mode
                    print(f"Using replace_target mode with RefineNet")
                    
                    # Get batch size
                    B, T, D = GT_motion.shape
                    
                    # Make a copy of GT_motion to modify
                    modified_GT_motion = GT_motion.clone()
                    
                    # Perform circular replacement of target frames within the same batch
                    target_frame_idx = T - 1  # Target frame is the last frame
                    
                    # Use simple circular shift as before
                    # Since data is already shuffled with a fixed seed, this provides sufficient diversity
                    # Create a circular shift for sequence indices: 0→1, 1→2, ..., B-1→0
                    next_seq_indices = (torch.arange(B, device=GT_motion.device) + 1) % B
                    
                    # Print detailed information about the replacement pattern
                    replacements = [f"{i}→{next_seq_indices[i].item()}" for i in range(B)]
                    print(f"Replace Target Mode - Using circular replacement pattern (0→1, 1→2, ..., {B-1}→0)")
                    print(f"Replacement pattern: {', '.join(replacements[:10])}{'...' if B > 10 else ''}")
                    
                    # Replace each target frame with the target frame from the next sequence
                    for seq_idx in range(B):
                        next_seq_idx = next_seq_indices[seq_idx].item()
                        
                        # Get rotations from next sequence's target frame (preserve root position)
                        next_local_rots = GT_motion[next_seq_idx, target_frame_idx, :-3].clone()
                        
                        # Replace rotations only, keep original root position
                        modified_GT_motion[seq_idx, target_frame_idx, :-3] = next_local_rots
                    
                    # Simply call _ours_transition with the modified motion
                    ours = _ours_transition(
                        self.ref_configs[idx], 
                        kf_model, 
                        ref_model, 
                        modified_GT_motion,  # Use the modified GT motion with replaced target frames
                        self.mean, 
                        self.std, 
                        GT_contact, 
                        GT_phase, 
                        GT_traj, 
                        GT_score, 
                        self.traj_mean, 
                        self.traj_std, 
                        kf_sampling
                    )
                else:
                    # Standard mode
                    ours = _ours_transition(self.ref_configs[idx], kf_model, ref_model, GT_motion, self.mean, self.std, GT_contact, GT_phase, GT_traj, GT_score, self.traj_mean, self.traj_std, kf_sampling)
                
                debug_print(f"Keyframes data available: {ours['keyframes'] is not None}")
                if ours['keyframes'] is not None:
                    debug_print(f"Keyframes shape: {len(ours['keyframes'])}")
                    for b_idx, kf in enumerate(ours['keyframes']):
                        debug_print(f"Batch {b_idx} keyframes: {kf}")
                
                res["motions"].append(ours["motion"])
                res["contacts"].append(ours["contact"])
                res["keyframes"].append(ours["keyframes"])
                res["trajs"].append(GT_traj)
                if len(self.kf_models) > 1:
                    res["tags"].append(f"Ours-{idx}")
                else:
                    res["tags"].append("Ours")

            return res
            
        debug_print(f"Batch {batch_id} not found")
        return None
        
    @torch.no_grad()
    def eval(self, num_frames=None, kf_sampling=["score"], traj_option=None):
        for i, batch in enumerate(self.dataloader):
            res = {
                "motions": [],
                "tags": [],
                "trajs": [],
                "contacts": [],
                "keyframes": [],
                "skeleton": self.skeleton,
            }

            # GT data
            GT_motion  = batch["motion"].to(self.device)
            GT_phase   = batch["phase"].to(self.device)
            GT_traj    = batch["traj"].to(self.device)
            GT_score   = batch["score"].to(self.device)
            GT_contact = ops.get_contact(GT_motion, self.skeleton, self.contact_idx, self.config.contact_threshold)

            if num_frames is not None:
                GT_motion = GT_motion[:, :num_frames]
                GT_phase = GT_phase[:, :num_frames]
                GT_traj = GT_traj[:, :num_frames]
                GT_score = GT_score[:, :num_frames]
                GT_contact = GT_contact[:, :num_frames]

            res["motions"].append(GT_motion.clone())
            res["contacts"].append(GT_contact.clone())
            res["keyframes"].append(None)
            res["tags"].append("GT")
            res["trajs"].append(GT_traj.clone())

            # interpolate motion
            B, T, D = GT_motion.shape
            if self.args.interp:
                keyframes = [self.config.context_frames-1, T-1]
                interp_motion = ops.interpolate_motion_by_keyframes(GT_motion, keyframes)
                res["motions"].append(interp_motion)
                res["contacts"].append(GT_contact)
                res["keyframes"].append(None)
                res["tags"].append("Interp")
                res["trajs"].append(GT_traj)

            # forward rmi model
            for idx, rmi_model in enumerate(self.rmi_models):
                rmi = _rmi_transition(self.config, rmi_model, GT_motion, self.mean, self.std, GT_contact)
                res["motions"].append(rmi["motion"])
                res["contacts"].append(rmi["contact"])
                res["keyframes"].append(None)
                res["trajs"].append(GT_traj)
                if len(self.rmi_models) > 1:
                    res["tags"].append(f"ERD-QV-{idx}")
                else:
                    res["tags"].append("ERD-QV")

            # forward two-stage model
            for idx, (ctx_model, det_model) in enumerate(zip(self.ctx_models, self.det_models)):
                twostage = _twostage_transition(self.config, ctx_model, det_model, GT_motion, self.mean, self.std, GT_contact)

                res["motions"].append(twostage["motion"])
                res["contacts"].append(twostage["contact"])
                res["keyframes"].append(None)
                res["trajs"].append(GT_traj)
                if len(self.ctx_models) > 1:
                    res["tags"].append(f"TS-Trans-{idx}")
                else:
                    res["tags"].append("TS-Trans")

            # forward our model
            for idx, (kf_model, ref_model) in enumerate(zip(self.kf_models, self.ref_models)):
                ctx_frames = self.config.context_frames
                if traj_option == "interp":
                    t = torch.linspace(0, 1, T-ctx_frames+1).to(self.device)
                    traj_pos = GT_traj[..., 0:2]
                    traj_dir = GT_traj[..., 2:4] # (sin, cos)
                    traj_ang = torch.atan2(traj_dir[..., 0:1], traj_dir[..., 1:2])
                    
                    traj_pos_from = GT_traj[:, ctx_frames-1:ctx_frames, 0:2]
                    traj_pos_to = GT_traj[:, -1:, 0:2]

                    traj_ang_from = traj_ang[:, ctx_frames-1:ctx_frames]
                    traj_ang_to = traj_ang[:, -1:]

                    traj_pos = traj_pos_from + (traj_pos_to - traj_pos_from) * t[None, :, None]
                    traj_ang = traj_ang_from + (traj_ang_to - traj_ang_from) * t[None, :, None]
                    traj_dir = torch.cat([torch.sin(traj_ang), torch.cos(traj_ang)], dim=-1)

                    GT_traj[:, ctx_frames-1:, 0:2] = traj_pos
                    GT_traj[:, ctx_frames-1:, 2:4] = traj_dir

                elif traj_option == "scale":
                    traj_pos = GT_traj[..., 0:2]
                    traj_vel = traj_pos[:, 1:] - traj_pos[:, :-1]
                    traj_vel = torch.cat([torch.zeros_like(traj_vel[:, :1]), traj_vel], dim=1)
                    traj_vel *= 1.2 # velocity scale
                    traj_pos = torch.cumsum(traj_vel, dim=1)
                    traj_pos = traj_pos - traj_pos[:, ctx_frames-1:ctx_frames]
                    GT_traj[:, ctx_frames-1:, 0:2] = traj_pos[:, ctx_frames-1:]
                    GT_motion[:, ctx_frames-1:, (-3, -1)] = traj_pos[:, ctx_frames-1:]
                
                elif traj_option in ["replace", "random"]:
                    batch_idx = torch.arange(B)
                    shuffle_idx = torch.randperm(B)
                    GT_traj[batch_idx, ctx_frames:] = GT_traj[shuffle_idx, ctx_frames:]

                    fwd_from = torch.matmul(trf.t_ortho6d.to_rotmat(GT_motion[batch_idx, -1, 0:6]), torch.tensor([0, 0, 1.0]).to(self.device))
                    fwd_to   = torch.matmul(trf.t_ortho6d.to_rotmat(GT_motion[shuffle_idx, -1, 0:6]), torch.tensor([0, 0, 1.0]).to(self.device))

                    up_axis  = torch.tensor([0, 1.0, 0]).to(self.device)
                    signed_angles = ops.get_signed_angle_torch(fwd_from, fwd_to, up_axis)
                    delta_R = trf.t_rotmat.from_aaxis(up_axis * signed_angles)

                    GT_root_rotmat = trf.t_ortho6d.to_rotmat(GT_motion[:, -1, 0:6])
                    new_root_rotmat = torch.matmul(delta_R, GT_root_rotmat)
                    new_rot6d = trf.t_ortho6d.from_rotmat(new_root_rotmat)

                    GT_motion[batch_idx, -1, 0:6] = new_rot6d
                    GT_motion[:, -1, (-3, -1)] = GT_traj[:, -1, 0:2]

                elif traj_option is not None:
                    raise NotImplementedError(f"Invalid traj_option: {traj_option}")
                
                # Get flags for different modes
                replace_target_with_segment_flag = hasattr(self, 'replace_target_with_segment') and self.replace_target_with_segment
                replace_target_with_refine_flag = hasattr(self, 'replace_target_with_refine') and self.replace_target_with_refine
                
                # Choose which function to call based on flags
                if replace_target_with_segment_flag:
                    # Use the replace_target_with_segment mode
                    print(f"Using replace_target_with_segment mode with SegmentNet")
                    
                    # Use the pre-loaded SegmentNet model
                    seg_model_idx = 0 if idx >= len(self.seg_models) else idx
                    seg_model = self.seg_models[seg_model_idx]
                    seg_config = self.seg_configs[0]
                    
                    ours = _segment_net_transition(
                        seg_config, 
                        kf_model, 
                        seg_model, 
                        GT_motion, 
                        self.mean, 
                        self.std, 
                        GT_contact, 
                        GT_phase, 
                        GT_traj, 
                        GT_score, 
                        self.traj_mean, 
                        self.traj_std, 
                        kf_sampling
                    )
                elif replace_target_with_refine_flag:
                    # Use the replace_target mode
                    print(f"Using replace_target mode with RefineNet")
                    
                    # Get batch size
                    B, T, D = GT_motion.shape
                    
                    # Make a copy of GT_motion to modify
                    modified_GT_motion = GT_motion.clone()
                    
                    # Perform circular replacement of target frames within the same batch
                    target_frame_idx = T - 1  # Target frame is the last frame
                    
                    # Use simple circular shift as before
                    # Since data is already shuffled with a fixed seed, this provides sufficient diversity
                    # Create a circular shift for sequence indices: 0→1, 1→2, ..., B-1→0
                    next_seq_indices = (torch.arange(B, device=GT_motion.device) + 1) % B
                    
                    # Print detailed information about the replacement pattern
                    replacements = [f"{i}→{next_seq_indices[i].item()}" for i in range(B)]
                    print(f"Replace Target Mode - Using circular replacement pattern (0→1, 1→2, ..., {B-1}→0)")
                    print(f"Replacement pattern: {', '.join(replacements[:10])}{'...' if B > 10 else ''}")
                    
                    # Replace each target frame with the target frame from the next sequence
                    for seq_idx in range(B):
                        next_seq_idx = next_seq_indices[seq_idx].item()
                        
                        # Get rotations from next sequence's target frame (preserve root position)
                        next_local_rots = GT_motion[next_seq_idx, target_frame_idx, :-3].clone()
                        
                        # Replace rotations only, keep original root position
                        modified_GT_motion[seq_idx, target_frame_idx, :-3] = next_local_rots
                    
                    # Simply call _ours_transition with the modified motion
                    ours = _ours_transition(
                        self.ref_configs[idx], 
                        kf_model, 
                        ref_model, 
                        modified_GT_motion,  # Use the modified GT motion with replaced target frames
                        self.mean, 
                        self.std, 
                        GT_contact, 
                        GT_phase, 
                        GT_traj, 
                        GT_score, 
                        self.traj_mean, 
                        self.traj_std, 
                        kf_sampling
                    )
                else:
                    # Standard mode
                    ours = _ours_transition(self.ref_configs[idx], kf_model, ref_model, GT_motion, self.mean, self.std, GT_contact, GT_phase, GT_traj, GT_score, self.traj_mean, self.traj_std, kf_sampling)
                res["motions"].append(ours["motion"])
                res["contacts"].append(ours["contact"])
                res["keyframes"].append(ours["keyframes"])
                res["trajs"].append(GT_traj)
                if len(self.kf_models) > 1:
                    res["tags"].append(f"Ours-{idx}")
                else:
                    res["tags"].append("Ours")

            yield res

# 下面是从eval.py导入的其他函数
from utils.eval_backup import _ours_transition, _segment_net_transition, _rmi_transition, _twostage_transition