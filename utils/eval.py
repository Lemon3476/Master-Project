import torch
from torch.utils.data import DataLoader
from aPyOpenGL import transforms as trf

from model.twostage import ContextTransformer, DetailTransformer
from model.rmi import RmiGenerator
from utils.dataset import MotionDataset
from utils import ops, utils

class Evaluator:
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
                    
                    ours = _replace_target_with_segment_transition(
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

@torch.no_grad()
def _twostage_transition(config, ctx_model: ContextTransformer, det_model: DetailTransformer, GT_motion, mean, std, GT_contact):
    motion = (GT_motion - mean) / std

    # forward ContextTransformer
    ctx_out, midway_targets = ctx_model.forward(motion, train=False)
    ctx_motion = ctx_out["motion"]

    # restore constrained frames
    pred_ctx_motion = motion.clone().detach()
    pred_ctx_motion[:, config.context_frames:-1] = ctx_motion[:, config.context_frames:-1]

    # forward DetailTransformer
    det_out = det_model.forward(pred_ctx_motion, midway_targets)
    det_motion = det_out["motion"]
    det_contact = det_out["contact"]

    # restore constrained frames
    pred_det_motion = motion.clone().detach()
    pred_det_motion[:, config.context_frames:-1] = det_motion[:, config.context_frames:-1]

    pred_det_contact = GT_contact.clone().detach()
    pred_det_contact[:, config.context_frames:-1] = det_contact[:, config.context_frames:-1]

    # denormalize
    # pred_det_motion = pred_det_motion * std + mean

    return {
        "motion": pred_det_motion * std + mean,
        "contact": pred_det_contact,
    }

@torch.no_grad()
def _rmi_transition(config, generator: RmiGenerator, GT_motion, mean, std, GT_contact):
    B, T, D = GT_motion.shape
    motion = (GT_motion - mean) / std

    local_rot, root_pos = torch.split(motion, [D-3, 3], dim=-1)
    root_vel = root_pos[:, 1:] - root_pos[:, :-1]
    root_vel = torch.cat([root_vel[:, 0:1], root_vel], dim=1)

    target = motion[:, -1]
    target_local_rot, target_root_pos = torch.split(target, [D-3, 3], dim=-1)
    
    generator.init_hidden(B, motion.device)
    pred_rot, pred_root_pos, pred_contact = [local_rot[:, 0]], [root_pos[:, 0]], [GT_contact[:, 0]]
    for i in range(config.context_frames):
        tta = T - i - 1
        lr, rp, c = generator.forward(local_rot[:, i], root_pos[:, i], root_vel[:, i], GT_contact[:, i], target_local_rot, target_root_pos, tta)
        pred_rot.append(lr)
        pred_root_pos.append(rp)
        pred_contact.append(c)
    for i in range(config.context_frames, T-1):
        tta = T - i - 1
        lr, rp, c = generator.forward(lr, rp, rp - pred_root_pos[-1], c, target_local_rot, target_root_pos, tta)
        pred_rot.append(lr)
        pred_root_pos.append(rp)
        pred_contact.append(c)
    
    # stack transition frames without context frames
    pred_rot = torch.stack(pred_rot, dim=1)
    pred_root_pos = torch.stack(pred_root_pos, dim=1)
    pred_contact = torch.stack(pred_contact, dim=1)

    motion = torch.cat([pred_rot, pred_root_pos], dim=-1)
    motion = motion * std + mean

    pred_motion = torch.cat([GT_motion[:, :config.context_frames], motion[:, config.context_frames:]], dim=1)
    pred_contact = torch.cat([GT_contact[:, :config.context_frames], pred_contact[:, config.context_frames:]], dim=1)

    return {
        "motion": pred_motion,
        "contact": pred_contact,
    }

@torch.no_grad()
def _ours_transition(config,
                     kf_model: ContextTransformer,
                     ref_model: DetailTransformer,
                     GT_motion,
                     mean,
                     std,
                     GT_contact,
                     GT_phase=None,
                     GT_traj=None,
                     GT_score=None,
                     traj_mean=None,
                     traj_std=None,
                     kf_sampling=["score"],):
    """
    config: config of RefineNet
    """
    motion = (GT_motion - mean) / std
    if config.use_traj:
        GT_traj = (GT_traj - traj_mean) / traj_std

    # forward ContextTransformer (KeyframeNet)
    # -----------------------------------------------------------------------------
    # NOTE: KeyframeNet always receives the full GT data as input (GT_phase, GT_traj),
    # but depending on its configuration, it may or may not use this data internally.
    # The KeyframeNet model will return predictions only for the features it was 
    # trained to predict (based on its config settings like use_phase, use_traj, etc.)
    # -----------------------------------------------------------------------------
    ctx_out, midway_targets = kf_model.forward(motion, phase=GT_phase, traj=GT_traj, train=False)
    ctx_motion = ctx_out["motion"]
    
    # Check if KeyframeNet was configured to predict phase
    if config.get("use_phase", False) and "phase" in ctx_out:
        ctx_phase = ctx_out["phase"]
        # If this condition is false, we'll use GT_phase directly for RefineNet
    
    # Check if KeyframeNet was configured to predict score
    if config.get("use_score", False) and "score" in ctx_out:
        ctx_score = ctx_out["score"]
        # If this condition is false, we'll use GT_score directly for RefineNet

    # restore constrained frames
    pred_ctx_motion = motion.clone().detach()
    pred_ctx_motion[:, config.context_frames:-1] = ctx_motion[:, config.context_frames:-1]
    pred_ctx_motion[:, midway_targets] = motion[:, midway_targets]

    # Initialize pred_ctx_phase based on whether KeyframeNet predicted it
    # -----------------------------------------------------------------------------
    # PHASE HANDLING LOGIC:
    # 1. If KeyframeNet predicted phase (ctx_phase exists), we use a combination of:
    #    - KeyframeNet's predicted phase for non-constrained frames
    #    - GT phase for constrained frames (context_frames and midway_targets)
    # 2. If KeyframeNet did NOT predict phase (no ctx_phase), we:
    #    - Set pred_ctx_phase to None initially
    #    - Will decide later whether to pass GT_phase directly to RefineNet
    # -----------------------------------------------------------------------------
    pred_ctx_phase = None
    
    # Check if KeyframeNet produced phase predictions
    if config.get("use_phase", False) and 'ctx_phase' in locals():
        # KeyframeNet predicted phase - create a blend of predicted and GT phase
        pred_ctx_phase = GT_phase.clone().detach()
        pred_ctx_phase[:, config.context_frames:-1] = ctx_phase[:, config.context_frames:-1]
        pred_ctx_phase[:, midway_targets] = GT_phase[:, midway_targets]
        # Note: For constrained frames (context_frames and midway_targets), we always use GT_phase

    # Initialize keyframes as None by default
    keyframes = None
    
    # Only process score if the config supports it and we have ctx_score defined
    if config.get("use_score", False) and 'ctx_score' in locals():
        pred_score = GT_score.clone().detach()
        pred_score[:, config.context_frames:-1] = ctx_score[:, config.context_frames:-1]
        
        pred_ctx_motion = pred_ctx_motion * std + mean
        if kf_sampling[0] == "score":
            keyframes = ops.get_keyframes_by_score(config, pred_score)
        elif kf_sampling[0] == "threshold":
            keyframes = ops.get_keyframes_by_score_threshold(config, pred_score, threshold=kf_sampling[1])
        elif kf_sampling[0] == "topk":
            keyframes = ops.get_keyframes_by_topk(config, pred_score, top=kf_sampling[1])
        elif kf_sampling[0] == "random":
            keyframes = ops.get_keyframes_by_random(config, pred_score, prob=kf_sampling[1])
        elif kf_sampling[0] == "uniform":
            keyframes = ops.get_keyframes_by_uniform(config, pred_score, step=kf_sampling[1])
        else:
            raise NotImplementedError(f"Invalid keyframe sampling method: {kf_sampling}")
        for b in range(motion.shape[0]):
            pred_ctx_motion[b:b+1] = ops.interpolate_motion_by_keyframes(pred_ctx_motion[b:b+1], keyframes[b])
        pred_ctx_motion = (pred_ctx_motion - mean) / std

    # forward DetailTransformer (RefineNet)
    # -----------------------------------------------------------------------------
    # REFINE NET INPUT PREPARATION:
    # The RefineNet requires specific conditions to properly refine the motion.
    # If RefineNet expects a condition (phase, trajectory) but KeyframeNet didn't 
    # provide it, we use Ground Truth data as a substitute.
    #
    # This is critical for ablation studies where we disable certain features in
    # KeyframeNet but still want to evaluate the complete pipeline.
    # -----------------------------------------------------------------------------
    
    # Check which inputs RefineNet expects based on its configuration
    ref_use_phase = hasattr(config, 'use_phase') and config.use_phase
    ref_use_traj = hasattr(config, 'use_traj') and config.use_traj
    
    # Prepare phase input for RefineNet
    if ref_use_phase:
        # RefineNet expects phase input
        if pred_ctx_phase is not None:
            # Use the phase prepared earlier (blend of KF prediction and GT)
            phase_arg = pred_ctx_phase
        else:
            # KeyframeNet didn't predict phase, so use GT phase directly
            # This happens in ablation studies where KeyframeNet has use_phase=false
            phase_arg = GT_phase
            print("Using GT_phase directly for RefineNet (KeyframeNet didn't predict phase)")
    else:
        # RefineNet doesn't expect phase input
        phase_arg = None
    
    # Prepare trajectory input for RefineNet
    traj_arg = GT_traj if ref_use_traj else None
    
    # Forward pass through RefineNet with prepared inputs
    det_out = ref_model.forward(pred_ctx_motion, midway_targets, phase=phase_arg, traj=traj_arg)
    det_motion = det_out["motion"]
    det_contact = det_out["contact"]

    # restore constrained frames
    pred_det_motion = motion.clone().detach()
    pred_det_motion[:, config.context_frames:-1] = det_motion[:, config.context_frames:-1]

    pred_det_contact = GT_contact.clone().detach()
    pred_det_contact[:, config.context_frames:-1] = det_contact[:, config.context_frames:-1]

    return {
        "motion": pred_det_motion * std + mean,
        "contact": pred_det_contact,
        "keyframes": keyframes,
    }

#####################################
# benchmark metrics
#####################################
def l2p(GT_motion, pred_motion, skeleton, l2p_mean, l2p_std, ctx_frames=10):
    B, T, D = GT_motion.shape

    def convert(motion):
        rot, pos = torch.split(motion, [D-3, 3], dim=-1)
        rot = rot.reshape(B, T, skeleton.num_joints, 6)
        _, gp = trf.t_ortho6d.fk(rot, pos, skeleton)
        gp = gp[:, ctx_frames:-1]
        return (gp - l2p_mean) / l2p_std
    
    GT_gp = convert(GT_motion)
    pred_gp = convert(pred_motion)

    norm = torch.sqrt(torch.sum((GT_gp - pred_gp) ** 2, dim=(2, 3)))
    return torch.mean(norm).item()

def l2q(GT_motion, pred_motion, ctx_frames=10):
    B, T, D = GT_motion.shape

    def convert(motion):
        rot, pos = torch.split(motion, [D-3, 3], dim=-1)
        rot = trf.t_quat.from_ortho6d(rot.reshape(B, T, -1, 6))
        rot = ops.remove_quat_discontinuities(rot)
        rot = rot[:, ctx_frames:-1]
        return rot
    
    GT_rot = convert(GT_motion)
    pred_rot = convert(pred_motion)
    norm = torch.sqrt(torch.sum((GT_rot - pred_rot) ** 2, dim=(2, 3)))
    return torch.mean(norm).item()

def npss(GT_motion, pred_motion, ctx_frames=10):
    B, T, D = GT_motion.shape

    def convert(motion):
        rot, pos = torch.split(motion, [D-3, 3], dim=-1)
        rot = trf.t_quat.from_ortho6d(rot.reshape(B, T, -1, 6))
        rot = ops.remove_quat_discontinuities(rot).reshape(B, T, -1)
        rot = rot[:, ctx_frames:-1]

        # Fourier coefficients along the time dimension
        fourier_coeffs = torch.real(torch.fft.fft(rot, dim=1))

        # square of the Fourier coefficients
        power = torch.square(fourier_coeffs)

        # sum of powers over time
        total_power = torch.sum(power, dim=1)

        # normalize powers with total
        norm_power = power / (total_power[:, None] + 1e-8)

        # cumulative sum over time
        cdf_power = torch.cumsum(norm_power, dim=1)

        return cdf_power, total_power
    
    GT_cdf_power, GT_total_power = convert(GT_motion)
    pred_cdf_power, _ = convert(pred_motion)

    # earth mover distance
    emd = torch.norm((pred_cdf_power - GT_cdf_power), p=1, dim=1)

    # weighted EMD
    power_weighted_emd = torch.sum(emd * GT_total_power) / torch.sum(GT_total_power)

    return power_weighted_emd.item()

def foot_skate(pred_motion, pred_contact, skeleton, foot_ids, ctx_frames=10):
    B, T, D = pred_motion.shape
    
    rot, pos = torch.split(pred_motion, [D-3, 3], dim=-1)
    rot = rot.reshape(B, T, -1, 6)
    _, gp = trf.t_ortho6d.fk(rot, pos, skeleton)

    # foot velocity
    fv = gp[:, 1:, foot_ids] - gp[:, :-1, foot_ids]
    fv = torch.sum(fv ** 2, dim=-1) # (B, T-1, 4)
    fv = torch.cat([fv[:, 0:1], fv], dim=1) # (B, T, 4)

    # # foot position
    # fp = gp[:, :, foot_ids] # (B, T, 4, 3)

    # # weight
    # weight = torch.clamp(2.0 - 2.0 ** (fp[..., 1] / height_threshold), min=0, max=1) # (B, T, 4)

    # # # mask - if all weights are zero, skip this sample
    # # mask = torch.sum(weight.reshape(B, -1), dim=-1) > 0 # (B)
    # # fv = fv[mask]
    # # weight = weight[mask]

    metric = torch.sum(fv * pred_contact, dim=-1)
    metric = torch.mean(metric[:, ctx_frames:-1], dim=-1)

    # # metric
    # metric = torch.sum(fv * weight, dim=-1) # (B, T)
    # metric = torch.mean(metric[:, ctx_frames:-1], dim=-1) # (B)

    return torch.mean(metric).item() * 100

@torch.no_grad()
def _replace_target_with_segment_transition(config,
                     kf_model: ContextTransformer,
                     seg_model,
                     GT_motion,
                     mean,
                     std,
                     GT_contact,
                     GT_phase=None,
                     GT_traj=None,
                     GT_score=None,
                     traj_mean=None,
                     traj_std=None,
                     kf_sampling=["score"]):
    """
    Replace the target frame (last frame) of each sequence with the target frame from another sequence
    in the same batch, then generate motion using the SegmentNet approach (similar to _segment_net_transition).
    
    This mode:
    1. Replaces only the target frame (last frame) of each sequence
    2. Uses a circular replacement pattern (0→1, 1→2, etc.) within the same batch
    3. Uses KeyframeNet to predict keyframes
    4. Uses SegmentNet to generate transitions between keyframes
    """
    B, T, D = GT_motion.shape
    device = GT_motion.device
    
    # Create a copy of GT_motion to modify the target frames
    modified_GT_motion = GT_motion.clone()
    
    # Collect modified pose positions for visualization
    modified_positions = []
    replacement_data = []
    
    # Perform circular replacement of target frames within the same batch
    total_replaced = 0
    print("Replace Target With Segment Mode - Target Frame Replacements:")
    
    # Target frame is the last frame
    target_frame_idx = T - 1
    
    # One batch already contains multiple sequences (typically 64)
    # Get all sequence indices in this batch
    seq_indices = torch.arange(B, device=device)
    
    # Create a circular shift for the sequence indices: 0→1, 1→2, ..., B-1→0
    next_seq_indices = (seq_indices + 1) % B
    
    print(f"Total sequences in this batch: {B}")
    print(f"Target frame replacement pattern (within the same batch): {seq_indices[:5].cpu().numpy()}→{next_seq_indices[:5].cpu().numpy()}...")
    
    for seq_idx in range(B):
        # Initialize modified positions for this sequence
        batch_modified_positions = []
        batch_replacement_data = []
        
        # Circular replacement within the same batch: 0→1, 1→2, ..., B-1→0
        next_seq_idx = next_seq_indices[seq_idx].item()
        
        # Get rotations from next sequence's target frame (preserve root position)
        next_local_rots = GT_motion[next_seq_idx, target_frame_idx, :-3].clone()
        
        # Replace rotations only, keep original root position
        modified_GT_motion[seq_idx, target_frame_idx, :-3] = next_local_rots
        
        # Record the modified position for visualization
        batch_modified_positions.append(target_frame_idx)
        
        # Record detailed replacement info
        batch_replacement_data.append({
            "target_idx": target_frame_idx,
            "next_seq_idx": next_seq_idx,
            "next_target_idx": target_frame_idx
        })
        
        # Print info for the first few sequences
        if seq_idx < 5:
            print(f"  Sequence {seq_idx}: Replaced target frame {target_frame_idx} with frame from sequence {next_seq_idx}")
        
        # Increase counter for successful replacements
        total_replaced += 1
        
        # Add batch modified positions to the list
        modified_positions.append(batch_modified_positions)
        replacement_data.append(batch_replacement_data)
    
    print(f"  Total replaced target frames: {total_replaced}")
    
    # Now we proceed with the segment_net transition logic, similar to _segment_net_transition
    # but using our modified_GT_motion
    
    # Normalize motion and trajectories
    motion = (modified_GT_motion - mean) / std
    if config.use_traj and GT_traj is not None:
        normalized_GT_traj = (GT_traj - traj_mean) / traj_std
    else:
        normalized_GT_traj = None
    
    # Step 1: Forward KeyframeNet to get keyframes
    ctx_out, midway_targets = kf_model.forward(motion, phase=GT_phase, traj=normalized_GT_traj, train=False)
    
    # Get predicted scores for keyframe selection
    if config.use_score and GT_score is not None:
        ctx_score = ctx_out["score"]
        
        # Get keyframes based on scores
        if kf_sampling[0] == "score":
            keyframes = ops.get_keyframes_by_score(config, ctx_score)
        elif kf_sampling[0] == "threshold":
            keyframes = ops.get_keyframes_by_score_threshold(config, ctx_score, threshold=kf_sampling[1])
        elif kf_sampling[0] == "topk":
            keyframes = ops.get_keyframes_by_topk(config, ctx_score, top=kf_sampling[1])
        elif kf_sampling[0] == "random":
            keyframes = ops.get_keyframes_by_random(config, ctx_score, prob=kf_sampling[1])
        elif kf_sampling[0] == "uniform":
            keyframes = ops.get_keyframes_by_uniform(config, ctx_score, step=kf_sampling[1])
        else:
            raise NotImplementedError(f"Invalid keyframe sampling method: {kf_sampling}")
    else:
        # Fallback to random keyframes if scores not available
        keyframes = []
        for b in range(B):
            kfs = ops.get_random_keyframe(config, T)
            keyframes.append(kfs)
    
    # Ensure the target frame is included in keyframes for each sequence
    for b in range(B):
        if target_frame_idx not in keyframes[b]:
            keyframes[b].append(target_frame_idx)
            keyframes[b] = sorted(keyframes[b])
    
    # Results to return
    final_motions = []
    final_contacts = []
    final_keyframes = keyframes.copy()
    
    # Step 2: Generate transitions between keyframes
    for b in range(B):
        batch_keyframes = keyframes[b]
        
        # Segments will be stitched into this list
        final_motion_segments = []
        final_contact_segments = []
        
        # Process each pair of keyframes
        for i in range(len(batch_keyframes) - 1):
            kf_start, kf_end = batch_keyframes[i], batch_keyframes[i+1]
            
            # For the first segment, use original context frames
            if i == 0:
                # Get initial context frames from motion
                overlap_start = 0
                overlap_end = kf_start + 1  # Include the keyframe
                
                # Extract segment data
                context_frames = motion[b:b+1, overlap_start:overlap_end].clone()
                context_phase = GT_phase[b:b+1, overlap_start:overlap_end].clone() if GT_phase is not None else None
                context_traj = normalized_GT_traj[b:b+1, overlap_start:overlap_end].clone() if normalized_GT_traj is not None else None
            else:
                # For subsequent segments, use the overlap frames from the previous generated segment
                overlap_frames = min(config.overlap_frames, len(final_motion_segments))
                if overlap_frames > 0:
                    # Use the last overlap_frames from previous segment as context
                    generated_len = len(final_motion_segments)
                    overlap_start = max(0, generated_len - overlap_frames)
                    
                    # Stack the previously generated frames
                    stacked_segments = torch.stack(final_motion_segments[overlap_start:], dim=0)  # [frames, D]
                    
                    # Reshape to match the expected dimensions [B, frames, D]
                    stacked_segments = stacked_segments.unsqueeze(0)  # [1, frames, D]
                    
                    # Use as context frames
                    context_frames = stacked_segments
                    
                    # Make sure we have the right number of frames
                    if context_frames.shape[1] > overlap_frames:
                        context_frames = context_frames[:, -overlap_frames:]
                    
                    # For phase and trajectory, use GT data aligned with current segment
                    context_phase = GT_phase[b:b+1, kf_start-overlap_frames+1:kf_start+1].clone() if GT_phase is not None else None
                    context_traj = normalized_GT_traj[b:b+1, kf_start-overlap_frames+1:kf_start+1].clone() if normalized_GT_traj is not None else None
            
            # Target frame (end keyframe)
            target_frame = motion[b:b+1, kf_end:kf_end+1].clone()
            target_phase = GT_phase[b:b+1, kf_end:kf_end+1].clone() if GT_phase is not None else None
            target_traj = normalized_GT_traj[b:b+1, kf_end:kf_end+1].clone() if normalized_GT_traj is not None else None
            
            # Construct segment by combining context and target
            segment_motion = torch.cat([context_frames, target_frame], dim=1)
            segment_phase = torch.cat([context_phase, target_phase], dim=1) if GT_phase is not None else None
            segment_traj = torch.cat([context_traj, target_traj], dim=1) if normalized_GT_traj is not None else None
            
            # Create draft motion by linear interpolation between context and target
            segment_keyframes = [0, segment_motion.shape[1] - 1]  # First and last frames
            draft_motion = ops.interpolate_motion_by_keyframes(segment_motion, segment_keyframes)
            
            # Midway targets for SegmentNet - mark only the end of overlap as a constraint
            overlap_size = context_frames.shape[1]
            midway_targets = [overlap_size - 1] if overlap_size > 0 else []
            
            # Forward pass through SegmentNet
            seg_out = seg_model.forward(draft_motion, midway_targets, phase=segment_phase, traj=segment_traj)
            
            refined_motion = seg_out["motion"]
            refined_contact = seg_out["contact"]
            
            # Skip the overlap portion for all segments except the first one
            start_idx = 0 if i == 0 else overlap_size
            
            # Extract the relevant portion of the generated motion (excluding overlap with previous segment)
            segment_result = refined_motion[0, start_idx:-1]  # Exclude the target frame too
            contact_result = refined_contact[0, start_idx:-1] if refined_contact is not None else None
            
            # Add to the final segments list
            for frame_idx in range(segment_result.shape[0]):
                final_motion_segments.append(segment_result[frame_idx])
                if contact_result is not None:
                    final_contact_segments.append(contact_result[frame_idx])
        
        # Add the final keyframe
        final_frame = motion[b, batch_keyframes[-1]]
        final_motion_segments.append(final_frame)
        
        if GT_contact is not None:
            final_contact_frame = GT_contact[b, batch_keyframes[-1]]
            final_contact_segments.append(final_contact_frame)
        
        # Combine all segments into the final motion
        final_motion = torch.stack(final_motion_segments, dim=0).unsqueeze(0)
        final_motion = final_motion * std + mean
        
        final_motions.append(final_motion)
        
        if GT_contact is not None:
            final_contact = torch.stack(final_contact_segments, dim=0).unsqueeze(0)
            final_contacts.append(final_contact)
    
    # Combine results from all batches
    combined_motion = torch.cat(final_motions, dim=0)
    
    # Create modified frames info
    modified_frames_info = {
        "positions": modified_positions,
        "replacement_data": replacement_data
    }
    
    # Print keyframe positions for debugging
    print("Replace Target With Segment Mode - Keyframes Positions:")
    if keyframes is not None and len(keyframes) > 0:
        kf_list = keyframes[0] if keyframes[0] else []
        print(f"  Sequence 0 keyframes: {kf_list}")
    print()  # Add empty line for readability
    
    result = {
        "motion": combined_motion,
        "keyframes": final_keyframes,
        "modified_frames": modified_frames_info,
    }
    
    if GT_contact is not None:
        combined_contact = torch.cat(final_contacts, dim=0)
        result["contact"] = combined_contact
    
    return result

@torch.no_grad()
def _segment_net_transition(config,
                           kf_model: ContextTransformer,
                           seg_model,
                           GT_motion,
                           mean,
                           std,
                           GT_contact,
                           GT_phase=None,
                           GT_traj=None,
                           GT_score=None,
                           traj_mean=None,
                           traj_std=None,
                           kf_sampling=["score"]):
    """
    Generate motion using the SegmentNet approach:
    1. Use KeyframeNet to predict keyframes
    2. For each pair of adjacent keyframes, generate a transition using SegmentNet
    3. Stitch these transitions together to create the full motion

    Args:
        config: SegmentNet configuration
        kf_model: KeyframeNet model (ContextTransformer)
        seg_model: SegmentNet model
        GT_motion: Ground truth motion data
        mean, std: Motion statistics for normalization
        GT_contact: Ground truth contact data
        GT_phase, GT_traj, GT_score: Optional additional data
        traj_mean, traj_std: Trajectory statistics for normalization
        kf_sampling: Keyframe sampling method
    """
    B, T, D = GT_motion.shape
    device = GT_motion.device
    
    # Normalize motion and trajectories
    motion = (GT_motion - mean) / std
    if config.use_traj and GT_traj is not None:
        normalized_GT_traj = (GT_traj - traj_mean) / traj_std
    else:
        normalized_GT_traj = None
    
    # Step 1: Forward KeyframeNet to get keyframes
    with torch.no_grad():
        ctx_out, midway_targets = kf_model.forward(motion, phase=GT_phase, traj=normalized_GT_traj, train=False)
        
        # Get predicted scores for keyframe selection
        if config.use_score and GT_score is not None:
            ctx_score = ctx_out["score"]
            
            # Get keyframes based on scores
            if kf_sampling[0] == "score":
                keyframes = ops.get_keyframes_by_score(config, ctx_score)
            elif kf_sampling[0] == "threshold":
                keyframes = ops.get_keyframes_by_score_threshold(config, ctx_score, threshold=kf_sampling[1])
            elif kf_sampling[0] == "topk":
                keyframes = ops.get_keyframes_by_topk(config, ctx_score, top=kf_sampling[1])
            elif kf_sampling[0] == "random":
                keyframes = ops.get_keyframes_by_random(config, ctx_score, prob=kf_sampling[1])
            elif kf_sampling[0] == "uniform":
                keyframes = ops.get_keyframes_by_uniform(config, ctx_score, step=kf_sampling[1])
            else:
                raise NotImplementedError(f"Invalid keyframe sampling method: {kf_sampling}")
        else:
            # Fallback to random keyframes if scores not available
            keyframes = []
            for b in range(B):
                kfs = ops.get_random_keyframe(config, T)
                keyframes.append(kfs)
    
    # Results to return
    final_motions = []
    final_contacts = []
    final_keyframes = keyframes.copy()
    
    # Step 2: Generate transitions between keyframes
    for b in range(B):
        batch_keyframes = keyframes[b]
        
        # Segments will be stitched into this list
        final_motion_segments = []
        final_contact_segments = []
        
        # Process each pair of keyframes
        for i in range(len(batch_keyframes) - 1):
            kf_start, kf_end = batch_keyframes[i], batch_keyframes[i+1]
            
            # For the first segment, use original context frames
            if i == 0:
                # Get initial context frames from GT
                overlap_start = 0
                overlap_end = kf_start + 1  # Include the keyframe
                
                # Extract segment data
                context_frames = motion[b:b+1, overlap_start:overlap_end].clone()
                context_phase = GT_phase[b:b+1, overlap_start:overlap_end].clone() if GT_phase is not None else None
                context_traj = normalized_GT_traj[b:b+1, overlap_start:overlap_end].clone() if normalized_GT_traj is not None else None
            else:
                # For subsequent segments, use the overlap frames from the previous generated segment
                overlap_frames = min(config.overlap_frames, len(final_motion_segments))
                if overlap_frames > 0:
                    # Use the last overlap_frames from previous segment as context
                    generated_len = len(final_motion_segments)
                    overlap_start = max(0, generated_len - overlap_frames)
                    
                    # Stack the previously generated frames
                    stacked_segments = torch.stack(final_motion_segments[overlap_start:], dim=0)  # [frames, D]
                    
                    # Reshape to match the expected dimensions [B, frames, D]
                    stacked_segments = stacked_segments.unsqueeze(0)  # [1, frames, D]
                    
                    # Use as context frames
                    context_frames = stacked_segments
                    
                    # Make sure we have the right number of frames
                    if context_frames.shape[1] > overlap_frames:
                        context_frames = context_frames[:, -overlap_frames:]
                    
                    # For phase and trajectory, use GT data aligned with current segment
                    context_phase = GT_phase[b:b+1, kf_start-overlap_frames+1:kf_start+1].clone() if GT_phase is not None else None
                    context_traj = normalized_GT_traj[b:b+1, kf_start-overlap_frames+1:kf_start+1].clone() if normalized_GT_traj is not None else None
            
            # Target frame (end keyframe)
            target_frame = motion[b:b+1, kf_end:kf_end+1].clone()
            target_phase = GT_phase[b:b+1, kf_end:kf_end+1].clone() if GT_phase is not None else None
            target_traj = normalized_GT_traj[b:b+1, kf_end:kf_end+1].clone() if normalized_GT_traj is not None else None
            
            # Construct segment by combining context and target
            segment_motion = torch.cat([context_frames, target_frame], dim=1)
            segment_phase = torch.cat([context_phase, target_phase], dim=1) if GT_phase is not None else None
            segment_traj = torch.cat([context_traj, target_traj], dim=1) if normalized_GT_traj is not None else None
            
            # Create draft motion by linear interpolation between context and target
            segment_keyframes = [0, segment_motion.shape[1] - 1]  # First and last frames
            draft_motion = ops.interpolate_motion_by_keyframes(segment_motion, segment_keyframes)
            
            # Midway targets for SegmentNet - mark only the end of overlap as a constraint
            overlap_size = context_frames.shape[1]
            midway_targets = [overlap_size - 1] if overlap_size > 0 else []
            
            # Forward pass through SegmentNet
            seg_out = seg_model.forward(draft_motion, midway_targets, phase=segment_phase, traj=segment_traj)
            
            refined_motion = seg_out["motion"]
            refined_contact = seg_out["contact"]
            
            # Skip the overlap portion for all segments except the first one
            start_idx = 0 if i == 0 else overlap_size
            
            # Extract the relevant portion of the generated motion (excluding overlap with previous segment)
            segment_result = refined_motion[0, start_idx:-1]  # Exclude the target frame too
            contact_result = refined_contact[0, start_idx:-1] if refined_contact is not None else None
            
            # Add to the final segments list
            for frame_idx in range(segment_result.shape[0]):
                final_motion_segments.append(segment_result[frame_idx])
                if contact_result is not None:
                    final_contact_segments.append(contact_result[frame_idx])
        
        # Add the final keyframe
        final_frame = motion[b, batch_keyframes[-1]]
        final_motion_segments.append(final_frame)
        
        if GT_contact is not None:
            final_contact_frame = GT_contact[b, batch_keyframes[-1]]
            final_contact_segments.append(final_contact_frame)
        
        # Combine all segments into the final motion
        final_motion = torch.stack(final_motion_segments, dim=0).unsqueeze(0)
        final_motion = final_motion * std + mean
        
        final_motions.append(final_motion)
        
        if GT_contact is not None:
            final_contact = torch.stack(final_contact_segments, dim=0).unsqueeze(0)
            final_contacts.append(final_contact)
    
    # Combine results from all batches
    combined_motion = torch.cat(final_motions, dim=0)
    
    # Ensure the output has the same number of frames as the input
    B, orig_T, D = GT_motion.shape
    if combined_motion.shape[1] != orig_T:
        print(f"WARNING: SegmentNet output shape ({combined_motion.shape}) doesn't match input shape ({GT_motion.shape})")
        if combined_motion.shape[1] < orig_T:
            # Pad with zeros to match frame count
            padding = torch.zeros((combined_motion.shape[0], orig_T - combined_motion.shape[1], combined_motion.shape[2]), 
                                 device=combined_motion.device)
            padded_motion = torch.cat([combined_motion, padding], dim=1)
            print(f"Padded SegmentNet motion to {padded_motion.shape}")
            combined_motion = padded_motion
        else:
            # Trim to match frame count
            combined_motion = combined_motion[:, :orig_T]
            print(f"Trimmed SegmentNet motion to {combined_motion.shape}")
    
    result = {
        "motion": combined_motion,
        "keyframes": final_keyframes,
    }
    
    if GT_contact is not None:
        combined_contact = torch.cat(final_contacts, dim=0)
        
        # Ensure contact has the same number of frames
        if combined_contact.shape[1] != orig_T:
            if combined_contact.shape[1] < orig_T:
                # Pad with zeros
                padding = torch.zeros((combined_contact.shape[0], orig_T - combined_contact.shape[1], combined_contact.shape[2]), 
                                     device=combined_contact.device)
                padded_contact = torch.cat([combined_contact, padding], dim=1)
                print(f"Padded SegmentNet contact to {padded_contact.shape}")
                combined_contact = padded_contact
            else:
                # Trim
                combined_contact = combined_contact[:, :orig_T]
                print(f"Trimmed SegmentNet contact to {combined_contact.shape}")
        
        result["contact"] = combined_contact
    
    return result

def traj_pos_error(GT_traj, pred_motion, ctx_frames=10):
    B, T, D = pred_motion.shape
    GT_pos = GT_traj[:, :, 0:2]
    _, pred_pos = torch.split(pred_motion, [D-3, 3], dim=-1)
    pred_pos = pred_pos[..., (0, 2)]

    norm = torch.sqrt(torch.sum((GT_pos - pred_pos) ** 2, dim=-1))
    return torch.mean(norm[:, ctx_frames:-1]).item() * 100