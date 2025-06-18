import random
import numpy as np

import torch
import torch.nn as nn

from . import transformer, multilinear
from utils.dataset import MotionDataset

class ContextTransformer(nn.Module):
    def __init__(self, config, dataset: MotionDataset):
        super(ContextTransformer, self).__init__()
        self.config = config

        # data to use
        self.use_phase = config.use_phase
        self.use_traj  = config.use_traj
        self.use_score = config.use_score
        self.decoupled_traj_encoder = config.get("decoupled_traj_encoder", False)
        self.decoupled_phase_encoder = config.get("decoupled_phase_encoder", False)
        self.decoupled_traj_decoder = config.get("decoupled_traj_decoder", False)

        # input dimension
        self.d_motion    = dataset.motion_dim
        self.d_phase     = dataset.phase_dim if self.use_phase else 0
        self.d_traj      = dataset.traj_dim  if self.use_traj  else 0
        self.d_score     = dataset.score_dim if self.use_score else 0
        self.num_experts = self.d_phase // 2

        # model parameters
        self.d_mask = config.d_mask
        self.dropout = config.dropout
        self.pre_lnorm = config.pre_lnorm
        self.n_layer = config.n_layer

        # network modules
        # Determine encoder architecture based on configuration
        if self.decoupled_traj_encoder or self.decoupled_phase_encoder:
            # Decoupled encoders architecture
            # 1. Pose+Mask Encoder (always present)
            pose_input_dim = self.d_motion + self.config.d_mask
            if not self.decoupled_phase_encoder and self.use_phase:
                # Include phase in pose encoder if phase is not decoupled
                pose_input_dim += self.d_phase
            
            self.pose_encoder = nn.Sequential(
                nn.Linear(pose_input_dim, self.config.d_encoder_h),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.config.d_encoder_h, self.config.d_model),
                nn.PReLU(),
                nn.Dropout(self.dropout)
            )
            
            # 2. Phase Encoder (if decoupled)
            if self.decoupled_phase_encoder and self.use_phase:
                self.phase_encoder = nn.Sequential(
                    nn.Linear(self.d_phase, self.config.d_encoder_h),
                    nn.PReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.config.d_encoder_h, self.config.d_model),
                    nn.PReLU(),
                    nn.Dropout(self.dropout)
                )
            
            # 3. Trajectory Encoder (if decoupled)
            if self.decoupled_traj_encoder and self.use_traj:
                self.traj_encoder = nn.Sequential(
                    nn.Linear(self.d_traj, self.config.d_encoder_h),
                    nn.PReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.config.d_encoder_h, self.config.d_model),
                    nn.PReLU(),
                    nn.Dropout(self.dropout)
                )
        else:
            # Original unified encoder
            self.encoder = nn.Sequential(
                nn.Linear(self.d_motion + self.config.d_mask + self.d_phase + self.d_traj, self.config.d_encoder_h),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.config.d_encoder_h, self.config.d_model),
                nn.PReLU(),
                nn.Dropout(self.dropout)
            )

        # Decoders
        if self.use_phase:
            # phase decoder
            self.phase_decoder = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_decoder_h),
                nn.PReLU(),
                nn.Linear(self.config.d_decoder_h, self.d_phase)
            )

            # gating network
            self.gating = nn.Sequential(
                nn.Linear(self.d_phase, self.config.d_gating_h),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.config.d_gating_h, self.config.d_gating_h),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.config.d_gating_h, self.num_experts),
                nn.Softmax(dim=-1)
            )

            # motion decoder
            self.motion_decoder = nn.ModuleList()
            input_dims = [self.config.d_model, self.config.d_decoder_h]
            for input_dim in input_dims:
                self.motion_decoder.append(multilinear.MultiLinear(self.num_experts, input_dim, self.config.d_decoder_h))
                self.motion_decoder.append(nn.PReLU())
            self.motion_decoder.append(multilinear.MultiLinear(self.num_experts, self.config.d_decoder_h, self.d_motion + self.d_score))

        else:
            self.decoder = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_decoder_h),
                nn.PReLU(),
                nn.Linear(self.config.d_decoder_h, self.d_motion + self.d_score)
            )
            
        # Trajectory decoder (if configured)
        if self.use_traj and self.decoupled_traj_decoder:
            self.traj_decoder = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_decoder_h),
                nn.PReLU(),
                nn.Linear(self.config.d_decoder_h, self.d_traj)
            )
            
            # Combined gating network for phase and trajectory (if needed)
            if self.use_phase and self.config.get("combined_gating", False):
                combined_input_dim = self.d_phase + self.d_traj
                self.combined_gating = nn.Sequential(
                    nn.Linear(combined_input_dim, self.config.d_gating_h),
                    nn.PReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.config.d_gating_h, self.config.d_gating_h),
                    nn.PReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.config.d_gating_h, self.num_experts),
                    nn.Softmax(dim=-1)
                )

        self.rel_pos_layer = nn.Sequential(
            nn.Linear(1, self.config.d_head),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config.d_head, self.config.d_head),
            nn.Dropout(self.dropout)
        )

        self.keyframe_pos_layer = nn.Sequential(
            nn.Linear(2, self.config.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.Dropout(self.dropout)
        )

        self.layer_norm = nn.LayerNorm(self.config.d_model)
        self.att_layers = nn.ModuleList()
        self.pff_layers = nn.ModuleList()

        for i in range(self.n_layer):
            self.att_layers.append(
                transformer.RelMultiHeadedAttention(
                    self.config.n_head, self.config.d_model,
                    self.config.d_head, dropout=self.config.dropout,
                    pre_lnorm=self.config.pre_lnorm,
                    bias=self.config.atten_bias
                )
                # transformer.ConvBlockAttention(
                #     d_model=self.config.d_model,
                #     kernel_size=self.config.kernel_size,
                #     dropout=self.config.dropout,
                #     pre_lnorm=self.config.pre_lnorm,
                # )
            )

            self.pff_layers.append(
                transformer.PositionwiseFeedForward(
                    self.config.d_model, self.config.d_pff_inner,
                    dropout=self.config.dropout,
                    pre_lnorm=self.config.pre_lnorm
                )
            )

    def get_rel_pos_idx(self, window_len):
        pos_idx = torch.arange(-window_len + 1, window_len, dtype=torch.float32) # (1, seq, 1)
        pos_idx = pos_idx[None, :, None]
        return pos_idx

    def get_midway_targets(self, num_frames, train=True):
        if not train:
            return list()
        
        targets = list()
        midway_amount = int(self.config.midway_amount * (num_frames - self.config.context_frames - 1)) # 1 for the target frame
        target_candidates = random.sample(range(self.config.context_frames, num_frames-1), midway_amount)
        for target in target_candidates:
            if random.random() < self.config.midway_prob:
                targets.append(target)
        return targets
    
    def get_attention_mask(self, num_frames, midway_targets):
        atten_mask = torch.ones(num_frames, num_frames, dtype=torch.bool)
        atten_mask[:, -1] = False
        atten_mask[:, :self.config.context_frames] = False
        atten_mask[:, midway_targets] = False
        atten_mask = atten_mask.unsqueeze(0)

        # (1, seq, seq)
        return atten_mask
    
    def get_data_mask(self, num_frames, midway_targets):
        # 0 for unknown and 1 for known
        data_mask = torch.zeros((num_frames, self.config.d_mask), dtype=torch.float32)
        data_mask[:self.config.context_frames, :] = 1
        data_mask[-1, :] = 1
        data_mask[midway_targets] = 1

        # (seq, d_mask)
        return data_mask
    
    def get_keyframe_pos_indices(self, num_frames):
        # position index relative to context and target frame
        ctx_idx = torch.arange(num_frames, dtype=torch.float32)
        ctx_idx = ctx_idx - (self.config.context_frames - 1)
        ctx_idx = ctx_idx[..., None]

        tgt_idx = torch.arange(num_frames, dtype=torch.float32)
        tgt_idx = -(tgt_idx - (num_frames-1))
        tgt_idx = tgt_idx[..., None]

        # ctx_idx: (seq, 1), tgt_idx: (seq, 1)
        keyframe_pos_indices = torch.cat([ctx_idx, tgt_idx], dim=-1)

        # (1, seq, 2)
        return keyframe_pos_indices[None]
    
    def forward(self, motion, phase=None, traj=None, train=True):
        B, T, M = motion.shape
        device  = motion.device
        # motion: (B, T, d_motion)
        # phase:  (B, T, d_phase)
        # traj:   (B, T, d_traj)

        # random midway target frames
        midway_targets = self.get_midway_targets(T, train=train)

        # attention mask
        atten_mask = self.get_attention_mask(T, midway_targets).to(device)

        # data mask
        data_mask = self.get_data_mask(T, midway_targets).to(device)
        data_mask = data_mask.expand(B, T, 1)

        # position index relative to context and target frame
        keyframe_pos = self.get_keyframe_pos_indices(T).to(device)

        # input processing based on encoder architecture
        if self.decoupled_traj_encoder or self.decoupled_phase_encoder:
            # Prepare pose+mask input
            pose_input = torch.cat([motion * data_mask, data_mask], dim=-1)
            
            # Add phase to pose input if phase is not decoupled
            if self.use_phase and phase is not None and not self.decoupled_phase_encoder:
                pose_input = torch.cat([pose_input, phase * data_mask], dim=-1)
            
            # Encode pose inputs
            x = self.pose_encoder(pose_input)
            
            # Encode phase separately if decoupled
            if self.decoupled_phase_encoder and self.use_phase and phase is not None:
                phase_embedding = self.phase_encoder(phase * data_mask)
                # Feature fusion with phase embedding
                x = x + phase_embedding
            
            # Encode trajectory separately and apply soft guidance
            if self.decoupled_traj_encoder and self.use_traj and traj is not None:
                traj_embedding = self.traj_encoder(traj)
                
                # Apply trajectory guidance scale
                guidance_scale = self.config.get("traj_guidance_scale", 1.0)
                
                # Feature fusion by addition with scaling
                x = x + traj_embedding * guidance_scale
        else:
            # Original unified encoding
            x = torch.cat([motion * data_mask, data_mask], dim=-1)
            if self.use_phase and phase is not None:
                x = torch.cat([x, phase * data_mask], dim=-1)
            if self.use_traj and traj is not None:
                x = torch.cat([x, traj], dim=-1)

            # encoder
            x = self.encoder(x)

        # positional embedding
        x = x + self.keyframe_pos_layer(keyframe_pos)

        rel_pos_idx = self.get_rel_pos_idx(T).to(device)
        rel_pos_emb = self.rel_pos_layer(rel_pos_idx)

        # Transformer layers
        for i in range(self.n_layer):
            x = self.att_layers[i](x, rel_pos_emb, mask=atten_mask)
            x = self.pff_layers[i](x)
        if self.pre_lnorm:
            x = self.layer_norm(x)

        # decoder
        if self.use_phase:
            # phase decoder
            phase = self.phase_decoder(x)
            
        # Trajectory decoder (if configured)
        traj_out = None
        if self.use_traj and self.decoupled_traj_decoder:
            traj_out = self.traj_decoder(x)

        # Gating network (motion decoder)
        if self.use_phase:
            # Use phase for gating if available
            expert_weights = self.gating(phase) # (B, T, num_experts)
            
            # For scheme B: If configured to use both phase and traj for gating
            if self.use_traj and self.decoupled_traj_decoder and self.config.get("combined_gating", False):
                # Use pre-defined combined gating network (initialized in __init__)
                # Combine phase and traj for gating
                combined_input = torch.cat([phase, traj_out], dim=-1)
                expert_weights = self.combined_gating(combined_input)

            # motion decoder
            for i in range(len(self.motion_decoder) // 2):
                # x.shape == (B, T, d_model)
                # weight.shape == (num_experts, d_model, d_out)

                d_model = x.shape[-1]
                x = x.reshape(B*T, 1, d_model).transpose(0, 1) # (1, B*T, d_model)
                x = self.motion_decoder[i*2](x) # (num_experts, B*T, d_out)
                x = x.transpose(0, 1).reshape(B, T, self.num_experts, -1)
                x = torch.sum(x * expert_weights[..., None], dim=-2) # (B, T, d_out)
                x = self.motion_decoder[i*2+1](x)
            
            # motion decoder - last layer
            d_model = x.shape[-1]
            x = x.reshape(B*T, 1, d_model).transpose(0, 1) # (1, B*T, d_model)
            x = self.motion_decoder[-1](x) # (num_experts, B*T, d_out)
            x = x.transpose(0, 1).reshape(B, T, self.num_experts, -1)
            x = torch.sum(x * expert_weights[..., None], dim=-2)
        else:
            x = self.decoder(x)

        # output
        res = {
            "motion": x
        }
        if self.use_phase:
            res["phase"] = phase
        if self.use_score:
            x, score = torch.split(x, [self.d_motion, self.d_score], dim=-1)
            score = torch.sigmoid(score)
            res["motion"] = x
            res["score"] = score
        if self.use_traj and self.decoupled_traj_decoder:
            res["traj"] = traj_out

        return res, midway_targets

class DetailTransformer(nn.Module):
    def __init__(self, config, dataset: MotionDataset):
        super(DetailTransformer, self).__init__()
        self.config = config

        # data to use
        self.use_phase = config.use_phase
        self.use_traj  = config.use_traj
        self.use_score = config.use_score
        self.use_kf_emb = config.get("use_kf_emb", False)
        self.decoupled_traj_encoder = config.get("decoupled_traj_encoder", False)
        self.decoupled_phase_encoder = config.get("decoupled_phase_encoder", False)
        self.decoupled_traj_decoder = config.get("decoupled_traj_decoder", False)

        # input dimension
        self.d_motion    = dataset.motion_dim
        self.d_contact   = len(config.contact_joints)
        self.d_phase     = dataset.phase_dim if self.use_phase else 0
        self.d_traj      = dataset.traj_dim  if self.use_traj  else 0
        self.d_score     = dataset.score_dim if self.use_score else 0
        self.num_experts = self.d_phase // 2
        
        # model parameters
        self.d_mask = config.d_mask
        self.dropout = config.dropout
        self.pre_lnorm = config.pre_lnorm
        self.n_layer = config.n_layer

        # network modules
        # Determine encoder architecture based on configuration
        if self.decoupled_traj_encoder or self.decoupled_phase_encoder:
            # Decoupled encoders architecture
            # 1. Pose+Mask Encoder (always present)
            pose_input_dim = self.d_motion + self.config.d_mask
            if not self.decoupled_phase_encoder and self.use_phase:
                # Include phase in pose encoder if phase is not decoupled
                pose_input_dim += self.d_phase
            
            self.pose_encoder = nn.Sequential(
                nn.Linear(pose_input_dim, self.config.d_encoder_h),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.config.d_encoder_h, self.config.d_model),
                nn.PReLU(),
                nn.Dropout(self.dropout)
            )
            
            # 2. Phase Encoder (if decoupled)
            if self.decoupled_phase_encoder and self.use_phase:
                self.phase_encoder = nn.Sequential(
                    nn.Linear(self.d_phase, self.config.d_encoder_h),
                    nn.PReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.config.d_encoder_h, self.config.d_model),
                    nn.PReLU(),
                    nn.Dropout(self.dropout)
                )
            
            # 3. Trajectory Encoder (if decoupled)
            if self.decoupled_traj_encoder and self.use_traj:
                self.traj_encoder = nn.Sequential(
                    nn.Linear(self.d_traj, self.config.d_encoder_h),
                    nn.PReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.config.d_encoder_h, self.config.d_model),
                    nn.PReLU(),
                    nn.Dropout(self.dropout)
                )
        else:
            # Original unified encoder
            self.encoder = nn.Sequential(
                nn.Linear(self.d_motion + self.config.d_mask + self.d_phase + self.d_traj, self.config.d_encoder_h),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.config.d_encoder_h, self.config.d_model),
                nn.PReLU(),
                nn.Dropout(self.dropout)
            )
            
        if self.use_kf_emb:
            self.keyframe_pos_layer = nn.Sequential(
                nn.Linear(2, self.config.d_model),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.config.d_model, self.config.d_model),
                nn.Dropout(self.dropout)
            )

        # Decoders
        if self.use_phase:
            # phase decoder
            self.phase_decoder = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_decoder_h),
                nn.PReLU(),
                nn.Linear(self.config.d_decoder_h, self.d_phase)
            )

            # gating network
            self.gating = nn.Sequential(
                nn.Linear(self.d_phase, self.config.d_gating_h),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.config.d_gating_h, self.config.d_gating_h),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.config.d_gating_h, self.num_experts),
                nn.Softmax(dim=-1)
            )

            # motion decoder
            self.motion_decoder = nn.ModuleList()
            input_dims = [self.config.d_model, self.config.d_decoder_h]
            for input_dim in input_dims:
                self.motion_decoder.append(multilinear.MultiLinear(self.num_experts, input_dim, self.config.d_decoder_h))
                self.motion_decoder.append(nn.PReLU())
            self.motion_decoder.append(multilinear.MultiLinear(self.num_experts, self.config.d_decoder_h, self.d_motion + self.d_contact))

        # Trajectory decoder (if configured)
        if self.use_traj and self.decoupled_traj_decoder:
            self.traj_decoder = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_decoder_h),
                nn.PReLU(),
                nn.Linear(self.config.d_decoder_h, self.d_traj)
            )
            
            # Combined gating for phase and trajectory (Scheme B)
            if self.use_phase and self.config.get("combined_gating", False):
                self.combined_gating = nn.Sequential(
                    nn.Linear(self.d_phase + self.d_traj, self.config.d_gating_h),
                    nn.PReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.config.d_gating_h, self.config.d_gating_h),
                    nn.PReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.config.d_gating_h, self.num_experts),
                    nn.Softmax(dim=-1)
                )

        else:
            self.decoder = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_decoder_h),
                nn.PReLU(),
                nn.Linear(self.config.d_decoder_h, self.d_motion + self.d_contact)
            )

        self.rel_pos_layer = nn.Sequential(
            nn.Linear(1, self.config.d_head),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config.d_head, self.config.d_head),
            nn.Dropout(self.dropout)
        )

        self.layer_norm = nn.LayerNorm(self.config.d_model)
        self.att_layers = nn.ModuleList()
        self.pff_layers = nn.ModuleList()

        for i in range(self.n_layer):
            self.att_layers.append(
                transformer.RelMultiHeadedAttention(
                    self.config.n_head, self.config.d_model,
                    self.config.d_head, dropout=self.config.dropout,
                    pre_lnorm=self.config.pre_lnorm,
                    bias=self.config.atten_bias
                )
            )

            self.pff_layers.append(
                transformer.PositionwiseFeedForward(
                    self.config.d_model, self.config.d_pff_inner,
                    dropout=self.config.dropout,
                    pre_lnorm=self.config.pre_lnorm
                )
            )

    def get_rel_pos_emb(self, window_len, dtype, device):
        pos_idx = torch.arange(-window_len + 1, window_len,
                               dtype=dtype, device=device)
        pos_idx = pos_idx[None, :, None]        # (1, seq, 1)
        rel_pos_emb = self.rel_pos_layer(pos_idx)
        return rel_pos_emb

    def get_data_mask(self, num_frames, midway_targets):
        # 0 for unknown and 1 for known
        data_mask = torch.zeros((num_frames, self.config.d_mask), dtype=torch.float32)
        data_mask[:self.config.context_frames, :] = 1
        data_mask[-1, :] = 1
        if len(midway_targets) == 1:
            data_mask[midway_targets] = 1
        elif len(midway_targets) > 1:
            for target in midway_targets:
                data_mask[target] = 1

        # (seq, d_mask)
        return data_mask
    
    def get_rel_pos_idx(self, window_len):
        pos_idx = torch.arange(-window_len + 1, window_len, dtype=torch.float32) # (1, seq, 1)
        pos_idx = pos_idx[None, :, None]
        return pos_idx
    
    def get_keyframe_pos_indices(self, num_frames):
            # position index relative to context and target frame
            ctx_idx = torch.arange(num_frames, dtype=torch.float32)
            ctx_idx = ctx_idx - (self.config.context_frames - 1)
            ctx_idx = ctx_idx[..., None]

            tgt_idx = torch.arange(num_frames, dtype=torch.float32)
            tgt_idx = -(tgt_idx - (num_frames-1))
            tgt_idx = tgt_idx[..., None]

            # ctx_idx: (seq, 1), tgt_idx: (seq, 1)
            keyframe_pos_indices = torch.cat([ctx_idx, tgt_idx], dim=-1)

            # (1, seq, 2)
            return keyframe_pos_indices[None]
    
    def forward(self, motion, midway_targets, phase=None, traj=None, attention_mask=None):
        B, T, M = motion.shape
        device  = motion.device

        # data mask
        data_mask = self.get_data_mask(T, midway_targets).to(device)
        data_mask = data_mask.expand(B, T, -1)

        # Store inputs for residual connections
        motion_in = motion
        if self.use_phase and phase is not None:
            phase_in = phase

        # Input processing based on encoder architecture
        if self.decoupled_traj_encoder or self.decoupled_phase_encoder:
            # Prepare pose+mask input
            pose_input = torch.cat([motion, data_mask], dim=-1)
            
            # Add phase to pose input if phase is not decoupled
            if self.use_phase and phase is not None and not self.decoupled_phase_encoder:
                pose_input = torch.cat([pose_input, phase], dim=-1)
            
            # Encode pose inputs
            x = self.pose_encoder(pose_input)
            
            # Encode phase separately if decoupled
            if self.decoupled_phase_encoder and self.use_phase and phase is not None:
                phase_embedding = self.phase_encoder(phase)
                # Feature fusion with phase embedding
                x = x + phase_embedding
            
            # Encode trajectory separately and apply soft guidance
            if self.decoupled_traj_encoder and self.use_traj and traj is not None:
                traj_embedding = self.traj_encoder(traj)
                
                # Apply trajectory guidance scale
                guidance_scale = self.config.get("traj_guidance_scale", 1.0)
                
                # Feature fusion by addition with scaling
                x = x + traj_embedding * guidance_scale
        else:
            # Original unified encoding
            x = torch.cat([motion, data_mask], dim=-1)
            if self.use_phase and phase is not None:
                x = torch.cat([x, phase], dim=-1)
            if self.use_traj and traj is not None:
                x = torch.cat([x, traj], dim=-1)

            # encoder
            x = self.encoder(x)

        # Add keyframe positional embedding if configured
        if self.use_kf_emb:
            keyframe_pos = self.get_keyframe_pos_indices(T).to(device)
            x = x + self.keyframe_pos_layer(keyframe_pos)

        # relative positional encoding
        rel_pos_idx = self.get_rel_pos_idx(T).to(device)
        rel_pos_emb = self.rel_pos_layer(rel_pos_idx)

        # Transformer layers
        for i in range(self.n_layer):
            x = self.att_layers[i](x, rel_pos_emb, mask=attention_mask)
            x = self.pff_layers[i](x)
        if self.pre_lnorm:
            x = self.layer_norm(x)

        # decoder
        if self.use_phase:
            # phase decoder - residual connection
            phase_out = self.phase_decoder(x)
            phase = phase_in + phase_out if self.config.residual else phase_out
            
            # Trajectory decoder (if configured)
            if self.use_traj and self.decoupled_traj_decoder:
                traj_out = self.traj_decoder(x)
                if self.config.get("residual", True) and traj is not None:
                    traj_out = traj + traj_out

            # gating - choose appropriate gating mechanism based on configuration
            if self.use_traj and self.decoupled_traj_decoder and self.config.get("combined_gating", False):
                # Combined gating using both phase and trajectory (Scheme B)
                combined_input = torch.cat([phase, traj_out], dim=-1)
                expert_weights = self.combined_gating(combined_input)  # (B, T, num_experts)
            else:
                # Default: use only phase for gating
                expert_weights = self.gating(phase)  # (B, T, num_experts)

            # motion decoder
            for i in range(len(self.motion_decoder) // 2):
                # x.shape == (B, T, d_model)
                # weight.shape == (num_experts, d_model, d_out)

                d_model = x.shape[-1]
                x = x.reshape(B*T, 1, d_model).transpose(0, 1) # (1, B*T, d_model)
                x = self.motion_decoder[i*2](x) # (num_experts, B*T, d_out)
                x = x.transpose(0, 1).reshape(B, T, self.num_experts, -1)
                x = torch.sum(x * expert_weights[..., None], dim=-2) # (B, T, d_out)
                x = self.motion_decoder[i*2+1](x) # activation
            
            # motion decoder - last layer
            d_model = x.shape[-1]
            x = x.reshape(B*T, 1, d_model).transpose(0, 1) # (1, B*T, d_model)
            x = self.motion_decoder[-1](x) # (num_experts, B*T, d_out)
            x = x.transpose(0, 1).reshape(B, T, self.num_experts, -1)
            x = torch.sum(x * expert_weights[..., None], dim=-2)
        else:
            x = self.decoder(x)
        
        # split motion and contact
        x, contact = torch.split(x, [M, 4], dim=-1)
        contact = torch.sigmoid(contact)

        if self.config.residual:
            x = motion_in + x

        # return dict
        res = {
            "motion": x,
            "contact": contact
        }
        if self.use_phase:
            res["phase"] = phase
        if self.use_traj and self.decoupled_traj_decoder:
            res["traj"] = traj_out

        return res