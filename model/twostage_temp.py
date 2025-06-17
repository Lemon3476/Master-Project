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
        self.decoupled_encoders = config.get("decoupled_encoders", False)
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
        if self.decoupled_encoders or self.decoupled_traj_encoder or self.decoupled_phase_encoder:
            # Determine main encoder input dimensions
            main_input_dim = self.d_motion + self.config.d_mask
            
            # 1. Main Encoder (always processes motion and mask)
            self.pose_encoder = nn.Sequential(
                nn.Linear(main_input_dim, self.config.d_encoder_h),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.config.d_encoder_h, self.config.d_model),
                nn.PReLU(),
                nn.Dropout(self.dropout)
            )
            
            # 2. Trajectory Encoder (optional)
            if self.use_traj and (self.decoupled_encoders or self.decoupled_traj_encoder):
                self.traj_encoder = nn.Sequential(
                    nn.Linear(self.d_traj, self.config.d_encoder_h),
                    nn.PReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.config.d_encoder_h, self.config.d_model),
                    nn.PReLU(),
                    nn.Dropout(self.dropout)
                )
                
            # 3. Phase Encoder (optional)
            if self.use_phase and self.decoupled_phase_encoder:
                self.phase_encoder = nn.Sequential(
                    nn.Linear(self.d_phase, self.config.d_encoder_h),
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

        # Phase decoder
        if self.use_phase:
            self.phase_decoder = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_decoder_h),
                nn.PReLU(),
                nn.Linear(self.config.d_decoder_h, self.d_phase)
            )
        
        # Trajectory decoder (optional)
        if self.use_traj and self.decoupled_traj_decoder:
            self.traj_decoder = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_decoder_h),
                nn.PReLU(),
                nn.Linear(self.config.d_decoder_h, self.d_traj)
            )

        # Spatio-Temporal Gating Network - uses both phase and trajectory as input for enhanced control
        if self.use_phase:
            # Determine gating input dimensions based on available features
            gating_input_dim = self.d_phase  # Base: phase only
            
            # Add trajectory dimension if using decoupled trajectory decoder
            if self.use_traj and self.decoupled_traj_decoder:
                gating_input_dim += self.d_traj  # Add trajectory dimension for spatial awareness
            
            # Create enhanced gating network
            self.gating = nn.Sequential(
                nn.Linear(gating_input_dim, self.config.d_gating_h),
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

        # Encode inputs with decoupled encoders
        if self.decoupled_encoders or self.decoupled_traj_encoder or self.decoupled_phase_encoder:
            # 1. Main motion encoder with mask
            main_input = torch.cat([motion * data_mask, data_mask], dim=-1)
            motion_embedding = self.pose_encoder(main_input)
            
            # Initialize transformer input with motion embedding
            x = motion_embedding
            
            # 2. Add phase embedding if using decoupled phase encoder
            if self.use_phase and phase is not None and self.decoupled_phase_encoder:
                # Apply mask to phase before encoding
                masked_phase = phase * data_mask
                phase_embedding = self.phase_encoder(masked_phase)
                # Add phase embedding
                x = x + phase_embedding
            
            # 3. Add trajectory embedding if using decoupled trajectory encoder
            if self.use_traj and traj is not None and (self.decoupled_encoders or self.decoupled_traj_encoder):
                traj_embedding = self.traj_encoder(traj)
                # Add trajectory embedding
                x = x + traj_embedding
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

        # Store transformer output for decoders
        zL = x
        
        # Phase Decoding
        if self.use_phase:
            predicted_phase = self.phase_decoder(zL)
        else:
            predicted_phase = None
            
        # Trajectory Decoding (if applicable)
        if self.use_traj and self.decoupled_traj_decoder:
            predicted_traj = self.traj_decoder(zL)
        else:
            predicted_traj = None
            
        # 3. Create gating signal - ONLY use phase for gating
        if self.use_phase and predicted_phase is not None:
            # Phase-only gating for MoE control
            expert_weights = self.gating(predicted_phase)
                
            # 4. Final Motion Generation with MoE - from transformer output zL
            motion_x = zL  # Use transformer output for motion decoding
            for i in range(len(self.motion_decoder) // 2):
                d_model = motion_x.shape[-1]
                motion_x = motion_x.reshape(B*T, 1, d_model).transpose(0, 1)
                motion_x = self.motion_decoder[i*2](motion_x)
                motion_x = motion_x.transpose(0, 1).reshape(B, T, self.num_experts, -1)
                motion_x = torch.sum(motion_x * expert_weights[..., None], dim=-2)
                motion_x = self.motion_decoder[i*2+1](motion_x)
            
            # motion decoder - last layer
            d_model = motion_x.shape[-1]
            motion_x = motion_x.reshape(B*T, 1, d_model).transpose(0, 1)
            motion_x = self.motion_decoder[-1](motion_x)
            motion_x = motion_x.transpose(0, 1).reshape(B, T, self.num_experts, -1)
            motion_x = torch.sum(motion_x * expert_weights[..., None], dim=-2)
            
            # Set output
            x = motion_x
        else:
            # Non-phase case: use standard decoder
            x = self.decoder(x)

        # 6. Prepare outputs
        res = {
            "motion": x
        }
        
        # Add phase output if available
        if self.use_phase:
            res["phase"] = predicted_phase
            
        # Add trajectory output if available
        if self.use_traj and self.decoupled_traj_decoder:
            res["traj"] = predicted_traj
            
        # Handle score if used
        if self.use_score:
            x, score = torch.split(x, [self.d_motion, self.d_score], dim=-1)
            score = torch.sigmoid(score)
            res["motion"] = x
            res["score"] = score

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
        self.decoupled_encoders = config.get("decoupled_encoders", False)
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
        if self.decoupled_encoders or self.decoupled_traj_encoder or self.decoupled_phase_encoder:
            # Determine main encoder input dimensions
            main_input_dim = self.d_motion + self.config.d_mask
            
            # 1. Main Encoder (always processes motion and mask)
            self.pose_encoder = nn.Sequential(
                nn.Linear(main_input_dim, self.config.d_encoder_h),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.config.d_encoder_h, self.config.d_model),
                nn.PReLU(),
                nn.Dropout(self.dropout)
            )
            
            # 2. Trajectory Encoder (optional)
            if self.use_traj and (self.decoupled_encoders or self.decoupled_traj_encoder):
                self.traj_encoder = nn.Sequential(
                    nn.Linear(self.d_traj, self.config.d_encoder_h),
                    nn.PReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.config.d_encoder_h, self.config.d_model),
                    nn.PReLU(),
                    nn.Dropout(self.dropout)
                )
                
            # 3. Phase Encoder (optional)
            if self.use_phase and self.decoupled_phase_encoder:
                self.phase_encoder = nn.Sequential(
                    nn.Linear(self.d_phase, self.config.d_encoder_h),
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

        # Phase decoder
        if self.use_phase:
            self.phase_decoder = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_decoder_h),
                nn.PReLU(),
                nn.Linear(self.config.d_decoder_h, self.d_phase)
            )
        
        # Trajectory decoder (optional)
        if self.use_traj and self.decoupled_traj_decoder:
            self.traj_decoder = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_decoder_h),
                nn.PReLU(),
                nn.Linear(self.config.d_decoder_h, self.d_traj)
            )

        # Spatio-Temporal Gating Network - uses both phase and trajectory as input for enhanced control
        if self.use_phase:
            # Determine gating input dimensions based on available features
            gating_input_dim = self.d_phase  # Base: phase only
            
            # Add trajectory dimension if using decoupled trajectory decoder
            if self.use_traj and self.decoupled_traj_decoder:
                gating_input_dim += self.d_traj  # Add trajectory dimension for spatial awareness
            
            # Create enhanced gating network
            self.gating = nn.Sequential(
                nn.Linear(gating_input_dim, self.config.d_gating_h),
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
        # motion: (B, T, d_motion)
        # phase:  (B, T, d_phase)
        # traj:   (B, T, d_traj)

        # data mask
        data_mask = self.get_data_mask(T, midway_targets).to(device)
        data_mask = data_mask.expand(B, T, -1)

        # Store inputs for residual connections
        motion_in = motion
        if self.use_phase and phase is not None:
            phase_in = phase

        # Encode inputs
        if self.decoupled_encoders or self.decoupled_traj_encoder or self.decoupled_phase_encoder:
            # Prepare main input
            main_input = torch.cat([motion, data_mask], dim=-1)
            x = self.pose_encoder(main_input)
            
            # Add trajectory embedding if using decoupled trajectory encoder
            if self.use_traj and traj is not None and (self.decoupled_encoders or self.decoupled_traj_encoder):
                traj_embedding = self.traj_encoder(traj)
                x = x + traj_embedding
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

        # Store transformer output for decoders
        zL = x
        
        # Phase Decoding
        if self.use_phase:
            phase_out = self.phase_decoder(zL)
            # Apply residual connection if configured
            predicted_phase = phase_in + phase_out if self.config.residual else phase_out
        else:
            predicted_phase = None
            
        # Trajectory Decoding (if applicable)
        if self.use_traj and self.decoupled_traj_decoder:
            predicted_traj = self.traj_decoder(zL)
        else:
            predicted_traj = None
            
        # 3. Create gating signal - ONLY use phase for gating
        if self.use_phase and predicted_phase is not None:
            # Phase-only gating for MoE control
            expert_weights = self.gating(predicted_phase)
                
            # 4. Final Motion Generation with MoE - from transformer output zL
            motion_x = zL  # Use transformer output for motion decoding
            for i in range(len(self.motion_decoder) // 2):
                d_model = motion_x.shape[-1]
                motion_x = motion_x.reshape(B*T, 1, d_model).transpose(0, 1)
                motion_x = self.motion_decoder[i*2](motion_x)
                motion_x = motion_x.transpose(0, 1).reshape(B, T, self.num_experts, -1)
                motion_x = torch.sum(motion_x * expert_weights[..., None], dim=-2)
                motion_x = self.motion_decoder[i*2+1](motion_x)
            
            # motion decoder - last layer
            d_model = motion_x.shape[-1]
            motion_x = motion_x.reshape(B*T, 1, d_model).transpose(0, 1)
            motion_x = self.motion_decoder[-1](motion_x)
            motion_x = motion_x.transpose(0, 1).reshape(B, T, self.num_experts, -1)
            motion_x = torch.sum(motion_x * expert_weights[..., None], dim=-2)
            
            # Set output
            x = motion_x
        else:
            # Non-phase case: use standard decoder
            x = self.decoder(x)
        
        # Split motion and contact
        x, contact = torch.split(x, [M, 4], dim=-1)
        contact = torch.sigmoid(contact)

        # Apply residual connection to motion if configured
        if self.config.residual:
            x = motion_in + x

        # 6. Prepare outputs
        res = {
            "motion": x,
            "contact": contact
        }
        
        # Add phase output if available
        if self.use_phase:
            res["phase"] = predicted_phase
            
        # Add trajectory output if available
        if self.use_traj and self.decoupled_traj_decoder:
            res["traj"] = predicted_traj

        return res