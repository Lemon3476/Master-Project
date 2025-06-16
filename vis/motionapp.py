from __future__ import annotations
from typing import Union

import copy
import numpy as np
import torch
import glfw
import glm
import cv2
import os
from OpenGL.GL import *

from aPyOpenGL import agl, transforms as trf

H36M_BVH2FBX = {
    "Hips": "mixamorig:Hips",
    "Spine": "mixamorig:Spine",
    "Spine1": "mixamorig:Spine1",
    "Neck": "mixamorig:Neck",
    "Head": "mixamorig:Head",
    "LeftShoulder": "mixamorig:LeftShoulder",
    "LeftUpArm": "mixamorig:LeftArm",
    "LeftForeArm": "mixamorig:LeftForeArm",
    "LeftHand": "mixamorig:LeftHand",
    "LeftHandThumb": "mixamorig:LeftHand",
    "L_Wrist_End": "mixamorig:LeftHand",
    "RightShoulder": "mixamorig:RightShoulder",
    "RightUpArm": "mixamorig:RightArm",
    "RightForeArm": "mixamorig:RightForeArm",
    "RightHand": "mixamorig:RightHand",
    "RightHandThumb": "mixamorig:RightHand",
    "R_Wrist_End": "mixamorig:RightHand",
    "LeftUpLeg": "mixamorig:LeftUpLeg",
    "LeftLowLeg": "mixamorig:LeftLeg",
    "LeftFoot": "mixamorig:LeftFoot",
    "LeftToeBase": "mixamorig:LeftToeBase",
    "RightUpLeg": "mixamorig:RightUpLeg",
    "RightLowLeg": "mixamorig:RightLeg",
    "RightFoot": "mixamorig:RightFoot",
    "RightToeBase": "mixamorig:RightToeBase",
}


FBX2FBX = {
    "mixamorig:Hips": "mixamorig:Hips",
    "mixamorig:Spine": "mixamorig:Spine",
    "mixamorig:Spine1": "mixamorig:Spine1",
    "mixamorig:Spine2": "mixamorig:Spine2",
    "mixamorig:Neck": "mixamorig:Neck",
    "mixamorig:Head": "mixamorig:Head",
    "mixamorig:LeftShoulder": "mixamorig:LeftShoulder",
    "mixamorig:LeftArm": "mixamorig:LeftArm",
    "mixamorig:LeftForeArm": "mixamorig:LeftForeArm",
    "mixamorig:LeftHand": "mixamorig:LeftHand",
    "mixamorig:RightShoulder": "mixamorig:RightShoulder",
    "mixamorig:RightArm": "mixamorig:RightArm",
    "mixamorig:RightForeArm": "mixamorig:RightForeArm",
    "mixamorig:RightHand": "mixamorig:RightHand",
    "mixamorig:LeftUpLeg": "mixamorig:LeftUpLeg",
    "mixamorig:LeftLeg": "mixamorig:LeftLeg",
    "mixamorig:LeftFoot": "mixamorig:LeftFoot",
    "mixamorig:LeftToeBase": "mixamorig:LeftToeBase",
    "mixamorig:RightUpLeg": "mixamorig:RightUpLeg",
    "mixamorig:RightLeg": "mixamorig:RightLeg",
    "mixamorig:RightFoot": "mixamorig:RightFoot",
    "mixamorig:RightToeBase": "mixamorig:RightToeBase",
}

def _reshape(x, traj=False):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if traj:
        x = x[..., :2]
        x = np.concatenate([x[..., 0:1], np.zeros_like(x[..., 0:1]), x[..., 1:2]], axis=-1)
    return x.reshape(-1, x.shape[-1])

class MotionStruct:
    def __init__(
        self,
        features: Union[torch.Tensor, np.ndarray],
        num_batches: int,
        skeleton: agl.Skeleton,
        tag: str,
        traj: Union[torch.Tensor, np.ndarray] = None,
        kf_indices=None,
        contact=None,
    ):
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if len(features.shape) != 2:
            raise ValueError(f"features must be a 2D tensor, got {features.shape}")
        
        # convert local rotations to quaternions
        local_rots, root_pos = features[:, :-3], features[:, -3:]
        local_quats = trf.n_quat.from_ortho6d(local_rots.reshape(-1, skeleton.num_joints, 6)) # [B*T, J, 4]
        root_pos = root_pos.reshape(-1, 3)

        # set attributes
        self.num_batches = num_batches
        self.frame_per_batch = features.shape[0] // num_batches
        self.skeleton = skeleton
        self.poses = [agl.Pose(skeleton, lq, rp) for lq, rp in zip(local_quats, root_pos)]
        self.tag = tag
        self.traj = traj
        self.kf_indices = kf_indices
        self.contact = contact
        self.offset = np.array([0, 0, 0], dtype=np.float32)
        self.visible = False
        # Set color based on tag
        if tag == "Multi-IB":
            self.character_color = [0.8, 0.2, 0.2]  # Red for Multi-IB
        elif tag == "Intra-Batch":
            self.character_color = [0.2, 0.8, 0.2]  # Green for Intra-Batch
        elif tag == "Ours" and "replace_target" in tag:
            self.character_color = [0.2, 0.2, 0.8]  # Blue for replace_target mode
        else:
            self.character_color = [0.5, 0.5, 0.5]  # Default gray for all other modes (including Ours)

        # traj and contact rendering
        self.traj_spheres = agl.Render.sphere(0.05).albedo([0.8, 0.1, 0.1]).instance_num(min(features.shape[0] // num_batches, 100))
        self.curr_traj_sphere = agl.Render.sphere(0.05).albedo([0.1, 0.8, 0.1])
        self.contact_spheres = [ agl.Render.sphere(0.05).albedo([0.1, 0.8, 0.1]) for _ in range(4) ]
    
    def render(self, character, frame, alpha=1.0, traj=False):
        if not self.visible:
            return
        
        # Boundary check for frame index
        if frame < 0 or frame >= len(self.poses):
            return  # Skip rendering if frame is out of bounds
            
        character.set_pose(agl.Pose(self.skeleton, self.poses[frame].local_quats, self.poses[frame].root_pos + self.offset))
        # Apply custom color if available - must set before each draw call
        if hasattr(self, 'character_color'):
            character.meshes[0].materials[0].albedo = glm.vec3(self.character_color)
# print(f"Applied color {self.character_color} to {self.tag}")  # Debug disabled
        agl.Render.model(character).alpha(alpha).draw()

        if traj and self.traj is not None:
            try:
                traj_idx = np.arange(self.frame_per_batch) + (frame // self.frame_per_batch) * self.frame_per_batch
                curr_idx = (frame % self.frame_per_batch) + (frame // self.frame_per_batch) * self.frame_per_batch
                
                # Boundary check for trajectory indices
                if curr_idx >= 0 and curr_idx < len(self.traj):
                    other_idx = np.setdiff1d(traj_idx, curr_idx)
                    # Filter out invalid indices
                    other_idx = other_idx[(other_idx >= 0) & (other_idx < len(self.traj))]
                    
                    if len(other_idx) > 0:
                        self.traj_spheres.position(self.traj[other_idx] + self.offset).draw()
                    self.curr_traj_sphere.position(self.traj[curr_idx] + self.offset).draw()
            except (IndexError, ValueError):
                pass  # Skip trajectory rendering if error
        
    def render_tag(self, frame: int, offset=[0, 0.8, 0]):
        if not self.visible:
            return
        
        # Boundary check
        if frame < 0 or frame >= len(self.poses):
            return
            
        pos = self.poses[frame].root_pos + np.array(offset) + self.offset
        agl.Render.text(self.tag).position(pos).scale(0.5).draw()
    
    def render_xray(self, frame):
        if not self.visible:
            return
        
        # Boundary check
        if frame < 0 or frame >= len(self.poses):
            return
            
        pose = agl.Pose(self.skeleton, self.poses[frame].local_quats, self.poses[frame].root_pos + self.offset)
        agl.Render.skeleton(pose).draw()

    def render_contact(self, frame, contact=False, contact_idx=None):
        # contact
        if contact and contact_idx is not None and self.contact is not None:
            try:
                # Boundary check
                if frame < 0 or frame >= len(self.poses) or frame >= len(self.contact):
                    return
                    
                on_contact = self.contact[frame]
                _, gp = trf.n_quat.fk(self.poses[frame].local_quats, self.poses[frame].root_pos, self.skeleton)
                contact_pos = gp[contact_idx]
                for i, idx in enumerate(contact_idx):
                    if on_contact[i]:
                        self.contact_spheres[i].position(contact_pos[i] + self.offset).draw()
            except (IndexError, ValueError):
                pass  # Skip contact rendering if error
    
    
    def switch_visible(self):
        self.visible = not self.visible

    def get_base(self, frame):
        # Boundary check
        if frame < 0 or frame >= len(self.poses):
            # Return default values if frame is out of bounds
            return glm.vec3(0, 0, 0), glm.mat4(1.0)
            
        pos = self.poses[frame].root_pos + self.offset
        pos = glm.vec3(pos[0], 0, pos[2])

        dir = trf.n_quat.mul_vec(self.poses[frame].local_quats[0], np.array([0, 0, 1]))
        dir = np.array([dir[0], 0, dir[2]])
        dir = dir / (np.linalg.norm(dir) + 1e-8)

        q = trf.n_quat.between_vecs(np.array([0, 0, 1]), dir)
        q = trf.n_quat.to_rotmat(q)
        r = glm.mat3(*q.transpose().flatten())

        return pos, r

class MotionApp(agl.App):
    def __init__(
        self,
        motions: Union[list[torch.Tensor], list[np.ndarray], torch.Tensor, np.ndarray], # list of (B, T, D) tensors
        tags: Union[list[str], str],
        skeleton: agl.Skeleton,
        dataset: str = "lafan1",
        trajs: Union[list[torch.Tensor], list[np.ndarray]] = None, # list of (B, T, D) tensors
        contacts: Union[list[torch.Tensor], list[np.ndarray]] = None, # list of (B, T, 4) tensors
        contact_idx: list[int] = None,
        kf_indices: list[list[int]] = None,
        modified_frames: Union[list[dict], dict] = None,  # Information about modified frames
        paused: bool = True,  # Start in paused mode
        save_videos: bool = False,  # Whether to save videos
        output_dir: str = "output_videos",  # Output directory for videos
        record_sequence_idx: int = None,  # Specific sequence index to record (0-63)
        compare_mode: bool = False,  # Whether in compare mode (auto-play both modes)
        show_keyframes: bool = False  # Whether to show keyframes in compare mode
    ):
        super().__init__()

        self.motions = motions if isinstance(motions, list) else [motions]
        self.compare_mode = compare_mode
        # 3-stage playback: intra -> multi -> intra+multi
        self.playback_stages = ["intra", "multi", "combined"] if compare_mode else [None]
        self.current_stage = 0 if compare_mode else 0
        self.stage_frame_count = 0  # Track frames played in current stage
        self.compare_state = self.playback_stages[0] if compare_mode else None  # Legacy compatibility
        self.compare_completed = False  # Legacy compatibility
        self.compare_frame_count = 0  # Legacy compatibility
        self.tags = tags if isinstance(tags, list) else [tags]
        self.skeleton = skeleton
        self.dataset = dataset

        if contacts is None:
            self.contacts = [None for _ in range(len(motions))]
        else:
            self.contacts = contacts if isinstance(contacts, list) else [contacts]

        if trajs is None:
            self.trajs = [None for _ in range(len(motions))]
        else:
            self.trajs = trajs if isinstance(trajs, list) else [trajs]

        if kf_indices is None:
            self.kf_indices = [None for _ in range(len(motions))]
        else:
            self.kf_indices = kf_indices if isinstance(kf_indices, list) else [kf_indices]
            
        if modified_frames is None:
            self.modified_frames = [None for _ in range(len(motions))]
        else:
            self.modified_frames = modified_frames if isinstance(modified_frames, list) else [modified_frames]
            

        # motion info
        self.num_batches = self.motions[0].shape[0]
        self.frame_per_batch = self.motions[0].shape[1]
        self.total_frames = self.motions[0].shape[0] * self.motions[0].shape[1]
        self.contact_idx = contact_idx
        
        # video recording settings
        self.save_videos = save_videos
        self.output_dir = output_dir
        self.record_sequence_idx = record_sequence_idx
        self.video_writers = {}
        self.video_width = 1920
        self.video_height = 1080
        self.video_fps = 30
        
        
        # recording mode setup
        self.recording_mode = False
        self.recording_batch = 0
        self.recording_frame = 0
        
        # offscreen rendering setup
        if self.save_videos:
            self.fbo = None
            self.color_texture = None
            self.depth_renderbuffer = None

        # reshape
        self.motions = [_reshape(m) for m in self.motions]
        if self.trajs is not None:
            self.trajs = [_reshape(t, traj=True) for t in self.trajs]
        if self.contacts is not None:
            self.contacts = [_reshape(c) for c in self.contacts]

        # rendering settings
        self.move_character = False
        self._show = {
            "arrow": False,  # Hide coordinate arrows by default in all modes
            "target": True,
            "tag": False,   # Hide character name tags
            "xray": False,
            "info": True,
            "traj": True,
            "every_ten": False,
            "keyframe": show_keyframes,  # Control keyframes visibility with parameter
            "contact": False,
            "transition": False,
            "modified": True,  # Show modified poses by default
            "alpha": 0.2
        }
    
    def _switch_move_character(self):
        self.move_character = not self.move_character
        for idx, motion in enumerate(self.motions):
            motion.offset = np.array([(idx * 1.5 if self.move_character else 0), 0, 0], dtype=np.float32)
    
    def _switch_show(self, key):
        self._show[key] = not self._show[key]
    
    def _get_first_ours_index(self):
        """Get the index of the first 'Ours' motion"""
        for i, tag in enumerate(self.tags):
            if tag == "Ours" or tag.startswith("Ours-"):
                return i
        return 0
    
    def _setup_offscreen_rendering(self):
        """Setup offscreen rendering using framebuffer objects"""
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        
        # Create color texture
        self.color_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.video_width, self.video_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_texture, 0)
        
        # Create depth renderbuffer
        self.depth_renderbuffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_renderbuffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.video_width, self.video_height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_renderbuffer)
        
        # Check framebuffer completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("Framebuffer not complete!")
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def _start_video_recording(self, batch_idx):
        """Start video recording for a specific batch"""
        if not self.save_videos:
            return
            
        # ALWAYS ensure move_character is enabled for video recording
        # This is critical to make sure animations are separated in the videos
        self.move_character = True
        
        # Create videos and images subdirectories with sequence-specific structure
        if self.record_sequence_idx is not None:
            # Create sequence-specific subdirectory
            # Add _with_keyframes suffix if keyframes are shown
            keyframes_suffix = "_with_keyframes" if self._show["keyframe"] else ""
            sequence_subdir = f"batch_{batch_idx:03d}_seq_{self.record_sequence_idx:02d}{keyframes_suffix}"
            self.sequence_dir = os.path.join(self.output_dir, sequence_subdir)
            self.videos_dir = os.path.join(self.sequence_dir, "videos")
            self.images_dir = os.path.join(self.sequence_dir, "images")
            
            # Print information about directory structure
            print(f"Recording to directory: {self.sequence_dir}")
            if self._show["keyframe"]:
                print(f"Directory includes '_with_keyframes' suffix to indicate keyframe visibility")
        else:
            # Use original structure for batch recording
            # Add _with_keyframes suffix if keyframes are shown
            keyframes_suffix = "_with_keyframes" if self._show["keyframe"] else ""
            batch_subdir = f"batch_{batch_idx:03d}{keyframes_suffix}"
            self.batch_dir = os.path.join(self.output_dir, batch_subdir)
            self.videos_dir = os.path.join(self.batch_dir, "videos")
            self.images_dir = os.path.join(self.batch_dir, "images")
            
            # Print information about directory structure
            print(f"Recording to directory: {self.batch_dir}")
            if self._show["keyframe"]:
                print(f"Directory includes '_with_keyframes' suffix to indicate keyframe visibility")
        
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Apply offsets to ALL motions before creating video writers
        # This ensures consistent spacing from the beginning of recording
        visible_idx = 0
        for motion in self.motions:
            if motion.visible:
                # Apply consistent offset for each visible motion
                motion.offset = np.array([visible_idx * 1.5, 0, 0], dtype=np.float32)
                visible_idx += 1
        
        print(f"Applied offsets to {visible_idx} visible animations for video recording")
        
        # Create video writers for each visible motion
        for motion in self.motions:
            if motion.visible:
                if self.record_sequence_idx is not None:
                    # Simplified filename since sequence info is in directory name
                    filename = f"{motion.tag}.mp4"
                else:
                    filename = f"batch_{batch_idx:03d}_{motion.tag}.mp4"
                filepath = os.path.join(self.videos_dir, filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(filepath, fourcc, self.video_fps, (self.video_width, self.video_height))
                self.video_writers[motion.tag] = writer
                print(f"Created video writer for {motion.tag}: {filepath}")
        
        print(f"Total video writers created: {len(self.video_writers)}")
    
    def _stop_video_recording(self):
        """Stop video recording and release writers"""
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()
    
    def _start_compare_video_recording(self):
        """Start video recording for compare mode - creates intra_batch directory first"""
        if not self.save_videos:
            return
        
        # Create intra_batch directory first
        self._create_compare_directories("intra_batch")
        self._start_compare_mode_recording("intra_batch")
    
    def _create_compare_directories(self, mode_name):
        """Create directories for compare mode recording - use single compare directory"""
        # Create base output directory for compare mode (ignore mode_name, use single directory)
        base_output_dir = os.path.dirname(self.output_dir)  # Get parent of output dir
        self.current_mode_dir = os.path.join(base_output_dir, "lafan1_keyframe_compare")
        
        if self.record_sequence_idx is not None:
            # Add _with_keyframes suffix if keyframes are shown
            keyframes_suffix = "_with_keyframes" if self._show["keyframe"] else ""
            sequence_subdir = f"batch_000_seq_{self.record_sequence_idx:02d}{keyframes_suffix}"
            self.sequence_dir = os.path.join(self.current_mode_dir, sequence_subdir)
            self.videos_dir = os.path.join(self.sequence_dir, "videos")
            self.images_dir = os.path.join(self.sequence_dir, "images")
            
            # Print information about directory structure
            print(f"Recording compare mode to directory: {self.sequence_dir}")
            if self._show["keyframe"]:
                print(f"Directory includes '_with_keyframes' suffix to indicate keyframe visibility")
        else:
            # Add _with_keyframes suffix if keyframes are shown
            keyframes_suffix = "_with_keyframes" if self._show["keyframe"] else ""
            compare_subdir = f"compare{keyframes_suffix}"
            self.compare_dir = os.path.join(self.current_mode_dir, compare_subdir)
            self.videos_dir = os.path.join(self.compare_dir, "videos")
            self.images_dir = os.path.join(self.compare_dir, "images")
            
            # Print information about directory structure
            print(f"Recording compare mode to directory: {self.compare_dir}")
            if self._show["keyframe"]:
                print(f"Directory includes '_with_keyframes' suffix to indicate keyframe visibility")
        
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
    
    def _start_compare_mode_recording(self, mode_name):
        """Start recording for current compare mode"""
        # Create video writers for each visible motion
        for motion in self.motions:
            if motion.visible:
                if self.record_sequence_idx is not None:
                    filename = f"{motion.tag}.mp4"
                else:
                    filename = f"batch_000_{motion.tag}.mp4"
                filepath = os.path.join(self.videos_dir, filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(filepath, fourcc, self.video_fps, (self.video_width, self.video_height))
                self.video_writers[motion.tag] = writer
                print(f"Recording {mode_name}: {filepath}")
    
    def _switch_stage_recording(self, stage_name):
        """Switch recording to new stage - but keep using same directory"""
        # Stop current recording
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()
        
        # Use the same compare directory for all stages (no need to recreate directories)
        # Just restart recording with existing directory structure
        self._start_compare_mode_recording("compare")
        
    def _switch_to_multi_ib_recording(self):
        """Switch from intra_batch to multi_ib recording (legacy method)"""
        self._switch_stage_recording("multi")
    
    def _capture_frame(self, motion_tag, frame_idx, batch_idx):
        """Capture current frame for video recording"""
        if not self.save_videos or motion_tag not in self.video_writers:
            return
            
        # Bind offscreen framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.video_width, self.video_height)
        
        # Clear and render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Render the scene (we'll implement this)
        self._render_for_video()
        
        # Read pixels
        pixels = glReadPixels(0, 0, self.video_width, self.video_height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8)
        image = image.reshape((self.video_height, self.video_width, 3))
        image = np.flip(image, 0)  # Flip vertically
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        
        # Write frame to video
        self.video_writers[motion_tag].write(image)
        
        # Screenshots handled by main recording logic, not here
        
        # Restore default framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def _render_for_video(self):
        """Render scene for video capture (simplified version of main render)"""
        # Set up basic OpenGL state
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.2, 0.2, 0.2, 1.0)
        
        if self.character1 is None or self.character2 is None:
            return
            
        character = self.character1
        
        # ALWAYS apply offsets for video recording to ensure animations are separated
        # This is critical for proper video recording regardless of move_character setting
        visible_count = 0
        for idx, motion in enumerate(self.motions):
            if motion.visible:
                # Apply offset for each visible motion based on its index
                # Force consistent offsets for video recording
                motion.offset = np.array([visible_count * 1.5, 0, 0], dtype=np.float32)
                visible_count += 1
        
        # Render current frame for all visible motions
        for idx, motion in enumerate(self.motions):
            if motion.visible:
                motion.render(character, self.frame, traj=self._show["traj"])
        
        # Render target frame if enabled
        if self._show["target"]:
            for motion in self.motions:
                if motion.visible:
                    motion.render(character, (self.bidx + 1) * self.frame_per_batch - 1, alpha=self._show["alpha"])
    
    def process_batch_for_video(self, batch_idx, start_app=True):
        """Process a single batch for video recording using GUI with auto-close
        
        Args:
            batch_idx: The batch index to process
            start_app: Whether to start the app (set to False if app is already running)
        """
        print(f"Processing batch {batch_idx} for video recording...")
        
        # Set up automatic recording mode
        self.recording_batch = batch_idx
        self.recording_mode = True
        self.recording_frame = 0
        self.playing = True  # Ensure auto-play is enabled
        
        # ALWAYS enable animation separation in video recording mode
        # This is critical for proper video recording
        self.move_character = True
        print("Video recording mode enabled: Animation separation will be applied automatically")
        
        # Make sure the bidx is set to the record_sequence_idx if specified
        if self.record_sequence_idx is not None:
            # For sequences > 63, we need to map to sequence index within batch
            seq_within_batch = self.record_sequence_idx % 64  # Get 0-63 position within batch
            
            # Ensure we're using the correct batch for this sequence
            if self.record_sequence_idx >= 64:
                correct_batch = self.record_sequence_idx // 64
                if batch_idx != correct_batch:
                    print(f"WARNING: Sequence {self.record_sequence_idx} should be in batch {correct_batch}, but current batch is {batch_idx}")
                    print(f"This may cause the wrong sequence to be displayed")
            
            print(f"Setting active sequence to {self.record_sequence_idx} (position {seq_within_batch} in batch {batch_idx})")
            self.bidx = seq_within_batch  # Use position within batch
            
            # Also set the global frame accordingly to ensure we see the correct sequence
            self.frame = self.bidx * self.frame_per_batch
            
            # Debug info to help identify sequence issues
            print(f"Frame range for sequence {self.record_sequence_idx}: {self.frame} to {self.frame + self.frame_per_batch - 1}")
            print(f"Target frame position: {(self.bidx + 1) * self.frame_per_batch - 1}")
        
        # Start the video recording process
        self._start_video_recording(batch_idx)
        print("=== RECORDING STARTED ===")
        
        # Start the app only if requested (may already be running)
        if start_app:
            agl.AppManager.start(self)
        else:
            # Initialize character models and other resources if needed
            if not hasattr(self, 'character1') or self.character1 is None:
                # Ensure character models are loaded
                if self.dataset == "lafan1":
                    self.character1 = agl.FBX("dataset/fbx-models/ybot.fbx").model()
                    self.character2 = agl.FBX("dataset/fbx-models/ybot.fbx").model()
                    self.character2.meshes[0].materials[0].albedo = glm.vec3([0.5, 0.5, 0.5])
                elif self.dataset == "100style":
                    self.character1 = agl.FBX("dataset/fbx-models/ybot-fingers.fbx").model()
                    self.character2 = agl.FBX("dataset/fbx-models/ybot-fingers.fbx").model()
                    self.character2.meshes[0].materials[0].albedo = glm.vec3([0.5, 0.5, 0.5])
                    self.character1.set_joint_map(FBX2FBX)
                    self.character2.set_joint_map(FBX2FBX)
            
            # Make sure animation is playing in recording mode
            self.playing = True
            
            # Set a flag to identify non-interactive recording mode
            # This prevents double-incrementing of recording_frame
            self._recording_in_noninteractive = True
            
            # Set all visible motions to have proper offsets
            ours_idx = 0
            for motion in self.motions:
                if motion.visible and (motion.tag == "Ours" or motion.tag.startswith("Ours")):
                    motion.offset = np.array([ours_idx * 1.5, 0, 0], dtype=np.float32)
                    ours_idx += 1
            
            print(f"Applied offsets to {ours_idx} visible Ours animations for recording")
                
            # Just run the update loop directly until recording completes
            while self.recording_mode:
                # Must manually handle the entire update-render cycle
                self.update()
                
                # Manually render the frame
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                self.render()
                self.render_text()
                
                # Force immediate recording of this frame
                self.should_record_frame = True
                self._record_current_frame()
                
                # Increment the recording frame counter
                self.recording_frame += 1
                
                # Check for completion
                if self.recording_frame >= self.frame_per_batch:
                    print(f"\nCompleted recording sequence {self.record_sequence_idx} in batch {batch_idx}")
                    self._stop_video_recording()
                    self.recording_mode = False
                    break
                    
                # No looping, progress through frames sequentially
    
    def _render_motion_frame_offline(self, motion, frame_number, batch_idx, frame_idx):
        """Render a single motion frame offline and return the image"""
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up basic rendering state
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, self.video_width, self.video_height)
        
        if self.character1 is None:
            # Create black image if no character
            return np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        
        # Render current frame motion
        motion.render(self.character1, frame_number, traj=True)
        
        # Render target frame (last frame of sequence) with transparency  
        target_frame = (batch_idx + 1) * self.frame_per_batch - 1
        motion.render(self.character2, target_frame, alpha=0.3)
        
        # Render modified poses if available
        if hasattr(self, 'modified_frames') and len(self.modified_frames) > 0:
            motion_idx = -1
            for i, m in enumerate(self.motions):
                if m.tag == motion.tag:
                    motion_idx = i
                    break
            
            if motion_idx >= 0 and motion_idx < len(self.modified_frames):
                modified_info = self.modified_frames[motion_idx]
                if modified_info is not None and batch_idx < len(modified_info.get("positions", [])):
                    modified_positions = modified_info["positions"][batch_idx]
                    if modified_positions is not None:
                        for pos in modified_positions:
                            # Render modified pose with transparency
                            motion.render(self.character2, pos + batch_idx * self.frame_per_batch, alpha=0.4)
        
        # Read pixels
        glFinish()
        pixels = glReadPixels(0, 0, self.video_width, self.video_height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8)
        image = image.reshape((self.video_height, self.video_width, 3))
        image = np.flip(image, 0)  # Flip vertically
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        
        return image
    
    def _init_headless_context(self):
        """Initialize OpenGL context for recording"""
        print("Initializing OpenGL context for recording...")
        
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")
        
        # Try invisible window first, fallback to visible if needed
        try:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            self.headless_window = glfw.create_window(self.video_width, self.video_height, "Video Recording", None, None)
            
            if not self.headless_window:
                print("Failed to create invisible window, trying visible window...")
                glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
                self.headless_window = glfw.create_window(self.video_width, self.video_height, "Video Recording", None, None)
            
            if not self.headless_window:
                raise Exception("Failed to create any window")
        
        except Exception as e:
            print(f"Window creation failed: {e}, trying with default hints...")
            glfw.default_window_hints()
            self.headless_window = glfw.create_window(self.video_width, self.video_height, "Video Recording", None, None)
            
            if not self.headless_window:
                raise Exception("Failed to create window with default hints")
        
        glfw.make_context_current(self.headless_window)
        print("OpenGL context initialized successfully")
    
    def _init_rendering_resources(self):
        """Initialize all rendering resources needed for offline rendering"""
        # Load character models
        if self.dataset == "lafan1":
            self.character1 = agl.FBX("dataset/fbx-models/ybot.fbx").model()
            self.character2 = agl.FBX("dataset/fbx-models/ybot.fbx").model()
            self.character2.meshes[0].materials[0].albedo = glm.vec3([0.5, 0.5, 0.5])
        elif self.dataset == "100style":
            self.character1 = agl.FBX("dataset/fbx-models/ybot-fingers.fbx").model()
            self.character2 = agl.FBX("dataset/fbx-models/ybot-fingers.fbx").model()
            self.character2.meshes[0].materials[0].albedo = glm.vec3([0.5, 0.5, 0.5])
            self.character1.set_joint_map(FBX2FBX)
            self.character2.set_joint_map(FBX2FBX)
        elif self.dataset in ["human36m", "mann"]:
            self.character1, self.character2 = None, None
        
        # Convert motions to MotionStruct
        motions = []
        for i in range(len(self.motions)):
            motion_struct = MotionStruct(self.motions[i], self.num_batches, self.skeleton, self.tags[i], 
                                        traj=self.trajs[i], kf_indices=self.kf_indices[i], contact=self.contacts[i])
            
            # Make all motions visible for recording to debug
            motion_struct.visible = True
            print(f"Motion {i}: {self.tags[i]} - visible: {motion_struct.visible}")
                
            motions.append(motion_struct)
        self.motions = motions
        
        # Setup basic OpenGL state
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glViewport(0, 0, self.video_width, self.video_height)
        
        # Setup rendering show options for recording
        self._show = {
            "target": True,    # Show target frame
            "modified": True,  # Show modified poses
            "traj": True,      # Show trajectory
            "alpha": 0.3       # Transparency for overlays
        }
    
    def _render_and_capture_offline(self, motion, frame_idx, batch_idx):
        """Render and capture a single motion for current frame"""
        if self.character1 is None:
            return
            
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup camera/view (basic setup)
        glViewport(0, 0, self.video_width, self.video_height)
        
        # Render current frame motion
        character = self.character1
        motion.render(character, self.frame, traj=self._show["traj"])
        
        # Render target frame (last frame of sequence) with transparency
        target_frame = (batch_idx + 1) * self.frame_per_batch - 1
        motion.render(self.character2, target_frame, alpha=0.3)
        
        # Render modified poses if available (same logic as in main render)
        if self._show["modified"] and motion.visible and hasattr(self, 'modified_frames') and len(self.modified_frames) > 0:
            # Find the motion index to get corresponding modified poses
            motion_idx = -1
            for i, m in enumerate(self.motions):
                if m.tag == motion.tag:
                    motion_idx = i
                    break
            
            if motion_idx >= 0 and motion_idx < len(self.modified_frames):
                modified_info = self.modified_frames[motion_idx]
                if modified_info is not None and batch_idx < len(modified_info.get("positions", [])):
                    modified_positions = modified_info["positions"][batch_idx]
                    if modified_positions is not None:
                        for pos in modified_positions:
                            # Render modified pose with gray character and transparency
                            motion.render(self.character2, pos + batch_idx * self.frame_per_batch, alpha=0.4)
        
        # Read pixels and save
        pixels = glReadPixels(0, 0, self.video_width, self.video_height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8)
        image = image.reshape((self.video_height, self.video_width, 3))
        image = np.flip(image, 0)  # Flip vertically
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        
        # Write frame to video
        if motion.tag in self.video_writers:
            self.video_writers[motion.tag].write(image)
        
        # Screenshots handled by main recording logic, not here
    
    def _cleanup_headless_context(self):
        """Clean up headless rendering context"""
        try:
            if hasattr(self, 'headless_window') and self.headless_window:
                glfw.destroy_window(self.headless_window)
                self.headless_window = None
            glfw.terminate()
            print("Headless OpenGL context cleaned up")
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def _save_current_frame_as_image(self, motion_tag, frame_idx, batch_idx):
        """Save current rendered frame as image and add to video"""
        if not self.save_videos:
            return
            
        try:
            # Read pixels from current framebuffer
            viewport = glGetIntegerv(GL_VIEWPORT)
            width, height = viewport[2], viewport[3]
            
            pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(pixels, dtype=np.uint8)
            image = image.reshape((height, width, 3))
            image = np.flip(image, 0)  # Flip vertically
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            
            # Resize to video resolution if needed
            if width != self.video_width or height != self.video_height:
                image = cv2.resize(image, (self.video_width, self.video_height))
            
            # Write frame to video if writer exists
            if motion_tag in self.video_writers:
                self.video_writers[motion_tag].write(image)
            
            # Save screenshot for keyframes
            if frame_idx % 10 == 0:  # Save every 10th frame
                screenshot_path = os.path.join(self.images_dir, f"batch_{batch_idx:03d}_{motion_tag}_frame_{frame_idx:04d}.png")
                cv2.imwrite(screenshot_path, image)
                
        except Exception as e:
            print(f"Error saving frame for {motion_tag}: {e}")
    
    def _cleanup_offscreen_rendering(self):
        """Clean up offscreen rendering resources"""
        try:
            if hasattr(self, 'fbo') and self.fbo:
                glDeleteFramebuffers(1, [self.fbo])
                self.fbo = None
            if hasattr(self, 'color_texture') and self.color_texture:
                glDeleteTextures([self.color_texture])
                self.color_texture = None
            if hasattr(self, 'depth_renderbuffer') and self.depth_renderbuffer:
                glDeleteRenderbuffers(1, [self.depth_renderbuffer])
                self.depth_renderbuffer = None
        except:
            pass  # Ignore cleanup errors

    def start(self):
        super().start()
        
        # Set to paused state for normal mode, but enable for recording or compare mode
        # Also auto-play when save_videos_and_images is used
        self.playing = self.recording_mode or self.compare_mode or self.save_videos
        self.should_record_frame = False
        
        # Initialize frame indices if not already done
        if not hasattr(self, 'frame'):
            self.frame = 0
        if not hasattr(self, 'bidx'):
            self.bidx = 0
        if not hasattr(self, 'fidx'):
            self.fidx = 0
        
        # Check if we need to enable split view on start (set by vis_mib.py for replace_target mode)
        self._enable_split_view = False
        if hasattr(self, '_enable_split_view_on_start') and self._enable_split_view_on_start:
            self._enable_split_view = True
            
        # ALWAYS enable split view and move_character in video recording mode
        # This is critical for proper animation separation in videos
        if self.save_videos:
            self._enable_split_view = True
            self.move_character = True
            print("Video recording mode: Enabling split view for separated animations")
            
            # Initialize batch ID for recording if not in recording mode
            if not self.recording_mode:
                self.recording_batch = 0
                
                # Initialize frame indices for specific sequence recording
                if self.record_sequence_idx is not None:
                    self.bidx = self.record_sequence_idx % 64  # Sequence within batch
                    self.fidx = 0
                    self.frame = self.bidx * self.frame_per_batch
                    print(f"Initializing playback for sequence {self.record_sequence_idx}")
                
                # We'll start the recording after motions are initialized
        
        # character model
        if self.dataset == "lafan1":
            self.character1 = agl.FBX("dataset/fbx-models/ybot.fbx").model()
            self.character2 = agl.FBX("dataset/fbx-models/ybot.fbx").model()
            self.character2.meshes[0].materials[0].albedo = glm.vec3([0.5, 0.5, 0.5])
        elif self.dataset == "100style":
            self.character1 = agl.FBX("dataset/fbx-models/ybot-fingers.fbx").model()
            self.character2 = agl.FBX("dataset/fbx-models/ybot-fingers.fbx").model()
            self.character2.meshes[0].materials[0].albedo = glm.vec3([0.5, 0.5, 0.5])
            self.character1.set_joint_map(FBX2FBX)
            self.character2.set_joint_map(FBX2FBX)
        elif self.dataset in ["human36m", "mann"]:
            self.character1, self.character2 = None, None
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        
        # model
        self.arrow = agl.FBX("dataset/fbx-models/arrow.fbx").model()

        # convert motions to MotionStruct
        motions = []
        for i in range(len(self.motions)):
            motion_struct = MotionStruct(self.motions[i], self.num_batches, self.skeleton, self.tags[i], 
                                        traj=self.trajs[i], kf_indices=self.kf_indices[i], contact=self.contacts[i])
            
            # Default setting: show animations with "Ours" tags and compare mode tags
            if (self.tags[i] == "Ours" or self.tags[i].startswith("Ours-") or 
                self.tags[i] == "Intra-Batch" or self.tags[i] == "Multi-IB"):
                motion_struct.visible = True
# print(f"Motion {i} ({self.tags[i]}) set to visible=True, color={getattr(motion_struct, 'character_color', 'default')}")  # Debug disabled
            
            motions.append(motion_struct)
        self.motions = motions
        
        # Now that motions are initialized, start video recording if needed
        if self.save_videos and not self.recording_mode:
            self._start_video_recording(self.recording_batch)
            print("=== AUTO-PLAY RECORDING STARTED ===")
        
        # Apply split view if requested (for replace_target mode)
        if self._enable_split_view:
            # Force enable split view mode
            self.move_character = True
            print("  Split view mode enabled - spreading animations apart")
            
            # Apply offsets to spread the animations apart (only for Ours animations)
            ours_idx = 0
            has_ours = False
            
            for idx, motion in enumerate(self.motions):
                # Only show and spread Ours animations
                if motion.tag == "Ours" or motion.tag.startswith("Ours"):
                    motion.offset = np.array([ours_idx * 1.5, 0, 0], dtype=np.float32)
                    motion.visible = True
                    ours_idx += 1
                    has_ours = True
                else:
                    # Hide GT and other non-Ours animations
                    motion.visible = False
            
            if has_ours:
                print(f"  Separated {ours_idx} 'Ours' animations (GT hidden)")
            else:
                print("  Warning: No 'Ours' animations found to display")
        
        # Start compare mode if enabled
        if self.compare_mode:
            if self.current_stage < len(self.playback_stages):
                stage_name = self.playback_stages[self.current_stage]
                print(f"Compare mode: Starting with stage 1: {stage_name}")
                
                # Keyframe visibility controlled by parameter
                keyframe_status = "shown" if self._show["keyframe"] else "hidden"
                print(f"Compare mode: Keyframes will be {keyframe_status}")
                
                if self.save_videos:
                    # First create the compare directories
                    self._create_compare_directories("compare")
                    self._start_compare_mode_recording("compare")  # Start recording for first stage
            
        # Start recording if in recording mode
        if self.recording_mode:
            print("=== RECORDING MODE ACTIVATED ===")
            # Show all available motions
            print("Available motions:")
            for i, motion in enumerate(self.motions):
                print(f"  {i}: {motion.tag}")
            
            # Only make Ours-related motions visible for recording
            # Also ensure they have proper offsets if move_character is enabled
            ours_idx = 0
            for motion in self.motions:
                if motion.tag == "Ours" or motion.tag.startswith("Ours-"):
                    motion.visible = True
                    
                    # Apply offset if move_character is enabled (for animation separation)
                    if self.move_character:
                        motion.offset = np.array([ours_idx * 1.5, 0, 0], dtype=np.float32)
                        ours_idx += 1
                        
                    print(f"Motion: {motion.tag} - set visible: True, offset: {motion.offset}")
                else:
                    motion.visible = False
                    print(f"Motion: {motion.tag} - set visible: False")
            
            if self.move_character and ours_idx > 0:
                print(f"Applied offsets to {ours_idx} 'Ours' animations for separation")
            
            # Initialize recording_frame and frame indices for recording
            self.recording_frame = 0
            if self.record_sequence_idx is not None:
                self.bidx = self.record_sequence_idx
                self.fidx = 0
                self.frame = self.bidx * self.frame_per_batch
                
            self._start_video_recording(self.recording_batch)
            print("=== RECORDING STARTED ===")

        # ui
        self.ui.add_menu("MotionApp")
        self.ui.add_menu_item("MotionApp", "Move Character", self._switch_move_character, glfw.KEY_M)
        self.ui.add_menu_item("MotionApp", "Show Target", lambda: self._switch_show("target"), glfw.KEY_T)
        self.ui.add_menu_item("MotionApp", "Show X-Ray", lambda: self._switch_show("xray"), glfw.KEY_X)
        self.ui.add_menu_item("MotionApp", "Show Tag", lambda: self._switch_show("tag"), glfw.KEY_Y)
        self.ui.add_menu_item("MotionApp", "Show Info", lambda: self._switch_show("info"), glfw.KEY_I)
        self.ui.add_menu_item("MotionApp", "Show Traj", lambda: self._switch_show("traj"), glfw.KEY_J)
        self.ui.add_menu_item("MotionApp", "Show Every Ten", lambda: self._switch_show("every_ten"), glfw.KEY_E)
        self.ui.add_menu_item("MotionApp", "Toggle Keyframes (K)", lambda: self._switch_show("keyframe"), glfw.KEY_K)
        self.ui.add_menu_item("MotionApp", "Show Arrow", lambda: self._switch_show("arrow"), glfw.KEY_R)
        self.ui.add_menu_item("MotionApp", "Show Contact", lambda: self._switch_show("contact"), glfw.KEY_C)
        self.ui.add_menu_item("MotionApp", "Show Transition", lambda: self._switch_show("transition"), glfw.KEY_N)
        self.ui.add_menu_item("MotionApp", "Show Modified Frames", lambda: self._switch_show("modified"), glfw.KEY_Q)
        
        # Add batch navigation shortcut hints
        self.ui.add_menu_item("MotionApp", "Next Batch (B)", lambda: None, glfw.KEY_B)
        self.ui.add_menu_item("MotionApp", "Prev Batch (V)", lambda: None, glfw.KEY_V)
        
        for motion in self.motions:
            self.ui.add_menu_item("MotionApp", f"Show {motion.tag}", motion.switch_visible)
        self.ui.add_menu_item("MotionApp", "Show All", lambda: [motion.switch_visible() for motion in self.motions])
    
    def update(self):
        super().update()
        
        # Handle recording mode
        if self.recording_mode and self.save_videos:
            if self.recording_frame < self.frame_per_batch:
                # Calculate the specific sequence frame to record
                if self.record_sequence_idx is not None:
                    # Record specific sequence: map to sequence position within batch (0-63)
                    seq_within_batch = self.record_sequence_idx % 64
                    
                    # Set bidx to the sequence position within the batch
                    self.bidx = seq_within_batch
                    
                    # Calculate the start frame for this specific sequence in the batch
                    sequence_start_frame = self.bidx * self.frame_per_batch
                    
                    # Set frame to the current position within the specific sequence
                    self.frame = sequence_start_frame + self.recording_frame
                    self.fidx = self.recording_frame
                    
                    # Debug info on first and last frame only
                    if self.recording_frame == 0:
                        print(f"Recording sequence {self.record_sequence_idx} (position {seq_within_batch}) - Start frame: {sequence_start_frame}")
                        target_frame = sequence_start_frame + self.frame_per_batch - 1
                        print(f"Target frame will be at position: {target_frame}")
                    elif self.recording_frame == self.frame_per_batch - 1:
                        print(f"Reached target frame for sequence {self.record_sequence_idx} at position {self.frame}")
                else:
                    # Record all sequences in batch (original behavior)
                    self.frame = self.recording_batch * self.frame_per_batch + self.recording_frame
                    self.bidx, self.fidx = self.frame // self.frame_per_batch, self.frame % self.frame_per_batch
                
                # Show progress bar
                progress = (self.recording_frame + 1) / self.frame_per_batch
                bar_length = 20
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                if self.record_sequence_idx is not None:
                    print(f"\rRecording seq {self.record_sequence_idx:02d} [{bar}] {self.recording_frame + 1:3d}/{self.frame_per_batch} ({progress:.0%})", end='', flush=True)
                else:
                    print(f"\rRecording [{bar}] {self.recording_frame + 1:3d}/{self.frame_per_batch} ({progress:.0%})", end='', flush=True)
                
                # Record after next render
                self.should_record_frame = True
                
                # Only increment recording_frame in interactive mode
                # In non-interactive mode, process_batch_for_video handles incrementing
                if not hasattr(self, '_recording_in_noninteractive') or not self._recording_in_noninteractive:
                    self.recording_frame += 1
            else:
                # Finished recording
                print()  # New line after progress bar
                if self.record_sequence_idx is not None:
                    print(f"✓ Completed batch {self.recording_batch}, sequence {self.record_sequence_idx}")
                else:
                    print(f"✓ Completed batch {self.recording_batch}")
                self._stop_video_recording()
                self.recording_mode = False
                # Close window
                if hasattr(self, 'window') and self.window:
                    glfw.set_window_should_close(self.window, True)
                return
        elif self.save_videos:
            # Auto-play mode with saving videos and images
            # No looping - progress sequentially through the frames
            if self.record_sequence_idx is not None:
                # Display only specific sequence
                self.bidx = self.record_sequence_idx
                
                # Make sure fidx is initialized correctly
                if not hasattr(self, 'fidx') or self.fidx is None:
                    self.fidx = 0
                    self.frame = self.bidx * self.frame_per_batch
                
                # Progress through frames sequentially without looping
                self.fidx = (self.fidx + 1) % self.frame_per_batch
                self.frame = self.bidx * self.frame_per_batch + self.fidx
                
                # Capture current frame
                self.should_record_frame = True
                
                # When we reach the end, stop playback and save videos
                if self.fidx == self.frame_per_batch - 1:
                    # At the last frame, save everything and exit
                    print(f"Completed playback of sequence {self.record_sequence_idx}")
                    
                    # Ensure proper closing
                    if hasattr(self, 'video_writers') and self.video_writers:
                        self._stop_video_recording()
                        print("Videos saved successfully")
                    
                    # Close the window
                    glfw.set_window_should_close(self.window, True)
            else:
                # Display all sequences
                # Progress forward without looping
                self.frame = (self.frame + 1) % self.total_frames
                self.bidx, self.fidx = self.frame // self.frame_per_batch, self.frame % self.frame_per_batch
                
                # Capture current frame
                self.should_record_frame = True
                
                # When we reach the end of the total frames, save and exit
                if self.frame == self.total_frames - 1:
                    print(f"Completed playback of all sequences")
                    
                    # Stop recording and exit
                    if hasattr(self, 'video_writers') and self.video_writers:
                        self._stop_video_recording()
                        print("Videos saved successfully")
                    
                    # Close the window
                    glfw.set_window_should_close(self.window, True)
        else:
            # Normal interactive mode
            if self.record_sequence_idx is not None:
                # Display only specific sequence
                self.bidx = self.record_sequence_idx
                
                if self.compare_mode:
                    # In compare mode, use frame count to track progress
                    self.fidx = self.stage_frame_count % self.frame_per_batch
                    self.frame = self.bidx * self.frame_per_batch + self.fidx
                    
                    
                    # Check if current stage completed
                    if self.stage_frame_count >= self.frame_per_batch:
                        self.current_stage += 1
                        if self.current_stage >= len(self.playback_stages):
                            # All stages completed
                            print("All 3 stages completed: intra -> multi -> combined")
                            if self.save_videos:
                                self._stop_video_recording()
                                print("Videos saved")
                            print("Exiting program")
                            glfw.set_window_should_close(self.window, True)
                            return
                        else:
                            # Switch to next stage
                            if self.current_stage < len(self.playback_stages):
                                stage_name = self.playback_stages[self.current_stage]
                                print(f"Switching to stage {self.current_stage + 1}: {stage_name}")
                                if self.save_videos:
                                    self._switch_stage_recording(stage_name)
                                self.stage_frame_count = 0  # Reset frame count
                                self.fidx = 0
                                
                                # Reset keyframe message flag to show message for new stage
                                if hasattr(self, '_keyframe_message_shown'):
                                    delattr(self, '_keyframe_message_shown')
                            self.frame = self.bidx * self.frame_per_batch
                    
                    # Increment frame count for current stage
                    self.stage_frame_count += 1
                else:
                    # Normal single sequence mode
                    sequence_frame = self.frame % self.frame_per_batch
                    self.frame = self.bidx * self.frame_per_batch + sequence_frame
                    self.fidx = sequence_frame
            else:
                # Display all sequences
                self.frame = self.frame % self.total_frames
                self.bidx, self.fidx = self.frame // self.frame_per_batch, self.frame % self.frame_per_batch

    def render(self):
        try:
            super().render()
            if self._show["arrow"]:
                for idx, motion in enumerate(self.motions):
                    pos, dir = motion.get_base(self.frame)
                    agl.Render.model(self.arrow).position(pos).orientation(dir).draw()
        except Exception as e:
            print(f"Shadow render error: {e}")
            return  # Skip this render frame

        try:
            if self.character1 is None or self.character2 is None:
                return

            if self._show["transition"]:
                if self.fidx < 10 or self.fidx == self.frame_per_batch - 1:
                    character = self.character1
                else:
                    character = self.character2
            else:
                character = self.character1
            
            # current frame - render based on stage
            for idx, motion in enumerate(self.motions):
                should_render = True
                if self.compare_mode:
                    if self.current_stage < len(self.playback_stages):
                        current_stage_name = self.playback_stages[self.current_stage]
                        if current_stage_name == "intra" and motion.tag != "Intra-Batch":
                            should_render = False
                        elif current_stage_name == "multi" and motion.tag != "Multi-IB":
                            should_render = False
                    # "combined" stage shows both motions
                
                if should_render and motion.visible:
                    motion.render(self.character1 if motion.tag in ["GT", "Dataset"] else character, self.frame, traj=self._show["traj"])

                # every ten frames
                if self._show["every_ten"]:
                    motion.render(self.character1, self.bidx * self.frame_per_batch)
                    motion.render(self.character1, (self.bidx + 1) * self.frame_per_batch - 1)
                    for i in range(10, self.frame_per_batch - 10, 10):
                        idx = self.bidx * self.frame_per_batch + i
                        motion.render(self.character2, idx)
                
                # keyframe
                if self._show["keyframe"] and motion.kf_indices is not None:
                    # Determine if we should render keyframes
                    should_render_keyframes = True
                    
                    # In compare mode, only render keyframes for the active stage motion
                    if self.compare_mode:
                        if self.current_stage < len(self.playback_stages):
                            current_stage_name = self.playback_stages[self.current_stage]
                            if current_stage_name == "intra" and motion.tag != "Intra-Batch":
                                should_render_keyframes = False
                            elif current_stage_name == "multi" and motion.tag != "Multi-IB":
                                should_render_keyframes = False
                    
                    # Only render keyframes for visible motions
                    if not motion.visible:
                        should_render_keyframes = False
                    
                    if should_render_keyframes:
                        try:
                            if self.bidx < len(motion.kf_indices) and len(motion.kf_indices[self.bidx]) > 2:
                                # Message on first frame
                                if self.fidx == 0 and not hasattr(self, f'_keyframe_message_shown_{motion.tag}'):
                                    print(f"Showing keyframes for {motion.tag}")
                                    setattr(self, f'_keyframe_message_shown_{motion.tag}', True)
                                
                                # Render all keyframes except first and last (they're context frames)
                                for kf in motion.kf_indices[self.bidx][1:-1]:
                                    kf_frame = kf + self.bidx * self.frame_per_batch
                                    
                                    # Set a unique color for keyframes based on motion tag
                                    original_motion_color = getattr(motion, 'character_color', None)
                                    
                                    # Use a brighter color for keyframes
                                    # if motion.tag == "GT" or motion.tag == "Dataset":
                                    #     motion.character_color = [0.2, 0.8, 0.2]  # Green for GT
                                    # elif motion.tag == "Ours" or motion.tag.startswith("Ours"):
                                    #     motion.character_color = [0.8, 0.2, 0.2]  # Red for Ours
                                    if motion.tag == "Intra-Batch":
                                        motion.character_color = [0.2, 0.8, 0.8]  # Cyan for Intra-Batch
                                    elif motion.tag == "Multi-IB":
                                        motion.character_color = [0.8, 0.2, 0.8]  # Magenta for Multi-IB

                                    
                                    # Render keyframe with distinguishable color and transparency
                                    motion.render(self.character2, kf_frame, alpha=0.3)  # Increased alpha for better visibility
                                    
                                    # Restore original color
                                    if original_motion_color is not None:
                                        motion.character_color = original_motion_color
                                    else:
                                        delattr(motion, 'character_color')
                                    
                                    # Also render a small text label "KF" at keyframe positions
                                    if self._show["tag"]:
                                        pos = motion.poses[kf_frame].root_pos + np.array([0, 1.2, 0]) + motion.offset
                                        agl.Render.text("KF").position(pos).scale(0.4).draw()
                        except (IndexError, TypeError):
                            pass  # Skip keyframe rendering if index error
                
                # modified frames (for intra_batch_replace and multi_ib modes)
                if self._show["modified"] and motion.visible and hasattr(self, 'modified_frames') and len(self.modified_frames) > 0:
                    # 在比较模式下，确保每个模式只显示自己的修改帧
                    should_show_modified = True
                    if self.compare_mode:
                        if self.current_stage < len(self.playback_stages):
                            current_stage_name = self.playback_stages[self.current_stage]
                            # intra阶段只显示Intra-Batch的修改帧
                            if current_stage_name == "intra" and motion.tag != "Intra-Batch":
                                should_show_modified = False
                            # multi阶段只显示Multi-IB的修改帧
                            elif current_stage_name == "multi" and motion.tag != "Multi-IB":
                                should_show_modified = False
                    
                    if should_show_modified:
                        # Find the correct modified_frames data by matching motion tag
                        motion_idx = -1
                        for i, m in enumerate(self.motions):
                            if m.tag == motion.tag:
                                motion_idx = i
                                break
                        
                        if motion_idx >= 0 and motion_idx < len(self.modified_frames):
                            try:
                                modified_info = self.modified_frames[motion_idx]
                                if modified_info is not None:
                                    positions = modified_info.get("positions", [])
                                    if positions and self.bidx < len(positions):
                                        # Get the modified pose positions for current batch
                                        modified_positions = positions[self.bidx]
                                    if modified_positions is not None:
                                        for pos in modified_positions:
                                            # Temporarily override motion's color to gray for modified frames
                                            original_motion_color = getattr(motion, 'character_color', None)
                                            motion.character_color = [0.5, 0.5, 0.5]  # Set gray color
                                            
                                            # Render modified pose with gray color and slight transparency
                                            motion.render(self.character2, pos + self.bidx * self.frame_per_batch, alpha=0.3)
                                            
                                            # Restore motion's original color
                                            if original_motion_color is not None:
                                                motion.character_color = original_motion_color
                                            else:
                                                delattr(motion, 'character_color')
                            except (IndexError, TypeError, KeyError):
                                pass  # Skip modified frames rendering if error
            
            # target frame
            if self._show["target"]:
                for motion in self.motions:
                    should_render = True
                    if self.compare_mode:
                        if self.current_stage < len(self.playback_stages):
                            current_stage_name = self.playback_stages[self.current_stage]
                            if current_stage_name == "intra" and motion.tag != "Intra-Batch":
                                should_render = False
                            elif current_stage_name == "multi" and motion.tag != "Multi-IB":
                                should_render = False
                    
                    if should_render and motion.visible:
                        try:
                            # Temporarily override motion's color to gray for target frame (same as other modes)
                            original_motion_color = getattr(motion, 'character_color', None)
                            motion.character_color = [0.5, 0.5, 0.5]  # Set gray color
                            
                            # Calculate target frame position - always use the last frame of the CURRENT sequence
                            # This ensures target frame is aligned with current sequence
                            target_frame_pos = (self.bidx + 1) * self.frame_per_batch - 1
                            
                            # Render target frame with gray color
                            motion.render(self.character1, target_frame_pos, alpha=0.5)  # Use 0.5 alpha for better visibility
                            
                            # Restore motion's original color
                            if original_motion_color is not None:
                                motion.character_color = original_motion_color
                            else:
                                delattr(motion, 'character_color')
                                
                        except (IndexError, AttributeError):
                            # Fallback to normal rendering without color change
                            target_frame_pos = (self.bidx + 1) * self.frame_per_batch - 1
                            motion.render(self.character1, target_frame_pos, alpha=0.5)
        
            # Handle frame recording after rendering (moved to render_text for proper timing)
            pass
        except Exception as e:
            import traceback
            print(f"Scene render error: {e}")
            print(f"Error details: {traceback.format_exc()}")
            # Continue execution instead of crashing
    
    def _record_current_frame(self):
        """Record current rendered frame for all visible motions"""
        try:
            # Ensure we're reading from the front buffer after rendering
            glFinish()  # Wait for all OpenGL commands to complete
            
            # Get window size instead of viewport (which might be wrong)
            if hasattr(self, 'window') and self.window:
                width, height = glfw.get_framebuffer_size(self.window)
            else:
                # Fallback to reasonable size
                width, height = 1280, 720
            
            # Removed: too verbose for progress bar mode
            
            # Use the front buffer
            glReadBuffer(GL_FRONT)
            pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(pixels, dtype=np.uint8)
            image = image.reshape((height, width, 3))
            image = np.flip(image, 0)  # Flip vertically
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            
            # Resize to target video resolution
            if width != self.video_width or height != self.video_height:
                image = cv2.resize(image, (self.video_width, self.video_height))
            
            # Save video frames for each visible motion
            saved_count = 0
            for motion in self.motions:
                if motion.visible:
                    # Write to video
                    if motion.tag in self.video_writers:
                        self.video_writers[motion.tag].write(image)
                        saved_count += 1
            
            # Save ONE screenshot per frame (not per motion)
            if self.compare_mode:
                # In compare mode, use stage_frame_count for each stage (0-100 for each stage)
                frame_num = self.stage_frame_count
                stage_name = self.playback_stages[self.current_stage] if self.current_stage < len(self.playback_stages) else "unknown"
                screenshot_path = os.path.join(self.images_dir, f"frame_{stage_name}_{frame_num:04d}.png")
            elif self.record_sequence_idx is not None:
                # Include sequence number in the filename
                seq_within_batch = self.record_sequence_idx % 64
                if hasattr(self, 'recording_frame') and self.recording_frame > 0:
                    # Use recording_frame in recording mode
                    frame_num = self.recording_frame - 1
                else:
                    # Use fidx in auto-play mode
                    frame_num = self.fidx
                screenshot_path = os.path.join(self.images_dir, f"seq_{seq_within_batch:02d}_frame_{frame_num:04d}.png")
            else:
                if hasattr(self, 'recording_frame') and self.recording_frame > 0:
                    # Use recording_frame in recording mode
                    frame_num = self.recording_frame - 1
                else:
                    # Use current frame in auto-play mode
                    frame_num = self.frame % self.frame_per_batch
                screenshot_path = os.path.join(self.images_dir, f"batch_{self.recording_batch:03d}_frame_{frame_num:04d}.png")
            
            # Save the image
            cv2.imwrite(screenshot_path, image)
            
            # Debug info to verify the correct sequence is being recorded
            if (hasattr(self, 'recording_frame') and self.recording_frame == 1) or self.frame % self.frame_per_batch == 0:
                # Print at start of recording or at first frame of sequence in auto-play mode
                seq_info = f"seq_{self.record_sequence_idx}" if self.record_sequence_idx is not None else "all sequences"
                print(f"Saving screenshots to: {self.images_dir} ({seq_info})")
                    
            # Removed: too verbose for progress bar mode
                    
        except Exception as e:
            print(f"Error recording frame: {e}")
            import traceback
            traceback.print_exc()

    def render_text(self):
        super().render_text()

        # motion tags
        if self._show["tag"]:
            for motion in self.motions:
                motion.render_tag(self.frame)

        # motion info
        if self._show["info"]:
            agl.Render.text_on_screen(f"Motion {self.bidx + 1} / {self.num_batches}").position([0, 0.1, 0]).scale(0.5).draw()
            agl.Render.text_on_screen(f"Frame {self.fidx + 1} / {self.frame_per_batch}").position([0, 0.05, 0]).scale(0.5).draw()
            
            # Show compare mode status
            if self.compare_mode:
                if self.current_stage < len(self.playback_stages):
                    stage_name = self.playback_stages[self.current_stage]
                    if stage_name == "intra":
                        agl.Render.text_on_screen("Mode: Intra-batch Replace").position([0, 0.0, 0]).scale(0.5).draw()
                    elif stage_name == "multi":
                        agl.Render.text_on_screen("Mode: Multi-IB").position([0, 0.0, 0]).scale(0.5).draw()
                    elif stage_name == "combined":
                        agl.Render.text_on_screen("Mode: Combined View").position([0, 0.0, 0]).scale(0.5).draw()
        
        # Handle frame recording after all rendering is complete
        if (self.recording_mode and hasattr(self, 'should_record_frame') and self.should_record_frame) or \
           (self.compare_mode and self.save_videos) or \
           (self.save_videos and self.playing):  # Also record in auto-play mode
            self._record_current_frame()
            if hasattr(self, 'should_record_frame'):
                self.should_record_frame = False

    def render_xray(self):
        super().render_xray()
        if self._show["contact"]:
            for motion in self.motions:
                motion.render_contact(self.frame, contact=self._show["contact"], contact_idx=self.contact_idx)

        if not self._show["xray"]:
            return

        # current frame
        for motion in self.motions:
            motion.render_xray(self.frame)

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if action != glfw.PRESS:
            return
        
        if mods & glfw.MOD_ALT:
            if (glfw.KEY_1 <= key <= len(self.motions) + glfw.KEY_1):
                self.motions[key - glfw.KEY_1].switch_visible()
        elif (glfw.KEY_0 <= key <= glfw.KEY_9):
            self.frame = (self.total_frames * (key - glfw.KEY_0)) // 10
            glfw.set_time(self.frame / self.fps)