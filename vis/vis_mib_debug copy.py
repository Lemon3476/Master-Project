import sys
sys.path.append(".")

import argparse
import os
import random
import numpy as np
import torch
from aPyOpenGL import agl

from utils.eval_debug import EvaluatorDebug, _segment_net_transition
from utils import ops
from vis.motionapp import MotionApp
from model.segment_net import SegmentNet
from model.twostage import ContextTransformer
from utils.dataset import MotionDataset
from utils import utils

# 添加调试功能
import traceback

def debug_print(message):
    """调试信息打印函数"""
    print(f"[DEBUG] {message}")

def get_output_directory(args, mode_override=None):
    """Generate output directory path based on mode and parameters"""
    base_dir = args.output_dir
    
    # Generate subdirectory name based on active modes
    mode_parts = []
    
    # Dataset and config
    mode_parts.append(f"{args.dataset}_{args.config.replace('.yaml', '')}")
    
    # Use mode override if provided (for 3-mode generation)
    if mode_override:
        mode_parts.append(mode_override)
    else:
        # Replacement modes
        # Removed random_kf_replace, reverse_kf_replace, and cross_seq_kf_replace modes
        if args.intra_batch_replace:
            mode_parts.append("intra_batch")
        if args.replace_target:  # Changed from replace_target_with_refine to replace_target
            mode_parts.append("replace_target")
        if args.multi_ib:
            mode_parts.append("multi_ib")
        if args.skip_interpolation:
            mode_parts.append("sparse_kf")
        if args.preserve_keyframes and not args.skip_interpolation:
            mode_parts.append("preserve_kf")
        if args.kf_config is not None and args.seg_config is not None:
            mode_parts.append("segment_net")
    
    # Other settings
    if args.no_shuffle:
        mode_parts.append("no_shuffle")
    if args.traj_edit:
        mode_parts.append(f"traj_{args.traj_edit}")
    
    # Batch ID (for single batch processing in interactive mode)
    if not args.save_videos_and_images:
        mode_parts.append(f"batch_{args.batch_id}")
    
    # If no special modes, use "default"
    if len(mode_parts) == 1:  # Only dataset_config
        mode_parts.append("default")
    
    # Combine parts
    mode_dir = "_".join(mode_parts)
    
    return os.path.join(base_dir, mode_dir)

# 设置随机种子，确保每次运行结果一致
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ =="__main__":
    try:
        # arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--interp", type=lambda s: s.lower() in ['true', '1'])
        parser.add_argument("--config", type=str, default="default.yaml")
        parser.add_argument("--traj_edit", type=str, default=None)
        parser.add_argument("--ts_configs", type=str, nargs="+", default=[]) # odd: context, even: detail
        parser.add_argument("--rmi_configs", type=str, nargs="+", default=[])
        parser.add_argument("--ours_configs", type=str, nargs="+", default=[]) # odd: keyframe, even: refine
        # Removed random_kf_replace, reverse_kf_replace, and cross_seq_kf_replace arguments
        parser.add_argument("--intra_batch_replace", action="store_true", help="Replace middle keyframe with keyframe from other sequences in same batch (0->1, 1->2, ..., 63->0)")
        parser.add_argument("--replace_target", action="store_true", help="Replace target frame with frame from other sequences in same batch and refine")
        parser.add_argument("--replace_target_with_segment", action="store_true", help="Replace target frame with frame from other sequences in same batch and use segment_net")
        parser.add_argument("--multi_ib", action="store_true", help="Multi-stage In-Betweening generation with frame modification")
        parser.add_argument("--middle_frame", type=int, default=None, help="Specify custom middle frame index for multi_ib mode")
        parser.add_argument("--compare_modes", action="store_true", help="Compare intra_batch_replace and multi_ib modes using exact same data")
        parser.add_argument("--show_keyframes", action="store_true", help="Show keyframes in compare mode (default: hide keyframes)")
        parser.add_argument("--seq", type=int, default=None, help="Specific sequence index to display (0-63) when using compare_modes")
        parser.add_argument("--preserve_middle_frame", action="store_true", help="Preserve middle frame in intra mode (RefineNet won't modify it)")
        parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling of sequences (default: shuffling enabled)")
        parser.add_argument("--skip_interpolation", action="store_true", help="Skip linear interpolation between keyframes and directly feed sparse keyframes to RefineNet")
        parser.add_argument("--preserve_keyframes", action="store_true", help="Preserve keyframes in standard mode, preventing RefineNet from modifying them")
        parser.add_argument("--batch_id", type=int, default=0, help="Batch ID to display")
        parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
        parser.add_argument("--save_videos_and_images", action="store_true", help="Save videos and screenshots for all 64 sequences")
        parser.add_argument("--output_dir", type=str, default="output", help="Base directory to save videos and screenshots")
        parser.add_argument("--record_sequence_idx", type=int, default=None, help="Record specific sequence index (0-63) within the batch. If not specified, records all sequences in the batch.")
        # SegmentNet specific arguments
        parser.add_argument("--kf_config", type=str, default=None, help="KeyframeNet config file (for SegmentNet)")
        parser.add_argument("--seg_config", type=str, default=None, help="SegmentNet config file")
        parser.add_argument("--kf_sampling", type=str, default="score", help="Keyframe sampling method (score, topk, threshold, random, uniform)")
        parser.add_argument("--kf_param", type=float, default=None, help="Parameter for keyframe sampling")
        args = parser.parse_args()
        
        debug_print("Parsing arguments completed")
        
        # Always use a fixed seed 1234 for data loading
        # This ensures consistent data order across all modes and runs
        fixed_data_seed = 1234
        print(f"Using fixed seed {fixed_data_seed} for data loading to ensure consistency")
        set_seed(fixed_data_seed)
        
        # Force data shuffle for all modes to ensure consistency
        # Since we use a fixed seed, the shuffle pattern will be identical across runs
        args.no_shuffle = False  # Enable shuffle (with fixed seed) for consistent data ordering
        print("Enabling data shuffle with fixed seed - this ensures consistent sequence ordering across all modes")
        
        # Check if we should use SegmentNet (when both kf_config and seg_config are provided)
        use_segment_mode = args.kf_config is not None and args.seg_config is not None
        
        # Add ours_configs for proper model loading (keyframe + refine pair)
        if not hasattr(args, 'ours_configs') or not args.ours_configs:
            # Use keyframe.yaml and refine.yaml as default pair (for both regular and segment mode)
            args.ours_configs = ["keyframe.yaml", "refine.yaml"]
        
        debug_print(f"Creating evaluator with configs: {args.ours_configs}")
        
        # Create evaluator with current parameters
        evaluator = EvaluatorDebug(args)
        
        debug_print("Evaluator created successfully")
        
        # If intra_batch_replace is enabled but not set in evaluator, set it manually
        if args.intra_batch_replace and not (hasattr(evaluator, 'intra_batch_replace') and evaluator.intra_batch_replace):
            evaluator.intra_batch_replace = True
        
        # If replace_target is enabled, set it in evaluator
        if args.replace_target:
            evaluator.replace_target_with_refine = True
            print("\n=== REPLACE TARGET MODE SETTINGS ===")
            print("- Using fixed seed (1234) for deterministic data loading")
            print("- Using fixed seed (1234) for deterministic target frame replacement")
            print("- Data consistently shuffled with fixed seed across all modes")
            print("- This ensures the same motion transitions in every run")
            print("======================================\n")
            
        # If replace_target_with_segment is enabled, set it in evaluator
        if args.replace_target_with_segment:
            evaluator.replace_target_with_segment = True
            
        # If multi_ib is enabled, set it in evaluator
        if args.multi_ib:
            evaluator.multi_ib = True
            
        # If custom middle frame is specified, set it in evaluator
        if args.middle_frame is not None:
            evaluator.custom_middle_frame = args.middle_frame
            print(f"Using custom middle frame index: {args.middle_frame}")
            
        # If specific sequence is specified, store it for keyframe display
        if args.seq is not None:
            # Store in a global variable that can be accessed by other modules
            import builtins
            builtins.seq_for_keyframes = args.seq
            print(f"Showing keyframes for sequence: {args.seq}")
            
        # Enable keyframe display if requested with --show_keyframes
        if args.show_keyframes:
            print("Keyframe visualization enabled - keyframes will be displayed for all modes")
            # Will be passed to MotionApp constructor
        
        # If skip_interpolation is enabled, set it in evaluator
        if args.skip_interpolation:
            evaluator.skip_interpolation = True
            print("Skip interpolation mode enabled - directly feeding sparse keyframes to RefineNet")
            
        # If preserve_keyframes is enabled, set it in evaluator
        if args.preserve_keyframes:
            evaluator.preserve_keyframes = True
            print("Preserve keyframes mode enabled - keyframes will not be modified by RefineNet")
        
        # If preserve_middle_frame is enabled, set it in evaluator
        if args.preserve_middle_frame:
            evaluator.preserve_middle_frame = True
            print("Preserve middle frame mode enabled - middle frame will not be modified by RefineNet")
            
        # If compare_modes is enabled, set both modes
        if args.compare_modes:
            evaluator.intra_batch_replace = True
            evaluator.multi_ib = True
            print("Compare modes enabled - using same data for fair comparison")
            
        # Set shuffle preference if specified
        if hasattr(args, 'no_shuffle'):
            evaluator.no_shuffle = args.no_shuffle
        
        # Skip previous batches based on batch_id
        skip_count = args.batch_id
        
        if args.intra_batch_replace:
            shuffle_status = "disabled" if args.no_shuffle else "enabled"
            print("\nNote: Intra-batch circular replacement mode enabled (0→1, 1→2, ..., 63→0)")
            print(f"      The middle keyframe in each sequence will be replaced")
            print(f"      Shuffle mode: {shuffle_status}")
            
        if args.replace_target:
            shuffle_status = "disabled" if args.no_shuffle else "enabled"
            print("\nNote: Replace Target mode enabled (0→1, 1→2, ..., 63→0)")
            print(f"      The target frame (last frame) in each sequence will be replaced")
            print(f"      Shuffle mode: {shuffle_status}")
            print(f"      Display: Ours animations visible by default")
            
        if args.replace_target_with_segment:
            shuffle_status = "disabled" if args.no_shuffle else "enabled"
            print("\nNote: Replace Target with Segment mode enabled (0→1, 1→2, ..., 63→0)")
            print(f"      The target frame (last frame) in each sequence will be replaced")
            print(f"      Using SegmentNet for motion generation")
            print(f"      Shuffle mode: {shuffle_status}")
            print(f"      Display: SegmentNet animations visible by default")
            
        if args.multi_ib:
            shuffle_status = "disabled" if args.no_shuffle else "enabled"
            middle_frame_info = f"custom position {args.middle_frame}" if args.middle_frame is not None else "calculated middle position"
            print("\nNote: Two-stage generation mode enabled")
            print(f"      First stage: Generate and modify target frame at {middle_frame_info}")
            print(f"      Second stage: Generate remaining sequence with context alignment")
            print(f"      Shuffle mode: {shuffle_status}")
            print(f"      Display: Ours animations visible by default")
            
        if args.skip_interpolation:
            print("\nNote: Skip interpolation mode enabled")
            print(f"      Keyframes will be preserved exactly as generated")
            print(f"      RefineNet will only modify non-keyframe positions")
            print(f"      This mode tests RefineNet's ability to generate from sparse keyframes")
            
        if args.preserve_keyframes and not args.skip_interpolation:
            print("\nNote: Preserve keyframes mode enabled")
            print(f"      Standard linear interpolation will be applied between keyframes")
            print(f"      But keyframes will be preserved exactly as generated by KeyframeNet")
            print(f"      RefineNet will only modify non-keyframe positions")
        
        # Store results for later processing
        all_results = []
        
        # Parse keyframe sampling arguments
        if args.kf_param is not None:
            kf_sampling = [args.kf_sampling, args.kf_param]
        else:
            kf_sampling = [args.kf_sampling]
            
        # Load directly the specified batch
        print(f"Directly loading batch {args.batch_id}...")
        
        debug_print(f"Loading batch {args.batch_id} for sequence {args.seq}")
        
        # Special case for SegmentNet visualization
        if use_segment_mode:
            print(f"SegmentNet mode enabled - loading KeyframeNet ({args.kf_config}) and SegmentNet ({args.seg_config}) models...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load dataset for statistics
            seg_config = utils.load_config(f"config/{args.dataset}/{args.seg_config}")
            dataset = MotionDataset(train=False, config=seg_config)
            
            # Statistics for normalization
            mean, std = dataset.motion_statistics()
            mean, std = mean.to(device), std.to(device)
            
            traj_mean, traj_std = dataset.traj_statistics()
            traj_mean, traj_std = traj_mean.to(device), traj_std.to(device)
            
            # Load KeyframeNet model
            kf_config = utils.load_config(f"config/{args.dataset}/{args.kf_config}")
            kf_model = ContextTransformer(kf_config, dataset).to(device)
            utils.load_model(kf_model, kf_config)
            kf_model.eval()
            
            # Load SegmentNet model
            seg_config = utils.load_config(f"config/{args.dataset}/{args.seg_config}")
            seg_model = SegmentNet(seg_config, dataset).to(device)
            utils.load_model(seg_model, seg_config)
            seg_model.eval()
            
            # Get results from baseline methods - directly load specified batch
            baseline_batches = []
            res = evaluator.eval_direct(batch_id=args.batch_id, traj_option=args.traj_edit)
            if res is not None:
                baseline_batches.append(res)
                debug_print(f"Loaded batch {args.batch_id}")
            else:
                debug_print(f"Failed to load batch {args.batch_id}")
            
            # Now add SegmentNet results to each batch
            for batch_idx, res in enumerate(baseline_batches):
                if batch_idx == args.batch_id:  # Only process the requested batch
                    # Extract GT motion and related data
                    GT_motion = res["motions"][0]  # GT motion
                    GT_contact = res["contacts"][0]  # GT contact
                    GT_traj = res["trajs"][0]  # GT trajectory
                    
                    # Get phase from dataset if needed
                    GT_phase = None
                    if seg_config.use_phase:
                        # Try to get phase from the evaluator result first
                        if "phase" in res and res["phase"] is not None:
                            GT_phase = res["phase"][0]  # Use GT phase from results
                        else:
                            # Fall back to loading from dataset
                            batch_data = next(iter(torch.utils.data.DataLoader(dataset, batch_size=GT_motion.shape[0])))
                            GT_phase = batch_data["phase"].to(device)
                            if GT_phase.shape[1] > GT_motion.shape[1]:
                                GT_phase = GT_phase[:, :GT_motion.shape[1]]
                    
                    # Get score from dataset if needed
                    GT_score = None
                    if seg_config.use_score:
                        # Try to get score from the evaluator result first
                        if "score" in res and res["score"] is not None:
                            GT_score = res["score"][0]  # Use GT score from results
                        else:
                            # Fall back to loading from dataset
                            batch_data = next(iter(torch.utils.data.DataLoader(dataset, batch_size=GT_motion.shape[0])))
                            GT_score = batch_data["score"].to(device)
                            if GT_score.shape[1] > GT_motion.shape[1]:
                                GT_score = GT_score[:, :GT_motion.shape[1]]
                    
                    # Generate motion using SegmentNet
                    debug_print("Generating motion with SegmentNet")
                    with torch.no_grad():
                        segment_result = _segment_net_transition(
                            seg_config,
                            kf_model,
                            seg_model,
                            GT_motion,
                            mean,
                            std,
                            GT_contact,
                            GT_phase,
                            GT_traj,
                            GT_score,
                            traj_mean,
                            traj_std,
                            kf_sampling
                        )
                    
                    # Add SegmentNet results to the batch
                    res["motions"].append(segment_result["motion"])
                    res["contacts"].append(segment_result.get("contact", GT_contact))  # Use GT if not available
                    res["trajs"].append(GT_traj)  # Use GT trajectory
                    res["keyframes"].append(segment_result.get("keyframes", []))
                    
                    # Add modified frames information if available
                    if "keyframes" in segment_result:
                        # Create proper modified_frames structure (dictionary with positions)
                        all_kfs = []
                        for b in range(GT_motion.shape[0]):
                            all_kfs.append(segment_result["keyframes"][b])
                        
                        # Create dictionary structure to match expected format in motionapp.py
                        segment_modified_frames = {"positions": all_kfs}
                        
                        # If modified_frames doesn't exist in res, create it
                        if "modified_frames" not in res:
                            res["modified_frames"] = [None] * len(res["tags"])
                        
                        # Add modified frames for SegmentNet
                        res["modified_frames"].append(segment_modified_frames)
                    
                    res["tags"].append("SegmentNet")
                
                all_results.append(res)
        else:
            # Standard loading - directly load specified batch
            debug_print("Starting targeted batch loading")
            
            res = evaluator.eval_direct(batch_id=args.batch_id, traj_option=args.traj_edit)
            if res is not None:
                debug_print(f"Found and loaded batch {args.batch_id}")
                all_results.append(res)
            else:
                debug_print(f"Failed to load batch {args.batch_id}")
            
            debug_print(f"Loaded {len(all_results)} batches in total")
        
        # Handle video saving mode
        if args.save_videos_and_images and not args.compare_modes:
            # Generate output directory based on current mode
            output_dir = get_output_directory(args)
            print(f"Output directory: {output_dir}")
            print(f"Saving videos and images for all {len(all_results)} batches...")
            
            for batch_idx in range(len(all_results)):
                print(f"Processing batch {batch_idx + 1}/{len(all_results)}...")
                
                # Get the current batch
                selected_res = all_results[batch_idx]
                
                # Process sequences based on command line arguments
                num_sequences = selected_res["motions"][0].shape[0]  # Get number of sequences in batch
                
                # If seq is specified, we only want to save that specific sequence
                if args.seq is not None:
                    print(f"Saving only sequence {args.seq} from batch {batch_idx} (as specified by --seq)...")
                    # Set record_sequence_idx to use the specified sequence
                    args.record_sequence_idx = args.seq
                    initial_seq = args.seq
                    # Skip to batch that contains this sequence if needed
                    if args.seq >= 64:
                        target_batch = args.seq // 64
                        if batch_idx != target_batch:
                            print(f"Skipping batch {batch_idx}, sequence {args.seq} is in batch {target_batch}")
                            continue
                elif args.record_sequence_idx is not None:
                    # Only process the specific sequence requested by --record_sequence_idx
                    print(f"Saving only sequence {args.record_sequence_idx} from batch {batch_idx}...")
                    initial_seq = args.record_sequence_idx
                else:
                    # Process all sequences in the batch
                    print(f"Saving all {num_sequences} sequences in batch {batch_idx}...")
                    initial_seq = 0
                    
                # Create MotionApp with video saving enabled
                debug_print("Creating MotionApp for video saving")
                app = MotionApp(selected_res["motions"], selected_res["tags"], selected_res["skeleton"],
                             trajs=selected_res["trajs"],
                             kf_indices=selected_res["keyframes"],
                             modified_frames=selected_res.get("modified_frames", None),
                             dataset=args.dataset,
                             paused=True,
                             save_videos=True,
                             output_dir=output_dir,
                             record_sequence_idx=initial_seq,  # Use specified sequence or start with first
                             show_keyframes=args.show_keyframes)  # Pass keyframe visibility parameter
                
                # Always enable split view for replace_target mode to ensure animations are separated
                if args.replace_target:
                    app._enable_split_view_on_start = True
                    app._enable_split_view = True  # Enable split view immediately
                    app.move_character = True  # Ensure move_character is enabled directly
                    print("Replace target mode with video recording: Enabling animation separation")
                
                # Set visibility based on mode
                for motion in app.motions:
                    if hasattr(motion, 'tag'):
                        tag = motion.tag
                        if args.multi_ib or args.intra_batch_replace or args.replace_target:
                            if tag == "Ours" or tag.startswith("Ours") or tag == "Intra-Batch" or tag == "Multi-IB":
                                motion.visible = True
                                print(f"  Setting {tag} visible for recording")
                
                # Process sequence(s)
                if args.seq is not None or args.record_sequence_idx is not None:
                    # Process just the specific sequence
                    seq_to_record = initial_seq
                    # When seq is provided, ensure we're using the correct sequence index
                    if args.seq is not None:
                        # Calculate correct batch and sequence-within-batch
                        target_batch = args.seq // 64
                        seq_within_batch = args.seq % 64
                        print(f"Recording sequence {args.seq} (global) = sequence {seq_within_batch} in batch {target_batch}...")
                        # Ensure we're using the correct batch
                        if batch_idx == target_batch:
                            # First start the app to initialize MotionStruct objects
                            agl.AppManager.start(app)
                            # Now process_batch_for_video is safe to call
                            app.process_batch_for_video(batch_idx, start_app=False)
                        else:
                            print(f"Warning: Current batch {batch_idx} doesn't match target batch {target_batch} for sequence {args.seq}")
                    else:
                        # Normal record_sequence_idx behavior
                        print(f"Recording sequence {seq_to_record} in batch {batch_idx}...")
                        # First start the app to initialize MotionStruct objects
                        agl.AppManager.start(app)
                        # Now process_batch_for_video is safe to call
                        app.process_batch_for_video(batch_idx, start_app=False)
                else:
                    # Process all sequences in batch
                    # Process first sequence - this starts the app
                    print(f"Recording sequence 0 in batch {batch_idx}...")
                    # First start the app to initialize MotionStruct objects
                    agl.AppManager.start(app)
                    # Now process_batch_for_video is safe to call
                    app.process_batch_for_video(batch_idx, start_app=False)
                    
                    # Process remaining sequences without starting new app
                    for seq_idx in range(1, num_sequences):
                        print(f"Processing sequence {seq_idx + 1}/{num_sequences} in batch {batch_idx}...")
                        
                        # Update sequence index for next recording
                        app.record_sequence_idx = seq_idx
                    
                        # Start new recording session for this sequence
                        app._start_video_recording(batch_idx)
                        
                        # Process this sequence (don't start a new app)
                        print(f"Recording sequence {seq_idx} in batch {batch_idx}...")
                        app.process_batch_for_video(batch_idx, start_app=False)
                
                # Close the window after all sequences are processed
                if hasattr(app, 'window') and app.window:
                    glfw.set_window_should_close(app.window, True)
                
                # Release resources
                del app
                
            print(f"All videos and images saved to {output_dir}")
            
        elif args.save_videos_and_images and args.compare_modes:
            # Handle compare_modes with video saving - show both motions together
            output_dir = get_output_directory(args, "compare")
            print(f"Compare modes: Saving videos to {output_dir}")
            
            # Convert sequence number to batch index and within-batch sequence index
            if args.seq is not None:
                # Calculate which batch contains this sequence (assuming 64 sequences per batch)
                batch_idx = args.seq // 64
                seq_in_batch = args.seq % 64  # Sequence index within the batch
                # Ensure the batch index is within bounds
                if batch_idx >= len(all_results):
                    print(f"Error: Sequence {args.seq} (batch {batch_idx}) not found in {len(all_results)} loaded batches")
                    sys.exit(1)
                print(f"Sequence {args.seq} is in batch {batch_idx}, position {seq_in_batch} within batch")
            else:
                batch_idx = args.batch_id
                seq_in_batch = None
                
            # Ensure batch_idx is within bounds
            if batch_idx >= len(all_results):
                print(f"Error: Batch index {batch_idx} out of range (available: 0-{len(all_results)-1})")
                sys.exit(1)
                
            selected_res = all_results[batch_idx]
            
            # Debug: print what we have
            print(f"Available motions: {len(selected_res['motions'])}")
            print(f"Tags: {selected_res['tags']}")
            
            # Create MotionApp with video saving enabled
            app = MotionApp(selected_res["motions"], selected_res["tags"], selected_res["skeleton"],
                         trajs=selected_res["trajs"],
                         kf_indices=selected_res["keyframes"],
                         modified_frames=selected_res.get("modified_frames", None),
                         dataset=args.dataset,
                         paused=True,
                         save_videos=True,
                         output_dir=output_dir,
                         record_sequence_idx=seq_in_batch,  # Use sequence index within batch
                         compare_mode=True,
                         show_keyframes=args.show_keyframes)  # Control keyframe visibility
            
            # Start the recording process
            agl.AppManager.start(app)
            
        else:
            # Original interactive mode
            # If the specified batch was successfully loaded
            if len(all_results) > 0:
                # Removed cross-sequence replacement logic
                
                # Get and display the specified batch
                selected_res = all_results[0]  # When directly loading, result is always at index 0
                
                debug_print(f"Interactive mode: showing batch {args.batch_id}")
                debug_print(f"Motions shape: {selected_res['motions'][0].shape}")
                debug_print(f"Tags: {selected_res['tags']}")
                
                # Check keyframes 
                if args.seq is not None:
                    seq_idx = args.seq
                    debug_print(f"Examining keyframes for sequence {seq_idx}")
                    for i, kf in enumerate(selected_res["keyframes"]):
                        if kf is not None:
                            debug_print(f"Keyframes for '{selected_res['tags'][i]}': {kf[seq_idx] if seq_idx < len(kf) else 'sequence index out of range'}")
                
                # Create MotionApp with modified poses support
                debug_print("Creating MotionApp instance")
                app = MotionApp(selected_res["motions"], selected_res["tags"], selected_res["skeleton"],
                             trajs=selected_res["trajs"],
                             kf_indices=selected_res["keyframes"],
                             modified_frames=selected_res.get("modified_frames", None),
                             dataset=args.dataset,
                             paused=True,  # Start in paused state
                             record_sequence_idx=args.seq,  # Show specific sequence if specified
                             compare_mode=args.compare_modes,  # Enable compare mode behavior
                             show_keyframes=args.show_keyframes)  # Control keyframe visibility
                
                # For two-stage generation or segment_net, relevant motions are visible by default
                if args.multi_ib:
                    # All Ours motions should be visible
                    for motion in app.motions:
                        if hasattr(motion, 'tag'):
                            tag = motion.tag
                            if tag.startswith("Ours"):
                                motion.visible = True
                
                # For replace_target mode, enable split view by default
                # but do this AFTER app.start() is called, when MotionStruct objects are created
                # We'll just set a flag to remember we want to do this
                app._enable_split_view_on_start = True
                print("  Split view mode will be enabled for replace_target animations")
                
                # For replace_target_with_segment or SegmentNet mode, SegmentNet motions are visible by default
                if args.replace_target_with_segment or use_segment_mode:
                    # All SegmentNet motions should be visible
                    for motion in app.motions:
                        if hasattr(motion, 'tag'):
                            tag = motion.tag
                            if tag == "SegmentNet":
                                motion.visible = True
                
                debug_print("Starting AppManager")
                agl.AppManager.start(app)
            else:
                print(f"Error: Could not find batch {args.batch_id}, maximum batch is {len(all_results)-1}")
                sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")