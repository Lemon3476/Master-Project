import sys
sys.path.append(".")

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader

from utils import eval_backup, utils
from utils.dataset import MotionDataset
from model.twostage import ContextTransformer, DetailTransformer

def calculate_height_range(motion_tensor, ctx_frames=None):
    """Calculate the height range (max height - min height) of the motion sequence.
    
    Args:
        motion_tensor: Motion data tensor of shape [B, T, D]
        ctx_frames: Optional context frames to exclude from calculation
    
    Returns:
        Tensor of shape [B] containing height range for each sequence
    """
    # Extract root position (last 3 dimensions)
    B, T, D = motion_tensor.shape
    local_rot, root_pos = torch.split(motion_tensor, [D-3, 3], dim=-1)
    
    # Extract Y-axis component (index 1)
    root_pos_y = root_pos[:, :, 1]  # Shape: [B, T]
    
    # If context frames are specified, only calculate on transition frames
    if ctx_frames is not None:
        root_pos_y = root_pos_y[:, ctx_frames:-1]
    
    # Calculate height range for each sequence
    height_range = torch.max(root_pos_y, dim=1)[0] - torch.min(root_pos_y, dim=1)[0]  # Shape: [B]
    
    return height_range

def visualize_results(results_df, output_file, transition_lengths=[15, 30, 60, 90]):
    """Create scatter plots showing the relationship between height range and performance metrics.
    
    Args:
        results_df: DataFrame containing all results
        output_file: Output file path for the visualization
        transition_lengths: List of transition lengths to include
    """
    # Create 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Define color map for different transition lengths
    colors = ['blue', 'green', 'orange', 'red']
    markers = ['o', 's', '^', 'd']
    
    metric_names = ['L2P', 'L2Q', 'NPSS', 'Foot_Skate']
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # Create each subplot
    for (metric, pos) in zip(metric_names, subplot_positions):
        ax = axs[pos]
        
        # Plot data for each transition length
        for i, trans_len in enumerate(transition_lengths):
            # Filter data for this transition length
            trans_data = results_df[results_df['Transition_Length'] == trans_len]
            
            # Skip if no data for this transition length
            if len(trans_data) == 0:
                continue
                
            # Plot scatter points
            ax.scatter(
                trans_data['Height_Range'], 
                trans_data[metric], 
                color=colors[i], 
                marker=markers[i],
                alpha=0.7, 
                s=30, 
                edgecolors='k', 
                linewidths=0.5,
                label=f'T={trans_len}'
            )
            
            # Add trend line
            x = trans_data['Height_Range']
            y = trans_data[metric]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(x), max(x), 100)
            ax.plot(x_range, p(x_range), color=colors[i], linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Calculate correlation
            corr = np.corrcoef(x, y)[0, 1]
            
            # Add correlation text near the end of the line
            text_x = max(x) * 0.9
            text_y = p(text_x)
            ax.text(text_x, text_y, f'r={corr:.3f}', color=colors[i], fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Set subplot title and labels
        ax.set_title(f'{metric} vs Height Range', fontsize=14, fontweight='bold')
        ax.set_xlabel('Height Range (m)', fontsize=12)
        ax.set_ylabel(f'{metric} Score', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()
    
    # Add overall title
    plt.suptitle("Motion Complexity vs Performance Analysis", fontsize=16, y=0.98)
    
    # Add footnote with sample count
    sample_counts = results_df.groupby('Transition_Length').size()
    sample_text = ", ".join([f"T={t}: {n} samples" for t, n in sample_counts.items()])
    plt.figtext(0.5, 0.01, f"Analysis includes {sample_text}", 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()

def run_analysis(args):
    """Main function to run the complexity vs performance analysis"""
    print(f"Starting complexity vs performance analysis for dataset: {args.dataset}")
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = utils.load_config(f"config/{args.dataset}/{args.config}")
    
    # Load model configurations
    if args.ours_configs:
        model_configs = []
        for i in range(0, len(args.ours_configs), 2):
            kf_config_path = args.ours_configs[i]
            ref_config_path = args.ours_configs[i+1] if i+1 < len(args.ours_configs) else None
            
            kf_config = utils.load_config(f"config/{args.dataset}/{kf_config_path}")
            model_configs.append({"kf_config": kf_config, "kf_path": kf_config_path})
            
            if ref_config_path:
                ref_config = utils.load_config(f"config/{args.dataset}/{ref_config_path}")
                model_configs[-1]["ref_config"] = ref_config
                model_configs[-1]["ref_path"] = ref_config_path
    else:
        # Use default configurations specified in args
        kf_config = utils.load_config(f"config/{args.dataset}/{args.kf_config}")
        ref_config = utils.load_config(f"config/{args.dataset}/{args.ref_config}")
        model_configs = [{"kf_config": kf_config, "ref_config": ref_config, 
                          "kf_path": args.kf_config, "ref_path": args.ref_config}]
    
    # Load test dataset
    test_dataset = MotionDataset(train=False, config=config)
    
    # Get statistics
    mean, std = test_dataset.motion_statistics(device)
    traj_mean, traj_std = test_dataset.traj_statistics(device)
    l2p_mean, l2p_std = test_dataset.l2p_statistics(device)
    skeleton = test_dataset.skeleton
    
    # Set contact joints
    contact_idx = []
    for joint in config.contact_joints:
        contact_idx.append(skeleton.idx_by_name[joint])
    
    # Define the transition lengths to evaluate
    transitions = [15, 30, 60, 90]
    print(f"Evaluating transition lengths: {transitions}")
    
    # Initialize results storage
    all_results = []
    
    # Process each model configuration
    for model_idx, model_config in enumerate(model_configs):
        print(f"\nProcessing model {model_idx+1}/{len(model_configs)}:")
        print(f"  KeyframeNet: {model_config['kf_path']}")
        print(f"  RefineNet: {model_config.get('ref_path', 'None')}")
        
        # Load models
        kf_model = ContextTransformer(model_config['kf_config'], test_dataset).to(device)
        utils.load_model(kf_model, model_config['kf_config'])
        kf_model.eval()
        
        if 'ref_config' in model_config:
            ref_model = DetailTransformer(model_config['ref_config'], test_dataset).to(device)
            utils.load_model(ref_model, model_config['ref_config'])
            ref_model.eval()
        else:
            ref_model = None
        
        # Set keyframe sampling method
        if args.kf_param is not None:
            kf_sampling = [args.kf_sampling, args.kf_param]
        else:
            kf_sampling = [args.kf_sampling]
        print(f"  Using keyframe sampling method: {kf_sampling}")
        
        # Process each transition length
        for transition in transitions:
            print(f"\nEvaluating transition length: {transition}")
            
            # Create data loader for this batch size
            num_frames = config.context_frames + transition + 1
            dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
            
            # Process each batch
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing batches (T={transition})")):
                    # Stop if max batches is reached
                    if args.max_batches is not None and batch_idx >= args.max_batches:
                        print(f"Reached maximum number of batches ({args.max_batches}), stopping.")
                        break
                    
                    # Get data
                    GT_motion = batch["motion"].to(device)
                    GT_phase = batch["phase"].to(device) if "phase" in batch else None
                    GT_traj = batch["traj"].to(device) if "traj" in batch else None
                    GT_score = batch["score"].to(device) if "score" in batch else None
                    
                    # Limit sequence length
                    GT_motion = GT_motion[:, :num_frames]
                    if GT_phase is not None:
                        GT_phase = GT_phase[:, :num_frames]
                    if GT_traj is not None:
                        GT_traj = GT_traj[:, :num_frames]
                    if GT_score is not None:
                        GT_score = GT_score[:, :num_frames]
                    
                    # Calculate contact data
                    GT_contact = eval_backup._get_foot_velocity(GT_motion, skeleton, contact_idx)
                    GT_contact = (GT_contact < config.contact_threshold).float()
                    
                    # Generate predictions using our model
                    if ref_model is not None:
                        ref_config = model_config['ref_config']
                        ours_out = eval_backup._ours_transition(
                            ref_config, kf_model, ref_model, GT_motion, mean, std, 
                            GT_contact, GT_phase, GT_traj, GT_score, traj_mean, traj_std, kf_sampling
                        )
                    else:
                        # Fall back to just using KeyframeNet if RefineNet is not available
                        print("  Warning: RefineNet not available, using only KeyframeNet")
                        ours_out = {"motion": GT_motion, "contact": GT_contact}
                    
                    pred_motion = ours_out["motion"]
                    pred_contact = ours_out["contact"]
                    
                    # Calculate height range (complexity metric)
                    height_ranges = calculate_height_range(GT_motion, config.context_frames)
                    
                    # Process each sequence in the batch individually
                    for seq_idx in range(GT_motion.shape[0]):
                        # Extract single sequence
                        gt_motion_single = GT_motion[seq_idx:seq_idx+1]
                        pred_motion_single = pred_motion[seq_idx:seq_idx+1]
                        pred_contact_single = pred_contact[seq_idx:seq_idx+1]
                        
                        # Calculate metrics
                        l2p = eval_backup.l2p(gt_motion_single, pred_motion_single, skeleton, l2p_mean, l2p_std, config.context_frames)
                        l2q = eval_backup.l2q(gt_motion_single, pred_motion_single, config.context_frames)
                        npss = eval_backup.npss(gt_motion_single, pred_motion_single, config.context_frames)
                        foot_skate = eval_backup.foot_skate(pred_motion_single, pred_contact_single, skeleton, contact_idx, ctx_frames=config.context_frames)
                        
                        # Store results
                        result = {
                            "Batch": batch_idx,
                            "Sequence": seq_idx,
                            "Transition_Length": transition,
                            "Height_Range": height_ranges[seq_idx].item(),
                            "Model": model_idx,
                            "KF_Config": model_config['kf_path'],
                            "RF_Config": model_config.get('ref_path', 'None'),
                            "L2P": l2p,
                            "L2Q": l2q,
                            "NPSS": npss * 10,  # Scale NPSS to be in same range as other metrics
                            "Foot_Skate": foot_skate * 10  # Scale foot skate for visibility
                        }
                        all_results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save to Excel file with multiple sheets
    excel_file = os.path.join(args.output_dir, args.output_file)
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Create a sheet for each transition length
        for trans_len in transitions:
            trans_data = results_df[results_df['Transition_Length'] == trans_len].copy()
            
            # Only create sheet if we have data for this transition length
            if len(trans_data) > 0:
                sheet_name = f'Transition_{trans_len}'
                
                # Sort by height range
                trans_data = trans_data.sort_values('Height_Range')
                
                # Drop the now-redundant transition length column
                trans_data = trans_data.drop('Transition_Length', axis=1)
                
                # Write to Excel
                trans_data.to_excel(writer, sheet_name=sheet_name, index=False, float_format="%.4f")
                print(f"  - Sheet '{sheet_name}' created with {len(trans_data)} rows")
        
        # Create a summary statistics sheet
        summary_stats = []
        for trans_len in transitions:
            trans_data = results_df[results_df['Transition_Length'] == trans_len]
            
            if len(trans_data) > 0:
                # Calculate correlation between height range and each metric
                corr_l2p = trans_data['Height_Range'].corr(trans_data['L2P'])
                corr_l2q = trans_data['Height_Range'].corr(trans_data['L2Q'])
                corr_npss = trans_data['Height_Range'].corr(trans_data['NPSS'])
                corr_fs = trans_data['Height_Range'].corr(trans_data['Foot_Skate'])
                
                # Calculate mean and std for each metric
                stats = {
                    'Transition_Length': trans_len,
                    'Count': len(trans_data),
                    'Height_Range_Mean': trans_data['Height_Range'].mean(),
                    'Height_Range_Std': trans_data['Height_Range'].std(),
                    'L2P_Mean': trans_data['L2P'].mean(),
                    'L2P_Std': trans_data['L2P'].std(),
                    'L2P_Correlation': corr_l2p,
                    'L2Q_Mean': trans_data['L2Q'].mean(),
                    'L2Q_Std': trans_data['L2Q'].std(),
                    'L2Q_Correlation': corr_l2q,
                    'NPSS_Mean': trans_data['NPSS'].mean(),
                    'NPSS_Std': trans_data['NPSS'].std(),
                    'NPSS_Correlation': corr_npss,
                    'Foot_Skate_Mean': trans_data['Foot_Skate'].mean(),
                    'Foot_Skate_Std': trans_data['Foot_Skate'].std(),
                    'Foot_Skate_Correlation': corr_fs
                }
                summary_stats.append(stats)
        
        # Create summary sheet
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_excel(writer, sheet_name='Summary_Stats', index=False, float_format="%.4f")
            print(f"  - Sheet 'Summary_Stats' created")
    
    print(f"Results saved to {excel_file}")
    
    # Generate visualization
    if len(all_results) > 0:
        png_file = os.path.join(args.output_dir, "complexity_vs_performance_scatter.png")
        visualize_results(results_df, png_file, transitions)
    
    print("Analysis completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze the relationship between motion complexity and performance metrics')
    parser.add_argument('--dataset', type=str, default='lafan1', help='Dataset name')
    parser.add_argument('--config', type=str, default='default.yaml', help='Configuration file')
    
    # Model configuration
    parser.add_argument('--kf_config', type=str, default='keyframe-d-enc-dec.yaml', help='Default KeyframeNet configuration file')
    parser.add_argument('--ref_config', type=str, default='refine-d-enc-dec-fc.yaml', help='Default RefineNet configuration file')
    parser.add_argument('--ours_configs', type=str, nargs='+', default=[], 
                        help='Custom model configurations to evaluate (pairs of keyframe and refine configs)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='eval/complexity_analysis', help='Output directory for results')
    parser.add_argument('--output_file', type=str, default='pose_complexity_metrics.xlsx', help='Output Excel file name')
    
    # Runtime configuration
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--max_batches', type=int, default=None, help='Maximum number of batches to process (default: all)')
    parser.add_argument('--kf_sampling', type=str, default="score", help='Keyframe sampling method: score, threshold, topk, random, uniform')
    parser.add_argument('--kf_param', type=float, default=None, help='Parameter for keyframe sampling method')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Motion Complexity vs Performance Analysis")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    if args.ours_configs:
        print(f"Custom model configurations: {args.ours_configs}")
    else:
        print(f"Models: KeyframeNet={args.kf_config}, RefineNet={args.ref_config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output file: {args.output_file}")
    print("-" * 80)
    
    run_analysis(args)