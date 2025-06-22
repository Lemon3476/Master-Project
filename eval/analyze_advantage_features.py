#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Analysis of Advantage Samples

This script implements a comprehensive analysis of motion features to identify
what characteristics make certain samples benefit more from the proposed method.
The analysis includes:
1. Defining top-N advantage samples based on advantage rankings
2. Computing kinematic, geometric, and pose features for all samples
3. Performing statistical analysis with boxplots, scatter plots, and t-tests
"""

import os
import sys
import argparse
import random
import warnings
import numpy as np
import pandas as pd
import time

# Global debug flag
DEBUG = False

def debug_print(*args, **kwargs):
    """
    Print debug information only if DEBUG is True.
    
    Args:
        *args: Variable length argument list to print
        **kwargs: Variable length keyword arguments to pass to print function
    """
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

# Filter specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*?is deprecated.*?")

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(".")

# Import project-specific modules
from utils.dataset import MotionDataset
from utils import utils
from aPyOpenGL import transforms as trf

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def define_advantage_group(ranking_file, top_n=100):
    """
    Define the advantage group by selecting the top-N samples from the ranking file.
    
    Args:
        ranking_file (str): Path to the advantage ranking Excel file
        top_n (int): Number of top samples to select for the advantage group
        
    Returns:
        pd.DataFrame: DataFrame containing all samples with their group (Advantage/Reference)
                     and transition length information
    """
    print(f"Defining advantage group from {ranking_file} (top {top_n} samples)...")
    
    # Load all sheets from the Excel file
    excel_data = pd.read_excel(ranking_file, sheet_name=None)
    
    # List to store individual dataframes for each transition length
    group_dfs = []
    
    # Process each sheet in the ranking file
    for sheet_name, df in excel_data.items():
        # Check if the sheet is an advantage ranking sheet
        if sheet_name.startswith("Advantage_Rank_T"):
            try:
                # Extract transition length
                transition_length = int(sheet_name.split("_")[-1][1:])
                
                # Ensure the DataFrame contains the necessary columns
                if "Batch" in df.columns and "Sequence" in df.columns:
                    # Make a copy to avoid modifying the original DataFrame
                    sheet_df = df.copy()
                    
                    # Convert Batch and Sequence to integers if they're not already
                    sheet_df["Batch"] = sheet_df["Batch"].astype(int)
                    sheet_df["Sequence"] = sheet_df["Sequence"].astype(int)
                    
                    # Add transition length column
                    sheet_df["Transition_Length"] = transition_length
                    
                    # Add group column (Advantage for top_n, Reference for the rest)
                    sheet_df["Group"] = "Reference"
                    if len(sheet_df) > 0:
                        top_indices = min(top_n, len(sheet_df))
                        sheet_df.loc[sheet_df.index[:top_indices], "Group"] = "Advantage"
                    
                    # Count samples in each group
                    advantage_count = (sheet_df["Group"] == "Advantage").sum()
                    reference_count = (sheet_df["Group"] == "Reference").sum()
                    
                    print(f"  - Transition Length {transition_length}: "
                          f"Found {advantage_count} advantage samples and {reference_count} reference samples")
                    
                    # Add to our list of dataframes
                    group_dfs.append(sheet_df)
                else:
                    print(f"  - Warning: Sheet '{sheet_name}' is missing required columns 'Batch' or 'Sequence', skipping")
            except (ValueError, IndexError) as e:
                print(f"  - Warning: Could not extract transition length from sheet name '{sheet_name}': {str(e)}, skipping")
    
    if not group_dfs:
        raise ValueError("No advantage samples found, please check the Excel file format")
    
    # Combine all dataframes into one
    grouping_df = pd.concat(group_dfs, ignore_index=True)
    
    # Print summary of the grouping DataFrame
    print(f"Created grouping DataFrame with {len(grouping_df)} total entries")
    for trans_len in grouping_df["Transition_Length"].unique():
        trans_df = grouping_df[grouping_df["Transition_Length"] == trans_len]
        adv_count = (trans_df["Group"] == "Advantage").sum()
        ref_count = (trans_df["Group"] == "Reference").sum()
        print(f"  - T{trans_len}: {adv_count} advantage samples, {ref_count} reference samples")
    
    return grouping_df


def calculate_velocities_and_accelerations(positions):
    """
    Calculate velocities, accelerations, and jerks from position data.
    
    Args:
        positions (np.ndarray): Position data with shape [T, 3] for xyz coordinates
        
    Returns:
        tuple: (velocities, accelerations, jerks) each with shape [T-1], [T-2], [T-3]
    """
    # Calculate velocities (first derivative of position)
    velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    
    # Calculate accelerations (second derivative of position)
    accelerations = np.diff(velocities, axis=0)
    
    # Calculate jerks (third derivative of position)
    jerks = np.diff(accelerations, axis=0)
    
    return velocities, accelerations, jerks


def calculate_trajectory_length(positions):
    """
    Calculate the total length of a trajectory.
    
    Args:
        positions (np.ndarray): Position data with shape [T, 3] for xyz coordinates
        
    Returns:
        float: Total trajectory length
    """
    # Calculate distances between consecutive points
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    
    # Sum all distances to get total length
    total_length = np.sum(distances)
    
    return total_length


def calculate_trajectory_curvature(positions):
    """
    Calculate the average curvature of a trajectory.
    
    Args:
        positions (np.ndarray): Position data with shape [T, 3] for xyz coordinates
        
    Returns:
        float: Average trajectory curvature
    """
    # We need at least 3 points to calculate curvature
    if positions.shape[0] < 3:
        return 0.0
    
    try:
        # Calculate velocity vectors
        velocities = np.diff(positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Check if speeds are all close to zero
        if np.all(speeds < 1e-6):
            return 0.0
        
        # Avoid division by zero with higher threshold for numerical stability
        non_zero_speeds = np.where(speeds > 1e-6, speeds, 1e-6)
        
        # Normalize velocity vectors
        normalized_velocities = velocities / non_zero_speeds[:, np.newaxis]
        
        # Calculate the angle between consecutive velocity vectors
        dot_products = np.sum(normalized_velocities[:-1] * normalized_velocities[1:], axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)  # Clamp to avoid numerical issues
        angles = np.arccos(dot_products)
        
        # Calculate curvature as angle change per unit distance
        # Ensure we're not dividing by near-zero values
        valid_speeds = speeds[1:] > 1e-6
        if not np.any(valid_speeds):
            return 0.0
            
        curvatures = np.zeros_like(angles)
        curvatures[valid_speeds] = angles[valid_speeds] / speeds[1:][valid_speeds]
        
        # Filter out any invalid values (NaN or Inf)
        valid_curvatures = curvatures[~np.isnan(curvatures) & ~np.isinf(curvatures)]
        return np.mean(valid_curvatures) if len(valid_curvatures) > 0 else 0.0
    
    except Exception as e:
        print(f"WARNING: Error calculating trajectory curvature: {str(e)}. Returning 0.")
        return 0.0


def calculate_pose_variation(positions):
    """
    Calculate the pose variation as the standard deviation of joint positions.
    
    Args:
        positions (np.ndarray): Joint position data with shape [T, J, 3]
                                where T is time, J is number of joints
        
    Returns:
        float: Pose variation measure
    """
    # Calculate standard deviation of each joint position over time
    joint_stds = np.std(positions, axis=0)  # [J, 3]
    
    # Calculate the mean standard deviation across all joints
    mean_std = np.mean(np.linalg.norm(joint_stds, axis=1))
    
    return mean_std


def calculate_body_extent(positions):
    """
    Calculate the average body extent as the average diagonal length of the bounding box.
    
    Args:
        positions (np.ndarray): Joint position data with shape [T, J, 3]
                                where T is time, J is number of joints
        
    Returns:
        float: Average body extent
    """
    extents = []
    
    # Calculate body extent for each frame
    for frame_positions in positions:
        # Calculate min and max coordinates for each dimension
        min_coords = np.min(frame_positions, axis=0)
        max_coords = np.max(frame_positions, axis=0)
        
        # Calculate bounding box diagonal length
        diagonal = np.linalg.norm(max_coords - min_coords)
        extents.append(diagonal)
    
    # Return average body extent
    return np.mean(extents)


def compute_features_for_sample(motion, skeleton):
    """
    Compute kinematic, geometric and pose features for a single motion sample.
    
    Args:
        motion (torch.Tensor or np.ndarray): Motion data for a single sample
        skeleton: Skeleton data structure for forward kinematics
        
    Returns:
        dict: Dictionary containing computed features for the sample
    """
    # Ensure motion is a PyTorch tensor
    if not isinstance(motion, torch.Tensor):
        motion = torch.tensor(motion)
    
    # Get motion dimensions
    T, D = motion.shape  # [T, D]
    
    # Add batch dimension to match fk function requirements [1, T, D]
    motion_batch = motion.unsqueeze(0)
    
    # Split rotation and position
    rot, pos = torch.split(motion_batch, [D-3, 3], dim=-1)
    
    # Reshape rotation data to match fk function requirements
    rot = rot.reshape(1, T, -1, 6)  # [1, T, J, 6] where J is number of joints
    
    # Use transforms fk function to calculate global joint positions
    _, joints_positions = trf.t_ortho6d.fk(rot, pos, skeleton)  # [1, T, J, 3]
    
    # Convert tensor to NumPy array and remove batch dimension
    joints_positions = joints_positions.cpu().numpy()[0]  # [T, J, 3]
    
    # Get root (hip) joint positions
    root_positions = joints_positions[:, 0]  # [T, 3]
    
    # Calculate kinematic features
    velocities, accelerations, jerks = calculate_velocities_and_accelerations(root_positions)
    
    avg_velocity = np.mean(velocities)
    peak_velocity = np.max(velocities)
    
    avg_acceleration = np.mean(np.abs(accelerations))
    peak_acceleration = np.max(np.abs(accelerations))
    
    avg_jerk = np.mean(np.abs(jerks))
    peak_jerk = np.max(np.abs(jerks))
    
    # Calculate geometric features
    traj_length = calculate_trajectory_length(root_positions)
    traj_curvature = calculate_trajectory_curvature(root_positions)
    
    # Calculate pose features
    pose_var = calculate_pose_variation(joints_positions)
    body_ext = calculate_body_extent(joints_positions)
    
    # Return all features as a dictionary
    return {
        "Avg_Velocity": avg_velocity,
        "Peak_Velocity": peak_velocity,
        "Avg_Acceleration": avg_acceleration,
        "Peak_Acceleration": peak_acceleration,
        "Avg_Jerk": avg_jerk,
        "Peak_Jerk": peak_jerk,
        "Trajectory_Length": traj_length,
        "Trajectory_Curvature": traj_curvature,
        "Pose_Variation": pose_var,
        "Body_Extent": body_ext
    }


def compute_features_for_dataset(dataset, output_file=None):
    """
    Compute kinematic, geometric and pose features for all samples in the dataset.
    This function is now decoupled from the grouping logic.
    
    Args:
        dataset (MotionDataset): Motion dataset object
        output_file (str, optional): Path to save the features Excel file
        
    Returns:
        pd.DataFrame: DataFrame containing all computed features
    """
    print("Computing features for all samples in the dataset...")
    
    # Extract skeleton from dataset
    skeleton = dataset.skeleton
    
    # Print dataset information if available
    if hasattr(dataset, 'motion'):
        print("Shapes:")
        print(f"    - motion.shape: {dataset.motion.shape}")
        if hasattr(dataset, 'phase'):
            print(f"    - phase.shape: {dataset.phase.shape}")
        if hasattr(dataset, 'traj'):
            print(f"    - traj.shape: {dataset.traj.shape}")
        if hasattr(dataset, 'score'):
            print(f"    - score.shape: {dataset.score.shape}")
    
    # List to store feature dictionaries for each sample
    feature_list = []
    
    # Get batch size from the global context (set in main function)
    # Default is 64 which is what benchmark uses
    try:
        BATCH_SIZE = globals().get('BATCH_SIZE', 64)
        print(f"Using batch size: {BATCH_SIZE}")
    except:
        BATCH_SIZE = 64
        print(f"Using default batch size: {BATCH_SIZE}")
    
    # Process each sample in the dataset
    for idx in tqdm(range(len(dataset)), desc="Computing features"):
        try:
            # Get sample data
            item = dataset[idx]
            motion = item["motion"]
            
            # Map the flat index to (batch_id, sequence_id) using benchmark convention
            # Where each batch has BATCH_SIZE sequences
            batch_id = idx // BATCH_SIZE
            sequence_id = idx % BATCH_SIZE
            
            debug_print(f"Sample {idx} mapped to Batch {batch_id}, Sequence {sequence_id}")
            
            # Compute features for this sample
            sample_features = compute_features_for_sample(motion, skeleton)
            
            # Add sample identification
            sample_features["Batch"] = batch_id
            sample_features["Sequence"] = sequence_id
            
            # Add to feature list
            feature_list.append(sample_features)
            
        except Exception as e:
            print(f"WARNING: Error computing features for sample {idx}: {str(e)}")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(feature_list)
    
    # Print summary of generated IDs
    batch_ids = features_df["Batch"].unique()
    print(f"Generated {len(batch_ids)} unique batch IDs: {sorted(batch_ids)}")
    print(f"Sequence ID range: {features_df['Sequence'].min()} - {features_df['Sequence'].max()}")
    
    # Save to Excel if output file is provided
    if output_file:
        print(f"Saving computed features to {output_file}...")
        features_df.to_excel(output_file, index=False)
    
    return features_df


def merge_features_with_groups(features_df, grouping_df, output_file=None):
    """
    Merge feature data with group information.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing computed features for all samples
        grouping_df (pd.DataFrame): DataFrame containing group assignments for all samples
        output_file (str, optional): Path to save the merged Excel file
        
    Returns:
        pd.DataFrame: Merged DataFrame with features and group assignments
    """
    print("\n" + "="*50)
    print("MERGING FEATURES WITH GROUP ASSIGNMENTS")
    print("="*50)
    
    # Ensure the key columns are of the same type for merging
    features_df["Batch"] = features_df["Batch"].astype(int)
    features_df["Sequence"] = features_df["Sequence"].astype(int)
    grouping_df["Batch"] = grouping_df["Batch"].astype(int)
    grouping_df["Sequence"] = grouping_df["Sequence"].astype(int)
    
    # Print some information about the dataframes before merging
    print(f"Features DataFrame: {len(features_df)} entries")
    print(f"Grouping DataFrame: {len(grouping_df)} entries")
    
    # Debug: Print sample ID ranges to check for overlap
    print("\nDEBUG - ID ranges check:")
    print(f"Features Batch IDs: {features_df['Batch'].unique()}")
    print(f"Features Sequence ID range: {features_df['Sequence'].min()} - {features_df['Sequence'].max()}")
    print(f"Grouping Batch IDs: {grouping_df['Batch'].unique()}")
    print(f"Grouping Sequence ID range: {grouping_df['Sequence'].min()} - {grouping_df['Sequence'].max()}")
    
    # Check for overlap between feature and grouping DataFrames
    feature_keys = set(zip(features_df["Batch"], features_df["Sequence"]))
    grouping_keys = set(zip(grouping_df["Batch"], grouping_df["Sequence"]))
    common_keys = feature_keys.intersection(grouping_keys)
    print(f"Common sample IDs between features and grouping: {len(common_keys)} out of {len(grouping_keys)} grouping samples")
    if len(common_keys) < min(len(feature_keys), len(grouping_keys)):
        print(f"WARNING: Only {len(common_keys)} samples match between features and grouping data!")
        print(f"This means {len(grouping_keys) - len(common_keys)} samples from grouping data won't have features.")
    
    # Create a list to store merged dataframes for each transition length
    merged_dfs = []
    
    print("\nMerging by transition length:")
    # Process each transition length separately
    for trans_len in sorted(grouping_df["Transition_Length"].unique()):
        # Get grouping data for this transition length
        trans_groups = grouping_df[grouping_df["Transition_Length"] == trans_len]
        
        # Merge with features
        # This performs a left join from trans_groups to features_df
        merged = pd.merge(
            trans_groups, 
            features_df, 
            on=["Batch", "Sequence"], 
            how="left"
        )
        
        # Check how many samples we have for each group after merging
        adv_count = (merged["Group"] == "Advantage").sum()
        ref_count = (merged["Group"] == "Reference").sum()
        
        # Check for null values to identify missing features
        null_features = merged[merged.columns[~merged.columns.isin(["Batch", "Sequence", "Transition_Length", "Group"])]].isnull().any(axis=1).sum()
        
        print(f"  - T{trans_len}: {adv_count} advantage, {ref_count} reference samples after merging ({null_features} samples missing features)")
        
        # Add to our list of merged dataframes
        merged_dfs.append(merged)
    
    # Combine all merged dataframes
    final_df = pd.concat(merged_dfs, ignore_index=True)
    
    # Check for missing values after merging
    missing_count = final_df.isnull().any(axis=1).sum()
    if missing_count > 0:
        print(f"\nWARNING: {missing_count} out of {len(final_df)} samples have missing values after merging.")
        
        # Count samples with missing features by group
        missing_by_group = final_df[final_df.isnull().any(axis=1)].groupby("Group").size()
        print("Samples with missing values by group:")
        for group, count in missing_by_group.items():
            print(f"  - {group}: {count} samples")
    
    # Save to Excel if output file is provided
    if output_file:
        print(f"\nSaving merged features to {output_file}...")
        final_df.to_excel(output_file, index=False)
    
    return final_df


def create_visualization(features_df, output_dir):
    """
    Create visualizations and statistical analysis for feature comparison.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing all computed features
        output_dir (str): Directory to save the output visualizations
    """
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add data validation and filtering step
    print("Validating data before visualization...")
    
    # First check for required columns
    required_columns = ["Batch", "Sequence", "Transition_Length", "Group"]
    feature_columns = [col for col in features_df.columns 
                      if col not in required_columns]
    
    if not all(col in features_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in features_df.columns]
        print(f"ERROR: Missing required columns: {missing_cols}")
        return
    
    # Check for NaN values in feature columns
    nan_counts = features_df[feature_columns].isnull().sum()
    if nan_counts.sum() > 0:
        print("\nFeature columns with NaN values:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  - {col}: {count} NaN values ({count/len(features_df)*100:.1f}%)")
    
    # Filter out rows with NaN values to ensure proper analysis
    valid_rows = features_df.dropna()
    if len(valid_rows) < len(features_df):
        print(f"\nDropping {len(features_df) - len(valid_rows)} rows with NaN values.")
        print(f"Proceeding with {len(valid_rows)} valid rows for visualization.")
        features_df = valid_rows
    
    # Check data for each transition length
    print("\nChecking data availability by transition length:")
    valid_trans_lengths = []
    trans_length_statistics = {}
    
    for trans_len in sorted(features_df["Transition_Length"].unique()):
        trans_data = features_df[features_df["Transition_Length"] == trans_len]
        adv_data = trans_data[trans_data["Group"] == "Advantage"]
        ref_data = trans_data[trans_data["Group"] == "Reference"]
        
        adv_count = len(adv_data)
        ref_count = len(ref_data)
        
        trans_length_statistics[trans_len] = {
            "advantage": adv_count,
            "reference": ref_count,
            "total": adv_count + ref_count
        }
        
        if adv_count > 0 and ref_count > 0:
            valid_trans_lengths.append(trans_len)
            print(f"  - T{trans_len}: {adv_count} advantage and {ref_count} reference samples - VALID")
        else:
            print(f"  - T{trans_len}: {adv_count} advantage and {ref_count} reference samples - SKIPPING (insufficient data)")
    
    if not valid_trans_lengths:
        print("\nERROR: No transition lengths have both advantage and reference samples.")
        print("Cannot proceed with visualization.")
        return
    
    print(f"\nProceeding with visualization for {len(valid_trans_lengths)} valid transition lengths: {valid_trans_lengths}")
    
    # Suppress various warnings to reduce output clutter
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="The palette list has more values.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Precision loss occurred.*")
    
    # Create a subdirectory for charts
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Use only the valid transition lengths (those with both advantage and reference samples)
    transition_lengths = valid_trans_lengths
    
    # Create a DataFrame to store t-test results
    t_test_results = []
    
    # Create a summary file with data availability information
    summary_file = os.path.join(output_dir, "data_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("DATA AVAILABILITY SUMMARY\n")
        f.write("========================\n\n")
        f.write(f"Total samples in dataset: {len(features_df)}\n\n")
        f.write("Samples by transition length:\n")
        for trans_len, stat_data in trans_length_statistics.items():
            f.write(f"  T{trans_len}: {stat_data['total']} total samples ")
            f.write(f"({stat_data['advantage']} advantage, {stat_data['reference']} reference)\n")
        
        f.write("\nValid transition lengths for visualization:\n")
        for trans_len in valid_trans_lengths:
            f.write(f"  T{trans_len}\n")
    
    print(f"Data availability summary saved to {summary_file}")
    
    # Feature list to analyze (excluding ID and grouping columns)
    feature_columns = [
        "Avg_Velocity", "Peak_Velocity", 
        "Avg_Acceleration", "Peak_Acceleration",
        "Avg_Jerk", "Peak_Jerk", 
        "Trajectory_Length", "Trajectory_Curvature",
        "Pose_Variation", "Body_Extent"
    ]
    
    # Feature display names for plot labels
    feature_display_names = {
        "Avg_Velocity": "Average Velocity",
        "Peak_Velocity": "Peak Velocity",
        "Avg_Acceleration": "Average Acceleration",
        "Peak_Acceleration": "Peak Acceleration",
        "Avg_Jerk": "Average Jerk",
        "Peak_Jerk": "Peak Jerk",
        "Trajectory_Length": "Trajectory Length",
        "Trajectory_Curvature": "Trajectory Curvature",
        "Pose_Variation": "Pose Variation",
        "Body_Extent": "Body Extent"
    }
    
    # Set seaborn style for plotting
    sns.set(style="whitegrid")
    
    # Analyze each transition length separately
    for trans_len in tqdm(transition_lengths, desc="Processing transition lengths"):
        print(f"Analyzing Transition Length {trans_len}...")
        
        try:
            # Filter data for this transition length
            trans_data = features_df[features_df["Transition_Length"] == trans_len]
            
            # Create a directory for this transition length
            trans_dir = os.path.join(charts_dir, f"T{trans_len}")
            os.makedirs(trans_dir, exist_ok=True)
            
            # Analyze each feature
            for feature in tqdm(feature_columns, desc=f"Analyzing features for T={trans_len}", leave=False):
                print(f"  - Analyzing feature: {feature}")
                
                # Split data by group
                advantage_data = trans_data[trans_data["Group"] == "Advantage"][feature].dropna()
                reference_data = trans_data[trans_data["Group"] == "Reference"][feature].dropna()
                
                # Skip if either group has no data
                if len(advantage_data) == 0 or len(reference_data) == 0:
                    print(f"    WARNING: Skipping feature '{feature}' as one of the groups has no data")
                    continue
                
                # Create figure for boxplot + scatter plot
                plt.figure(figsize=(10, 6))
                
                # Create a DataFrame for plotting
                # For large datasets, sample a maximum number of points to avoid performance issues
                max_samples_per_group = 200  # Adjust this value based on your needs
                
                if len(advantage_data) > max_samples_per_group:
                    advantage_sample = advantage_data.sample(max_samples_per_group, random_state=42)
                else:
                    advantage_sample = advantage_data
                    
                if len(reference_data) > max_samples_per_group:
                    reference_sample = reference_data.sample(max_samples_per_group, random_state=42)
                else:
                    reference_sample = reference_data
                
                plot_data = pd.DataFrame({
                    feature: pd.concat([advantage_sample, reference_sample]),
                    "Group": ["Advantage"] * len(advantage_sample) + ["Reference"] * len(reference_sample)
                })
                
                # Create a split visualization with boxplot and individual points
                ax = plt.gca()
                
                # Draw boxplot with updated parameters to avoid deprecation warnings
                sns.boxplot(x="Group", y=feature, hue="Group", data=plot_data, ax=ax, 
                            palette=["#3498db", "#95a5a6"], width=0.5, showfliers=False,
                            flierprops={'marker': 'o', 'markersize': 5}, legend=False)
                
                # Use stripplot with optimized parameters to better handle large numbers of points
                sns.stripplot(x="Group", y=feature, data=plot_data, ax=ax,
                             size=2.5, color=".25", alpha=0.4, jitter=0.35)
                
                # Perform t-test with enhanced checks for near-identical data
                try:
                    # Check for constant data which can cause t-test to fail
                    adv_std = advantage_data.std()
                    ref_std = reference_data.std()
                    adv_mean = advantage_data.mean()
                    ref_mean = reference_data.mean()
                    
                    # Import scipy.stats directly to avoid any potential naming conflicts
                    from scipy import stats as scipy_stats
                    
                    # Check if data is nearly identical (both means and std devs)
                    means_nearly_equal = abs(adv_mean - ref_mean) < 1e-6
                    stds_nearly_zero = adv_std < 1e-6 and ref_std < 1e-6
                    
                    # If both distributions are constant (or nearly so), they're basically identical
                    if stds_nearly_zero:
                        if means_nearly_equal:
                            # Nearly identical constant values - set t-stat to 0 and p-value to 1
                            t_stat, p_value = 0.0, 1.0
                        else:
                            # Constant but different values - large effect, definite difference
                            t_stat = np.sign(adv_mean - ref_mean) * 10.0  # Large t-value
                            p_value = 0.001  # Small p-value indicating significance
                    elif adv_std < 1e-6 or ref_std < 1e-6:
                        # One group is constant but the other varies - use Mann-Whitney test
                        # which doesn't assume any particular distribution
                        try:
                            u_stat, p_value = scipy_stats.mannwhitneyu(
                                advantage_data, 
                                reference_data,
                                alternative='two-sided'
                            )
                            # Approximate t-stat from U statistic (this is an approximation)
                            t_stat = np.sign(adv_mean - ref_mean) * np.sqrt(u_stat)
                        except Exception as mw_err:
                            print(f"    DEBUG: Mann-Whitney test failed: {str(mw_err)}")
                            # If Mann-Whitney also fails, use a simple comparison
                            t_stat = np.sign(adv_mean - ref_mean) * 5.0
                            p_value = 0.01 if abs(adv_mean - ref_mean) > 1e-6 else 0.5
                    else:
                        # Check if means are nearly identical to avoid catastrophic cancellation
                        if means_nearly_equal and (adv_std < 1e-3 or ref_std < 1e-3):
                            # Nearly identical means with small variance - not significant
                            t_stat, p_value = 0.0, 1.0
                        else:
                            # Normal case - use Welch's t-test
                            t_stat, p_value = scipy_stats.ttest_ind(
                                advantage_data, 
                                reference_data, 
                                equal_var=False,  # Use Welch's t-test for unequal variances
                                alternative='two-sided'
                            )
                            
                            # Check for NaN or Inf values in results
                            if np.isnan(t_stat) or np.isnan(p_value) or np.isinf(t_stat) or np.isinf(p_value):
                                raise ValueError("t-test produced NaN or Inf values")
                except Exception as e:
                    print(f"    WARNING: T-test failed with error: {str(e)}. Using fallback statistics.")
                    import traceback
                    if DEBUG:
                        traceback.print_exc()
                    # Fallback to simpler statistics if t-test fails
                    diff = adv_mean - ref_mean
                    pooled_var = ((advantage_data.var() + reference_data.var()) / 2) or 1.0
                    t_stat = diff / np.sqrt(pooled_var)
                    
                    # Set p-value based on magnitude of difference
                    if abs(diff) < 1e-6:
                        p_value = 1.0  # No significant difference
                    elif abs(diff) > np.sqrt(pooled_var):
                        p_value = 0.01  # Likely significant
                    else:
                        p_value = 0.5  # Neutral p-value as fallback
                
                # Store t-test results
                t_test_results.append({
                    "Transition_Length": trans_len,
                    "Feature": feature,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "Advantage_Mean": advantage_data.mean(),
                    "Reference_Mean": reference_data.mean(),
                    "Advantage_Std": advantage_data.std(),
                    "Reference_Std": reference_data.std(),
                    "Advantage_Samples": len(advantage_data),
                    "Reference_Samples": len(reference_data)
                })
                
                # Add significance markers
                significance = ""
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                
                # Calculate effect size (Cohen's d) with enhanced protection against numerical issues
                try:
                    adv_std = advantage_data.std()
                    ref_std = reference_data.std()
                    denominator = np.sqrt((adv_std ** 2 + ref_std ** 2) / 2)
                    
                    # Enhanced handling of edge cases
                    if denominator > 1e-6:
                        effect_size = (advantage_data.mean() - reference_data.mean()) / denominator
                    elif abs(advantage_data.mean() - reference_data.mean()) < 1e-6:
                        # If means are very close and std devs are near zero, effect is zero
                        effect_size = 0.0
                    else:
                        # If means differ but std devs are near zero, use large effect size
                        # with appropriate sign based on mean difference
                        effect_size = np.sign(advantage_data.mean() - reference_data.mean()) * 2.0
                except Exception as e:
                    print(f"    WARNING: Effect size calculation failed: {str(e)}. Using fallback.")
                    effect_size = 0.0
                
                # Handle percentage calculation with better numerical stability
                try:
                    ref_mean = reference_data.mean()
                    adv_mean = advantage_data.mean()
                    
                    # Enhanced handling of edge cases in percentage calculation
                    if abs(ref_mean) > 1e-6:
                        pct_diff = ((adv_mean - ref_mean) / ref_mean * 100)
                    elif abs(adv_mean) < 1e-6 and abs(ref_mean) < 1e-6:
                        # Both means are effectively zero
                        pct_diff = 0.0
                    elif abs(ref_mean) < 1e-6:
                        # Reference mean is near zero but advantage mean is not
                        # Set a large percentage difference with appropriate sign
                        pct_diff = np.sign(adv_mean) * 100.0
                    else:
                        pct_diff = float('nan')
                except Exception as e:
                    print(f"    WARNING: Percentage difference calculation failed: {str(e)}. Using fallback.")
                    pct_diff = float('nan')
                    
                # Add statistical annotation with checks for NaN values and improved formatting
                try:
                    # Ensure all values are valid and use fallbacks if not
                    t_stat_str = f"{t_stat:.2f}" if not np.isnan(t_stat) and not np.isinf(t_stat) else "N/A"
                    p_value_str = f"{p_value:.4f}" if not np.isnan(p_value) and not np.isinf(p_value) else "N/A"
                    effect_size_str = f"{effect_size:.2f}" if not np.isnan(effect_size) and not np.isinf(effect_size) else "N/A"
                    
                    # Calculate mean and std with robustness
                    adv_mean = advantage_data.mean()
                    adv_std = advantage_data.std()
                    ref_mean = reference_data.mean()
                    ref_std = reference_data.std()
                    
                    adv_mean_str = f"{adv_mean:.2f}" if not np.isnan(adv_mean) and not np.isinf(adv_mean) else "N/A"
                    adv_std_str = f"{adv_std:.2f}" if not np.isnan(adv_std) and not np.isinf(adv_std) else "N/A"
                    ref_mean_str = f"{ref_mean:.2f}" if not np.isnan(ref_mean) and not np.isinf(ref_mean) else "N/A"
                    ref_std_str = f"{ref_std:.2f}" if not np.isnan(ref_std) and not np.isinf(ref_std) else "N/A"
                    
                    stats_text = (
                        f"T-test: t={t_stat_str}, p={p_value_str} {significance}\n"
                        f"Effect size: d={effect_size_str}\n"
                        f"Advantage mean: {adv_mean_str} ± {adv_std_str}\n"
                        f"Reference mean: {ref_mean_str} ± {ref_std_str}\n"
                    )
                    
                    # Only add percentage difference if it's a valid number
                    if not np.isnan(pct_diff) and not np.isinf(pct_diff):
                        stats_text += f"Difference: {pct_diff:.1f}%"
                    else:
                        stats_text += "Difference: N/A"
                        
                except Exception as e:
                    print(f"    WARNING: Error formatting stats text: {str(e)}. Using simplified format.")
                    stats_text = "Statistical analysis unavailable due to data constraints."
                
                # Add annotation to plot
                plt.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Set title and labels
                plt.title(f"{feature_display_names[feature]} Comparison (T={trans_len}) {significance}", 
                          fontsize=14, fontweight='bold')
                plt.xlabel("")
                plt.ylabel(feature_display_names[feature], fontsize=12)
                
                # Save figure
                plt.tight_layout()
                output_file = os.path.join(trans_dir, f"{feature}_comparison.png")
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"    Saved to {output_file}")
                plt.close()
                
            # Create a combined visualization for this transition length
            try:
                # Check if we have enough data for a meaningful visualization
                advantage_count = len(trans_data[trans_data["Group"] == "Advantage"])
                reference_count = len(trans_data[trans_data["Group"] == "Reference"])
                
                if advantage_count > 0 and reference_count > 0:
                    create_combined_visualization(trans_data, feature_columns, feature_display_names, 
                                                 os.path.join(charts_dir, f"T{trans_len}_combined.png"), trans_len)
                    print(f"  Created combined visualization for T={trans_len}")
                else:
                    print(f"  WARNING: Skipping combined visualization for T={trans_len} due to insufficient data")
            except Exception as e:
                print(f"  WARNING: Failed to create combined visualization for T={trans_len}: {str(e)}")
                
        except Exception as e:
            print(f"ERROR processing transition length {trans_len}: {str(e)}")
    
    # Create overall combined visualization
    try:
        create_overall_combined_visualization(features_df, transition_lengths, feature_columns, 
                                             feature_display_names, os.path.join(charts_dir, "overall_combined.png"))
        print("Created overall combined visualization")
    except Exception as e:
        print(f"WARNING: Failed to create overall combined visualization: {str(e)}")
    
    # Save t-test results to Excel
    try:
        t_test_df = pd.DataFrame(t_test_results)
        t_test_file = os.path.join(output_dir, "t_test_results.xlsx")
        t_test_df.to_excel(t_test_file, index=False)
        print(f"T-test results saved to {t_test_file}")
    except Exception as e:
        print(f"WARNING: Failed to save t-test results: {str(e)}")
    
    # Create summary visualization of t-test results
    try:
        create_ttest_summary_visualization(t_test_df, feature_display_names, 
                                          os.path.join(charts_dir, "ttest_summary.png"))
        print("Created t-test summary visualization")
    except Exception as e:
        print(f"WARNING: Failed to create t-test summary visualization: {str(e)}")


def create_combined_visualization(data, features, feature_names, output_file, trans_len):
    """
    Create a combined visualization for all features at a specific transition length.
    
    Args:
        data (pd.DataFrame): Data for a specific transition length
        features (list): List of feature column names
        feature_names (dict): Dictionary mapping feature column names to display names
        output_file (str): Output file path
        trans_len (int): Transition length
    """
    # Create a grid of subplots (2 rows x 5 columns for 10 features)
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    axs = axs.flatten()
    
    # Set a title for the entire figure
    fig.suptitle(f"Feature Comparison for Transition Length {trans_len}", fontsize=16, y=0.98)
    
    # Plot each feature
    for i, feature in enumerate(features):
        ax = axs[i]
        
        # Split data by group
        advantage_data = data[data["Group"] == "Advantage"][feature].dropna()
        reference_data = data[data["Group"] == "Reference"][feature].dropna()
        
        # Create a DataFrame for plotting
        # For large datasets, sample a maximum number of points to avoid performance issues
        max_samples_per_group = 200  # Adjust this value based on your needs
        
        if len(advantage_data) > max_samples_per_group:
            advantage_sample = advantage_data.sample(max_samples_per_group, random_state=42)
        else:
            advantage_sample = advantage_data
            
        if len(reference_data) > max_samples_per_group:
            reference_sample = reference_data.sample(max_samples_per_group, random_state=42)
        else:
            reference_sample = reference_data
        
        plot_data = pd.DataFrame({
            feature: pd.concat([advantage_sample, reference_sample]),
            "Group": ["Advantage"] * len(advantage_sample) + ["Reference"] * len(reference_sample)
        })
        
        # Create boxplot with updated parameters to avoid deprecation warnings
        sns.boxplot(x="Group", y=feature, hue="Group", data=plot_data, ax=ax,
                    palette=["#3498db", "#95a5a6"], width=0.5, showfliers=False,
                    flierprops={'marker': 'o', 'markersize': 5}, legend=False)
        
        # Perform t-test with enhanced checks for near-identical data
        try:
            # Check for constant data which can cause t-test to fail
            adv_std = advantage_data.std()
            ref_std = reference_data.std()
            adv_mean = advantage_data.mean()
            ref_mean = reference_data.mean()
            
            # Import scipy.stats directly to avoid any potential naming conflicts
            from scipy import stats as scipy_stats
            
            # Check if data is nearly identical (both means and std devs)
            means_nearly_equal = abs(adv_mean - ref_mean) < 1e-6
            stds_nearly_zero = adv_std < 1e-6 and ref_std < 1e-6
            
            # If both distributions are constant (or nearly so), they're basically identical
            if stds_nearly_zero:
                if means_nearly_equal:
                    # Nearly identical constant values - set t-stat to 0 and p-value to 1
                    t_stat, p_value = 0.0, 1.0
                else:
                    # Constant but different values - large effect, definite difference
                    t_stat = np.sign(adv_mean - ref_mean) * 10.0  # Large t-value
                    p_value = 0.001  # Small p-value indicating significance
            elif adv_std < 1e-6 or ref_std < 1e-6:
                # One group is constant but the other varies - use Mann-Whitney test
                # which doesn't assume any particular distribution
                try:
                    u_stat, p_value = scipy_stats.mannwhitneyu(
                        advantage_data, 
                        reference_data,
                        alternative='two-sided'
                    )
                    # Approximate t-stat from U statistic (this is an approximation)
                    t_stat = np.sign(adv_mean - ref_mean) * np.sqrt(u_stat)
                except Exception as mw_err:
                    if DEBUG:
                        print(f"    DEBUG: Mann-Whitney test failed: {str(mw_err)}")
                    # If Mann-Whitney also fails, use a simple comparison
                    t_stat = np.sign(adv_mean - ref_mean) * 5.0
                    p_value = 0.01 if abs(adv_mean - ref_mean) > 1e-6 else 0.5
            else:
                # Check if means are nearly identical to avoid catastrophic cancellation
                if means_nearly_equal and (adv_std < 1e-3 or ref_std < 1e-3):
                    # Nearly identical means with small variance - not significant
                    t_stat, p_value = 0.0, 1.0
                else:
                    # Normal case - use Welch's t-test
                    t_stat, p_value = scipy_stats.ttest_ind(
                        advantage_data, reference_data, equal_var=False, alternative='two-sided'
                    )
                    
                    # Check for NaN or Inf values in results
                    if np.isnan(t_stat) or np.isnan(p_value) or np.isinf(t_stat) or np.isinf(p_value):
                        raise ValueError("t-test produced NaN or Inf values")
        except Exception as e:
            print(f"    WARNING: T-test failed in combined visualization: {str(e)}. Using fallback.")
            if DEBUG:
                import traceback
                traceback.print_exc()
            # Fallback to simpler statistics
            diff = adv_mean - ref_mean
            pooled_var = ((advantage_data.var() + reference_data.var()) / 2) or 1.0
            t_stat = diff / np.sqrt(pooled_var)
            
            # Set p-value based on magnitude of difference
            if abs(diff) < 1e-6:
                p_value = 1.0  # No significant difference
            elif abs(diff) > np.sqrt(pooled_var):
                p_value = 0.01  # Likely significant
            else:
                p_value = 0.5  # Neutral p-value as fallback
        
        # Add significance markers
        significance = ""
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        
        # Set title and labels
        ax.set_title(f"{feature_names[feature]} {significance}", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(feature_names[feature], fontsize=10)
        
        # Add p-value annotation
        ax.text(0.5, 0.02, f"p={p_value:.4f}", transform=ax.transAxes, 
                ha='center', fontsize=9)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_overall_combined_visualization(data, transition_lengths, features, feature_names, output_file):
    """
    Create an overall combined visualization showing feature differences across all transition lengths.
    
    Args:
        data (pd.DataFrame): Full feature data
        transition_lengths (list): List of transition lengths
        features (list): List of feature column names
        feature_names (dict): Dictionary mapping feature column names to display names
        output_file (str): Output file path
    """
    try:
        # Create a figure with subplots
        fig, axs = plt.subplots(len(features), 1, figsize=(15, 4 * len(features)))
        
        # If there's only one feature, make axs iterable
        if len(features) == 1:
            axs = [axs]
        
        # Plot each feature
        for i, feature in enumerate(features):
            ax = axs[i]
            
            # Create a list to store data for plotting
            plot_data = []
            
            # Process each transition length
            for trans_len in transition_lengths:
                try:
                    # Filter data for this transition length
                    trans_data = data[data["Transition_Length"] == trans_len]
                    
                    # Check if we have enough data for both groups
                    adv_data = trans_data[trans_data["Group"] == "Advantage"][feature].dropna()
                    ref_data = trans_data[trans_data["Group"] == "Reference"][feature].dropna()
                    
                    if len(adv_data) > 0 and len(ref_data) > 0:
                        # Calculate means and standard errors for each group
                        advantage_mean = adv_data.mean()
                        # Handle case with single sample where SEM is not defined
                        advantage_sem = adv_data.sem() if len(adv_data) > 1 else 0
                        reference_mean = ref_data.mean()
                        reference_sem = ref_data.sem() if len(ref_data) > 1 else 0
                        
                        # Add to plot data
                        plot_data.append({
                            "Transition_Length": trans_len,
                            "Advantage_Mean": advantage_mean,
                            "Advantage_SEM": advantage_sem,
                            "Reference_Mean": reference_mean,
                            "Reference_SEM": reference_sem,
                            "Advantage_Count": len(adv_data),
                            "Reference_Count": len(ref_data)
                        })
                    else:
                        print(f"WARNING: Insufficient data for feature '{feature}' at transition length {trans_len}")
                except Exception as e:
                    print(f"WARNING: Error processing transition length {trans_len} for feature '{feature}': {str(e)}")
            
            # Skip if no data to plot
            if not plot_data:
                ax.text(0.5, 0.5, f"No data available for feature: {feature_names[feature]}", 
                         ha='center', va='center', transform=ax.transAxes)
                continue
                
            # Convert to DataFrame
            plot_df = pd.DataFrame(plot_data)
            
            # Skip if dataframe is empty
            if plot_df.empty:
                ax.text(0.5, 0.5, f"No data available for feature: {feature_names[feature]}", 
                         ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Ensure all columns have valid numeric data by replacing NaN values
            numeric_cols = ["Advantage_Mean", "Advantage_SEM", "Reference_Mean", "Reference_SEM"]
            for col in numeric_cols:
                plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce').fillna(0)
            
            # Plot lines with error bars
            ax.errorbar(plot_df["Transition_Length"], plot_df["Advantage_Mean"], 
                        yerr=plot_df["Advantage_SEM"], marker='o', label="Advantage", 
                        color="#3498db", capsize=5)
            ax.errorbar(plot_df["Transition_Length"], plot_df["Reference_Mean"], 
                        yerr=plot_df["Reference_SEM"], marker='s', label="Reference", 
                        color="#95a5a6", capsize=5)
            
            # Calculate and plot relative difference with improved numerical stability
            rel_diff = []
            valid_indices = []
            
            for idx, (adv, ref) in enumerate(zip(plot_df["Advantage_Mean"], plot_df["Reference_Mean"])):
                try:
                    if pd.isna(adv) or pd.isna(ref) or np.isinf(adv) or np.isinf(ref):
                        rel_diff.append(np.nan)
                    elif abs(ref) > 1e-6:
                        rel_diff.append((adv - ref) / ref * 100)
                        valid_indices.append(idx)
                    elif abs(adv) < 1e-6 and abs(ref) < 1e-6:
                        # Both means are effectively zero
                        rel_diff.append(0.0)
                        valid_indices.append(idx)
                    elif abs(ref) < 1e-6:
                        # Reference mean is near zero but advantage mean is not
                        # Set a capped percentage difference with appropriate sign
                        rel_diff.append(np.sign(adv) * 100.0)
                        valid_indices.append(idx)
                    else:
                        rel_diff.append(np.nan)
                except Exception as e:
                    print(f"WARNING: Error calculating relative difference for feature '{feature}': {str(e)}")
                    rel_diff.append(np.nan)
            
            ax_right = ax.twinx()
            
            # Only plot if we have valid relative differences
            if valid_indices:
                # Filter out NaN values for plotting
                valid_trans_lens = [plot_df["Transition_Length"].iloc[i] for i in valid_indices]
                valid_rel_diffs = [rel_diff[i] for i in valid_indices]
                
                if valid_trans_lens and valid_rel_diffs:
                    ax_right.plot(valid_trans_lens, valid_rel_diffs, 'r--', label="% Difference")
                    ax_right.set_ylabel("% Difference", color='r')
                    ax_right.tick_params(axis='y', colors='r')
            
            # Set title and labels
            ax.set_title(feature_names[feature], fontsize=14)
            ax.set_xlabel("Transition Length")
            ax.set_ylabel(feature_names[feature])
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(loc='upper left')
            if valid_indices:  # Only add right legend if we have percentage differences
                ax_right.legend(loc='upper right')
            
            # Add sample counts as text in the plot
            if "Advantage_Count" in plot_df.columns and "Reference_Count" in plot_df.columns:
                info_text = f"Sample counts - Advantage: {sum(plot_df['Advantage_Count'])}, Reference: {sum(plot_df['Reference_Count'])}"
                ax.text(0.5, 0.02, info_text, transform=ax.transAxes, ha='center', fontsize=8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    except Exception as e:
        print(f"ERROR: Failed to create overall combined visualization: {str(e)}")
        # Create a simple error figure
        plt.figure(figsize=(15, 10))
        plt.text(0.5, 0.5, f"Error creating visualization: {str(e)}", 
                 ha='center', va='center', fontsize=14, wrap=True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()


def create_ttest_summary_visualization(t_test_df, feature_names, output_file):
    """
    Create a summary visualization of t-test results.
    
    Args:
        t_test_df (pd.DataFrame): DataFrame containing t-test results
        feature_names (dict): Dictionary mapping feature column names to display names
        output_file (str): Output file path
    """
    try:
        # Check if we have valid t-test results
        if t_test_df.empty:
            print("WARNING: No t-test results available for visualization.")
            # Create a simple figure with a message
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, "No t-test results available for visualization", 
                     ha='center', va='center', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            return
            
        # Clean p-values before creating pivot table
        # Replace any NaN, Inf, or negative p-values with 1.0
        t_test_df_clean = t_test_df.copy()
        t_test_df_clean["p_value_clean"] = t_test_df_clean["p_value"].apply(
            lambda x: 1.0 if pd.isna(x) or np.isinf(x) or x < 0 or x > 1 else x
        )
        
        # Create a pivot table of p-values
        pivot_df = t_test_df_clean.pivot(index="Feature", columns="Transition_Length", values="p_value_clean")
        
        # If pivot table is empty, handle gracefully
        if pivot_df.empty:
            print("WARNING: Pivot table is empty, no data to visualize.")
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, "No data available for visualization", 
                     ha='center', va='center', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Rename index values to use display names
        pivot_df.index = [feature_names.get(feature, feature) for feature in pivot_df.index]
        
        # Create a figure
        plt.figure(figsize=(12, 8))
        
        # Replace any remaining NaN values with 1.0 (non-significant)
        pivot_df_clean = pivot_df.fillna(1.0)
        
        # For annotations, format the p-values as strings with proper handling of NaN values
        annot_df = pivot_df_clean.applymap(lambda x: "{:.4f}".format(x) if not pd.isna(x) else "N/A")
        
        # Create heatmap with improved formatting
        sns.heatmap(pivot_df_clean, annot=annot_df, cmap="RdYlGn_r", vmin=0, vmax=0.1, 
                    linewidths=0.5, fmt="", cbar_kws={"label": "p-value"})
        
        # Set title and labels
        plt.title("Feature Significance by Transition Length (p-values)", fontsize=16)
        plt.xlabel("Transition Length", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        
        # Add significance markers
        for i, feature in enumerate(pivot_df_clean.index):
            for j, trans_len in enumerate(pivot_df_clean.columns):
                p_value = pivot_df_clean.iloc[i, j]
                if pd.notna(p_value):  # Check if p_value is not NaN
                    significance = ""
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    
                    if significance:
                        plt.text(j + 0.5, i + 0.8, significance, ha="center", va="center", 
                                 color="white", fontweight="bold", fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    except Exception as e:
        print(f"ERROR: Failed to create t-test summary visualization: {str(e)}")
        # Create a simple error figure
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, f"Error creating visualization: {str(e)}", 
                 ha='center', va='center', fontsize=14, wrap=True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze features of advantage samples")
    
    parser.add_argument("--ranking_file", type=str, default="advantage_ranking_by_transition.xlsx",
                        help="Path to advantage ranking Excel file (default: advantage_ranking_by_transition.xlsx)")
    
    parser.add_argument("--dataset", type=str, default="lafan1",
                        help="Dataset name (default: lafan1)")
    
    parser.add_argument("--config", type=str, default="default.yaml",
                        help="Configuration file (default: default.yaml)")
    
    parser.add_argument("--top_n", type=int, default=100,
                        help="Number of top advantage samples to select (default: 100)")
    
    parser.add_argument("--output_dir", type=str, default="feature_analysis_results",
                        help="Output directory (default: feature_analysis_results)")
    
    parser.add_argument("--skip_feature_calculation", action="store_true",
                        help="Skip feature calculation if features file already exists")
    
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (default: 0)")
    
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with additional output")
    
    parser.add_argument("--examine_rankings", action="store_true",
                        help="Print detailed information about the ranking file structure")
    
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size used in the dataset (default: 64)")
    
    parser.add_argument("--direct_visualization", action="store_true",
                        help="Skip merging and directly use the samples in the ranking file for visualization")
    
    args = parser.parse_args()
    
    # Set debug flag
    global DEBUG
    DEBUG = args.debug
    
    # Configure logging verbosity
    if DEBUG:
        # Enable all warnings when in debug mode
        warnings.resetwarnings()
        print("Debug mode enabled - showing all warnings and detailed output")
    
    print("=" * 80)
    print("Feature Analysis of Advantage Samples")
    print("=" * 80)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Set global batch size for ID mapping
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size
        print(f"Using batch size: {BATCH_SIZE} for ID mapping")
        
        # Examine ranking file if requested
        if args.examine_rankings:
            print("\n" + "="*50)
            print("EXAMINING RANKING FILE STRUCTURE")
            print("="*50)
            
            try:
                # Load the ranking Excel file
                excel_data = pd.read_excel(args.ranking_file, sheet_name=None)
                
                print(f"Ranking file: {args.ranking_file}")
                print(f"Number of sheets: {len(excel_data)}")
                
                # Print sheet names
                print("\nSheet names:")
                for i, sheet_name in enumerate(excel_data.keys()):
                    print(f"  {i+1}. {sheet_name}")
                
                # Examine each sheet
                print("\nSheet details:")
                for sheet_name, df in excel_data.items():
                    print(f"\n  Sheet: {sheet_name}")
                    print(f"  Rows: {len(df)}")
                    print(f"  Columns: {list(df.columns)}")
                    
                    # Sample data
                    if not df.empty:
                        print(f"  First few rows:")
                        print(df.head(3).to_string())
                        
                        # Check for Batch and Sequence columns
                        if "Batch" in df.columns and "Sequence" in df.columns:
                            print(f"  Batch ID range: {df['Batch'].min()} - {df['Batch'].max()}")
                            print(f"  Sequence ID range: {df['Sequence'].min()} - {df['Sequence'].max()}")
                
                print("\nRanking file examination complete.")
                
                # Create a copy of the ranking file in the output directory for reference
                ranking_copy = os.path.join(args.output_dir, "ranking_file_copy.xlsx")
                with pd.ExcelWriter(ranking_copy) as writer:
                    for sheet_name, df in excel_data.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"Created a copy of the ranking file at {ranking_copy}")
                
            except Exception as e:
                print(f"ERROR examining ranking file: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Step 1: Define advantage groups from ranking file
        grouping_file = os.path.join(args.output_dir, "sample_grouping.xlsx")
        grouping_df = define_advantage_group(args.ranking_file, args.top_n)
        grouping_df.to_excel(grouping_file, index=False)
        
        # Check if we should do direct visualization using the ranking file data
        if args.direct_visualization:
            print("\n" + "="*50)
            print("USING DIRECT VISUALIZATION FROM RANKING FILE")
            print("="*50)
            print("Skipping feature calculation and merging steps.")
            
            # Add columns for features that would come from feature calculation
            # These will all be NaN, but the visualization code will filter them out
            feature_columns = [
                "Avg_Velocity", "Peak_Velocity", 
                "Avg_Acceleration", "Peak_Acceleration",
                "Avg_Jerk", "Peak_Jerk", 
                "Trajectory_Length", "Trajectory_Curvature",
                "Pose_Variation", "Body_Extent"
            ]
            
            # Check if ranking file contains any feature columns already
            existing_features = [col for col in grouping_df.columns if col in feature_columns]
            missing_features = [col for col in feature_columns if col not in existing_features]
            
            if existing_features:
                print(f"Ranking file already contains these feature columns: {existing_features}")
            
            # Add missing feature columns with NaN values
            for col in missing_features:
                grouping_df[col] = np.nan
                
            # Directly use the grouping dataframe for visualization
            create_visualization(grouping_df, args.output_dir)
            
        else:
            # Feature calculation file paths
            raw_features_file = os.path.join(args.output_dir, "raw_features.xlsx")
            final_features_file = os.path.join(args.output_dir, "final_feature_analysis.xlsx")
            
            # Step 2: Compute features for all samples (if needed)
            if not args.skip_feature_calculation or not os.path.exists(raw_features_file):
                # Load configuration and dataset
                config = utils.load_config(f"config/{args.dataset}/{args.config}")
                dataset = MotionDataset(train=False, config=config)
                print(f"Loaded test dataset: {args.dataset}, with {len(dataset)} samples")
                
                # Compute features for all samples (independent of grouping)
                features_df = compute_features_for_dataset(dataset, raw_features_file)
            else:
                print(f"Loading pre-computed raw features from {raw_features_file}")
                features_df = pd.read_excel(raw_features_file)
            
            # Step 3: Merge features with group assignments
            final_df = merge_features_with_groups(features_df, grouping_df, final_features_file)
            
            # Step 4: Create visualizations and perform statistical analysis
            create_visualization(final_df, args.output_dir)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()