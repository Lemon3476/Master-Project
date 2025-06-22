#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Advantage Analysis

This script implements an in-depth comparative analysis based on specific features (e.g., Body Extent).
The analysis includes:
1. Loading performance and feature data from final_feature_analysis.xlsx
2. Performing T-tests to analyze feature influence across different transition lengths (15,30,60,90)
3. Generating T-test result Excel files
4. Creating binned bar charts showing performance improvements across feature value ranges
5. Supporting command-line specification of baseline and proposed models

Usage:
    python eval/analyze_feature_advantage.py --feature body_extent --baseline_model Ours-0 --proposed_model Ours-1

Output:
    - T-test results in Excel format
    - Binned bar charts for each transition length showing performance improvement by feature level
    - Statistical summary in console output
"""

import os
import sys
sys.path.append(".")  # Add project root to path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt errors
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import time

# Define feature display names mapping
FEATURE_DISPLAY_NAMES = {
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

# Define metrics for performance analysis
METRICS = ["L2P", "L2Q", "NPSS", "Foot_Skate"]

def load_feature_data(feature_file):
    """
    Load feature data from the Excel file.
    
    Args:
        feature_file (str): Path to the feature analysis Excel file
        
    Returns:
        pd.DataFrame: DataFrame containing all feature data
    """
    print(f"Loading feature data from: {feature_file}")
    
    try:
        # Load data
        feature_df = pd.read_excel(feature_file)
        
        # Check required columns
        required_columns = ["Batch", "Sequence", "Transition_Length", "Group"]
        missing_columns = [col for col in required_columns if col not in feature_df.columns]
        
        if missing_columns:
            raise ValueError(f"Required columns missing in feature data: {missing_columns}")
        
        # Print dataset information
        print(f"Loaded {len(feature_df)} samples with {len(feature_df.columns)} features/attributes")
        print(f"Transition lengths: {sorted(feature_df['Transition_Length'].unique())}")
        if "Group" in feature_df.columns:
            group_counts = feature_df["Group"].value_counts()
            print(f"Sample groups: {dict(group_counts)}")
        
        return feature_df
    
    except Exception as e:
        print(f"Error loading feature data: {str(e)}")
        raise

def normalize_feature_name(feature_name):
    """
    Normalize feature name to match column names in the dataset.
    
    Args:
        feature_name (str): User-provided feature name
        
    Returns:
        str: Normalized feature name
    """
    # First, convert to lowercase and replace spaces with underscores for the lookup
    lookup_name = feature_name.lower().replace(" ", "_")
    
    # Map common variations to standard names
    feature_mapping = {
        "velocity": "avg_velocity",
        "acceleration": "avg_acceleration",
        "jerk": "avg_jerk",
        "curvature": "trajectory_curvature",
        "length": "trajectory_length",
        "pose": "pose_variation",
        "extent": "body_extent"
    }
    
    # Check if the normalized name is in the mapping
    for key, value in feature_mapping.items():
        if key in lookup_name and lookup_name != value:
            print(f"Note: Mapping feature name '{feature_name}' to '{value}'")
            # Return capitalized version that matches dataset (PascalCase format)
            return value.replace("_", " ").title().replace(" ", "_")
    
    # Handle known capitalization formats used in the dataset
    if lookup_name == "body_extent":
        return "Body_Extent"
    elif lookup_name == "pose_variation":
        return "Pose_Variation"
    elif lookup_name == "trajectory_curvature":
        return "Trajectory_Curvature"
    elif lookup_name == "trajectory_length":
        return "Trajectory_Length"
    elif lookup_name == "avg_velocity":
        return "Avg_Velocity"
    elif lookup_name == "peak_velocity":
        return "Peak_Velocity"
    elif lookup_name == "avg_acceleration":
        return "Avg_Acceleration"
    elif lookup_name == "peak_acceleration":
        return "Peak_Acceleration"
    elif lookup_name == "avg_jerk":
        return "Avg_Jerk"
    elif lookup_name == "peak_jerk":
        return "Peak_Jerk"
    
    # If we don't have a specific mapping, convert to PascalCase which matches the dataset format
    return "".join(word.capitalize() for word in lookup_name.split("_"))

def perform_ttest_analysis(feature_df, feature_name, baseline_model=None, proposed_model=None):
    """
    Perform T-test analysis for the specified feature across transition lengths.
    
    Args:
        feature_df (pd.DataFrame): Feature data DataFrame
        feature_name (str): Name of the feature to analyze
        baseline_model (str, optional): Baseline model name
        proposed_model (str, optional): Proposed model name
        
    Returns:
        pd.DataFrame: DataFrame with T-test results
    """
    print(f"Performing T-test analysis for feature: {feature_name}")
    
    # Normalize feature name
    normalized_feature = normalize_feature_name(feature_name)
    
    # Check if feature exists in the dataset
    if normalized_feature not in feature_df.columns and f"{normalized_feature.capitalize()}" not in feature_df.columns:
        available_features = [col for col in feature_df.columns 
                            if col.lower() not in ["batch", "sequence", "transition_length", "group"]]
        raise ValueError(f"Feature '{normalized_feature}' not found in the dataset. Available features: {available_features}")
    
    # Find exact feature column name (handling case sensitivity)
    feature_col = normalized_feature
    if feature_col not in feature_df.columns:
        for col in feature_df.columns:
            if col.lower() == normalized_feature.lower():
                feature_col = col
                break
    
    # Create list to store T-test results
    ttest_results = []
    
    # Get all transition lengths
    transition_lengths = sorted(feature_df["Transition_Length"].unique())
    
    # Analyze each transition length
    for trans_len in transition_lengths:
        print(f"  Analyzing transition length: {trans_len}")
        
        # Filter data for this transition length
        trans_data = feature_df[feature_df["Transition_Length"] == trans_len]
        
        # Check if we have improvement columns or need to calculate them
        if baseline_model is not None and proposed_model is not None:
            # Check for both models' columns - try both formats: "model_metric" and "metric_model"
            for metric in METRICS:
                # Try standard format "metric_model"
                baseline_col = f"{metric}_{baseline_model}"
                proposed_col = f"{metric}_{proposed_model}"
                
                # Check alternate format "model_metric" if needed
                if baseline_col not in trans_data.columns or proposed_col not in trans_data.columns:
                    alt_baseline_col = f"{baseline_model}_{metric}"
                    alt_proposed_col = f"{proposed_model}_{metric}"
                    
                    if alt_baseline_col in trans_data.columns and alt_proposed_col in trans_data.columns:
                        baseline_col = alt_baseline_col
                        proposed_col = alt_proposed_col
                
                # Look for existing improvement column with different formats
                improvement_col = f"{metric}_Improvement"
                if improvement_col not in trans_data.columns:
                    # Try alternative formats
                    alt_formats = [f"{metric}_Impr", f"{metric}_Impr_Ratio"]
                    
                    for alt_format in alt_formats:
                        if alt_format in trans_data.columns:
                            improvement_col = alt_format
                            print(f"    Using existing improvement column: {improvement_col}")
                            break
                
                # If we have both model columns, we can calculate improvement if needed
                if baseline_col in trans_data.columns and proposed_col in trans_data.columns:
                    if improvement_col not in trans_data.columns:
                        # Calculate improvement (baseline - proposed, since lower is better)
                        trans_data[improvement_col] = trans_data[baseline_col] - trans_data[proposed_col]
                        print(f"    Calculated improvement from {baseline_col} and {proposed_col}")
                    
                    # Now that we have the improvement column, analyze feature effect
                    # We'll compute correlation and T-test between feature and improvement
                    
                    # First, check for valid data points
                    valid_mask = ~np.isnan(trans_data[feature_col]) & ~np.isnan(trans_data[improvement_col])
                    valid_data = trans_data[valid_mask]
                    
                    if len(valid_data) < 5:
                        print(f"    Warning: Too few valid samples ({len(valid_data)}) for {metric} analysis")
                        continue
                    
                    # Calculate correlation
                    correlation = valid_data[[feature_col, improvement_col]].corr().iloc[0, 1]
                    
                    # Create feature groups based on feature value (low, medium, high)
                    bins = [
                        valid_data[feature_col].min(),
                        valid_data[feature_col].quantile(0.33),
                        valid_data[feature_col].quantile(0.66),
                        valid_data[feature_col].max()
                    ]
                    
                    # Add some margin to bins to avoid edge cases
                    if bins[0] == bins[1]:
                        bins[1] += 1e-6
                    if bins[1] == bins[2]:
                        bins[2] += 1e-6
                    if bins[2] == bins[3]:
                        bins[3] += 1e-6
                    
                    labels = ["Low", "Medium", "High"]
                    valid_data["Feature_Group"] = pd.cut(
                        valid_data[feature_col], 
                        bins=bins, 
                        labels=labels, 
                        include_lowest=True
                    )
                    
                    # Calculate statistics for each group
                    group_stats = []
                    for group in labels:
                        group_data = valid_data[valid_data["Feature_Group"] == group]
                        if len(group_data) > 0:
                            group_stats.append({
                                "Group": group,
                                "Count": len(group_data),
                                "Mean_Improvement": group_data[improvement_col].mean(),
                                "Std_Improvement": group_data[improvement_col].std(),
                                "Mean_Feature_Value": group_data[feature_col].mean()
                            })
                    
                    # Perform T-test between Low and High feature groups
                    low_group = valid_data[valid_data["Feature_Group"] == "Low"][improvement_col]
                    high_group = valid_data[valid_data["Feature_Group"] == "High"][improvement_col]
                    
                    if len(low_group) >= 2 and len(high_group) >= 2:
                        t_stat, p_value = stats.ttest_ind(low_group, high_group, equal_var=False)
                        
                        # Add to results
                        ttest_results.append({
                            "Transition_Length": trans_len,
                            "Metric": metric,
                            "Feature": feature_col,
                            "Correlation": correlation,
                            "T_Statistic": t_stat,
                            "P_Value": p_value,
                            "Low_Group_Count": len(low_group),
                            "Low_Group_Mean": low_group.mean(),
                            "Low_Group_Std": low_group.std(),
                            "High_Group_Count": len(high_group),
                            "High_Group_Mean": high_group.mean(),
                            "High_Group_Std": high_group.std(),
                            "Group_Stats": group_stats
                        })
                    else:
                        print(f"    Warning: Not enough samples in Low/High groups for T-test on {metric}")
                else:
                    if baseline_col not in trans_data.columns:
                        print(f"    Warning: Baseline model column '{baseline_col}' not found")
                    if proposed_col not in trans_data.columns:
                        print(f"    Warning: Proposed model column '{proposed_col}' not found")
        
        else:
            # If no models specified, just analyze the dataset structure
            print(f"    Note: No baseline/proposed models specified, skipping T-test")
            # Just add basic feature statistics
            ttest_results.append({
                "Transition_Length": trans_len,
                "Feature": feature_col,
                "Mean_Value": trans_data[feature_col].mean(),
                "Std_Value": trans_data[feature_col].std(),
                "Min_Value": trans_data[feature_col].min(),
                "Max_Value": trans_data[feature_col].max(),
                "Sample_Count": len(trans_data)
            })
    
    # Convert to DataFrame
    return pd.DataFrame(ttest_results)

def create_feature_binned_barchart(feature_df, feature_name, transition, output_dir, 
                                 metric="L2P", num_bins=3, baseline_model=None, proposed_model=None):
    """
    Create binned bar charts showing average performance improvement by feature value groups.
    
    Args:
        feature_df (pd.DataFrame): Feature data DataFrame
        feature_name (str): Name of the feature to analyze
        transition (int): Transition length
        output_dir (str): Output directory
        metric (str): Metric to analyze
        num_bins (int): Number of bins
        baseline_model (str, optional): Baseline model name
        proposed_model (str, optional): Proposed model name
    """
    # Filter data for this transition length
    trans_data = feature_df[feature_df["Transition_Length"] == transition].copy()
    
    # Normalize feature name
    normalized_feature = normalize_feature_name(feature_name)
    
    # Find exact feature column name (handling case sensitivity)
    feature_col = normalized_feature
    for col in trans_data.columns:
        if col.lower() == normalized_feature.lower():
            feature_col = col
            break
    
    # Check if feature exists
    if feature_col not in trans_data.columns:
        print(f"Error: Feature '{feature_col}' not found in dataset for T{transition}")
        return
    
    # Metric improvement column
    improvement_col = f"{metric}_Improvement"
    
    # Check if improvement column exists or calculate it
    if improvement_col not in trans_data.columns:
        # Try alternative formats
        alt_formats = [f"{metric}_Impr", f"{metric}_Impr_Ratio"]
        found_alt = False
        
        for alt_format in alt_formats:
            if alt_format in trans_data.columns:
                improvement_col = alt_format
                print(f"  Using existing improvement column: {improvement_col}")
                found_alt = True
                break
        
        if not found_alt:
            if baseline_model is None or proposed_model is None:
                print(f"Error: Cannot create chart - no improvement data and no models specified")
                return
            
            # Try both column formats
            baseline_col = f"{metric}_{baseline_model}"
            proposed_col = f"{metric}_{proposed_model}"
            
            # Check alternate format if needed
            if baseline_col not in trans_data.columns or proposed_col not in trans_data.columns:
                alt_baseline_col = f"{baseline_model}_{metric}"
                alt_proposed_col = f"{proposed_model}_{metric}"
                
                if alt_baseline_col in trans_data.columns and alt_proposed_col in trans_data.columns:
                    baseline_col = alt_baseline_col
                    proposed_col = alt_proposed_col
            
            if baseline_col not in trans_data.columns or proposed_col not in trans_data.columns:
                print(f"Error: Missing model columns for {metric} at T{transition}")
                return
            
            # Calculate improvement (baseline - proposed, since lower is better)
            trans_data[improvement_col] = trans_data[baseline_col] - trans_data[proposed_col]
            print(f"  Calculated improvement from {baseline_col} and {proposed_col}")
    
    # Remove rows with NaN values in feature or improvement
    valid_data = trans_data.dropna(subset=[feature_col, improvement_col])
    
    if len(valid_data) < 5:
        print(f"Warning: Too few valid samples ({len(valid_data)}) for feature binning at T{transition}")
        return
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Determine bins based on feature distribution
    feature_min = valid_data[feature_col].min()
    feature_max = valid_data[feature_col].max()
    
    if num_bins == 3:
        # Tertiles
        bins = [
            feature_min,
            valid_data[feature_col].quantile(0.33),
            valid_data[feature_col].quantile(0.66),
            feature_max
        ]
        labels = ["Low", "Medium", "High"]
    elif num_bins == 5:
        # Quintiles
        bins = [
            feature_min,
            valid_data[feature_col].quantile(0.2),
            valid_data[feature_col].quantile(0.4),
            valid_data[feature_col].quantile(0.6),
            valid_data[feature_col].quantile(0.8),
            feature_max
        ]
        labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    else:
        # Default to tertiles
        bins = [
            feature_min,
            valid_data[feature_col].quantile(0.33),
            valid_data[feature_col].quantile(0.66),
            feature_max
        ]
        labels = ["Low", "Medium", "High"]
    
    # Ensure bins are unique (add small epsilon if needed)
    for i in range(1, len(bins)):
        if bins[i] <= bins[i-1]:
            bins[i] = bins[i-1] + 1e-6
    
    # Add feature group column
    valid_data["Feature_Group"] = pd.cut(
        valid_data[feature_col], 
        bins=bins, 
        labels=labels, 
        include_lowest=True
    )
    
    # Calculate average improvement for each group
    grouped_data = valid_data.groupby("Feature_Group")[improvement_col].agg(
        ['mean', 'std', 'count']).reset_index()
    grouped_data.columns = ["Feature_Group", "Mean_Improvement", "Std_Improvement", "Count"]
    
    # Create bar chart
    ax = sns.barplot(x="Feature_Group", y="Mean_Improvement", data=grouped_data)
    
    # Add sample count and mean labels
    def add_labels(x, y, counts, stds):
        for i, (xi, yi, ci, si) in enumerate(zip(x, y, counts, stds)):
            # Label position depends on bar height
            if yi >= 0:
                va = 'bottom'
                offset = 0.01
            else:
                va = 'top'
                offset = -0.1
            
            # Add sample count
            plt.text(xi, yi + offset, 
                    f"n={ci}", 
                    ha='center', va=va, 
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add mean±std
            plt.text(xi, yi - offset*2 if yi >= 0 else yi + offset*2, 
                    f"μ={yi:.3f}±{si:.3f}", 
                    ha='center', va='top' if yi >= 0 else 'bottom', 
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add the labels
    add_labels(
        ax.get_xticks(), 
        grouped_data["Mean_Improvement"],
        grouped_data["Count"],
        grouped_data["Std_Improvement"]
    )
    
    # Add horizontal reference line at y=0
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Set title and labels
    display_feature_name = FEATURE_DISPLAY_NAMES.get(normalized_feature, feature_col)
    model_labels = ""
    if baseline_model is not None and proposed_model is not None:
        model_labels = f" ({baseline_model} vs {proposed_model})"
    
    plt.title(f"Transition Length T={transition}: Average {metric} Improvement by {display_feature_name} Group{model_labels}")
    plt.xlabel(f"{display_feature_name} Group")
    plt.ylabel(f"Average {metric} Improvement")
    plt.grid(True, alpha=0.3)
    
    # Save chart
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"T{transition}_{metric}_{normalized_feature}_binned_improvement.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  Created binned bar chart: {output_file}")
    
    return grouped_data

def create_combined_chart(feature_df, feature_name, output_dir, metric="L2P", 
                        baseline_model=None, proposed_model=None):
    """
    Create a combined chart showing the feature's impact across all transition lengths.
    
    Args:
        feature_df (pd.DataFrame): Feature data DataFrame
        feature_name (str): Name of the feature to analyze
        output_dir (str): Output directory
        metric (str): Metric to analyze
        baseline_model (str, optional): Baseline model name
        proposed_model (str, optional): Proposed model name
    """
    # Normalize feature name
    normalized_feature = normalize_feature_name(feature_name)
    
    # Find exact feature column name (handling case sensitivity)
    feature_col = normalized_feature
    for col in feature_df.columns:
        if col.lower() == normalized_feature.lower():
            feature_col = col
            break
    
    # Check if feature exists
    if feature_col not in feature_df.columns:
        print(f"Error: Feature '{feature_col}' not found in dataset")
        return
    
    # Create figure with 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()
    
    # Map transition lengths to subplot positions
    transition_lengths = sorted(feature_df["Transition_Length"].unique())
    positions = {}
    for i, trans in enumerate(transition_lengths[:4]):  # Limit to first 4 transition lengths
        positions[trans] = i
    
    # Process each transition length
    for trans, idx in positions.items():
        # Filter data for this transition length
        trans_data = feature_df[feature_df["Transition_Length"] == trans].copy()
        
        # Improvement column
        improvement_col = f"{metric}_Improvement"
        
        # Check if improvement column exists or calculate it
        if improvement_col not in trans_data.columns:
            # Try alternative formats
            alt_formats = [f"{metric}_Impr", f"{metric}_Impr_Ratio"]
            found_alt = False
            
            for alt_format in alt_formats:
                if alt_format in trans_data.columns:
                    improvement_col = alt_format
                    print(f"  Using existing improvement column: {improvement_col}")
                    found_alt = True
                    break
            
            if not found_alt:
                if baseline_model is None or proposed_model is None:
                    print(f"Error: Cannot create combined chart - no improvement data and no models specified")
                    continue
                
                # Try both column formats
                baseline_col = f"{metric}_{baseline_model}"
                proposed_col = f"{metric}_{proposed_model}"
                
                # Check alternate format if needed
                if baseline_col not in trans_data.columns or proposed_col not in trans_data.columns:
                    alt_baseline_col = f"{baseline_model}_{metric}"
                    alt_proposed_col = f"{proposed_model}_{metric}"
                    
                    if alt_baseline_col in trans_data.columns and alt_proposed_col in trans_data.columns:
                        baseline_col = alt_baseline_col
                        proposed_col = alt_proposed_col
                
                if baseline_col not in trans_data.columns or proposed_col not in trans_data.columns:
                    print(f"Error: Missing model columns for {metric} at T{trans}")
                    continue
                
                # Calculate improvement (baseline - proposed, since lower is better)
                trans_data[improvement_col] = trans_data[baseline_col] - trans_data[proposed_col]
                print(f"  Calculated improvement from {baseline_col} and {proposed_col}")
        
        # Get current subplot
        ax = axs[idx]
        
        # Remove rows with NaN values
        valid_data = trans_data.dropna(subset=[feature_col, improvement_col])
        
        if len(valid_data) < 5:
            ax.text(0.5, 0.5, f"Insufficient data for T{trans}", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create scatter plot
        sns.scatterplot(x=feature_col, y=improvement_col, data=valid_data, ax=ax, alpha=0.6)
        
        # Fit trend line
        try:
            # Calculate regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_data[feature_col], valid_data[improvement_col]
            )
            
            # Add trend line
            x_vals = np.array([valid_data[feature_col].min(), valid_data[feature_col].max()])
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals, y_vals, 'r-')
            
            # Add trend line equation and statistics
            ax.text(0.05, 0.95, 
                   f"y = {slope:.4f}x + {intercept:.4f}\n"
                   f"R² = {r_value**2:.4f}, p = {p_value:.4f}",
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        except Exception as e:
            print(f"Warning: Failed to fit trend line for T{trans}: {str(e)}")
        
        # Add horizontal reference line at y=0
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Set subplot title and labels
        display_feature_name = FEATURE_DISPLAY_NAMES.get(normalized_feature, feature_col)
        ax.set_title(f"Transition Length T={trans}")
        ax.set_xlabel(display_feature_name)
        ax.set_ylabel(f"{metric} Improvement")
        ax.grid(True, alpha=0.3)
    
    # Set overall title
    model_labels = ""
    if baseline_model is not None and proposed_model is not None:
        model_labels = f" ({baseline_model} vs {proposed_model})"
    
    display_feature_name = FEATURE_DISPLAY_NAMES.get(normalized_feature, feature_col)
    fig.suptitle(f"Impact of {display_feature_name} on {metric} Performance Improvement{model_labels}", 
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save chart
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{normalized_feature}_{metric}_combined.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Created combined chart: {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Feature Advantage Analysis")
    parser.add_argument("--feature", type=str, default="body_extent",
                        help="Feature to analyze (default: body_extent)")
    parser.add_argument("--feature_file", type=str, default="feature_analysis_results/final_feature_analysis.xlsx",
                        help="Path to feature analysis Excel file")
    parser.add_argument("--output_dir", type=str, default="feature_analysis_results/feature_advantage_analysis",
                        help="Output directory for analysis results")
    parser.add_argument("--baseline_model", type=str, default=None,
                        help="Baseline model label name")
    parser.add_argument("--proposed_model", type=str, default=None,
                        help="Proposed model label name")
    parser.add_argument("--num_bins", type=int, default=3,
                        help="Number of bins for binned bar charts (3 or 5)")
    
    args = parser.parse_args()
    
    # Print execution information
    print("=" * 80)
    print("Feature Advantage Analysis")
    print("=" * 80)
    print(f"Feature to analyze: {args.feature}")
    print(f"Feature data file: {args.feature_file}")
    print(f"Output directory: {args.output_dir}")
    if args.baseline_model and args.proposed_model:
        print(f"Comparing models: {args.baseline_model} (baseline) vs {args.proposed_model} (proposed)")
    else:
        print("No model comparison specified (use --baseline_model and --proposed_model)")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Step 1: Load feature data
        feature_df = load_feature_data(args.feature_file)
        
        # Step 2: Perform T-test analysis
        ttest_results = perform_ttest_analysis(
            feature_df, 
            args.feature, 
            baseline_model=args.baseline_model, 
            proposed_model=args.proposed_model
        )
        
        # Save T-test results
        normalized_feature = normalize_feature_name(args.feature)
        ttest_file = os.path.join(args.output_dir, f"{normalized_feature}_ttest_results.xlsx")
        ttest_results.to_excel(ttest_file, index=False)
        print(f"T-test results saved to: {ttest_file}")
        
        # Step 3: Create charts directory
        charts_dir = os.path.join(args.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Step 4: Create binned bar charts for each transition length and metric
        transition_lengths = sorted(feature_df["Transition_Length"].unique())
        
        for metric in METRICS:
            print(f"\nAnalyzing impact of {args.feature} on {metric}:")
            
            # Create per-transition charts
            all_bins_data = []
            for trans in transition_lengths:
                print(f"  Transition Length T{trans}:")
                bins_data = create_feature_binned_barchart(
                    feature_df, 
                    args.feature, 
                    trans, 
                    charts_dir, 
                    metric=metric, 
                    num_bins=args.num_bins,
                    baseline_model=args.baseline_model,
                    proposed_model=args.proposed_model
                )
                
                if bins_data is not None:
                    # Add transition length
                    bins_data["Transition_Length"] = trans
                    all_bins_data.append(bins_data)
            
            # Create combined chart showing feature impact across transition lengths
            create_combined_chart(
                feature_df, 
                args.feature, 
                charts_dir, 
                metric=metric,
                baseline_model=args.baseline_model,
                proposed_model=args.proposed_model
            )
            
            # Save binned data summary
            if all_bins_data:
                binned_df = pd.concat(all_bins_data, ignore_index=True)
                binned_file = os.path.join(args.output_dir, f"{normalized_feature}_{metric}_binned_data.xlsx")
                binned_df.to_excel(binned_file, index=False)
                print(f"Binned analysis data saved to: {binned_file}")
        
        # Print analysis summary
        print("\n" + "=" * 80)
        print(f"Analysis Summary for {args.feature}")
        print("=" * 80)
        
        # Extract key insights from T-test results
        if not ttest_results.empty and "P_Value" in ttest_results.columns:
            print("\nSignificant effects (p < 0.05):")
            significant = ttest_results[ttest_results["P_Value"] < 0.05]
            
            if len(significant) > 0:
                for _, row in significant.iterrows():
                    effect_direction = "positive" if row["T_Statistic"] > 0 else "negative"
                    significance = ""
                    if row["P_Value"] < 0.001:
                        significance = "***"
                    elif row["P_Value"] < 0.01:
                        significance = "**"
                    elif row["P_Value"] < 0.05:
                        significance = "*"
                    
                    print(f"  T{row['Transition_Length']} {row['Metric']}: {effect_direction} effect (p={row['P_Value']:.4f}{significance})")
            else:
                print("  No significant effects found")
            
            # Print correlations
            if "Correlation" in ttest_results.columns:
                print("\nCorrelations between feature and performance improvement:")
                for metric in METRICS:
                    metric_data = ttest_results[ttest_results["Metric"] == metric]
                    if not metric_data.empty:
                        print(f"  {metric}:")
                        for _, row in metric_data.iterrows():
                            print(f"    T{row['Transition_Length']}: r={row['Correlation']:.4f}")
        
        print("\nAnalysis complete! Results saved to:")
        print(f"  {args.output_dir}")
        print(f"  Charts: {charts_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()