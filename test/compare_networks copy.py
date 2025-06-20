import pandas as pd
from scipy import stats
import sys
import os

def perform_t_test_from_separate_files(
    file_method1, 
    file_method2, 
    sheet_name, 
    metric, 
    output_handle,
    name_method1='Method 1', 
    name_method2='Method 2'
):
    """
    Loads data from two separate Excel files, performs a t-test, 
    and writes the results to the provided file handle.
    (This function remains unchanged as it is already correct)
    """
    output_handle.write(f"\n--- T-Test Analysis for Metric: '{metric}' on Sheet: '{sheet_name}' ---\n")

    try:
        df1 = pd.read_excel(file_method1, sheet_name=sheet_name)
        df2 = pd.read_excel(file_method2, sheet_name=sheet_name)
    except FileNotFoundError as e:
        error_msg = f"Error: Could not find a file. Please ensure paths are correct. Details: {e}\n"
        print(error_msg, file=sys.stderr)
        output_handle.write(error_msg)
        return
    except ValueError:
        error_msg = f"Info: Sheet '{sheet_name}' not found in one of the files. Skipping this analysis.\n"
        print(error_msg, file=sys.stdout)
        output_handle.write(error_msg)
        return

    scores_method1 = df1[metric].dropna()
    scores_method2 = df2[metric].dropna()

    if scores_method1.empty or scores_method2.empty:
        warning_msg = f"Warning: No data found for metric '{metric}' in one or both files for sheet '{sheet_name}'. Skipping.\n"
        print(warning_msg, file=sys.stderr)
        output_handle.write(warning_msg)
        return

    if len(scores_method1) != len(scores_method2):
        output_handle.write(f"Warning: The number of samples is different between the two methods.\n")
    
    output_handle.write(f"Comparing {name_method1} (n={len(scores_method1)}) vs {name_method2} (n={len(scores_method2)})\n")
    output_handle.write(f"{name_method1} Mean: {scores_method1.mean():.4f}, Std: {scores_method1.std():.4f}\n")
    output_handle.write(f"{name_method2} Mean: {scores_method2.mean():.4f}, Std: {scores_method2.std():.4f}\n")

    t_statistic, p_value = stats.ttest_ind(
        scores_method1, 
        scores_method2, 
        equal_var=False,
        alternative='less'
    )

    output_handle.write(f"\nT-statistic = {t_statistic:.4f}\n")
    output_handle.write(f"P-value = {p_value:.6f}\n")

    alpha = 0.05
    if p_value < alpha:
        output_handle.write(f"Conclusion: The result is statistically significant (p < {alpha}).\n")
        output_handle.write(f"This suggests that '{name_method1}' performs significantly better (has a lower '{metric}' score) than '{name_method2}'.\n")
    else:
        output_handle.write(f"Conclusion: The result is not statistically significant (p >= {alpha}).\n")
        output_handle.write(f"There is not enough statistical evidence to conclude that '{name_method1}' is significantly better than '{name_method2}'.\n")

def main():
    """Main function to configure and run the entire analysis for all transitions."""
    # --- 1. Configure Your Analysis Here ---
    
    FILE_OURS = 'evaluation_results_keyframe_enc_refine_enc_fc.xlsx'
    FILE_BASELINE = 'evaluation_results_keyframe_refine.xlsx'
    
    # MODIFICATION: Define a list of all transition lengths to test
    TRANSITIONS_TO_TEST = [15, 30, 60, 90]

    METRICS_TO_TEST = ['L2P', 'L2Q', 'NPSS', 'Foot Skate']
    OUTPUT_FILENAME = 't_test_all_transitions.txt'

    # --- 2. Run the Analysis ---
    print(f"Starting T-Test Analysis for transitions: {TRANSITIONS_TO_TEST}...")
    
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        f.write("T-Test Analysis Results\n")
        f.write("=========================\n")

        # MODIFICATION: Loop over each transition length
        for trans_len in TRANSITIONS_TO_TEST:
            # Dynamically create the sheet name for each iteration
            sheet_name = f'Transition_{trans_len}'
            
            # Print progress to the console
            print(f"\n>>> Analyzing Sheet: '{sheet_name}'...")
            
            # Write a clear header in the output file for this section
            f.write(f"\n\n############################################################\n")
            f.write(f"# Analysis for Transition Length: {trans_len} (Sheet: {sheet_name})\n")
            f.write(f"############################################################\n")

            # The inner loop for metrics remains the same
            for metric in METRICS_TO_TEST:
                perform_t_test_from_separate_files(
                    file_method1=FILE_OURS,
                    file_method2=FILE_BASELINE,
                    sheet_name=sheet_name, # Use the dynamically generated sheet name
                    metric=metric,
                    output_handle=f,
                    name_method1='Ours',
                    name_method2='Baseline'
                )
                f.write("------------------------------------------------------\n")

    print(f"\nAnalysis complete. All results have been saved to '{OUTPUT_FILENAME}'.")

if __name__ == '__main__':
    main()