import pandas as pd
from scipy import stats
import sys

def perform_t_test_from_separate_files(
    file_method1, 
    file_method2, 
    sheet_name, 
    metric, 
    output_handle, # Argument to handle file writing
    name_method1='Method 1', 
    name_method2='Method 2'
):
    """
    Loads data from two separate Excel files, performs a t-test, 
    and writes the results to the provided file handle.
    """
    output_handle.write(f"\n--- T-Test Analysis for Metric: '{metric}' on Sheet: '{sheet_name}' ---\n")

    try:
        # Load data from the two separate Excel files
        df1 = pd.read_excel(file_method1, sheet_name=sheet_name)
        df2 = pd.read_excel(file_method2, sheet_name=sheet_name)
    except FileNotFoundError as e:
        error_msg = f"Error: Could not find a file. Please ensure paths are correct. Details: {e}\n"
        print(error_msg, file=sys.stderr)
        output_handle.write(error_msg)
        return
    except ValueError:
        # This error is expected if a sheet for a specific transition length doesn't exist in the file.
        error_msg = f"Info: Sheet '{sheet_name}' not found in one of the files. Skipping this transition length.\n"
        print(error_msg, file=sys.stdout) # Print as info, not error
        output_handle.write(error_msg)
        return

    # Extract the specified metric column from each DataFrame
    scores_method1 = df1[metric].dropna()
    scores_method2 = df2[metric].dropna()

    if scores_method1.empty or scores_method2.empty:
        warning_msg = f"Warning: No data found for metric '{metric}' in one or both files. Skipping.\n"
        print(warning_msg, file=sys.stderr)
        output_handle.write(warning_msg)
        return

    if len(scores_method1) != len(scores_method2):
        output_handle.write(f"Warning: The number of samples is different between the two methods.\n")
    
    output_handle.write(f"Comparing {name_method1} (n={len(scores_method1)}) vs {name_method2} (n={len(scores_method2)})\n")
    output_handle.write(f"{name_method1} Mean: {scores_method1.mean():.4f}, Std: {scores_method1.std():.4f}\n")
    output_handle.write(f"{name_method2} Mean: {scores_method2.mean():.4f}, Std: {scores_method2.std():.4f}\n")

    # Perform an independent two-sample t-test (Welch's t-test)
    t_statistic, p_value = stats.ttest_ind(
        scores_method1, 
        scores_method2, 
        equal_var=False,
        alternative='less'
    )

    output_handle.write(f"\nT-statistic = {t_statistic:.4f}\n")
    output_handle.write(f"P-value = {p_value:.6f}\n")

    # Interpret the p-value
    alpha = 0.05
    if p_value < alpha:
        output_handle.write(f"Conclusion: The result is statistically significant (p < {alpha}).\n")
        output_handle.write(f"This suggests that '{name_method1}' performs significantly better (has a lower '{metric}' score) than '{name_method2}'.\n")
    else:
        output_handle.write(f"Conclusion: The result is not statistically significant (p >= {alpha}).\n")
        output_handle.write(f"There is not enough statistical evidence to conclude that '{name_method1}' is significantly better than '{name_method2}'.\n")


if __name__ == '__main__':
    # --- 1. Configure Your Analysis Here ---
    
    # File paths for the two Excel result files
    FILE_OURS = 'evaluation_results_keyframe_enc_refine_enc_fc.xlsx'
    FILE_BASELINE = 'evaluation_results_keyframe_refine.xlsx'

    # MODIFICATION: Define a list of all transition lengths to test
    TRANSITIONS_TO_TEST = [15, 30, 60, 90]

    # A list of all metrics you want to test
    METRICS_TO_TEST = ['L2P', 'L2Q', 'NPSS', 'Foot Skate']
    
    # The name of the file where results will be saved
    OUTPUT_FILENAME = 't_test_results_all_transitions.txt'

    # --- 2. Run the Analysis ---
    print(f"Starting T-Test Analysis for transitions {TRANSITIONS_TO_TEST}...")
    print(f"Results will be saved to '{OUTPUT_FILENAME}'")
    
    # Open the output file once
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        f.write("T-Test Analysis Results\n")
        f.write("=========================\n")

        # MODIFICATION: Loop over each transition length
        for trans_len in TRANSITIONS_TO_TEST:
            # Dynamically create the sheet name based on the transition length
            sheet_name = f'Transition_{trans_len}'
            
            # Write a main header for the current transition length section
            f.write(f"\n\n############################################################\n")
            f.write(f"# Analysis for Transition Length: {trans_len} (Sheet: {sheet_name})\n")
            f.write(f"############################################################\n")

            # The inner loop for metrics remains the same
            for metric in METRICS_TO_TEST:
                perform_t_test_from_separate_files(
                    file_method1=FILE_OURS,
                    file_method2=FILE_BASELINE,
                    sheet_name=sheet_name,
                    metric=metric,
                    output_handle=f, # Pass the file handle to the function
                    name_method1='Ours',
                    name_method2='Long-MIB'
                )
                f.write("------------------------------------------------------\n")

    print(f"Analysis complete. All results have been saved to '{OUTPUT_FILENAME}'.")