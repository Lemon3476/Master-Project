import pandas as pd
from scipy import stats
import sys
import os
import argparse # 1. 导入 argparse 库

def perform_t_test_and_get_summary(
    data_file, 
    sheet_name, 
    metric, 
    name_method1='Ours', 
    name_method2='Baseline'
):
    """
    从单个Excel工作表中加载宽格式数据，进行t检验，并返回一个包含结果的字典。
    (此函数无需修改)
    """
    try:
        df = pd.read_excel(data_file, sheet_name=sheet_name)
    except (FileNotFoundError, ValueError):
        print(f"Info: Could not read sheet '{sheet_name}' from '{data_file}'. Skipping.", file=sys.stdout)
        return None

    metric_col_1 = f"{metric}_net1"
    metric_col_2 = f"{metric}_net2"

    if metric_col_1 not in df.columns or metric_col_2 not in df.columns:
        print(f"Warning: Columns '{metric_col_1}' or '{metric_col_2}' not found in sheet '{sheet_name}'.", file=sys.stderr)
        return None
        
    scores_method1 = df[metric_col_1].dropna()
    scores_method2 = df[metric_col_2].dropna()

    if scores_method1.empty or scores_method2.empty:
        return None

    t_statistic, p_value = stats.ttest_ind(
        scores_method1, 
        scores_method2, 
        equal_var=False,
        alternative='less'
    )
    
    summary_dict = {
        'Transition': sheet_name.split('_')[-1],
        'Metric': metric,
        f'{name_method1}_Mean': scores_method1.mean(),
        f'{name_method1}_Std': scores_method1.std(),
        f'{name_method2}_Mean': scores_method2.mean(),
        f'{name_method2}_Std': scores_method2.std(),
        'T-statistic': t_statistic,
        'P-value': p_value,
        'Significant (p<0.05)': p_value < 0.05
    }
    
    return summary_dict

def write_summary_to_text_file(summary, output_handle, name_method1='Ours', name_method2='Baseline'):
    """将单个摘要字典的内容写入到文本文件中。(此函数无需修改)"""
    output_handle.write(f"\n--- T-Test Analysis for Metric: '{summary['Metric']}' on Sheet: 'Comparison_{summary['Transition']}' ---\n")
    output_handle.write(f"Comparing {name_method1} vs {name_method2}\n")
    output_handle.write(f"  {name_method1} -> Mean: {summary[f'{name_method1}_Mean']:.4f}, Std: {summary[f'{name_method1}_Std']:.4f}\n")
    output_handle.write(f"  {name_method2} -> Mean: {summary[f'{name_method2}_Mean']:.4f}, Std: {summary[f'{name_method2}_Std']:.4f}\n")
    output_handle.write(f"\nT-statistic = {summary['T-statistic']:.4f}\n")
    output_handle.write(f"P-value = {summary['P-value']:.6f}\n")
    if summary['Significant (p<0.05)']:
        output_handle.write(f"Conclusion: The result is statistically significant (p < 0.05).\n")
    else:
        output_handle.write(f"Conclusion: The result is not statistically significant (p >= 0.05).\n")

def main():
    """主函数，配置并运行所有分析。"""
    # --- 1. 使用 argparse 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Perform T-test analysis on comparison results from a 'wide' format Excel file.")
    parser.add_argument(
        '--input_file', 
        type=str, 
        required=True, 
        help="Path to the input Excel file (e.g., 'final_comparison_summary.xlsx')."
    )
    parser.add_argument(
        '--text_report', 
        type=str, 
        default='final_t_test_report.txt',
        help="Name for the output detailed text report."
    )
    parser.add_argument(
        '--excel_summary', 
        type=str, 
        default='final_t_test_summary.xlsx',
        help="Name for the output summary Excel file."
    )
    args = parser.parse_args()

    # --- 2. 定义常量配置 ---
    TRANSITIONS_TO_TEST = [15, 30, 60, 90]
    METRICS_TO_TEST = ['L2P', 'L2Q', 'NPSS', 'Foot Skate']
    
    # --- 3. 运行分析 ---
    print(f"Starting T-Test Analysis on '{args.input_file}'...")
    
    all_summaries = [] 

    with open(args.text_report, 'w', encoding='utf-8') as f:
        f.write(f"T-Test Analysis Report for '{args.input_file}'\n")
        f.write("============================================\n")

        for trans_len in TRANSITIONS_TO_TEST:
            sheet_name_to_read = f'Comparison_{trans_len}'
            print(f"\n>>> Analyzing Sheet: '{sheet_name_to_read}'...")
            
            f.write(f"\n\n############################################################\n")
            f.write(f"# Analysis for Transition Length: {trans_len}\n")
            f.write(f"############################################################\n")
            
            for metric in METRICS_TO_TEST:
                # 使用 args.input_file 作为数据文件路径
                summary = perform_t_test_and_get_summary(
                    data_file=args.input_file,
                    sheet_name=sheet_name_to_read,
                    metric=metric,
                    name_method1='Ours',
                    name_method2='Baseline'
                )
                if summary:
                    all_summaries.append(summary)
                    write_summary_to_text_file(summary, f)
                f.write("------------------------------------------------------\n")

    print(f"\nDetailed text report saved to '{args.text_report}'.")

    if all_summaries:
        print(f"Creating final summary Excel file...")
        summary_df = pd.DataFrame(all_summaries)
        
        ordered_columns = [
            'Transition', 'Metric', 
            'Ours_Mean', 'Ours_Std', 'Baseline_Mean', 'Baseline_Std', 
            'T-statistic', 'P-value', 'Significant (p<0.05)'
        ]
        summary_df = summary_df[ordered_columns]
        
        try:
            # 使用 args.excel_summary 作为输出文件名
            summary_df.to_excel(args.excel_summary, index=False, float_format="%.4f")
            print(f"Summary Excel file saved to '{args.excel_summary}'.")
        except Exception as e:
            print(f"Error saving summary Excel file: {e}", file=sys.stderr)
    else:
        print("No valid data was analyzed, so no summary Excel file was created.")

if __name__ == '__main__':
    try:
        import openpyxl
    except ImportError:
        print("Error: 'openpyxl' library is required to write .xlsx files.", file=sys.stderr)
        print("Please install it using: pip install openpyxl", file=sys.stderr)
        sys.exit(1)
        
    main()