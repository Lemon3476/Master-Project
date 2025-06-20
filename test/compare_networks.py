import pandas as pd
import argparse
import sys
import os # 导入os模块

def analyze_sheet_comparison(file1_path, file2_path, sheet_name, method1, method2):
    """
    对指定的单个工作表进行比较分析，并返回包含结果的DataFrame。
    (此函数内部逻辑不变)
    """
    try:
        df1 = pd.read_excel(file1_path, sheet_name=sheet_name, engine='openpyxl')
        df2 = pd.read_excel(file2_path, sheet_name=sheet_name, engine='openpyxl')
    except FileNotFoundError as e:
        print(f"Error: File not found while processing sheet '{sheet_name}'.\nFull path checked: {e.filename}", file=sys.stderr)
        return None
    except ValueError:
        print(f"Info: Sheet '{sheet_name}' not found in one or both files. Skipping.", file=sys.stdout)
        return None
    except Exception as e:
        print(f"Error loading sheet '{sheet_name}'. Details: {e}", file=sys.stderr)
        return None

    # ... (此函数其余部分保持不变) ...
    df1_filtered = df1[df1['Method'] == method1].copy()
    df2_filtered = df2[df2['Method'] == method2].copy()

    if df1_filtered.empty:
        print(f"Error: Method '{method1}' not found in file '{os.path.basename(file1_path)}'. Available methods: {df1['Method'].unique()}", file=sys.stderr)
        return None
    if df2_filtered.empty:
        print(f"Error: Method '{method2}' not found in file '{os.path.basename(file2_path)}'. Available methods: {df2['Method'].unique()}", file=sys.stderr)
        return None

    merge_keys = ['Batch', 'Sequence']
    metric_columns = ['L2P', 'L2Q', 'NPSS', 'Foot Skate']
    
    df_merged = pd.merge(
        df1_filtered,
        df2_filtered,
        on=merge_keys,
        suffixes=('_net1', '_net2'),
        how='inner'
    )

    if df_merged.empty:
        print(f"Warning: No common sequences found in sheet '{sheet_name}'.", file=sys.stderr)
        return None

    diff_columns = []
    for metric in metric_columns:
        diff_col_name = f"{metric}_diff"
        diff_columns.append(diff_col_name)
        df_merged[diff_col_name] = df_merged[f'{metric}_net1'] - df_merged[f'{metric}_net2']

    zscore_columns = []
    for col in diff_columns:
        zscore_col_name = f"{col}_zscore"
        zscore_columns.append(zscore_col_name)
        mean, std = df_merged[col].mean(), df_merged[col].std()
        df_merged[zscore_col_name] = (df_merged[col] - mean) / std if std > 0 else 0.0

    df_merged['Advantage_Score'] = df_merged[zscore_columns].mean(axis=1)
    df_final = df_merged.sort_values(by='Advantage_Score', ascending=True).reset_index(drop=True)
    
    return df_final


def main():
    """主函数，用于解析参数和协调分析流程。"""
    parser = argparse.ArgumentParser(description="Compare two sets of network evaluation results to find advantages.")
    
    parser.add_argument("--file1", required=True, help="Path to the first results .xlsx file (Network 1).")
    parser.add_argument("--file2", required=True, help="Path to the second results .xlsx file (Network 2).")
    parser.add_argument("--method1", default="Ours", help="The name of the method for Network 1.")
    parser.add_argument("--method2", default="Long-MIB", help="The name of the method for Network 2.")
    parser.add_argument("--all-transitions", action="store_true", help="Analyze all standard transition lengths (15, 30, 60, 90).")
    parser.add_argument("--sheet_name", default="Transition_90", help="The name of the single sheet to compare if --all-transitions is not used.")
    parser.add_argument("--output_file", default="final_comparison_summary.xlsx", help="Path to save the output comparison Excel file.")
    parser.add_argument("--top_n", type=int, default=10, help="The number of top advantage sequences to display.")
    
    args = parser.parse_args()

    # ==================== 核心修改部分 ====================
    # 获取脚本所在的目录 (e.g., /.../long-mib/test)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 从脚本目录上溯一级，得到项目根目录 (e.g., /.../long-mib)
    project_root = os.path.dirname(script_dir)

    # 基于项目根目录，构建文件的绝对路径
    file1_abs_path = os.path.join(project_root, args.file1)
    file2_abs_path = os.path.join(project_root, args.file2)
    output_abs_path = os.path.join(project_root, args.output_file)
    
    print(f"Project root identified as: {project_root}")
    print(f"Looking for File 1 at: {file1_abs_path}")
    print(f"Looking for File 2 at: {file2_abs_path}")
    # =======================================================

    if args.all_transitions:
        transitions_to_test = [15, 30, 60, 90]
    else:
        transitions_to_test = [args.sheet_name.split('_')[-1]]

    with pd.ExcelWriter(output_abs_path, engine='openpyxl') as writer:
        print(f"Starting comparison analysis. Results will be saved to '{output_abs_path}'")
        for trans_len in transitions_to_test:
            sheet_name_to_read = f'Transition_{trans_len}'
            output_sheet_name = f'Comparison_{trans_len}'
            
            print(f"\n>>> Comparing data from sheet: '{sheet_name_to_read}'...")
            
            df_result = analyze_sheet_comparison(
                file1_path=file1_abs_path, # 使用绝对路径
                file2_path=file2_abs_path, # 使用绝对路径
                sheet_name=sheet_name_to_read,
                method1=args.method1,
                method2=args.method2
            )
            
            if df_result is not None:
                df_result.to_excel(writer, sheet_name=output_sheet_name, index=False, float_format="%.4f")
                print(f"    -> Results for '{sheet_name_to_read}' saved to sheet '{output_sheet_name}'.")
                
                print(f"--- Top {args.top_n} sequences where '{args.method1}' has advantage (Transition {trans_len}) ---")
                display_cols = ['Batch', 'Sequence', 'Advantage_Score'] + [f"{m}_diff" for m in ['L2P', 'L2Q', 'NPSS', 'Foot Skate']]
                print(df_result.head(args.top_n).to_string(columns=display_cols, index=False))
                print("------------------------------------------------------------------------------------")

    print(f"\nAnalysis complete. Full comparison saved to '{output_abs_path}'.")

if __name__ == "__main__":
    try:
        import openpyxl
    except ImportError:
        print("Error: 'openpyxl' library is required to write .xlsx files.", file=sys.stderr)
        print("Please install it using: pip install openpyxl", file=sys.stderr)
        sys.exit(1)
        
    main()