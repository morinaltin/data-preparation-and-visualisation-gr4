import pandas as pd
import numpy as np
import os
from utils import load_final_dataset, print_section_header

def remove_confirmed_outliers():
    print_section_header("OUTLIER REMOVAL & DATASET FINALIZATION")
    
    # Paths
    data_path = '../../data/processed/household_power_consumption_cleaned.csv'
    flags_path = '../../outputs/phase2/outlier_method_comparison.csv'
    output_path = '../../data/processed/household_power_consumption_phase2_clean.csv'
    
    # Load Data
    print(f"Loading main dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Loading outlier flags from: {flags_path}")
    if not os.path.exists(flags_path):
        print("Error: Outlier comparison results not found. Please run Step 12 (Method Comparison) first.")
        return
        
    flags_df = pd.read_csv(flags_path)
    
    # Verify alignment
    if len(df) != len(flags_df):
        print(f"Error: Dataset length ({len(df)}) and flags length ({len(flags_df)}) do not match!")
        return

    # Check for 'outlier_consensus' column (True if 2+ methods agree)
    if 'outlier_consensus' not in flags_df.columns:
        print("Error: 'outlier_consensus' column missing in flags file.")
        return

    # Filter
    n_total = len(df)
    
    # We remove rows where outlier_consensus is TRUE
    # consensus means at least 2 methods agreed it's an outlier
    outliers_mask = flags_df['outlier_consensus'].astype(bool)
    n_outliers = outliers_mask.sum()
    
    df_clean = df[~outliers_mask].copy()
    n_clean = len(df_clean)
    
    print(f"\nStats:")
    print(f"  Total rows before: {n_total:,}")
    print(f"  Outliers to remove: {n_outliers:,} (Consensus of 2+ methods)")
    print(f"  Rows remaining:    {n_clean:,}")
    print(f"  Percentage removed: {(n_outliers/n_total)*100:.2f}%")
    
    # Save
    print(f"\nSaving clean dataset to: {output_path}")
    df_clean.to_csv(output_path, index=False)
    print("✓ Saved successfully.")

    # Create a simple report
    report_path = '../../reports/phase2/outlier_removal_report.txt'
    with open(report_path, 'w') as f:
        f.write("OUTLIER REMOVAL REPORT\n")
        f.write("======================\n\n")
        f.write(f"Original Row Count: {n_total:,}\n")
        f.write(f"Outliers Removed:   {n_outliers:,} (Consensus > 1 method)\n")
        f.write(f"Final Row Count:    {n_clean:,}\n")
        f.write(f"Percentage Removed: {(n_outliers/n_total)*100:.2f}%\n")
        f.write("\nRationale:\n")
        f.write("Removed rows identified as outliers by at least 2 out of 3 methods (Z-Score, Isolation Forest, LOF).\n")
        f.write("Single-method outliers were kept as potentially valid extreme values (false positives elimination).\n")
    
    print(f"✓ Report saved to {report_path}")

if __name__ == "__main__":
    remove_confirmed_outliers()
