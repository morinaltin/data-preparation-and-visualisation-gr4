import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from utils import save_report, save_csv, print_section_header

def load_outlier_results():
    print_section_header("LOADING OUTLIER DETECTION RESULTS")
    
    zscore_df = pd.read_csv('../../outputs/phase2/outliers_zscore_flags.csv')
    iforest_df = pd.read_csv('../../outputs/phase2/outliers_iforest_flags.csv')
    lof_df = pd.read_csv('../../outputs/phase2/outliers_lof_flags.csv')
    
    print(f"✓ Loaded Z-Score results: {zscore_df['outlier_any'].sum():,} outliers")
    print(f"✓ Loaded Isolation Forest results: {iforest_df['outlier_iforest'].sum():,} outliers")
    print(f"✓ Loaded LOF results: {lof_df['outlier_lof'].sum():,} outliers")
    
    return zscore_df, iforest_df, lof_df

def create_comparison_dataframe(zscore_df, iforest_df, lof_df):
    comparison_df = pd.DataFrame({
        'outlier_zscore': zscore_df['outlier_any'],
        'outlier_iforest': iforest_df['outlier_iforest'],
        'outlier_lof': lof_df['outlier_lof']
    })
    
    comparison_df['num_methods'] = (
        comparison_df['outlier_zscore'].astype(int) +
        comparison_df['outlier_iforest'].astype(int) +
        comparison_df['outlier_lof'].astype(int)
    )
    
    comparison_df['outlier_consensus'] = comparison_df['num_methods'] >= 2
    comparison_df['outlier_all'] = comparison_df['num_methods'] == 3
    
    return comparison_df

def analyze_overlap(comparison_df):
    print_section_header("OUTLIER METHOD OVERLAP ANALYSIS")
    
    total_rows = len(comparison_df)
    
    only_zscore = (comparison_df['outlier_zscore'] & 
                   ~comparison_df['outlier_iforest'] & 
                   ~comparison_df['outlier_lof']).sum()
    
    only_iforest = (~comparison_df['outlier_zscore'] & 
                    comparison_df['outlier_iforest'] & 
                    ~comparison_df['outlier_lof']).sum()
    
    only_lof = (~comparison_df['outlier_zscore'] & 
                ~comparison_df['outlier_iforest'] & 
                comparison_df['outlier_lof']).sum()
    
    zscore_iforest = (comparison_df['outlier_zscore'] & 
                      comparison_df['outlier_iforest'] & 
                      ~comparison_df['outlier_lof']).sum()
    
    zscore_lof = (comparison_df['outlier_zscore'] & 
                  ~comparison_df['outlier_iforest'] & 
                  comparison_df['outlier_lof']).sum()
    
    iforest_lof = (~comparison_df['outlier_zscore'] & 
                   comparison_df['outlier_iforest'] & 
                   comparison_df['outlier_lof']).sum()
    
    all_three = comparison_df['outlier_all'].sum()
    
    print("Outlier Detection Overlap:")
    print("-" * 60)
    print(f"Only Z-Score: {only_zscore:,} ({only_zscore/total_rows*100:.2f}%)")
    print(f"Only Isolation Forest: {only_iforest:,} ({only_iforest/total_rows*100:.2f}%)")
    print(f"Only LOF: {only_lof:,} ({only_lof/total_rows*100:.2f}%)")
    print(f"Z-Score + Isolation Forest: {zscore_iforest:,} ({zscore_iforest/total_rows*100:.2f}%)")
    print(f"Z-Score + LOF: {zscore_lof:,} ({zscore_lof/total_rows*100:.2f}%)")
    print(f"Isolation Forest + LOF: {iforest_lof:,} ({iforest_lof/total_rows*100:.2f}%)")
    print(f"All 3 Methods: {all_three:,} ({all_three/total_rows*100:.2f}%)")
    print("-" * 60)
    
    consensus = comparison_df['outlier_consensus'].sum()
    print(f"\nConsensus outliers (2+ methods): {consensus:,} ({consensus/total_rows*100:.2f}%)")
    
    results = {
        'only_zscore': only_zscore,
        'only_iforest': only_iforest,
        'only_lof': only_lof,
        'zscore_iforest': zscore_iforest,
        'zscore_lof': zscore_lof,
        'iforest_lof': iforest_lof,
        'all_three': all_three,
        'consensus': consensus
    }
    
    return results

def create_venn_diagram(comparison_df):
    n_zscore = comparison_df['outlier_zscore'].sum()
    n_iforest = comparison_df['outlier_iforest'].sum()
    n_lof = comparison_df['outlier_lof'].sum()
    
    only_zscore = (comparison_df['outlier_zscore'] & 
                   ~comparison_df['outlier_iforest'] & 
                   ~comparison_df['outlier_lof']).sum()
    
    only_iforest = (~comparison_df['outlier_zscore'] & 
                    comparison_df['outlier_iforest'] & 
                    ~comparison_df['outlier_lof']).sum()
    
    only_lof = (~comparison_df['outlier_zscore'] & 
                ~comparison_df['outlier_iforest'] & 
                comparison_df['outlier_lof']).sum()
    
    zscore_iforest = (comparison_df['outlier_zscore'] & 
                      comparison_df['outlier_iforest'] & 
                      ~comparison_df['outlier_lof']).sum()
    
    zscore_lof = (comparison_df['outlier_zscore'] & 
                  ~comparison_df['outlier_iforest'] & 
                  comparison_df['outlier_lof']).sum()
    
    iforest_lof = (~comparison_df['outlier_zscore'] & 
                   comparison_df['outlier_iforest'] & 
                   comparison_df['outlier_lof']).sum()
    
    all_three = comparison_df['outlier_all'].sum()
    
    plt.figure(figsize=(12, 8))
    
    venn = venn3(
        subsets=(only_zscore, only_iforest, zscore_iforest, 
                 only_lof, zscore_lof, iforest_lof, all_three),
        set_labels=('Z-Score', 'Isolation Forest', 'LOF')
    )
    
    plt.title('Outlier Detection Method Overlap', fontsize=16, fontweight='bold', pad=20)
    
    plt.text(0.5, -0.15, f'Total unique outliers: {comparison_df["num_methods"].gt(0).sum():,}', 
             ha='center', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.5, -0.20, f'Consensus (2+ methods): {comparison_df["outlier_consensus"].sum():,}', 
             ha='center', transform=plt.gca().transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/outlier_method_venn.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: outlier_method_venn.png")
    plt.close()

def create_comparison_bar_chart(comparison_df):
    methods = ['Z-Score', 'Isolation Forest', 'LOF', 'Consensus (2+)', 'All 3']
    counts = [
        comparison_df['outlier_zscore'].sum(),
        comparison_df['outlier_iforest'].sum(),
        comparison_df['outlier_lof'].sum(),
        comparison_df['outlier_consensus'].sum(),
        comparison_df['outlier_all'].sum()
    ]
    percentages = [count / len(comparison_df) * 100 for count in counts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    ax1.bar(methods, counts, color=colors)
    ax1.set_ylabel('Number of Outliers', fontsize=12)
    ax1.set_title('Outliers Detected by Each Method', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(counts):
        ax1.text(i, v + max(counts)*0.02, f'{v:,}', ha='center', fontweight='bold')
    
    ax2.bar(methods, percentages, color=colors)
    ax2.set_ylabel('Percentage of Data (%)', fontsize=12)
    ax2.set_title('Percentage of Data Flagged as Outliers', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(percentages):
        ax2.text(i, v + max(percentages)*0.02, f'{v:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/outlier_method_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outlier_method_comparison.png")
    plt.close()

def generate_comparison_report(comparison_df, overlap_results):
    total = len(comparison_df)
    
    report = []
    report.append("Outlier Detection Method Comparison")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    report.append("Summary:")
    report.append("This analysis compares three outlier detection methods:")
    report.append("1. Z-Score (univariate statistical)")
    report.append("2. Isolation Forest (multivariate ML-based)")
    report.append("3. LOF (density-based)")
    report.append("")
    report.append("Individual Method Results:")
    report.append("")
    report.append(f"Z-Score: {comparison_df['outlier_zscore'].sum():,} outliers ({comparison_df['outlier_zscore'].sum()/total*100:.2f}%)")
    report.append(f"Isolation Forest: {comparison_df['outlier_iforest'].sum():,} outliers ({comparison_df['outlier_iforest'].sum()/total*100:.2f}%)")
    report.append(f"LOF: {comparison_df['outlier_lof'].sum():,} outliers ({comparison_df['outlier_lof'].sum()/total*100:.2f}%)")
    report.append("")
    report.append("Overlap Analysis:")
    report.append("")
    report.append(f"Only Z-Score: {overlap_results['only_zscore']:,} ({overlap_results['only_zscore']/total*100:.2f}%)")
    report.append(f"Only Isolation Forest: {overlap_results['only_iforest']:,} ({overlap_results['only_iforest']/total*100:.2f}%)")
    report.append(f"Only LOF: {overlap_results['only_lof']:,} ({overlap_results['only_lof']/total*100:.2f}%)")
    report.append(f"Z-Score + Isolation Forest: {overlap_results['zscore_iforest']:,} ({overlap_results['zscore_iforest']/total*100:.2f}%)")
    report.append(f"Z-Score + LOF: {overlap_results['zscore_lof']:,} ({overlap_results['zscore_lof']/total*100:.2f}%)")
    report.append(f"Isolation Forest + LOF: {overlap_results['iforest_lof']:,} ({overlap_results['iforest_lof']/total*100:.2f}%)")
    report.append(f"All 3 Methods: {overlap_results['all_three']:,} ({overlap_results['all_three']/total*100:.2f}%)")
    report.append("")
    report.append(f"Consensus Outliers (2+ methods agree): {overlap_results['consensus']:,} ({overlap_results['consensus']/total*100:.2f}%)")
    report.append("")
    report.append("Key Findings:")
    report.append(f"- {overlap_results['all_three']:,} outliers detected by all 3 methods (high confidence)")
    report.append(f"- {overlap_results['consensus']:,} outliers detected by at least 2 methods (moderate-high confidence)")
    report.append("- Isolation Forest detected most outliers (multivariate approach)")
    report.append("- LOF detected fewest outliers (density-based, more conservative)")
    report.append("- Z-Score middle ground (univariate statistical)")
    report.append("")
    report.append("Recommendation:")
    report.append("Use consensus outliers (2+ methods) for high-confidence anomaly detection.")
    report.append("This balances sensitivity with specificity.")
    
    return "\n".join(report)

def main():
    print_section_header("OUTLIER DETECTION METHOD COMPARISON")
    
    zscore_df, iforest_df, lof_df = load_outlier_results()
    
    comparison_df = create_comparison_dataframe(zscore_df, iforest_df, lof_df)
    
    overlap_results = analyze_overlap(comparison_df)
    
    create_venn_diagram(comparison_df)
    create_comparison_bar_chart(comparison_df)
    
    save_csv(comparison_df, 'outlier_method_comparison.csv')
    
    report = generate_comparison_report(comparison_df, overlap_results)
    save_report(report, 'outlier_method_comparison_report.txt')
    
    print_section_header("METHOD COMPARISON COMPLETE")
    print(f"✓ Total unique outliers: {comparison_df['num_methods'].gt(0).sum():,}")
    print(f"✓ Consensus outliers (2+ methods): {overlap_results['consensus']:,}")
    print(f"✓ High-confidence outliers (all 3): {overlap_results['all_three']:,}")

if __name__ == "__main__":
    main()
