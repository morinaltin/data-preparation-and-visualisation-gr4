import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_final_dataset, get_numeric_features, save_report, save_csv, print_section_header

def calculate_zscore(df, features):
    z_scores = pd.DataFrame(index=df.index)
    for col in features:
        mean = df[col].mean()
        std = df[col].std()
        z_scores[f'zscore_{col}'] = (df[col] - mean) / std
    return z_scores

def detect_outliers_zscore(z_scores, threshold):
    outlier_flags = pd.DataFrame(index=z_scores.index)
    for col in z_scores.columns:
        outlier_flags[col.replace('zscore_', 'outlier_')] = np.abs(z_scores[col]) > threshold
    total_outliers = outlier_flags.any(axis=1).sum()
    return outlier_flags, total_outliers

def experiment_thresholds(df, features, thresholds=[2.5, 3.0, 3.5]):
    print_section_header("Z-SCORE THRESHOLD EXPERIMENTATION")
    
    z_scores = calculate_zscore(df, features)
    results = {}
    
    for threshold in thresholds:
        print(f"\nThreshold |Z| > {threshold}:")
        print("-" * 50)
        
        outlier_flags, total_outliers = detect_outliers_zscore(z_scores, threshold)
        
        outlier_counts = {}
        for col in outlier_flags.columns:
            count = outlier_flags[col].sum()
            pct = (count / len(df)) * 100
            feature_name = col.replace('outlier_', '')
            outlier_counts[feature_name] = {'count': count, 'percentage': pct}
            print(f"  {feature_name}: {count:,} ({pct:.2f}%)")
        
        pct_total = (total_outliers / len(df)) * 100
        print(f"\n  Total rows with outliers: {total_outliers:,} ({pct_total:.2f}%)")
        
        results[threshold] = {
            'threshold': threshold,
            'total_outliers': total_outliers,
            'percentage': pct_total,
            'per_feature': outlier_counts,
            'flags': outlier_flags
        }
    
    return results, z_scores

def visualize_threshold_comparison(results):
    thresholds = list(results.keys())
    totals = [results[t]['total_outliers'] for t in thresholds]
    percentages = [results[t]['percentage'] for t in thresholds]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    ax1.bar([str(t) for t in thresholds], totals, color=colors)
    ax1.set_xlabel('Z-Score Threshold', fontsize=12)
    ax1.set_ylabel('Number of Outliers', fontsize=12)
    ax1.set_title('Outliers Detected by Threshold', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(totals):
        ax1.text(i, v + max(totals)*0.02, f'{v:,}', ha='center', fontweight='bold')
    
    ax2.bar([str(t) for t in thresholds], percentages, color=colors)
    ax2.set_xlabel('Z-Score Threshold', fontsize=12)
    ax2.set_ylabel('Percentage of Data (%)', fontsize=12)
    ax2.set_title('Percentage Flagged as Outliers', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(percentages):
        ax2.text(i, v + max(percentages)*0.02, f'{v:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/zscore_threshold_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: zscore_threshold_comparison.png")
    plt.close()

def select_optimal_threshold(results):
    print_section_header("THRESHOLD SELECTION ANALYSIS")
    
    print("Threshold Comparison Summary:")
    print("-" * 70)
    print(f"{'Threshold':<12} {'Outliers':<15} {'Percentage':<15} {'Assessment'}")
    print("-" * 70)
    
    for threshold in sorted(results.keys()):
        r = results[threshold]
        total = r['total_outliers']
        pct = r['percentage']
        
        if pct > 2.0:
            assessment = "Too strict"
        elif pct < 0.3:
            assessment = "Too lenient"
        else:
            assessment = "✓ Balanced"
        
        print(f"|Z| > {threshold:<6} {total:>10,}     {pct:>6.2f}%         {assessment}")
    
    print("-" * 70)
    
    recommended = 3.0
    print(f"\n📌 SELECTED THRESHOLD: |Z| > {recommended}")
    print("\nJustification:")
    print("- Statistical basis: 99.7% of normal data within ±3σ")
    print("- Balanced detection without excessive false positives")
    print("- Appropriate for sensor data anomaly detection")
    
    return recommended

def generate_report(results, selected_threshold, features):
    report = []
    report.append("="*70)
    report.append(" Z-SCORE OUTLIER DETECTION - ANALYSIS")
    report.append("="*70)
    report.append("")
    report.append("METHOD:")
    report.append("Z-Score measures standard deviations from mean")
    report.append("Formula: Z = (X - μ) / σ")
    report.append("")
    report.append("ADVANTAGES:")
    report.append("- Simple, interpretable")
    report.append("- Fast computation")
    report.append("")
    report.append("LIMITATIONS:")
    report.append("- Univariate (per feature)")
    report.append("- Assumes normal distribution")
    report.append("")
    report.append("="*70)
    report.append(" PARAMETER EXPERIMENTATION")
    report.append("="*70)
    report.append("")
    
    report.append(f"{'Threshold':<12} {'Outliers':<15} {'Percentage':<15}")
    report.append("-"*42)
    for threshold in sorted(results.keys()):
        r = results[threshold]
        report.append(f"|Z| > {threshold:<6} {r['total_outliers']:>10,}     {r['percentage']:>6.2f}%")
    
    report.append("")
    report.append("="*70)
    report.append(f" SELECTED THRESHOLD: |Z| > {selected_threshold}")
    report.append("="*70)
    report.append("")
    
    selected_result = results[selected_threshold]
    report.append(f"Total outliers: {selected_result['total_outliers']:,}")
    report.append(f"Percentage: {selected_result['percentage']:.2f}%")
    report.append("")
    report.append("OUTLIERS PER FEATURE:")
    report.append("-"*70)
    
    for feature, stats in selected_result['per_feature'].items():
        report.append(f"  {feature:<30} {stats['count']:>8,}  ({stats['percentage']:>5.2f}%)")
    
    report.append("")
    report.append(f"Analysis completed: {pd.Timestamp.now()}")
    report.append("="*70)
    
    return "\n".join(report)

def main():
    print_section_header("Z-SCORE OUTLIER DETECTION")
    
    df = load_final_dataset()
    features = get_numeric_features(df)
    
    print(f"\nAnalyzing {len(features)} numeric features")
    
    results, z_scores = experiment_thresholds(df, features)
    visualize_threshold_comparison(results)
    selected_threshold = select_optimal_threshold(results)
    
    selected_flags = results[selected_threshold]['flags']
    selected_flags['outlier_any'] = selected_flags.any(axis=1)
    save_csv(selected_flags, 'outliers_zscore_flags.csv')
    
    report = generate_report(results, selected_threshold, features)
    save_report(report, 'outlier_zscore_report.txt')
    
    print_section_header("Z-SCORE ANALYSIS COMPLETE")
    print(f"✓ Selected threshold: |Z| > {selected_threshold}")
    print(f"✓ Outliers detected: {results[selected_threshold]['total_outliers']:,}")
    print(f"✓ Percentage: {results[selected_threshold]['percentage']:.2f}%")

if __name__ == "__main__":
    main()
