import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_final_dataset, get_numeric_features, save_report, save_csv, print_section_header

def calculate_enhanced_statistics(df, features):
    print_section_header("ENHANCED STATISTICAL ANALYSIS")
    
    stats_list = []
    
    for feature in features:
        data = df[feature].dropna()
        
        mean = data.mean()
        median = data.median()
        std = data.std()
        variance = data.var()
        
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        ci_95 = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
        ci_lower = ci_95[0]
        ci_upper = ci_95[1]
        
        percentile_5 = data.quantile(0.05)
        percentile_95 = data.quantile(0.95)
        
        min_val = data.min()
        max_val = data.max()
        
        stats_dict = {
            'Feature': feature,
            'Count': len(data),
            'Mean': mean,
            'Median': median,
            'Std': std,
            'Variance': variance,
            'Min': min_val,
            'Max': max_val,
            'Q25': q25,
            'Q75': q75,
            'IQR': iqr,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'CI_95_Lower': ci_lower,
            'CI_95_Upper': ci_upper,
            'Percentile_5': percentile_5,
            'Percentile_95': percentile_95
        }
        
        stats_list.append(stats_dict)
        
        print(f"{feature}:")
        print(f"  Skewness: {skewness:.4f} {'(right-skewed)' if skewness > 0 else '(left-skewed)' if skewness < 0 else '(symmetric)'}")
        print(f"  Kurtosis: {kurtosis:.4f} {'(heavy-tailed)' if kurtosis > 0 else '(light-tailed)'}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print()
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df

def interpret_skewness(skewness):
    if abs(skewness) < 0.5:
        return "Approximately symmetric"
    elif skewness > 0:
        if skewness < 1:
            return "Moderately right-skewed"
        else:
            return "Highly right-skewed"
    else:
        if skewness > -1:
            return "Moderately left-skewed"
        else:
            return "Highly left-skewed"

def interpret_kurtosis(kurtosis):
    if abs(kurtosis) < 0.5:
        return "Normal distribution (mesokurtic)"
    elif kurtosis > 0:
        return "Heavy-tailed (leptokurtic) - more outliers"
    else:
        return "Light-tailed (platykurtic) - fewer outliers"

def visualize_distributions(df, features):
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        data = df[feature].dropna()
        
        ax.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
        ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{feature}\nSkew: {stats.skew(data):.2f}, Kurt: {stats.kurtosis(data):.2f}', 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/enhanced_statistics_distributions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: enhanced_statistics_distributions.png")
    plt.close()

def create_statistics_summary_plot(stats_df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    ax = axes[0, 0]
    features_short = [f[:15] for f in stats_df['Feature']]
    ax.barh(features_short, stats_df['Skewness'], color=['red' if x > 0 else 'blue' for x in stats_df['Skewness']])
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Skewness', fontsize=11)
    ax.set_title('Distribution Skewness by Feature', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    ax = axes[0, 1]
    ax.barh(features_short, stats_df['Kurtosis'], color=['red' if x > 0 else 'green' for x in stats_df['Kurtosis']])
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Kurtosis', fontsize=11)
    ax.set_title('Distribution Kurtosis by Feature', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    ax = axes[1, 0]
    cv = (stats_df['Std'] / stats_df['Mean']) * 100
    ax.barh(features_short, cv, color='steelblue')
    ax.set_xlabel('Coefficient of Variation (%)', fontsize=11)
    ax.set_title('Variability by Feature', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    ax = axes[1, 1]
    ci_width = stats_df['CI_95_Upper'] - stats_df['CI_95_Lower']
    ax.barh(features_short, ci_width, color='orange')
    ax.set_xlabel('95% Confidence Interval Width', fontsize=11)
    ax.set_title('Estimation Uncertainty by Feature', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/enhanced_statistics_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: enhanced_statistics_summary.png")
    plt.close()

def generate_report(stats_df):
    report = []
    report.append("Enhanced Statistical Analysis Report")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    report.append("This report provides advanced statistical measures beyond basic descriptive statistics.")
    report.append("")
    report.append("Key Metrics Explained:")
    report.append("")
    report.append("Skewness: Measures distribution asymmetry")
    report.append("  - Near 0: Symmetric distribution")
    report.append("  - Positive: Right-skewed (long tail on right)")
    report.append("  - Negative: Left-skewed (long tail on left)")
    report.append("")
    report.append("Kurtosis: Measures tail heaviness")
    report.append("  - Near 0: Normal distribution")
    report.append("  - Positive: Heavy tails (more extreme values)")
    report.append("  - Negative: Light tails (fewer extreme values)")
    report.append("")
    report.append("95% Confidence Interval: Range where true mean likely falls (95% confidence)")
    report.append("")
    report.append("="*70)
    report.append("FEATURE-BY-FEATURE ANALYSIS")
    report.append("="*70)
    report.append("")
    
    for _, row in stats_df.iterrows():
        report.append(f"{row['Feature']}:")
        report.append(f"  Mean: {row['Mean']:.4f} (95% CI: [{row['CI_95_Lower']:.4f}, {row['CI_95_Upper']:.4f}])")
        report.append(f"  Median: {row['Median']:.4f}")
        report.append(f"  Std Dev: {row['Std']:.4f}")
        report.append(f"  Range: [{row['Min']:.4f}, {row['Max']:.4f}]")
        report.append(f"  IQR: {row['IQR']:.4f} (Q1: {row['Q25']:.4f}, Q3: {row['Q75']:.4f})")
        report.append(f"  Skewness: {row['Skewness']:.4f} - {interpret_skewness(row['Skewness'])}")
        report.append(f"  Kurtosis: {row['Kurtosis']:.4f} - {interpret_kurtosis(row['Kurtosis'])}")
        report.append("")
    
    report.append("="*70)
    report.append("KEY FINDINGS")
    report.append("="*70)
    report.append("")
    
    high_skew = stats_df[abs(stats_df['Skewness']) > 1]['Feature'].tolist()
    if high_skew:
        report.append(f"Highly skewed features: {', '.join(high_skew)}")
        report.append("  Implication: Not normally distributed, outliers likely")
    
    heavy_tail = stats_df[stats_df['Kurtosis'] > 1]['Feature'].tolist()
    if heavy_tail:
        report.append(f"Heavy-tailed features: {', '.join(heavy_tail)}")
        report.append("  Implication: More extreme values than normal distribution")
    
    report.append("")
    report.append("Recommendation:")
    report.append("Features with high skewness/kurtosis may benefit from transformation")
    report.append("(e.g., log transform) before using in statistical models that assume normality.")
    
    return "\n".join(report)

def main():
    print_section_header("ENHANCED STATISTICS ANALYSIS")
    
    df = load_final_dataset()
    features = get_numeric_features(df)
    
    stats_df = calculate_enhanced_statistics(df, features)
    
    save_csv(stats_df, 'enhanced_statistics.csv')
    
    visualize_distributions(df, features)
    create_statistics_summary_plot(stats_df)
    
    report = generate_report(stats_df)
    save_report(report, 'enhanced_statistics_report.txt')
    
    print_section_header("ENHANCED STATISTICS COMPLETE")
    print(f"✓ Analyzed {len(features)} features")
    print(f"✓ Calculated skewness, kurtosis, and confidence intervals")
    print(f"✓ Generated distribution visualizations")

if __name__ == "__main__":
    main()
