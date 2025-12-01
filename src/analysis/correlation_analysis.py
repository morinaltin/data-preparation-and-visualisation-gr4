import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_final_dataset, get_numeric_features, save_report, save_csv, print_section_header

def calculate_correlation(df, features):
    print_section_header("CORRELATION ANALYSIS")
    
    correlation_matrix = df[features].corr()
    
    print("Correlation Matrix:")
    print(correlation_matrix)
    print()
    
    return correlation_matrix

def find_strong_correlations(correlation_matrix, threshold=0.7):
    strong_corr = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                strong_corr.append({
                    'Feature_1': correlation_matrix.columns[i],
                    'Feature_2': correlation_matrix.columns[j],
                    'Correlation': corr_value
                })
    
    return pd.DataFrame(strong_corr)

def create_correlation_heatmap(correlation_matrix):
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: correlation_heatmap.png")
    plt.close()

def create_covariance_heatmap(df, features):
    covariance_matrix = df[features].cov()
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(covariance_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='viridis',
                square=True,
                linewidths=1,
                cbar_kws={'label': 'Covariance'})
    
    plt.title('Feature Covariance Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/covariance_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: covariance_heatmap.png")
    plt.close()
    
    return covariance_matrix

def generate_report(correlation_matrix, strong_corr_df):
    report = []
    report.append("Correlation and Covariance Analysis Report")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    report.append("Correlation measures the strength of linear relationships between features.")
    report.append("Values range from -1 to +1:")
    report.append("  +1 = perfect positive correlation")
    report.append("   0 = no correlation")
    report.append("  -1 = perfect negative correlation")
    report.append("")
    report.append("Strong Correlations (|r| >= 0.7):")
    report.append("")
    
    if len(strong_corr_df) > 0:
        for _, row in strong_corr_df.iterrows():
            direction = "positive" if row['Correlation'] > 0 else "negative"
            report.append(f"{row['Feature_1']} ↔ {row['Feature_2']}: {row['Correlation']:.3f} ({direction})")
    else:
        report.append("No strong correlations found (all |r| < 0.7)")
    
    report.append("")
    report.append("Key Findings:")
    report.append("")
    
    max_corr = 0
    max_pair = None
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr = abs(correlation_matrix.iloc[i, j])
            if corr > max_corr:
                max_corr = corr
                max_pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])
    
    if max_pair:
        report.append(f"Strongest correlation: {max_pair[0]} ↔ {max_pair[1]} ({max_corr:.3f})")
    
    report.append("")
    report.append("Implications:")
    report.append("- Highly correlated features contain redundant information")
    report.append("- PCA will capture this correlation and reduce dimensionality")
    report.append("- Multivariate outlier detection benefits from understanding correlations")
    
    return "\n".join(report)

def main():
    print_section_header("CORRELATION & COVARIANCE ANALYSIS")
    
    df = load_final_dataset()
    features = get_numeric_features(df)
    
    correlation_matrix = calculate_correlation(df, features)
    strong_corr_df = find_strong_correlations(correlation_matrix, threshold=0.7)
    
    print(f"Found {len(strong_corr_df)} strong correlations (|r| >= 0.7)")
    if len(strong_corr_df) > 0:
        print("\nStrong Correlations:")
        for _, row in strong_corr_df.iterrows():
            print(f"  {row['Feature_1']} ↔ {row['Feature_2']}: {row['Correlation']:.3f}")
    
    create_correlation_heatmap(correlation_matrix)
    covariance_matrix = create_covariance_heatmap(df, features)
    
    save_csv(correlation_matrix, 'correlation_matrix.csv')
    save_csv(covariance_matrix, 'covariance_matrix.csv')
    if len(strong_corr_df) > 0:
        save_csv(strong_corr_df, 'strong_correlations.csv')
    
    report = generate_report(correlation_matrix, strong_corr_df)
    save_report(report, 'correlation_analysis_report.txt')
    
    print_section_header("CORRELATION ANALYSIS COMPLETE")
    print(f"✓ Analyzed correlations between {len(features)} features")
    print(f"✓ Generated correlation and covariance heatmaps")

if __name__ == "__main__":
    main()
