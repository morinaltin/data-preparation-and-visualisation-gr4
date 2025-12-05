import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_final_dataset, get_numeric_features, save_report, save_csv, print_section_header

def test_normality(df, features):
    print_section_header("NORMALITY TESTS")
    
    normality_results = []
    
    for feature in features:
        data = df[feature].dropna()
        
        shapiro_stat, shapiro_p = stats.shapiro(data[:5000])
        
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        
        anderson_result = stats.anderson(data)
        
        is_normal_shapiro = shapiro_p > 0.05
        is_normal_ks = ks_p > 0.05
        
        print(f"{feature}:")
        print(f"  Shapiro-Wilk: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4e}")
        print(f"  {'✓ Normal' if is_normal_shapiro else '✗ Not normal'} (α=0.05)")
        print(f"  Kolmogorov-Smirnov: statistic={ks_stat:.4f}, p-value={ks_p:.4e}")
        print(f"  {'✓ Normal' if is_normal_ks else '✗ Not normal'} (α=0.05)")
        print()
        
        normality_results.append({
            'Feature': feature,
            'Shapiro_Statistic': shapiro_stat,
            'Shapiro_P_Value': shapiro_p,
            'Shapiro_Normal': is_normal_shapiro,
            'KS_Statistic': ks_stat,
            'KS_P_Value': ks_p,
            'KS_Normal': is_normal_ks,
            'Anderson_Statistic': anderson_result.statistic
        })
    
    return pd.DataFrame(normality_results)

def create_qq_plots(df, features):
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        data = df[feature].dropna()
        
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: {feature}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
    
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/qq_plots.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: qq_plots.png")
    plt.close()

def create_distribution_comparison(df, features):
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        data = df[feature].dropna()
        
        ax.hist(data, bins=50, density=True, alpha=0.6, color='steelblue', 
                edgecolor='black', label='Observed')
        
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{feature} vs Normal Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/distribution_vs_normal.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: distribution_vs_normal.png")
    plt.close()

def create_kde_plots(df, features):
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        data = df[feature].dropna()
        
        sns.kdeplot(data=data, ax=ax, fill=True, color='steelblue', alpha=0.6)
        
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
        ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'KDE: {feature}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/kde_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: kde_plots.png")
    plt.close()

def generate_report(normality_df):
    report = []
    report.append("Distribution Analysis and Normality Test Report")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    report.append("This report tests whether features follow a normal (Gaussian) distribution.")
    report.append("")
    report.append("Tests Performed:")
    report.append("")
    report.append("1. Shapiro-Wilk Test")
    report.append("   - Tests null hypothesis: data is normally distributed")
    report.append("   - p-value < 0.05 → reject null → data is NOT normal")
    report.append("")
    report.append("2. Kolmogorov-Smirnov Test")
    report.append("   - Compares sample distribution to normal distribution")
    report.append("   - p-value < 0.05 → data significantly different from normal")
    report.append("")
    report.append("3. Q-Q Plots")
    report.append("   - Visual test: points should follow diagonal line if normal")
    report.append("   - Deviations indicate non-normality")
    report.append("")
    report.append("="*70)
    report.append("NORMALITY TEST RESULTS")
    report.append("="*70)
    report.append("")
    
    for _, row in normality_df.iterrows():
        report.append(f"{row['Feature']}:")
        report.append(f"  Shapiro-Wilk p-value: {row['Shapiro_P_Value']:.4e}")
        report.append(f"  Result: {'Normal' if row['Shapiro_Normal'] else 'Not Normal'}")
        report.append(f"  Kolmogorov-Smirnov p-value: {row['KS_P_Value']:.4e}")
        report.append(f"  Result: {'Normal' if row['KS_Normal'] else 'Not Normal'}")
        report.append("")
    
    report.append("="*70)
    report.append("SUMMARY")
    report.append("="*70)
    report.append("")
    
    normal_count = normality_df['Shapiro_Normal'].sum()
    total_count = len(normality_df)
    
    report.append(f"Features passing normality (Shapiro-Wilk): {normal_count}/{total_count}")
    report.append("")
    
    if normal_count == 0:
        report.append("FINDING: None of the features follow a normal distribution.")
        report.append("")
        report.append("Implications:")
        report.append("- Parametric tests assuming normality may not be appropriate")
        report.append("- Consider non-parametric alternatives")
        report.append("- Transformations (log, sqrt) may help normalize distributions")
        report.append("- Outlier detection methods not assuming normality preferred")
    elif normal_count < total_count / 2:
        report.append("FINDING: Most features are not normally distributed.")
        report.append("")
        report.append("Implications:")
        report.append("- Use caution with parametric statistical tests")
        report.append("- Non-parametric methods may be more robust")
    else:
        report.append("FINDING: Most features approximate normal distribution.")
        report.append("")
        report.append("Implications:")
        report.append("- Parametric statistical methods appropriate")
        report.append("- Z-score and Mahalanobis distance valid approaches")
    
    report.append("")
    report.append("Recommendation:")
    report.append("Given the non-normal distributions, our use of:")
    report.append("- Isolation Forest (no normality assumption) ✓")
    report.append("- LOF (no normality assumption) ✓")
    report.append("- Z-Score (assumes normality) - use with caution")
    report.append("was appropriate.")
    
    return "\n".join(report)

def main():
    print_section_header("DISTRIBUTION ANALYSIS & NORMALITY TESTS")
    
    df = load_final_dataset()
    features = get_numeric_features(df)
    
    normality_df = test_normality(df, features)
    
    create_qq_plots(df, features)
    create_distribution_comparison(df, features)
    create_kde_plots(df, features)
    
    save_csv(normality_df, 'normality_tests.csv')
    
    report = generate_report(normality_df)
    save_report(report, 'distribution_analysis_report.txt')
    
    print_section_header("DISTRIBUTION ANALYSIS COMPLETE")
    normal_count = normality_df['Shapiro_Normal'].sum()
    print(f"✓ Tested {len(features)} features for normality")
    print(f"✓ {normal_count}/{len(features)} features are normally distributed")
    print(f"✓ Generated Q-Q plots, distribution comparisons, and KDE plots")

if __name__ == "__main__":
    main()
