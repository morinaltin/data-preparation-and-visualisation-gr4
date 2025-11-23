import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from utils import load_final_dataset, get_numeric_features, save_report, save_csv, print_section_header

def detect_outliers_lof(df, features, n_neighbors):
    X = df[features].values
    
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination='auto',
        novelty=False
    )
    
    print(f"  Computing LOF (n_neighbors={n_neighbors})...")
    predictions = lof.fit_predict(X)
    scores = lof.negative_outlier_factor_
    
    n_outliers = (predictions == -1).sum()
    pct_outliers = (n_outliers / len(df)) * 100
    
    print(f"  Detected: {n_outliers:,} outliers ({pct_outliers:.2f}%)")
    
    return predictions, scores

def experiment_neighbors(df, features, n_neighbors_list=[10, 20, 50]):
    print_section_header("LOF N_NEIGHBORS EXPERIMENTATION")
    
    results = {}
    
    for n_neighbors in n_neighbors_list:
        print(f"\nn_neighbors = {n_neighbors}:")
        print("-" * 50)
        
        predictions, scores = detect_outliers_lof(df, features, n_neighbors)
        
        n_outliers = (predictions == -1).sum()
        pct_outliers = (n_outliers / len(df)) * 100
        
        outlier_scores = scores[predictions == -1]
        inlier_scores = scores[predictions == 1]
        
        lof_values = -scores
        outlier_lof = lof_values[predictions == -1]
        
        print(f"  LOF score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Outlier LOF mean: {outlier_lof.mean():.4f}")
        print(f"  Outliers with LOF > 1.5: {(outlier_lof > 1.5).sum():,}")
        
        results[n_neighbors] = {
            'n_neighbors': n_neighbors,
            'predictions': predictions,
            'scores': scores,
            'lof_values': lof_values,
            'n_outliers': n_outliers,
            'percentage': pct_outliers,
            'outlier_lof_mean': outlier_lof.mean(),
            'high_lof_count': (outlier_lof > 1.5).sum()
        }
    
    return results

def visualize_neighbors_comparison(results):
    n_neighbors_list = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    ax = axes[0, 0]
    counts = [results[n]['n_outliers'] for n in n_neighbors_list]
    ax.bar([str(n) for n in n_neighbors_list], counts, color=colors)
    ax.set_xlabel('n_neighbors Parameter', fontsize=11)
    ax.set_ylabel('Number of Outliers', fontsize=11)
    ax.set_title('Outliers Detected by n_neighbors', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts)*0.02, f'{v:,}', ha='center', fontweight='bold')
    
    ax = axes[0, 1]
    percentages = [results[n]['percentage'] for n in n_neighbors_list]
    ax.bar([str(n) for n in n_neighbors_list], percentages, color=colors)
    ax.set_xlabel('n_neighbors Parameter', fontsize=11)
    ax.set_ylabel('Percentage of Data (%)', fontsize=11)
    ax.set_title('Percentage Flagged as Outliers', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(percentages):
        ax.text(i, v + max(percentages)*0.02, f'{v:.2f}%', ha='center', fontweight='bold')
    
    ax = axes[1, 0]
    lof_means = [results[n]['outlier_lof_mean'] for n in n_neighbors_list]
    ax.bar([str(n) for n in n_neighbors_list], lof_means, color=colors)
    ax.set_xlabel('n_neighbors Parameter', fontsize=11)
    ax.set_ylabel('Mean LOF Score', fontsize=11)
    ax.set_title('Average LOF Score for Outliers', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(lof_means):
        ax.text(i, v + max(lof_means)*0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    ax = axes[1, 1]
    for i, n in enumerate(n_neighbors_list):
        lof_vals = results[n]['lof_values']
        ax.hist(lof_vals, bins=50, alpha=0.5, label=f'n={n}', color=colors[i])
    ax.set_xlabel('LOF Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('LOF Score Distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axvline(x=1.5, color='red', linestyle='--', label='LOF=1.5 threshold')
    
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/lof_neighbors_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: lof_neighbors_comparison.png")
    plt.close()

def select_optimal_neighbors(results):
    print_section_header("N_NEIGHBORS SELECTION ANALYSIS")
    
    print("n_neighbors Comparison Summary:")
    print("-" * 80)
    print(f"{'n_neighbors':<15} {'Outliers':<12} {'Percentage':<12} {'Avg LOF':<12} {'Assessment'}")
    print("-" * 80)
    
    for n in sorted(results.keys()):
        r = results[n]
        
        if r['percentage'] < 1.0:
            assessment = "Too conservative"
        elif r['percentage'] > 5.0:
            assessment = "Too aggressive"
        else:
            assessment = "✓ Balanced"
        
        print(f"{n:<15} {r['n_outliers']:>10,}  {r['percentage']:>6.2f}%      {r['outlier_lof_mean']:>6.4f}      {assessment}")
    
    print("-" * 80)
    
    recommended = 20
    
    print(f"\n📌 SELECTED N_NEIGHBORS: {recommended}")
    print("\nJustification:")
    print("- Balanced between local and global outlier detection")
    print(f"- Detects {results[recommended]['n_outliers']:,} density-based anomalies")
    print("- Not too sensitive to local variations (n=10) or too global (n=50)")
    print("- Standard choice in literature for density-based outlier detection")
    
    return recommended

def generate_report(results, selected_n_neighbors, features):
    report = []
    report.append("LOF (Local Outlier Factor) Outlier Detection Analysis")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    report.append("Method Overview:")
    report.append("LOF detects outliers based on local density.")
    report.append("Points in low-density areas compared to neighbors are flagged as outliers.")
    report.append("LOF score > 1 indicates outlier (lower density than neighbors).")
    report.append("")
    report.append("Advantages:")
    report.append("- Density-based: Detects local anomalies within clusters")
    report.append("- No distribution assumptions")
    report.append("- Good for data with varying densities")
    report.append("")
    report.append(f"Features analyzed: {len(features)}")
    report.append("")
    report.append("n_neighbors Parameter Experimentation:")
    report.append("")
    
    for n in sorted(results.keys()):
        r = results[n]
        report.append(f"n_neighbors = {n}:")
        report.append(f"  Outliers found: {r['n_outliers']:,} ({r['percentage']:.2f}% of data)")
        report.append(f"  Average LOF score: {r['outlier_lof_mean']:.4f}")
        report.append("")
    
    report.append(f"Selected n_neighbors: {selected_n_neighbors}")
    report.append("")
    
    r = results[selected_n_neighbors]
    report.append("Justification:")
    report.append("- Balanced approach for local and global anomaly detection")
    report.append(f"- Detected {r['n_outliers']:,} outliers ({r['percentage']:.2f}%)")
    report.append("- Standard parameter choice in density-based outlier detection")
    report.append("- Avoids over-sensitivity to very local variations")
    report.append("")
    report.append("Key Findings:")
    report.append("- LOF detects points in sparse regions (low local density)")
    report.append("- Different from Isolation Forest (which uses tree isolation)")
    report.append("- Captures local context that univariate methods miss")
    
    return "\n".join(report)

def main():
    print_section_header("LOF (LOCAL OUTLIER FACTOR) OUTLIER DETECTION")
    
    df = load_final_dataset()
    features = get_numeric_features(df)
    
    print(f"\nAnalyzing {len(features)} numeric features with LOF")
    
    results = experiment_neighbors(df, features)
    visualize_neighbors_comparison(results)
    selected_n_neighbors = select_optimal_neighbors(results)
    
    selected_result = results[selected_n_neighbors]
    output_df = pd.DataFrame({
        'outlier_lof': selected_result['predictions'] == -1,
        'lof_score': selected_result['lof_values']
    })
    save_csv(output_df, 'outliers_lof_flags.csv')
    
    report = generate_report(results, selected_n_neighbors, features)
    save_report(report, 'outlier_lof_report.txt')
    
    print_section_header("LOF ANALYSIS COMPLETE")
    print(f"✓ Selected n_neighbors: {selected_n_neighbors}")
    print(f"✓ Outliers detected: {selected_result['n_outliers']:,}")
    print(f"✓ Percentage: {selected_result['percentage']:.2f}%")

if __name__ == "__main__":
    main()
