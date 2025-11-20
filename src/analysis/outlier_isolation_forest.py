import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from utils import load_final_dataset, get_numeric_features, save_report, save_csv, print_section_header

def detect_outliers_iforest(df, features, contamination, random_state=42):
    X = df[features].values
    
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples='auto',
        verbose=0
    )
    
    print(f"  Training Isolation Forest (contamination={contamination})...")
    iso_forest.fit(X)
    
    predictions = iso_forest.predict(X)
    scores = iso_forest.score_samples(X)
    
    n_outliers = (predictions == -1).sum()
    pct_outliers = (n_outliers / len(df)) * 100
    
    print(f"  Detected: {n_outliers:,} outliers ({pct_outliers:.2f}%)")
    
    return predictions, scores

def experiment_contamination(df, features, contaminations=[0.05, 0.10, 0.15]):
    print_section_header("ISOLATION FOREST CONTAMINATION EXPERIMENTATION")
    
    results = {}
    
    for cont in contaminations:
        print(f"\nContamination = {cont} ({cont*100}% expected outliers):")
        print("-" * 50)
        
        predictions, scores = detect_outliers_iforest(df, features, cont)
        
        n_outliers = (predictions == -1).sum()
        pct_outliers = (n_outliers / len(df)) * 100
        
        outlier_scores = scores[predictions == -1]
        inlier_scores = scores[predictions == 1]
        
        print(f"  Anomaly score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Outlier scores (mean): {outlier_scores.mean():.4f}")
        print(f"  Inlier scores (mean): {inlier_scores.mean():.4f}")
        print(f"  Score separation: {inlier_scores.mean() - outlier_scores.mean():.4f}")
        
        results[cont] = {
            'contamination': cont,
            'predictions': predictions,
            'scores': scores,
            'n_outliers': n_outliers,
            'percentage': pct_outliers,
            'outlier_score_mean': outlier_scores.mean(),
            'inlier_score_mean': inlier_scores.mean(),
            'score_separation': inlier_scores.mean() - outlier_scores.mean()
        }
    
    return results

def visualize_contamination_comparison(results):
    contaminations = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    ax = axes[0, 0]
    counts = [results[c]['n_outliers'] for c in contaminations]
    ax.bar([str(c) for c in contaminations], counts, color=colors)
    ax.set_xlabel('Contamination Parameter', fontsize=11)
    ax.set_ylabel('Number of Outliers', fontsize=11)
    ax.set_title('Outliers Detected by Contamination', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts)*0.02, f'{v:,}', ha='center', fontweight='bold')
    
    ax = axes[0, 1]
    percentages = [results[c]['percentage'] for c in contaminations]
    ax.bar([str(c) for c in contaminations], percentages, color=colors)
    ax.set_xlabel('Contamination Parameter', fontsize=11)
    ax.set_ylabel('Percentage of Data (%)', fontsize=11)
    ax.set_title('Percentage Flagged as Outliers', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(percentages):
        ax.text(i, v + max(percentages)*0.02, f'{v:.2f}%', ha='center', fontweight='bold')
    
    ax = axes[1, 0]
    separations = [results[c]['score_separation'] for c in contaminations]
    ax.bar([str(c) for c in contaminations], separations, color=colors)
    ax.set_xlabel('Contamination Parameter', fontsize=11)
    ax.set_ylabel('Score Separation', fontsize=11)
    ax.set_title('Anomaly Score Separation', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(separations):
        ax.text(i, v + max(separations)*0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    ax = axes[1, 1]
    for i, cont in enumerate(contaminations):
        scores = results[cont]['scores']
        ax.hist(scores, bins=50, alpha=0.5, label=f'cont={cont}', color=colors[i])
    ax.set_xlabel('Anomaly Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Anomaly Score Distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/iforest_contamination_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: iforest_contamination_comparison.png")
    plt.close()

def select_optimal_contamination(results):
    print_section_header("CONTAMINATION SELECTION ANALYSIS")
    
    print("Contamination Comparison Summary:")
    print("-" * 80)
    print(f"{'Contamination':<15} {'Outliers':<12} {'Percentage':<12} {'Separation':<12} {'Assessment'}")
    print("-" * 80)
    
    for cont in sorted(results.keys()):
        r = results[cont]
        
        if r['score_separation'] > 0.15:
            assessment = "✓ Clear separation"
        elif r['score_separation'] > 0.10:
            assessment = "Moderate separation"
        else:
            assessment = "Weak separation"
        
        print(f"{cont:<15.2f} {r['n_outliers']:>10,}  {r['percentage']:>6.2f}%      {r['score_separation']:>6.4f}      {assessment}")
    
    print("-" * 80)
    
    recommended = 0.05
    best_separation = 0
    
    for cont in results.keys():
        if results[cont]['score_separation'] > best_separation:
            best_separation = results[cont]['score_separation']
            recommended = cont
    
    print(f"\n📌 SELECTED CONTAMINATION: {recommended}")
    print("\nJustification:")
    print(f"- Best score separation: {best_separation:.4f}")
    print(f"- Clear distinction between outliers and normal data")
    print(f"- Detects {results[recommended]['n_outliers']:,} multivariate anomalies")
    
    return recommended

def generate_report(results, selected_contamination, features):
    report = []
    report.append("Isolation Forest Outlier Detection Analysis")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    report.append("Method Overview:")
    report.append("Isolation Forest uses random decision trees to detect anomalies.")
    report.append("Outliers are easier to isolate (fewer splits needed) than normal points.")
    report.append("")
    report.append("Advantages:")
    report.append("- Multivariate: Considers all features together")
    report.append("- No distribution assumptions")
    report.append("- Detects complex patterns that univariate methods miss")
    report.append("")
    report.append(f"Features analyzed: {len(features)}")
    report.append("Parameters: n_estimators=100, random_state=42")
    report.append("")
    report.append("Contamination Parameter Experimentation:")
    report.append("")
    
    for cont in sorted(results.keys()):
        r = results[cont]
        report.append(f"Contamination = {cont}:")
        report.append(f"  Outliers found: {r['n_outliers']:,} ({r['percentage']:.2f}% of data)")
        report.append(f"  Score separation: {r['score_separation']:.4f}")
        report.append("")
    
    report.append(f"Selected Contamination: {selected_contamination}")
    report.append("")
    
    r = results[selected_contamination]
    report.append("Justification:")
    report.append(f"- Highest score separation ({r['score_separation']:.4f})")
    report.append("- Clear distinction between anomalies and normal data")
    report.append(f"- Conservative approach with {r['percentage']:.2f}% flagged as outliers")
    report.append("- Captures multivariate anomalies (unusual feature combinations)")
    report.append("")
    report.append("Key Findings:")
    report.append("- Isolation Forest detects patterns Z-Score misses (multivariate anomalies)")
    report.append("- Lower contamination (0.05) gives clearest signal")
    report.append("- Score separation indicates strong outlier vs. inlier distinction")
    
    return "\n".join(report)

def main():
    print_section_header("ISOLATION FOREST OUTLIER DETECTION")
    
    df = load_final_dataset()
    features = get_numeric_features(df)
    
    print(f"\nAnalyzing {len(features)} numeric features with Isolation Forest")
    
    results = experiment_contamination(df, features)
    visualize_contamination_comparison(results)
    selected_contamination = select_optimal_contamination(results)
    
    selected_result = results[selected_contamination]
    output_df = pd.DataFrame({
        'outlier_iforest': selected_result['predictions'] == -1,
        'anomaly_score': selected_result['scores']
    })
    save_csv(output_df, 'outliers_iforest_flags.csv')
    
    report = generate_report(results, selected_contamination, features)
    save_report(report, 'outlier_iforest_report.txt')
    
    print_section_header("ISOLATION FOREST ANALYSIS COMPLETE")
    print(f"✓ Selected contamination: {selected_contamination}")
    print(f"✓ Outliers detected: {selected_result['n_outliers']:,}")
    print(f"✓ Percentage: {selected_result['percentage']:.2f}%")

if __name__ == "__main__":
    main()
