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

