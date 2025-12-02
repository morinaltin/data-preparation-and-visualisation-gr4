import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import load_final_dataset, get_numeric_features, save_report, save_csv, print_section_header

def standardize_data(df, features):
    print_section_header("DATA STANDARDIZATION FOR PCA")
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=df.index)
    
    print(f"✓ Standardized {len(features)} features (mean=0, std=1)")
    print(f"  Original range example (Global_active_power): [{df['Global_active_power'].min():.2f}, {df['Global_active_power'].max():.2f}]")
    print(f"  Scaled range: [{scaled_df['Global_active_power'].min():.2f}, {scaled_df['Global_active_power'].max():.2f}]")
    
    return scaled_data, scaler

def perform_pca(scaled_data, n_components=None):
    print_section_header("PRINCIPAL COMPONENT ANALYSIS")
    
    if n_components is None:
        n_components = min(scaled_data.shape[1], 7)
    
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(scaled_data)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"Number of components: {n_components}")
    print(f"\nExplained variance by component:")
    for i, (var, cum) in enumerate(zip(explained_variance, cumulative_variance), 1):
        print(f"  PC{i}: {var*100:.2f}% (cumulative: {cum*100:.2f}%)")
    
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\nComponents needed for 95% variance: {n_95}")
    
    return pca, pca_components, explained_variance, cumulative_variance

def create_scree_plot(explained_variance, cumulative_variance):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    n_components = len(explained_variance)
    components = range(1, n_components + 1)
    
    ax1.bar(components, explained_variance * 100, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Variance Explained (%)', fontsize=12)
    ax1.set_title('Scree Plot - Variance per Component', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks(components)
    
    for i, v in enumerate(explained_variance * 100, 1):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    ax2.plot(components, cumulative_variance * 100, marker='o', linewidth=2, markersize=8, color='darkgreen')
    ax2.axhline(y=95, color='red', linestyle='--', label='95% threshold')
    ax2.fill_between(components, cumulative_variance * 100, alpha=0.3, color='green')
    ax2.set_xlabel('Principal Component', fontsize=12)
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xticks(components)
    ax2.legend()
    
    for i, v in enumerate(cumulative_variance * 100, 1):
        ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/pca_scree_plot.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: pca_scree_plot.png")
    plt.close()

def create_pca_scatter(pca_components, df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.scatter(pca_components[:, 0], pca_components[:, 1], 
                alpha=0.3, s=1, c='steelblue')
    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.set_title('PCA: First Two Principal Components', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    if pca_components.shape[1] >= 3:
        ax2.scatter(pca_components[:, 0], pca_components[:, 2], 
                    alpha=0.3, s=1, c='darkgreen')
        ax2.set_xlabel('PC1', fontsize=12)
        ax2.set_ylabel('PC3', fontsize=12)
        ax2.set_title('PCA: PC1 vs PC3', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/pca_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: pca_scatter.png")
    plt.close()

def create_component_heatmap(pca, features):
    components_df = pd.DataFrame(
        pca.components_,
        columns=features,
        index=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(components_df, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, linewidths=1, cbar_kws={'label': 'Component Loading'})
    plt.title('PCA Component Loadings', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Original Features', fontsize=12)
    plt.ylabel('Principal Components', fontsize=12)
    plt.tight_layout()
    plt.savefig('../../outputs/phase2/pca_component_loadings.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: pca_component_loadings.png")
    plt.close()
    
    return components_df

def generate_report(pca, explained_variance, cumulative_variance, components_df, features):
    report = []
    report.append("Principal Component Analysis (PCA) Report")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    report.append("PCA reduces data dimensions while preserving variance.")
    report.append("It transforms correlated features into uncorrelated principal components.")
    report.append("")
    report.append("Why PCA?")
    report.append("- Reduces 7 features to 2-3 components for visualization")
    report.append("- Removes redundancy (we found 0.999 correlation between power and intensity)")
    report.append("- Captures most important patterns in data")
    report.append("")
    report.append("Explained Variance:")
    report.append("")
    for i, (var, cum) in enumerate(zip(explained_variance, cumulative_variance), 1):
        report.append(f"PC{i}: {var*100:.2f}% (cumulative: {cum*100:.2f}%)")
    report.append("")
    
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    report.append(f"Components for 95% variance: {n_95}")
    report.append("")
    report.append("Component Interpretation:")
    report.append("")
    
    for i in range(min(3, pca.n_components_)):
        report.append(f"PC{i+1} (explained variance: {explained_variance[i]*100:.2f}%):")
        loadings = components_df.iloc[i].abs().sort_values(ascending=False)
        top_features = loadings.head(3)
        for feat, loading in top_features.items():
            report.append(f"  - {feat}: {components_df.iloc[i][feat]:.3f}")
        report.append("")
    
    report.append("Key Findings:")
    report.append(f"- First 2 components capture {cumulative_variance[1]*100:.1f}% of variance")
    report.append("- Data can be visualized in 2D with minimal information loss")
    report.append("- PC1 likely represents overall power consumption")
    report.append("- PC2 likely represents power distribution patterns")
    report.append("")
    report.append("Implications:")
    report.append("- Dimensionality reduction successful")
    report.append("- 2D visualization preserves most patterns")
    report.append("- Confirms feature redundancy (high correlation)")
    
    return "\n".join(report)

def main():
    print_section_header("PCA - PRINCIPAL COMPONENT ANALYSIS")
    
    df = load_final_dataset()
    features = get_numeric_features(df)
    
    features = [f for f in features if f != 'Sub_metering_1']
    
    scaled_data, scaler = standardize_data(df, features)
    
    pca, pca_components, explained_variance, cumulative_variance = perform_pca(scaled_data)
    
    create_scree_plot(explained_variance, cumulative_variance)
    create_pca_scatter(pca_components, df)
    components_df = create_component_heatmap(pca, features)
    
    pca_df = pd.DataFrame(
        pca_components[:, :3],
        columns=['PC1', 'PC2', 'PC3']
    )
    save_csv(pca_df, 'pca_components.csv')
    save_csv(components_df, 'pca_loadings.csv')
    
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
        'Explained_Variance': explained_variance * 100,
        'Cumulative_Variance': cumulative_variance * 100
    })
    save_csv(variance_df, 'pca_variance.csv')
    
    report = generate_report(pca, explained_variance, cumulative_variance, components_df, features)
    save_report(report, 'pca_analysis_report.txt')
    
    print_section_header("PCA ANALYSIS COMPLETE")
    print(f"✓ Reduced {len(features)} features to {pca.n_components_} components")
    print(f"✓ First 2 components explain {cumulative_variance[1]*100:.1f}% of variance")
    print(f"✓ Generated scree plot, scatter plots, and component loadings")

if __name__ == "__main__":
    main()
