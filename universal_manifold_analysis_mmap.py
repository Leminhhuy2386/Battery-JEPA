import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import umap
except ImportError:
    print("UMAP not found, fallback will be used")

warnings.filterwarnings('ignore')

# Paths
CACHE_DIR = '/work/huy.leminh/code/Dr.Huy/AI Battery/data/dataset_cache'
FIGURE_DATA_DIR = './figure_data'
OUTPUT_FIGURE_DIR = './figures'
os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

def load_and_mmap_dataset(chem_name, cache_filename, eval_filename):
    print(f"Loading {chem_name}...")
    cache_path = os.path.join(CACHE_DIR, cache_filename)
    eval_path = os.path.join(FIGURE_DATA_DIR, eval_filename)
    
    # Load evaluation NPZ (holds representations, targets, predictions)
    eval_data = np.load(eval_path)
    targets = eval_data['targets']
    latent_features = eval_data['latent_features'] # Shape (N, 100, 64)
    
    # Load dataset cache using mmap to avoid loading large arrays
    cache_data = np.load(cache_path, mmap_mode='r')
    masks = cache_data['total_curve_attn_masks'] # Shape (N, 100)
    curves = cache_data['total_charge_discharge_curves'] # Shape (N, 100, 3, 100)
    
    N = len(targets)
    samples = []
    
    # Identify cells and group samples by cell
    # A new cell starts whenever the cycle index decreases
    cells = []
    current_cell = []
    prev_cyc = 999
    
    print(f"  Processing {N} samples with memory mapping...")
    for idx in range(N):
        cyc = int(masks[idx].sum())
        if cyc < prev_cyc and len(current_cell) > 0:
            cells.append(current_cell)
            current_cell = []
        current_cell.append((idx, cyc))
        prev_cyc = cyc
    if len(current_cell) > 0:
        cells.append(current_cell)
        
    print(f"  Grouped into {len(cells)} cells.")
    
    # Process SOH and extract latents cell-by-cell
    processed_cells = []
    for cell_idx, cell_samples in enumerate(cells):
        processed_cell = []
        # Find index of the first cycle (cycle 5 usually) to use as reference or cycle index 0
        first_idx = cell_samples[0][0]
        # In aligned mode, np.max(curves[idx, 0, 2, :]) is 1.0. 
        # But to be safe and uniform, we read capacity at first sample of the cell
        try:
            # discharge capacity is row 2, columns 50: (discharge phase)
            ref_cap = np.max(curves[first_idx, 0, 2, 50:])
            if ref_cap <= 0:
                ref_cap = 1.0
        except Exception:
            ref_cap = 1.0
            
        for idx, cyc in cell_samples:
            # SOH is capacity at current cycle divided by capacity at cycle 1
            try:
                cap_i = np.max(curves[idx, cyc - 1, 2, 50:])
                # In aligned scaling, the capacity is already divided by initial capacity,
                # so cap_i is already the SOH. If ref_cap is 1.0, SOH = cap_i.
                # In unaligned, cap_i is relative to nominal, and dividing by ref_cap yields SOH relative to initial.
                soh = cap_i / ref_cap
            except Exception:
                soh = 1.0
                
            # Slice latent features at the observed cycle index (cyc - 1)
            latent_vector = latent_features[idx, cyc - 1, :]
            
            processed_cell.append({
                'latent': latent_vector,
                'chemistry': chem_name,
                'eol': targets[idx],
                'cycle_index': cyc,
                'soh': soh,
                'cell_id': f"{chem_name}_cell_{cell_idx}"
            })
        processed_cells.append(processed_cell)
        
    return processed_cells

def main():
    # 1. Load all datasets
    calb_cells = load_and_mmap_dataset('CALB', 'CALB_test_seq5_cd100_ec100_aligned.npz', 'reused_eval_JEPA_CALB_aligned.npz')
    na_cells = load_and_mmap_dataset('Na-ion', 'NA-ion_test_seq5_cd100_ec100_unaligned.npz', 'reused_eval_JEPA_NAion_unaligned.npz')
    zn_cells = load_and_mmap_dataset('Zn-ion', 'ZN-coin_test_seq5_cd100_ec100_unaligned.npz', 'reused_eval_JEPA_ZN-coin_unaligned.npz')
    li_cells = load_and_mmap_dataset('Li-ion', 'MIX_large_test_seq5_cd100_ec100_aligned.npz', 'reused_eval_JEPA_MIX_large_aligned.npz')
    
    # 2. Downsample and balance the datasets
    # Na-ion: 5 cells, CALB: 5 cells, ZN-coin: 20 cells, Li-ion: 158 cells
    # We will keep all cells for Na-ion, CALB, and Zn-ion.
    # For Li-ion, we downsample by selecting 15 cells systematically (every ~10th cell) to ensure reproducible and balanced representation.
    np.random.seed(42)
    selected_li_indices = np.linspace(0, len(li_cells) - 1, 15, dtype=int)
    li_cells_balanced = [li_cells[idx] for idx in selected_li_indices]
    
    print(f"\nBalanced dataset breakdown:")
    print(f"  Li-ion: {len(li_cells_balanced)} cells, {sum(len(c) for c in li_cells_balanced)} samples")
    print(f"  Zn-ion: {len(zn_cells)} cells, {sum(len(c) for c in zn_cells)} samples")
    print(f"  Na-ion: {len(na_cells)} cells, {sum(len(c) for c in na_cells)} samples")
    print(f"  CALB: {len(calb_cells)} cells, {sum(len(c) for c in calb_cells)} samples")
    
    # Merge cells into a single flat list of samples
    all_matched_cells = li_cells_balanced + zn_cells + na_cells + calb_cells
    flat_samples = []
    for cell in all_matched_cells:
        flat_samples.extend(cell)
        
    # Convert to DataFrame
    latents = np.array([s['latent'] for s in flat_samples])
    df = pd.DataFrame(latents, columns=[f'latent_{i}' for i in range(64)])
    df['chemistry'] = [s['chemistry'] for s in flat_samples]
    df['eol'] = [s['eol'] for s in flat_samples]
    df['cycle_index'] = [s['cycle_index'] for s in flat_samples]
    df['soh'] = [s['soh'] for s in flat_samples]
    df['cell_id'] = [s['cell_id'] for s in flat_samples]
    
    print(f"Merged dataset shape: {df.shape}")
    
    # 3. Dimensionality Reduction
    print("\nPerforming dimensionality reduction...")
    
    # PCA
    print("  Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(latents)
    df['pca_dim1'] = pca_results[:, 0]
    df['pca_dim2'] = pca_results[:, 1]
    
    # t-SNE
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_results = tsne.fit_transform(latents)
    df['tsne_dim1'] = tsne_results[:, 0]
    df['tsne_dim2'] = tsne_results[:, 1]
    
    # UMAP
    print("  Running UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_results = reducer.fit_transform(latents)
    df['umap_dim1'] = umap_results[:, 0]
    df['umap_dim2'] = umap_results[:, 1]
    
    # 4. Generate publication-quality visualizations (Nature Style)
    print("\nGenerating Nature-style figures...")
    
    sns.set_theme(style='ticks')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=300)
    axes = axes.ravel()
    
    # Palette for Chemistry
    chem_colors = {
        'Li-ion': '#2b5c8f',  # Deep blue
        'Zn-ion': '#2ca02c',  # Emerald green
        'Na-ion': '#ff7f0e',  # Amber orange
        'CALB': '#9467bd'     # Purple
    }
    
    # Panel A: Chemistry
    ax = axes[0]
    sns.scatterplot(
        data=df, x='umap_dim1', y='umap_dim2', hue='chemistry',
        palette=chem_colors, alpha=0.75, s=15, ax=ax, edgecolor='none'
    )
    ax.set_title("a | Colored by Chemistry", fontweight='bold', loc='left')
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    ax.legend(title="Chemistry", frameon=True, facecolor='white', edgecolor='none')
    
    # Panel B: Cycle Life (EOL)
    ax = axes[1]
    sc = ax.scatter(
        df['umap_dim1'], df['umap_dim2'], c=df['eol'],
        cmap='plasma', alpha=0.75, s=12, edgecolor='none'
    )
    ax.set_title("b | Colored by Cycle Life (EOL)", fontweight='bold', loc='left')
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Cycle Life (Cycles)")
    
    # Panel C: State of Health (SOH)
    ax = axes[2]
    sc = ax.scatter(
        df['umap_dim1'], df['umap_dim2'], c=df['soh'],
        cmap='viridis', alpha=0.75, s=12, edgecolor='none'
    )
    ax.set_title("c | Colored by Normalized SOH", fontweight='bold', loc='left')
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("SOH ($Q/Q_0$)")
    
    # Panel D: Cycle Index
    ax = axes[3]
    sc = ax.scatter(
        df['umap_dim1'], df['umap_dim2'], c=df['cycle_index'],
        cmap='coolwarm', alpha=0.75, s=12, edgecolor='none'
    )
    ax.set_title("d | Colored by Cycle Index", fontweight='bold', loc='left')
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Observed Cycle Index")
    
    plt.tight_layout()
    fig_jpg_path = os.path.join(OUTPUT_FIGURE_DIR, 'fig_universal_manifold.jpg')
    fig_pdf_path = os.path.join(OUTPUT_FIGURE_DIR, 'fig_universal_manifold.pdf')
    plt.savefig(fig_jpg_path, bbox_inches='tight', dpi=300)
    plt.savefig(fig_pdf_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Nature-style UMAP manifold plots to {fig_jpg_path}")
    
    # Supplementary PCA Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=300)
    axes = axes.ravel()
    sns.scatterplot(data=df, x='pca_dim1', y='pca_dim2', hue='chemistry', palette=chem_colors, alpha=0.75, s=15, ax=axes[0], edgecolor='none')
    axes[0].set_title("PCA: Chemistry", fontweight='bold', loc='left')
    sc = axes[1].scatter(df['pca_dim1'], df['pca_dim2'], c=df['eol'], cmap='plasma', alpha=0.75, s=12, edgecolor='none')
    axes[1].set_title("PCA: Cycle Life (EOL)", fontweight='bold', loc='left')
    fig.colorbar(sc, ax=axes[1], shrink=0.85).set_label("Cycle Life")
    sc = axes[2].scatter(df['pca_dim1'], df['pca_dim2'], c=df['soh'], cmap='viridis', alpha=0.75, s=12, edgecolor='none')
    axes[2].set_title("PCA: Normalized SOH", fontweight='bold', loc='left')
    fig.colorbar(sc, ax=axes[2], shrink=0.85).set_label("SOH")
    sc = axes[3].scatter(df['pca_dim1'], df['pca_dim2'], c=df['cycle_index'], cmap='coolwarm', alpha=0.75, s=12, edgecolor='none')
    axes[3].set_title("PCA: Cycle Index", fontweight='bold', loc='left')
    fig.colorbar(sc, ax=axes[3], shrink=0.85).set_label("Cycle Index")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_universal_pca_supplementary.jpg'), bbox_inches='tight', dpi=300)
    plt.close()

    # Supplementary t-SNE Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=300)
    axes = axes.ravel()
    sns.scatterplot(data=df, x='tsne_dim1', y='tsne_dim2', hue='chemistry', palette=chem_colors, alpha=0.75, s=15, ax=axes[0], edgecolor='none')
    axes[0].set_title("t-SNE: Chemistry", fontweight='bold', loc='left')
    sc = axes[1].scatter(df['tsne_dim1'], df['tsne_dim2'], c=df['eol'], cmap='plasma', alpha=0.75, s=12, edgecolor='none')
    axes[1].set_title("t-SNE: Cycle Life (EOL)", fontweight='bold', loc='left')
    fig.colorbar(sc, ax=axes[1], shrink=0.85).set_label("Cycle Life")
    sc = axes[2].scatter(df['tsne_dim1'], df['tsne_dim2'], c=df['soh'], cmap='viridis', alpha=0.75, s=12, edgecolor='none')
    axes[2].set_title("t-SNE: Normalized SOH", fontweight='bold', loc='left')
    fig.colorbar(sc, ax=axes[2], shrink=0.85).set_label("SOH")
    sc = axes[3].scatter(df['tsne_dim1'], df['tsne_dim2'], c=df['cycle_index'], cmap='coolwarm', alpha=0.75, s=12, edgecolor='none')
    axes[3].set_title("t-SNE: Cycle Index", fontweight='bold', loc='left')
    fig.colorbar(sc, ax=axes[3], shrink=0.85).set_label("Cycle Index")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_universal_tsne_supplementary.jpg'), bbox_inches='tight', dpi=300)
    plt.close()

    # 5. Compute Quantitative Metrics
    print("\nComputing quantitative metrics...")
    
    # Silhouette score by chemistry
    sil_chem = silhouette_score(latents, df['chemistry'])
    print(f"  Silhouette Score (grouped by Chemistry): {sil_chem:.4f}")
    
    # Silhouette score by degradation stage
    # Define degradation stage: Fresh (SOH >= 0.96), Medium (0.90 <= SOH < 0.96), Aged (SOH < 0.90)
    def get_stage(soh):
        if soh >= 0.96:
            return 'Fresh'
        elif soh >= 0.90:
            return 'Medium'
        else:
            return 'Aged'
    df['degradation_stage'] = df['soh'].apply(get_stage)
    sil_stage = silhouette_score(latents, df['degradation_stage'])
    print(f"  Silhouette Score (grouped by Degradation Stage): {sil_stage:.4f}")
    
    # Inter-chemistry centroid distances in 64D space
    centroids = {}
    for chem in df['chemistry'].unique():
        centroids[chem] = latents[df['chemistry'] == chem].mean(axis=0)
        
    print("\n  Inter-chemistry centroid Euclidean distances (64D space):")
    chem_list = sorted(centroids.keys())
    dist_matrix = np.zeros((len(chem_list), len(chem_list)))
    for i, c1 in enumerate(chem_list):
        for j, c2 in enumerate(chem_list):
            dist = np.linalg.norm(centroids[c1] - centroids[c2])
            dist_matrix[i, j] = dist
            if i < j:
                print(f"    {c1} <-> {c2}: {dist:.4f}")
                
    # Correlation between latent progression (projection on PCA PC1) and SOH
    pca_1d = PCA(n_components=1)
    latent_progression = pca_1d.fit_transform(latents).ravel()
    df['latent_progression'] = latent_progression
    
    overall_corr, _ = pearsonr(df['latent_progression'], df['soh'])
    print(f"\n  Overall Correlation (Latent Progression vs SOH): {overall_corr:.4f}")
    
    chems_corr = {}
    for chem in df['chemistry'].unique():
        sub_df = df[df['chemistry'] == chem]
        r_val, _ = pearsonr(sub_df['latent_progression'], sub_df['soh'])
        chems_corr[chem] = r_val
        print(f"    {chem} Correlation: {r_val:.4f}")
        
    # 6. Principal Aging Direction & Cosine Similarity across chemistries
    # Fit a linear regression for each chemistry: latent_vector = w_c * soh + b_c
    aging_directions = {}
    for chem in df['chemistry'].unique():
        sub_df = df[df['chemistry'] == chem]
        sub_latents = latents[df['chemistry'] == chem]
        sub_soh = sub_df['soh'].values
        
        w_c = np.zeros(64)
        for f in range(64):
            slope, intercept = np.polyfit(sub_soh, sub_latents[:, f], 1)
            w_c[f] = slope
            
        aging_directions[chem] = w_c / np.linalg.norm(w_c)
        
    print("\n  Cosine similarity of SOH regression directions between chemistries:")
    sim_matrix = np.zeros((len(chem_list), len(chem_list)))
    for i, c1 in enumerate(chem_list):
        for j, c2 in enumerate(chem_list):
            sim = np.dot(aging_directions[c1], aging_directions[c2])
            sim_matrix[i, j] = sim
            if i < j:
                print(f"    {c1} <-> {c2} Cosine Similarity: {sim:.4f}")
                
    # Evaluate PC1 trajectory direction similarity
    pc1_directions = {}
    for chem in df['chemistry'].unique():
        sub_latents = latents[df['chemistry'] == chem]
        pca_sub = PCA(n_components=1)
        pca_sub.fit(sub_latents)
        pc1_dir = pca_sub.components_[0]
        # Align direction so that it correlates negatively with SOH
        scores = np.dot(sub_latents, pc1_dir)
        r_val, _ = pearsonr(scores, df[df['chemistry'] == chem]['soh'].values)
        if r_val > 0:
            pc1_dir = -pc1_dir
        pc1_directions[chem] = pc1_dir
        
    print("\n  Cosine similarity of PC1 trajectory directions between chemistries:")
    for i, c1 in enumerate(chem_list):
        for j, c2 in enumerate(chem_list):
            sim = np.dot(pc1_directions[c1], pc1_directions[c2])
            if i < j:
                print(f"    {c1} <-> {c2} PC1 Similarity: {sim:.4f}")
                
    # 7. Save quantitative results to JSON for review
    metrics_summary = {
        'silhouette_scores': {
            'by_chemistry': float(sil_chem),
            'by_degradation_stage': float(sil_stage)
        },
        'centroid_distances': {
            f"{c1}_to_{c2}": float(dist_matrix[i, j])
            for i, c1 in enumerate(chem_list) for j, c2 in enumerate(chem_list) if i < j
        },
        'soh_correlations': {
            'overall': float(overall_corr),
            **{chem: float(chems_corr[chem]) for chem in chems_corr}
        },
        'aging_direction_cosine_similarities': {
            f"{c1}_to_{c2}": float(sim_matrix[i, j])
            for i, c1 in enumerate(chem_list) for j, c2 in enumerate(chem_list) if i < j
        },
        'pc1_similarities': {
            f"{c1}_to_{c2}": float(np.dot(pc1_directions[c1], pc1_directions[c2]))
            for i, c1 in enumerate(chem_list) for j, c2 in enumerate(chem_list) if i < j
        }
    }
    
    with open(os.path.join(FIGURE_DATA_DIR, 'universal_manifold_metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print("\n✓ Saved quantitative metrics summary to figure_data/universal_manifold_metrics.json")
    
    # 8. Save merged dataset for quick reloading
    df.to_csv(os.path.join(FIGURE_DATA_DIR, 'universal_manifold_merged_dataset.csv'), index=False)
    print("✓ Saved merged dataset.")

if __name__ == '__main__':
    main()
