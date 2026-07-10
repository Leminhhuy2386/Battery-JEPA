import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UMAP is imported
try:
    import umap
except ImportError:
    print("UMAP not found, fallback will be used")

warnings.filterwarnings('ignore')

# Paths
DATASET_PATH = './dataset'
FIGURE_DATA_DIR = './figure_data'
OUTPUT_FIGURE_DIR = './figures'
os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

# Helper functions to load raw labels and files
def get_eol_labels(prefix):
    if prefix == 'MICH':
        path = f'{DATASET_PATH}/Life labels/total_MICH_labels.json'
    elif prefix.startswith('Tongji'):
        path = f'{DATASET_PATH}/Life labels/Tongji_labels.json'
    else:
        path = f'{DATASET_PATH}/Life labels/{prefix}_labels.json'
    with open(path) as f:
        return json.load(f)

def get_cell_capacity_data(file_name):
    prefix = file_name.split('_')[0]
    folder = {
        'MATR': 'MATR',
        'HUST': 'HUST',
        'SNL': 'SNL',
        'CALCE': 'CALCE',
        'HNEI': 'HNEI',
        'RWTH': 'RWTH',
        'UL-PUR': 'UL_PUR',
        'BIT2': 'BIT2',
        'Tongji': 'Tongji',
        'Stanford': 'Stanford',
        'ISU-ILCC': 'ISU_ILCC',
        'XJTU': 'XJTU',
        'ZN-coin': 'ZN-coin',
        'CALB': 'CALB',
        'NA-ion': 'NA-ion'
    }.get(prefix, 'MICH')
    
    if prefix == 'MICH' or prefix == 'SMICH':
        path = f'{DATASET_PATH}/total_MICH/{file_name}'
        if not os.path.exists(path):
            path = f'{DATASET_PATH}/MICH/{file_name}'
            if not os.path.exists(path):
                path = f'{DATASET_PATH}/MICH_EXP/{file_name[1:] if file_name.startswith("S") else file_name}'
    else:
        path = f'{DATASET_PATH}/{folder}/{file_name}'
        
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return len(data['cycle_data']), data

def get_soh_capacity_at_cycle(data, cycle_num):
    cycle_data = data['cycle_data'][cycle_num - 1]
    caps = np.array(cycle_data['discharge_capacity_in_Ah'])
    return np.max(caps) if len(caps) > 0 else 0.0

def load_and_match_dataset(chem_name, test_files, npz_path):
    print(f"Loading and matching {chem_name}...")
    npz_data = np.load(npz_path)
    npz_targets = npz_data['targets']
    npz_reps = npz_data['latent_features']
    
    samples = []
    sample_idx = 0
    
    for file_name in test_files:
        prefix = file_name.split('_')[0]
        try:
            labels = get_eol_labels(prefix)
        except Exception:
            continue
            
        eol = labels.get(file_name, None)
        if eol is None or eol <= 100:
            continue
            
        try:
            valid_cycle_number, cell_data = get_cell_capacity_data(file_name)
        except Exception:
            continue
            
        try:
            initial_cap = get_soh_capacity_at_cycle(cell_data, 1)
        except Exception:
            initial_cap = 0.0
            
        if initial_cap <= 0:
            continue
            
        cell_samples = []
        for i in range(5, 101):
            if i >= eol or i > valid_cycle_number:
                break
                
            try:
                cap_i = get_soh_capacity_at_cycle(cell_data, i)
            except Exception:
                cap_i = 0.0
                
            soh = cap_i / initial_cap
            
            # Extract 64D representation at cycle index i
            rep_64d = npz_reps[sample_idx, i - 1, :]
            
            cell_samples.append({
                'latent': rep_64d,
                'chemistry': chem_name,
                'eol': eol,
                'cycle_index': i,
                'soh': soh,
                'cell_id': file_name
            })
            sample_idx += 1
            
        if len(cell_samples) > 0:
            samples.append(cell_samples)
            
    print(f"  Total cells matched: {len(samples)}, Total samples: {sum(len(c) for c in samples)}")
    return samples

def main():
    # Load test split file lists from split_recorder
    from data_provider.data_split_recorder import split_recorder
    
    # 1. Load and match all datasets
    calb_cells = load_and_match_dataset('CALB', split_recorder.CALB_test_files, f"{FIGURE_DATA_DIR}/reused_eval_JEPA_CALB_aligned.npz")
    na_cells = load_and_match_dataset('Na-ion', split_recorder.NAion_2021_test_files, f"{FIGURE_DATA_DIR}/reused_eval_JEPA_NAion_unaligned.npz")
    zn_cells = load_and_match_dataset('Zn-ion', split_recorder.ZNcoin_test_files, f"{FIGURE_DATA_DIR}/reused_eval_JEPA_ZN-coin_unaligned.npz")
    li_cells = load_and_match_dataset('Li-ion', split_recorder.MIX_large_test_files, f"{FIGURE_DATA_DIR}/reused_eval_JEPA_MIX_large_aligned.npz")
    
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
    
    # Set style
    sns.set_theme(style='ticks')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=300)
    axes = axes.ravel()
    
    # Palette for Chemistry
    chem_colors = {
        'Li-ion': '#2b5c8f',  # Elegant deep blue
        'Zn-ion': '#2ca02c',  # Emerald green
        'Na-ion': '#ff7f0e',  # Amber orange
        'CALB': '#9467bd'     # Pouch cell purple
    }
    
    # Panel A: Chemistry
    ax = axes[0]
    sns.scatterplot(
        data=df, x='umap_dim1', y='umap_dim2', hue='chemistry',
        palette=chem_colors, alpha=0.75, s=15, ax=ax, edgecolor='none'
    )
    ax.set_title("A. Colored by Chemistry", fontweight='bold', loc='left')
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    ax.legend(title="Chemistry", frameon=True, facecolor='white', edgecolor='none')
    
    # Panel B: Cycle Life (EOL)
    ax = axes[1]
    sc = ax.scatter(
        df['umap_dim1'], df['umap_dim2'], c=df['eol'],
        cmap='plasma', alpha=0.75, s=12, edgecolor='none'
    )
    ax.set_title("B. Colored by Cycle Life (EOL)", fontweight='bold', loc='left')
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
    ax.set_title("C. Colored by Normalized SOH", fontweight='bold', loc='left')
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
    ax.set_title("D. Colored by Cycle Index", fontweight='bold', loc='left')
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
    print(f"✓ Saved Nature-style UMAP manifold plots to {fig_jpg_path} and {fig_pdf_path}")
    
    # Re-run for PCA & t-SNE supplementary grids
    # PCA Plot
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
    
    # 5. Compute Quantitative Metrics
    print("\nComputing quantitative metrics...")
    
    # Silhouette score by chemistry
    sil_chem = silhouette_score(latents, df['chemistry'])
    print(f"  Silhouette Score (grouped by Chemistry): {sil_chem:.4f}")
    
    # Silhouette score by degradation stage
    # Define degradation stage: Fresh (SOH >= 0.95), Medium (0.90 <= SOH < 0.95), Aged (SOH < 0.90)
    def get_stage(soh):
        if soh >= 0.95:
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
    # We fit a PCA model on all latents to get the main axis of variation
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
    # Fit a linear regression for each chemistry in the 64D latent space:
    # latent_vector = w_c * soh + b_c
    # w_c represents the principal aging direction vector.
    aging_directions = {}
    for chem in df['chemistry'].unique():
        sub_df = df[df['chemistry'] == chem]
        sub_latents = latents[df['chemistry'] == chem]
        sub_soh = sub_df['soh'].values
        
        # Linear regression slope for each of the 64 dimensions against SOH
        w_c = np.zeros(64)
        for f in range(64):
            slope, intercept = np.polyfit(sub_soh, sub_latents[:, f], 1)
            w_c[f] = slope
            
        # Normalize to unit vector
        aging_directions[chem] = w_c / np.linalg.norm(w_c)
        
    print("\n  Cosine similarity of aging directions between chemistries:")
    sim_matrix = np.zeros((len(chem_list), len(chem_list)))
    for i, c1 in enumerate(chem_list):
        for j, c2 in enumerate(chem_list):
            sim = np.dot(aging_directions[c1], aging_directions[c2])
            sim_matrix[i, j] = sim
            if i < j:
                print(f"    {c1} <-> {c2} Cosine Similarity: {sim:.4f}")
                
    # Also evaluate PC1 direction similarity (PCA on each chemistry's trajectories)
    pc1_directions = {}
    for chem in df['chemistry'].unique():
        sub_latents = latents[df['chemistry'] == chem]
        pca_sub = PCA(n_components=1)
        pca_sub.fit(sub_latents)
        pc1_dir = pca_sub.components_[0]
        # Align direction so that it correlates negatively with SOH (increasing aging)
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
                
    # 7. Save quantitative results to JSON for scientific review
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
    print("✓ Saved merged dataset for visualization reproduction.")

if __name__ == '__main__':
    main()
