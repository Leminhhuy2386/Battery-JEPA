import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Paths
CACHE_DIR = '/work/huy.leminh/code/Dr.Huy/AI Battery/data/dataset_cache'
FIGURE_DATA_DIR = './figure_data'
OUTPUT_FIGURE_DIR = './figures'
os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

# Helvetica/Arial font style for publication-ready figures
plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'grid.alpha': 0.3,
})

# Colors for the 4 chemistries
CHEM_COLORS = {
    'Li-ion': '#2b5c8f',  # Elegant deep blue
    'Zn-ion': '#2ca02c',  # Emerald green
    'Na-ion': '#ff7f0e',  # Amber orange
    'CALB': '#9467bd'     # Pouch cell purple
}

def compute_physical_quantities(curve):
    v_dis = curve[0, 50:]
    i_dis = curve[1, 50:]
    q_dis = curve[2, 50:]
    
    # 1. Capacity
    capacity = np.max(q_dis)
    
    # 2 & 3. dQ/dV Peak and Plateau Voltage
    dq = np.diff(q_dis)
    dv = np.diff(v_dis)
    dq_dv = np.abs(dq / (dv + 1e-8))
    dq_dv_smooth = gaussian_filter1d(dq_dv, sigma=1.5)
    
    peak_idx = np.argmax(dq_dv_smooth)
    dq_dv_peak = dq_dv_smooth[peak_idx]
    plateau_voltage = v_dis[peak_idx]
    
    # 4. Internal Resistance Proxy
    v_drop = np.abs(curve[0, 49] - curve[0, 50])
    current_applied = np.abs(curve[1, 50])
    resistance = v_drop / (current_applied + 1e-8)
    
    return capacity, plateau_voltage, dq_dv_peak, resistance

def load_and_mmap_dataset(chem_name, cache_filename, eval_filename):
    print(f"Loading and processing {chem_name}...")
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
    cells = []
    current_cell = []
    prev_cyc = 999
    
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
    
    for cell_idx, cell_samples in enumerate(cells):
        first_idx = cell_samples[0][0]
        try:
            ref_cap = np.max(curves[first_idx, 0, 2, 50:])
            if ref_cap <= 0:
                ref_cap = 1.0
        except Exception:
            ref_cap = 1.0
            
        for idx, cyc in cell_samples:
            try:
                cap_i = np.max(curves[idx, cyc - 1, 2, 50:])
                soh = cap_i / ref_cap
            except Exception:
                soh = 1.0
                
            latent_vector = latent_features[idx, cyc - 1, :]
            
            try:
                curve = curves[idx, cyc - 1]
                capacity, plateau_voltage, dq_dv_peak, resistance = compute_physical_quantities(curve)
            except Exception:
                capacity, plateau_voltage, dq_dv_peak, resistance = 0.0, 0.0, 0.0, 0.0
            
            samples.append({
                'latent': latent_vector,
                'chemistry': chem_name,
                'eol': targets[idx],
                'cycle_index': cyc,
                'soh': soh,
                'capacity': capacity,
                'plateau_voltage': plateau_voltage,
                'dq_dv_peak': dq_dv_peak,
                'resistance': resistance,
                'cell_id': f"{chem_name}_cell_{cell_idx}"
            })
            
    print(f"  Processed {len(samples)} samples.")
    return samples

def main():
    # 1. Load all datasets
    calb_samples = load_and_mmap_dataset('CALB', 'CALB_test_seq5_cd100_ec100_aligned.npz', 'reused_eval_JEPA_CALB_aligned.npz')
    na_samples = load_and_mmap_dataset('Na-ion', 'NA-ion_test_seq5_cd100_ec100_unaligned.npz', 'reused_eval_JEPA_NAion_unaligned.npz')
    zn_samples = load_and_mmap_dataset('Zn-ion', 'ZN-coin_test_seq5_cd100_ec100_unaligned.npz', 'reused_eval_JEPA_ZN-coin_unaligned.npz')
    li_samples = load_and_mmap_dataset('Li-ion', 'MIX_large_test_seq5_cd100_ec100_aligned.npz', 'reused_eval_JEPA_MIX_large_aligned.npz')
    
    all_samples = li_samples + zn_samples + na_samples + calb_samples
    
    # Create main DataFrame
    latents = np.array([s['latent'] for s in all_samples])
    df = pd.DataFrame(latents, columns=[f'latent_{i}' for i in range(64)])
    df['chemistry'] = [s['chemistry'] for s in all_samples]
    df['eol'] = [s['eol'] for s in all_samples]
    df['cycle_index'] = [s['cycle_index'] for s in all_samples]
    df['soh'] = [s['soh'] for s in all_samples]
    df['capacity'] = [s['capacity'] for s in all_samples]
    df['plateau_voltage'] = [s['plateau_voltage'] for s in all_samples]
    df['dq_dv_peak'] = [s['dq_dv_peak'] for s in all_samples]
    df['resistance'] = [s['resistance'] for s in all_samples]
    df['cell_id'] = [s['cell_id'] for s in all_samples]
    
    print(f"\nTotal merged dataset shape: {df.shape}")
    
    # Save processed dataset
    df.to_csv(os.path.join(FIGURE_DATA_DIR, 'universal_physical_dataset.csv'), index=False)
    print("✓ Saved universal_physical_dataset.csv")
    
    # 2. Compute Correlations and p-values
    print("\nComputing correlations and statistical significance...")
    chemistries = ['Li-ion', 'Zn-ion', 'Na-ion', 'CALB']
    metrics = ['Capacity', 'Plateau Voltage', 'dQ/dV Peak', 'Resistance']
    metric_cols = ['capacity', 'plateau_voltage', 'dq_dv_peak', 'resistance']
    
    corr_results = []
    
    for chem in chemistries:
        chem_df = df[df['chemistry'] == chem]
        print(f"  Chemistry: {chem} | Samples: {len(chem_df)}")
        for dim in range(64):
            latent_vals = chem_df[f'latent_{dim}'].values
            for m_name, m_col in zip(metrics, metric_cols):
                phys_vals = chem_df[m_col].values
                # Filter out NaNs if any
                valid_mask = ~np.isnan(latent_vals) & ~np.isnan(phys_vals)
                if valid_mask.sum() > 5:
                    r_val, p_val = pearsonr(latent_vals[valid_mask], phys_vals[valid_mask])
                else:
                    r_val, p_val = 0.0, 1.0
                    
                corr_results.append({
                    'chemistry': chem,
                    'dimension': dim + 1,  # 1-indexed for figures
                    'metric': m_name,
                    'pearson_r': r_val if not np.isnan(r_val) else 0.0,
                    'p_value': p_val if not np.isnan(p_val) else 1.0
                })
                
    df_corr = pd.DataFrame(corr_results)
    df_corr.to_csv(os.path.join(FIGURE_DATA_DIR, 'universal_correlations_raw.csv'), index=False)
    print("✓ Saved universal_correlations_raw.csv")
    
    # 3. Calculate Universality Scores and Rank
    print("\nRanking universal latent dimensions...")
    universality_rows = []
    
    for metric in metrics:
        for dim in range(1, 65):
            sub_df = df_corr[(df_corr['metric'] == metric) & (df_corr['dimension'] == dim)]
            corrs = []
            p_vals = []
            for chem in chemistries:
                r = sub_df[sub_df['chemistry'] == chem]['pearson_r'].iloc[0]
                p = sub_df[sub_df['chemistry'] == chem]['p_value'].iloc[0]
                corrs.append(r)
                p_vals.append(p)
                
            corrs = np.array(corrs)
            p_vals = np.array(p_vals)
            abs_corrs = np.abs(corrs)
            
            # Harmonic mean of absolute correlations (penalizes low correlations)
            eps = 1e-9
            h_mean = 4.0 / np.sum(1.0 / (abs_corrs + eps))
            
            avg_abs = np.mean(abs_corrs)
            min_abs = np.min(abs_corrs)
            max_p = np.max(p_vals)
            
            sign_consistent = np.all(corrs >= 0) or np.all(corrs <= 0)
            sign_str = "+" if np.all(corrs >= 0) else ("-" if np.all(corrs <= 0) else "Mixed")
            
            universality_rows.append({
                'dimension': dim,
                'metric': metric,
                'harmonic_mean_r': h_mean,
                'avg_abs_r': avg_abs,
                'min_abs_r': min_abs,
                'max_p_value': max_p,
                'sign_consistency': sign_str,
                'is_sign_consistent': sign_consistent,
                'is_significant': max_p < 0.05,
                **{f'r_{c}': r for c, r in zip(chemistries, corrs)},
                **{f'p_{c}': p for c, p in zip(chemistries, p_vals)}
            })
            
    df_univ = pd.DataFrame(universality_rows)
    df_univ.to_csv(os.path.join(FIGURE_DATA_DIR, 'universal_dimensions_ranked.csv'), index=False)
    print("✓ Saved universal_dimensions_ranked.csv")
    
    # Save summary of top universal dimensions
    top_dims_dict = {}
    print("\n--- Top Universal Latent Dimensions by Metric ---")
    for metric in metrics:
        print(f"\nMetric: {metric}")
        m_df = df_univ[df_univ['metric'] == metric].sort_values('harmonic_mean_r', ascending=False)
        top_dims_dict[metric] = m_df.head(5)[['dimension', 'harmonic_mean_r', 'avg_abs_r', 'min_abs_r', 'max_p_value', 'sign_consistency']].to_dict(orient='records')
        for idx, r in m_df.head(5).iterrows():
            print(f"  Dim {r['dimension']:2d} | Score: {r['harmonic_mean_r']:.3f} | Min Abs: {r['min_abs_r']:.3f} | p-max: {r['max_p_value']:.1e} | Signs: {r['sign_consistency']}")
            
    with open(os.path.join(FIGURE_DATA_DIR, 'universal_dimensions_top_summary.json'), 'w') as f:
        json.dump(top_dims_dict, f, indent=4)
        
    # Find overall universal dimensions (correlate with ANY physical metric across all chemistries)
    # We will pick the top universal dimension for each metric for further plotting.
    top_dims = {}
    for metric in metrics:
        top_dims[metric] = int(df_univ[df_univ['metric'] == metric].sort_values('harmonic_mean_r', ascending=False)['dimension'].iloc[0])
    print(f"\nTop Universal Dimensions chosen for detailed plotting: {top_dims}")
    
    # 4. PLOTTING
    sns.set_theme(style='ticks')
    
    # ==========================================
    # FIGURE A: Heatmap of Universal Correlations
    # ==========================================
    print("\nGenerating Figure A: Heatmap of correlations...")
    # Select unique dimensions that are in the top 3 of any metric to display in the heatmap
    selected_dims = set()
    for metric in metrics:
        m_dims = df_univ[df_univ['metric'] == metric].sort_values('harmonic_mean_r', ascending=False).head(3)['dimension'].tolist()
        selected_dims.update(m_dims)
    selected_dims = sorted(list(selected_dims))
    
    # Create pivot matrix for the heatmap
    # Index: Dimension
    # Columns: Metric x Chemistry
    heatmap_data = []
    for dim in selected_dims:
        row = {'Dimension': f"Dim {dim}"}
        for metric in metrics:
            for chem in chemistries:
                r_val = df_corr[(df_corr['dimension'] == dim) & (df_corr['metric'] == metric) & (df_corr['chemistry'] == chem)]['pearson_r'].iloc[0]
                row[f"{metric}\n({chem})"] = r_val
        heatmap_data.append(row)
        
    df_heat = pd.DataFrame(heatmap_data).set_index('Dimension')
    
    plt.figure(figsize=(14, 8), dpi=300)
    sns.heatmap(df_heat, annot=True, cmap='RdBu_r', vmin=-1.0, vmax=1.0, fmt=".2f",
                cbar_kws={'label': 'Pearson Correlation Coefficient (r)', 'shrink': 0.8},
                linewidths=0.5, linecolor='#eeeeee', annot_kws={'size': 9})
    plt.title('Universal Latent Dimensions Correlation Matrix across Batteries', fontweight='bold', fontsize=14, pad=15)
    plt.xlabel('Physical Metric & Battery Chemistry', fontweight='bold', labelpad=12)
    plt.ylabel('Latent Dimension index (1-64)', fontweight='bold', labelpad=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_universal_heatmap.jpg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_universal_heatmap.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved fig_universal_heatmap.jpg")
    
    # ==========================================
    # FIGURE B: Universal Dimension Ranking
    # ==========================================
    print("Generating Figure B: Universal dimension ranking...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    axes = axes.ravel()
    
    metric_colors = {
        'Capacity': '#1f77b4',
        'Plateau Voltage': '#2ca02c',
        'dQ/dV Peak': '#ff7f0e',
        'Resistance': '#d62728'
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        m_df = df_univ[df_univ['metric'] == metric].sort_values('harmonic_mean_r', ascending=False).head(10)
        
        bars = ax.barh(np.arange(10), m_df['harmonic_mean_r'].values, color=metric_colors[metric], alpha=0.85, height=0.6)
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels([f"Dim {d}" for d in m_df['dimension'].values], fontsize=10)
        ax.invert_yaxis()  # top-down ranking
        
        # Add labels to the end of bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                    va='center', ha='left', fontsize=8, color='#333333')
            
        ax.set_title(f"{metric} Universality Ranking", fontweight='bold', fontsize=12)
        ax.set_xlabel('Universality Score (Harmonic Mean |r|)', fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.grid(axis='x', ls='--', alpha=0.5)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
            
    # Removed suptitle for publication standard
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_universal_ranking.jpg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_universal_ranking.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved fig_universal_ranking.jpg")
    
    # ==========================================
    # FIGURE C: Scatter Plots for Top Dimensions
    # ==========================================
    print("Generating Figure C: Scatter plots for top dimensions...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=300)
    axes = axes.ravel()
    
    scatter_configs = [
        ('Capacity', top_dims['Capacity'], 'capacity', 'Discharge Capacity (Ah)', 'lower left'),
        ('Plateau Voltage', top_dims['Plateau Voltage'], 'plateau_voltage', 'Plateau Voltage (V)', 'upper left'),
        ('dQ/dV Peak', top_dims['dQ/dV Peak'], 'dq_dv_peak', 'dQ/dV Peak Height (Ah/V)', 'lower left'),
        ('Resistance', top_dims['Resistance'], 'resistance', 'Resistance Proxy (Ohm)', 'upper left')
    ]
    
    for idx, (metric, dim_idx, col_name, y_lbl, loc) in enumerate(scatter_configs):
        ax = axes[idx]
        
        # Plot each chemistry with a distinct marker and color
        markers = {'Li-ion': 'o', 'Zn-ion': 's', 'Na-ion': '^', 'CALB': 'D'}
        
        for chem in chemistries:
            chem_df = df[df['chemistry'] == chem]
            x_vals = chem_df[f'latent_{dim_idx-1}'].values
            y_vals = chem_df[col_name].values
            
            # Subsample for very large datasets to prevent plot cluttering
            if len(chem_df) > 500:
                np.random.seed(42)
                sub_idx = np.random.choice(len(chem_df), 500, replace=False)
                x_vals = x_vals[sub_idx]
                y_vals = y_vals[sub_idx]
                
            r_val, p_val = pearsonr(x_vals, y_vals)
            
            # Regression line
            sns.regplot(x=x_vals, y=y_vals, ax=ax, scatter=False, color=CHEM_COLORS[chem],
                        line_kws={'lw': 1.5, 'ls': '--', 'alpha': 0.8})
            
            # Scatter points
            ax.scatter(x_vals, y_vals, color=CHEM_COLORS[chem], marker=markers[chem],
                       alpha=0.45, s=15, edgecolors='none', label=f"{chem} (r={r_val:+.3f})")
            
        ax.set_title(f"Universal Coordinates for {metric} (Dim {dim_idx})", fontweight='bold', fontsize=12)
        ax.set_xlabel(f"Latent Coordinate Value (Dim {dim_idx})", fontsize=10)
        ax.set_ylabel(y_lbl, fontsize=10)
        ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none')
        ax.grid(True, ls='--', alpha=0.4)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
            
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_universal_scatters.jpg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_universal_scatters.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved fig_universal_scatters.jpg")
    
    # ==========================================
    # FIGURE D: Cross-Chemistry Comparison Plots
    # ==========================================
    print("Generating Figure D: Cross-chemistry comparison plots...")
    # Plot how the universal latent coordinate progresses with SOH degradation across chemistries
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Left: SOH vs Capacity Dimension (Dim 48)
    ax = axes[0]
    dim_cap = top_dims['Capacity']
    for chem in chemistries:
        chem_df = df[df['chemistry'] == chem]
        # Bin SOH into 20 bins and compute mean and std of Dim 48 for each bin
        chem_df['soh_bin'] = pd.cut(chem_df['soh'], bins=np.linspace(0.7, 1.05, 15))
        bin_stats = chem_df.groupby('soh_bin').agg({f'latent_{dim_cap-1}': ['mean', 'std'], 'soh': 'mean'}).dropna()
        
        x = bin_stats['soh']['mean'].values
        y = bin_stats[f'latent_{dim_cap-1}']['mean'].values
        y_err = bin_stats[f'latent_{dim_cap-1}']['std'].values
        
        ax.plot(x, y, color=CHEM_COLORS[chem], lw=2, marker='o', label=chem)
        ax.fill_between(x, y - 0.5 * y_err, y + 0.5 * y_err, color=CHEM_COLORS[chem], alpha=0.15)
        
    ax.set_title(f"Capacity Encoder (Dim {dim_cap}) Trajectory vs SOH", fontweight='bold', fontsize=12)
    ax.set_xlabel("State of Health ($Q/Q_0$)", fontsize=10)
    ax.set_ylabel(f"Latent Value (Dim {dim_cap})", fontsize=10)
    ax.invert_xaxis()  # show progression as SOH degrades from 1.0 to 0.7
    ax.legend(loc='best', frameon=True, edgecolor='none')
    ax.grid(True, ls='--', alpha=0.4)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
        
    # Right: SOH vs Resistance Dimension (Dim 50 or 44)
    ax = axes[1]
    dim_res = top_dims['Resistance']
    for chem in chemistries:
        chem_df = df[df['chemistry'] == chem]
        chem_df['soh_bin'] = pd.cut(chem_df['soh'], bins=np.linspace(0.7, 1.05, 15))
        bin_stats = chem_df.groupby('soh_bin').agg({f'latent_{dim_res-1}': ['mean', 'std'], 'soh': 'mean'}).dropna()
        
        x = bin_stats['soh']['mean'].values
        y = bin_stats[f'latent_{dim_res-1}']['mean'].values
        y_err = bin_stats[f'latent_{dim_res-1}']['std'].values
        
        ax.plot(x, y, color=CHEM_COLORS[chem], lw=2, marker='o', label=chem)
        ax.fill_between(x, y - 0.5 * y_err, y + 0.5 * y_err, color=CHEM_COLORS[chem], alpha=0.15)
        
    ax.set_title(f"Resistance Encoder (Dim {dim_res}) Trajectory vs SOH", fontweight='bold', fontsize=12)
    ax.set_xlabel("State of Health ($Q/Q_0$)", fontsize=10)
    ax.set_ylabel(f"Latent Value (Dim {dim_res})", fontsize=10)
    ax.invert_xaxis()
    ax.legend(loc='best', frameon=True, edgecolor='none')
    ax.grid(True, ls='--', alpha=0.4)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_universal_trajectories.jpg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_universal_trajectories.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved fig_universal_trajectories.jpg")
    
    print("\n--- Pipeline Completed Successfully ---")

if __name__ == '__main__':
    main()
