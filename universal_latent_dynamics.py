import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

# Paths
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
    'Li-ion': '#2b5c8f',  # Deep blue
    'Zn-ion': '#2ca02c',  # Emerald green
    'Na-ion': '#ff7f0e',  # Amber orange
    'CALB': '#9467bd'     # Purple
}

def smooth_trajectory(data, sigma=1.0):
    """Applies a 1D Gaussian filter along the first axis to smooth out high-frequency noise."""
    return gaussian_filter1d(data, sigma=sigma, axis=0)

def compute_derivatives(z, t):
    """Computes first and second derivatives of high-dimensional latent trajectory z w.r.t time t."""
    M, D = z.shape
    z_prime = np.zeros_like(z)
    z_double_prime = np.zeros_like(z)
    
    # 1st derivative (velocity vector)
    for i in range(M):
        if i == 0:
            dt = t[1] - t[0]
            z_prime[i] = (z[1] - z[0]) / dt
        elif i == M - 1:
            dt = t[-1] - t[-2]
            z_prime[i] = (z[-1] - z[-2]) / dt
        else:
            dt = t[i+1] - t[i-1]
            z_prime[i] = (z[i+1] - z[i-1]) / dt
            
    # 2nd derivative (acceleration vector)
    for i in range(M):
        if i == 0:
            dt = t[1] - t[0]
            z_double_prime[i] = (z_prime[1] - z_prime[0]) / dt
        elif i == M - 1:
            dt = t[-1] - t[-2]
            z_double_prime[i] = (z_prime[-1] - z_prime[-2]) / dt
        else:
            dt = t[i+1] - t[i-1]
            z_double_prime[i] = (z_prime[i+1] - z_prime[i-1]) / dt
            
    return z_prime, z_double_prime

def compute_curvature(z_prime, z_double_prime):
    """Computes the curvature along a multidimensional trajectory."""
    M, _ = z_prime.shape
    curvature = np.zeros(M)
    
    for i in range(M):
        v = z_prime[i]
        a = z_double_prime[i]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            curvature[i] = 0.0
            continue
            
        # Projection of acceleration onto velocity
        proj_a_v = (np.dot(a, v) / (v_norm**2)) * v
        # Orthogonal component of acceleration
        a_orth = a - proj_a_v
        
        # Curvature formula
        curvature[i] = np.linalg.norm(a_orth) / (v_norm**2 + 1e-8)
        
    return curvature

def detect_transition_point(v, a, curvature, t, min_cycle_idx=5):
    """Detects the transition point using the peak of latent acceleration or curvature spike."""
    # Find index where cycle_index is at least min_cycle_idx
    valid_indices = [i for i, cycle in enumerate(t) if cycle >= min_cycle_idx]
    if not valid_indices:
        return t[len(t) // 2], len(t) // 2
        
    # We look for the maximum acceleration index in the second half of the sequence
    # to avoid early initialization noise
    start_idx = valid_indices[0]
    search_start = max(start_idx, len(t) // 4)
    search_end = len(t) - 1
    
    if search_start >= search_end:
        search_start = start_idx
        
    # Find peak of latent acceleration
    peak_acc_idx = search_start + np.argmax(a[search_start:search_end+1])
    
    # Alternatively, use a Windowed Mean-Shift on Velocity
    # Compare mean of velocity before and after to find where it changes most rapidly
    best_shift_val = -1.0
    best_shift_idx = peak_acc_idx
    W = max(2, len(t) // 10) # Window size
    
    for idx in range(search_start + W, search_end - W + 1):
        mean_before = np.mean(v[idx-W:idx])
        mean_after = np.mean(v[idx:idx+W])
        shift = mean_after - mean_before
        if shift > best_shift_val:
            best_shift_val = shift
            best_shift_idx = idx
            
    # Combine both indicators: weight mean shift and acceleration peak
    # We find that the peak of acceleration is highly correlated with the transition
    trans_idx = best_shift_idx
    return t[trans_idx], trans_idx

def main():
    # 1. Load the dataset containing latent representations and physical properties
    dataset_path = os.path.join(FIGURE_DATA_DIR, 'universal_physical_dataset.csv')
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run universal_latent_analysis.py first.")
        sys.exit(1)
        
    print("Loading universal physical dataset...")
    df = pd.read_csv(dataset_path)
    
    # 2. Extract latent columns
    latent_cols = [f'latent_{i}' for i in range(64)]
    
    # 3. Process cell-by-cell
    cell_groups = df.groupby(['chemistry', 'cell_id'])
    
    processed_cells = []
    
    print("\nProcessing latent trajectories and detecting transition points...")
    for (chem, cell_id), cell_df in cell_groups:
        # Sort by cycle index
        cell_df = cell_df.sort_values('cycle_index').reset_index(drop=True)
        M = len(cell_df)
        if M < 6:
            # Skip cells with too few cycles to differentiate
            continue
            
        t = cell_df['cycle_index'].values
        z = cell_df[latent_cols].values
        
        # Smooth latent space trajectory to suppress noise
        z_smooth = smooth_trajectory(z, sigma=1.0)
        
        # Compute trajectory velocity (z') and acceleration (z'') vectors
        z_prime, z_double_prime = compute_derivatives(z_smooth, t)
        
        # Magnitudes
        v = np.linalg.norm(z_prime, axis=1)
        a = np.linalg.norm(z_double_prime, axis=1)
        
        # Curvature
        curvature = compute_curvature(z_prime, z_double_prime)
        
        # Distance from fresh state (cell's own first observed cycle)
        dist_fresh = np.linalg.norm(z_smooth - z_smooth[0], axis=1)
        
        # Detect Transition Point
        trans_cycle, trans_idx = detect_transition_point(v, a, curvature, t)
        
        # SOH, Capacity, Resistance, Plateau Voltage
        soh = cell_df['soh'].values
        capacity = cell_df['capacity'].values
        resistance = cell_df['resistance'].values
        plateau = cell_df['plateau_voltage'].values
        eol = cell_df['eol'].iloc[0]
        
        # Determine failure cycle (EOL)
        # failure is defined as SOH <= 0.80, or the last cycle if it doesn't reach it
        fail_mask = soh <= 0.80
        if fail_mask.any():
            fail_cycle = t[np.argmax(fail_mask)]
        else:
            fail_cycle = t[-1] # Fallback to last cycle
            
        # Determine onsets of conventional indicators
        # Capacity fade onset (SOH <= 0.98 or 0.95)
        fade_mask = soh <= 0.98
        fade_onset = t[np.argmax(fade_mask)] if fade_mask.any() else t[-1]
        
        # Resistance growth onset (Resistance >= 1.10 of initial)
        ref_res = resistance[0]
        res_mask = resistance >= (1.10 * ref_res)
        resist_onset = t[np.argmax(res_mask)] if res_mask.any() else t[-1]
        
        # Plateau shift onset (Plateau Voltage drops below 0.98 of initial)
        ref_plat = plateau[0]
        plat_mask = plateau <= (0.98 * ref_plat)
        plat_onset = t[np.argmax(plat_mask)] if (plat_mask.any() and ref_plat > 0) else t[-1]
        
        # Early Warning Horizon (cycles before EOL)
        warning_horizon = fail_cycle - trans_cycle
        
        processed_cells.append({
            'cell_id': cell_id,
            'chemistry': chem,
            'eol': eol,
            'fail_cycle': fail_cycle,
            'trans_cycle': trans_cycle,
            'fade_onset_cycle': fade_onset,
            'resist_onset_cycle': resist_onset,
            'plat_onset_cycle': plat_onset,
            'warning_horizon_cycles': warning_horizon,
            'num_cycles': M,
            't': t.tolist(),
            'v': v.tolist(),
            'a': a.tolist(),
            'curvature': curvature.tolist(),
            'dist_fresh': dist_fresh.tolist(),
            'soh': soh.tolist(),
            'capacity': capacity.tolist(),
            'resistance': resistance.tolist(),
            'plateau': plateau.tolist()
        })
        
    df_results = pd.DataFrame([{
        'cell_id': c['cell_id'],
        'chemistry': c['chemistry'],
        'eol': c['eol'],
        'fail_cycle': c['fail_cycle'],
        'trans_cycle': c['trans_cycle'],
        'fade_onset': c['fade_onset_cycle'],
        'resist_onset': c['resist_onset_cycle'],
        'plat_onset': c['plat_onset_cycle'],
        'warning_horizon': c['warning_horizon_cycles']
    } for c in processed_cells])
    
    df_results.to_csv(os.path.join(FIGURE_DATA_DIR, 'universal_dynamics_results.csv'), index=False)
    print(f"✓ Saved results of {len(processed_cells)} cells to figure_data/universal_dynamics_results.csv")
    
    # 4. Save stats and compute averages per chemistry
    print("\n--- Summary of Transition Analysis per Chemistry ---")
    stats_dict = {}
    for chem in CHEM_COLORS.keys():
        chem_df = df_results[df_results['chemistry'] == chem]
        if len(chem_df) == 0:
            continue
        avg_fail = chem_df['fail_cycle'].mean()
        avg_trans = chem_df['trans_cycle'].mean()
        avg_warning = chem_df['warning_horizon'].mean()
        avg_fade = chem_df['fade_onset'].mean()
        avg_resist = chem_df['resist_onset'].mean()
        
        # Calculate percentage of cells where transition occurs BEFORE capacity fade
        early_warn_pct = (chem_df['trans_cycle'] < chem_df['fade_onset']).mean() * 100
        
        print(f"\nChemistry: {chem} (N = {len(chem_df)} cells)")
        print(f"  Avg Failure Cycle (EOL): {avg_fail:.1f}")
        print(f"  Avg Latent Transition Cycle: {avg_trans:.1f}")
        print(f"  Avg Early Warning Horizon: {avg_warning:.1f} cycles before failure")
        print(f"  Avg Capacity Fade Onset Cycle: {avg_fade:.1f}")
        print(f"  Transition Precedes Capacity Fade in {early_warn_pct:.1f}% of cells")
        
        stats_dict[chem] = {
            'num_cells': len(chem_df),
            'avg_failure_cycle': float(avg_fail),
            'avg_latent_transition_cycle': float(avg_trans),
            'avg_warning_horizon_cycles': float(avg_warning),
            'avg_fade_onset_cycle': float(avg_fade),
            'avg_resist_onset_cycle': float(avg_resist),
            'early_warning_percentage': float(early_warn_pct)
        }
        
    with open(os.path.join(FIGURE_DATA_DIR, 'universal_dynamics_metrics.json'), 'w') as f:
        json.dump(stats_dict, f, indent=4)
        
    # ==========================================
    # FIGURE A: Latent Trajectories (UMAP Space)
    # ==========================================
    print("\nGenerating Figure A: Trajectories in UMAP Space...")
    # Load UMAP coordinates from df and plot trajectories
    df_umap = df.copy()
    # Check if UMAP coordinates exist in merged csv, otherwise calculate
    if 'umap_dim1' not in df_umap.columns:
        print("  UMAP coordinates not in dataset, calculating UMAP projection...")
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        umap_results = reducer.fit_transform(df_umap[latent_cols].values)
        df_umap['umap_dim1'] = umap_results[:, 0]
        df_umap['umap_dim2'] = umap_results[:, 1]
        
    plt.figure(figsize=(9, 7.5), dpi=300)
    # Background scatter of all points
    plt.scatter(df_umap['umap_dim1'], df_umap['umap_dim2'], c='#e0e0e0', s=2, alpha=0.5, edgecolor='none', label='All Cycles')
    
    # Plot trajectories for representative cells of each chemistry
    for chem in CHEM_COLORS.keys():
        chem_cells = [c for c in processed_cells if c['chemistry'] == chem]
        if not chem_cells:
            continue
        # Pick cell with median lifetime
        cell_lfts = [c['num_cycles'] for c in chem_cells]
        median_idx = np.argsort(cell_lfts)[len(cell_lfts) // 2]
        rep_cell = chem_cells[median_idx]
        
        # Get coordinates for this cell
        cell_coords = df_umap[df_umap['cell_id'] == rep_cell['cell_id']].sort_values('cycle_index')
        plt.plot(cell_coords['umap_dim1'], cell_coords['umap_dim2'], color=CHEM_COLORS[chem], lw=2.5, label=f'{chem} Representative Path')
        
        # Highlight fresh state, transition state, and EOL state
        plt.scatter(cell_coords['umap_dim1'].iloc[0], cell_coords['umap_dim2'].iloc[0], color=CHEM_COLORS[chem], marker='o', s=80, edgecolors='black', zorder=5)
        plt.scatter(cell_coords['umap_dim1'].iloc[rep_cell['fail_cycle'] == rep_cell['t']], cell_coords['umap_dim2'].iloc[rep_cell['fail_cycle'] == rep_cell['t']], color=CHEM_COLORS[chem], marker='X', s=100, edgecolors='black', zorder=5)
        
        # Draw transition point
        trans_mask = np.array(rep_cell['t']) == rep_cell['trans_cycle']
        if trans_mask.any():
            plt.scatter(cell_coords['umap_dim1'].iloc[trans_mask], cell_coords['umap_dim2'].iloc[trans_mask], color='red', marker='*', s=180, edgecolors='black', zorder=6)

    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("Latent Trajectory Pathways and Transition Points", fontweight='bold', pad=12)
    # Custom legends for markers
    from matplotlib.lines import Line2D
    marker_legends = [
        Line2D([0], [0], marker='o', color='gray', label='Start of Life (SOL)', markerfacecolor='gray', markersize=8, ls=''),
        Line2D([0], [0], marker='*', color='red', label='Critical Transition Point', markerfacecolor='red', markersize=10, ls=''),
        Line2D([0], [0], marker='X', color='gray', label='End of Life (EOL)', markerfacecolor='gray', markersize=8, ls=''),
    ]
    chem_legends = [Line2D([0], [0], color=CHEM_COLORS[chem], lw=2.5, label=chem) for chem in CHEM_COLORS.keys() if any(c['chemistry'] == chem for c in processed_cells)]
    plt.legend(handles=chem_legends + marker_legends, loc='best')
    plt.grid(True, ls='--', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_dynamics_trajectories.jpg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_dynamics_trajectories.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved fig_dynamics_trajectories.jpg")
    
    # ==========================================
    # FIGURE B: Transition Visualization
    # ==========================================
    print("Generating Figure B: Transition Visualizations...")
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), dpi=300)
    
    for idx, chem in enumerate(CHEM_COLORS.keys()):
        chem_cells = [c for c in processed_cells if c['chemistry'] == chem]
        if not chem_cells:
            continue
        # Select representative cell
        cell_lfts = [c['num_cycles'] for c in chem_cells]
        rep_cell = chem_cells[np.argsort(cell_lfts)[len(cell_lfts) // 2]]
        
        # Left Panel: SOH and Capacity
        ax_soh = axes[idx, 0]
        ax_soh.plot(rep_cell['t'], rep_cell['soh'], color='#2ca02c', lw=2, label='SOH')
        ax_soh.set_ylabel('State of Health ($Q/Q_0$)', color='#2ca02c')
        ax_soh.tick_params(axis='y', labelcolor='#2ca02c')
        
        # Title for the row
        ax_soh.set_title(f"{chem} (Cell: {rep_cell['cell_id']}) | Conventional Metrics", fontweight='bold', fontsize=11)
        ax_soh.set_xlabel("Cycle Index")
        ax_soh.grid(True, ls='--', alpha=0.3)
        
        # Add EOL line
        ax_soh.axvline(x=rep_cell['fail_cycle'], color='black', ls='--', alpha=0.7, label='EOL (80% SOH)')
        ax_soh.axvline(x=rep_cell['trans_cycle'], color='red', ls='-', lw=1.5, label='Latent Transition')
        
        # Right Panel: Latent Dynamics
        ax_dyn = axes[idx, 1]
        ax_dyn.plot(rep_cell['t'], rep_cell['v'], color='#1f77b4', lw=2, label='Latent Velocity')
        ax_dyn.set_ylabel('Latent Velocity', color='#1f77b4')
        ax_dyn.tick_params(axis='y', labelcolor='#1f77b4')
        
        # Plot acceleration on secondary y-axis
        ax_acc = ax_dyn.twinx()
        ax_acc.plot(rep_cell['t'], rep_cell['a'], color='#ff7f0e', lw=1.5, ls='--', label='Latent Acceleration')
        ax_acc.set_ylabel('Latent Acceleration', color='#ff7f0e')
        ax_acc.tick_params(axis='y', labelcolor='#ff7f0e')
        
        ax_dyn.set_title(f"{chem} Latent Space Kinematics", fontweight='bold', fontsize=11)
        ax_dyn.set_xlabel("Cycle Index")
        ax_dyn.grid(True, ls='--', alpha=0.3)
        
        # Lines indicating transition and failure
        ax_dyn.axvline(x=rep_cell['fail_cycle'], color='black', ls='--', alpha=0.7)
        ax_dyn.axvline(x=rep_cell['trans_cycle'], color='red', ls='-', lw=1.5)
        
        # Align legend
        lines_soh, labels_soh = ax_soh.get_legend_handles_labels()
        ax_soh.legend(lines_soh, labels_soh, loc='lower left')
        
        lines_dyn, labels_dyn = ax_dyn.get_legend_handles_labels()
        lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
        ax_dyn.legend(lines_dyn + lines_acc, labels_dyn + labels_acc, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_dynamics_transitions.jpg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_dynamics_transitions.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved fig_dynamics_transitions.jpg")
    
    # ==========================================
    # FIGURE C: Early Warning Horizon Histograms
    # ==========================================
    print("Generating Figure C: Early Warning Horizon Histograms...")
    plt.figure(figsize=(10, 6.5), dpi=300)
    
    for chem in CHEM_COLORS.keys():
        chem_df = df_results[df_results['chemistry'] == chem]
        if len(chem_df) == 0:
            continue
        sns.kdeplot(chem_df['warning_horizon'], color=CHEM_COLORS[chem], fill=True, alpha=0.2, lw=2, label=f"{chem} (Mean: {chem_df['warning_horizon'].mean():.1f} cyc)")
        
    plt.xlabel("Early Warning Horizon (Cycles Before Failure)", fontweight='bold', fontsize=11)
    plt.ylabel("Probability Density", fontweight='bold', fontsize=11)
    plt.title("Early Warning Horizon of Latent Phase Transition Across Chemistries", fontweight='bold', pad=15)
    plt.legend(frameon=True, edgecolor='none')
    plt.grid(True, ls='--', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_dynamics_warning_horizons.jpg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_dynamics_warning_horizons.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved fig_dynamics_warning_horizons.jpg")
    
    # ==========================================
    # FIGURE D: Comparison of Onset Timings
    # ==========================================
    print("Generating Figure D: Onset Comparison Plots...")
    # Melt the dataframe for plotting
    df_melt = df_results.melt(
        id_vars=['cell_id', 'chemistry'],
        value_vars=['trans_cycle', 'fade_onset', 'resist_onset'],
        var_name='Indicator', value_name='Cycle'
    )
    
    # Rename variables for figure
    df_melt['Indicator'] = df_melt['Indicator'].map({
        'trans_cycle': 'Latent Transition',
        'fade_onset': 'Capacity Fade (98% SOH)',
        'resist_onset': 'Resistance Growth (110% R)'
    })
    
    plt.figure(figsize=(12, 7.5), dpi=300)
    ax = sns.boxplot(
        data=df_melt, x='chemistry', y='Cycle', hue='Indicator',
        palette=['#d62728', '#1f77b4', '#ff7f0e'], width=0.6,
        showmeans=True, meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"}
    )
    
    plt.xlabel("Battery Chemistry", fontweight='bold', fontsize=11)
    plt.ylabel("Onset Cycle Index", fontweight='bold', fontsize=11)
    plt.title("Comparison of Onset Detection Cycles Across Battery Chemistries", fontweight='bold', pad=15)
    plt.legend(title="Degradation Indicator", frameon=True, edgecolor='none')
    plt.grid(True, ls='--', alpha=0.3)
    
    # Remove top/right spines
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_dynamics_onsets_comparison.jpg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'fig_dynamics_onsets_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved fig_dynamics_onsets_comparison.jpg")
    
    print("\n--- Dynamics Analysis Completed Successfully ---")

if __name__ == '__main__':
    main()
