import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

warnings.filterwarnings('ignore')

# Helvetica/Arial font style for publication-ready figures
plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'axes.titlesize': 20, 'axes.labelsize': 18,
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold',
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'grid.alpha': 0.3,
})

# Colors for the 4 chemistries
CHEM_COLORS = {
    'Li-ion': '#00c853',   # Vibrant Green
    'Zn-ion': '#9c27b0',   # Vibrant Purple
    'Na-ion': '#ff1744',   # Vibrant Red
    'CALB': '#ff9100'      # Vibrant Orange
}

FIGURE_DATA_DIR = './figure_data'
FIGURES_DIR = './figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Helper function to check if data files exist
def check_data_file(filename):
    path = os.path.join(FIGURE_DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Some plots will be skipped.")
        return None
    return path

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1: 3D Ribbon/Waterfall Visualization of JEPA Pre-training Sequence & Masking
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure_1():
    path = check_data_file('jepa_input_masking_demo.csv')
    if not path:
        return
    print("Plotting Figure 1: 3D Ribbon of JEPA Sequence & Masking...")
    df = pd.read_csv(path)
    
    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    selected_cycles = sorted(df['cycle_num'].unique())
    
    for idx, cyc_num in enumerate(selected_cycles):
        cyc_df = df[df['cycle_num'] == cyc_num].sort_values('step_idx')
        x = cyc_df['step_idx'].values
        y = cyc_df['cycle_sequence_idx'].values
        z = cyc_df['voltage'].values
        is_context = cyc_df['is_context'].iloc[0] == 1
        
        color = '#3498db' if is_context else '#e74c3c'
        ls = '-' if is_context else '--'
        lw = 3.5 if is_context else 2.5
        alpha = 0.85 if is_context else 0.6
        label = "Context Cycle ($M_l=0$)" if is_context and idx == 0 else (
                "Masked Target ($M_l=1$, Predicted)" if not is_context and idx == 1 else None)
        
        ax.plot(x, y, z, color=color, linestyle=ls, linewidth=lw, alpha=alpha, label=label)
        
        # Add filled polygon under the curve to create a ribbon effect
        zs = np.concatenate([[2.0], z, [2.0]])
        poly_color = '#ebf5fb' if is_context else '#fdedec'
        ax.add_collection3d(plt.fill_between(x, zs[1:-1], 2.0, color=poly_color, alpha=0.35), zs=idx+1, zdir='y')
        
    # Stretch y-axis (depth) to separate ribbons, and adjust aspect ratio
    ax.set_box_aspect((1.2, 1.8, 0.8))
    
    ax.set_xlabel('Discharge Curve Step', labelpad=15)
    ax.set_ylabel('Cycle Sequence Index', labelpad=32)
    ax.set_zlabel('Voltage (V)', labelpad=15)
    ax.set_yticks([1, 2, 3, 4, 5])
    
    ytick_labels = []
    for idx, cyc_num in enumerate(selected_cycles):
        is_context = df[df['cycle_num'] == cyc_num]['is_context'].iloc[0] == 1
        if is_context:
            ytick_labels.append(f'Cycle {cyc_num}')
        else:
            ytick_labels.append(f'Cycle {cyc_num} (Masked)')
    ax.set_yticklabels(ytick_labels)
    
    ax.set_zlim(2.0, 3.8)
    ax.view_init(elev=20, azim=-55)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Set background pane colors to clean white/grey
    ax.xaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
    ax.yaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
    ax.zaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
    
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.92), framealpha=0.9)
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.02, right=0.98)
    
    plt.savefig(os.path.join(FIGURES_DIR, 'jepa_input_masking_demo.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'jepa_input_masking_demo.jpg'), bbox_inches='tight', dpi=600)
    plt.close()
    print("✓ Figure 1 plotted.")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Chemistry-Specific Degradation Envelopes and EOL Lifespan Distributions
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure_2():
    env_path = check_data_file('chemistry_degradation_envelope.csv')
    dist_path = check_data_file('eol_distributions.csv')
    if not env_path or not dist_path:
        return
    print("Plotting Figure 2: Degradation Envelopes and EOL Distributions...")
    
    df_env = pd.read_csv(env_path)
    df_life = pd.read_csv(dist_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # Panel A: SOH Trajectories Envelopes
    ax_env = axes[0]
    ax_env.set_title('(a) Multi-Chemistry SOH Degradation Envelopes', fontweight='bold', pad=12, fontsize=20)
    ax_env.set_xlabel('Normalized Cycle Lifetime ($N / N_{EOL}$)', fontsize=18)
    ax_env.set_ylabel('State of Health (Normalized Capacity)', fontsize=18)
    
    for chem in CHEM_COLORS.keys():
        chem_df = df_env[df_env['chemistry'] == chem].sort_values('norm_cycle')
        if len(chem_df) == 0:
            print(f"Warning: No data for envelope chemistry {chem}")
            continue
        x = chem_df['norm_cycle'].values
        mean_soh = chem_df['mean_soh'].values
        pct_10 = chem_df['pct_10'].values
        pct_90 = chem_df['pct_90'].values
        
        color = CHEM_COLORS[chem]
        ax_env.plot(x, mean_soh, color=color, lw=3.5, label=f'{chem} (Mean)')
        ax_env.fill_between(x, pct_10, pct_90, color=color, alpha=0.30)
        
    ax_env.grid(True, linestyle=':', alpha=0.7)
    ax_env.legend(loc='lower left', fontsize=16)
    ax_env.set_xlim(0, 1)
    ax_env.set_ylim(0.65, 1.18)
    ax_env.tick_params(axis='both', which='major', labelsize=16)
    for s in ['top', 'right']: ax_env.spines[s].set_visible(False)
    
    ax_env.annotate('Initial capacity activation\n(phase-change alternative chemistries)', 
                    xy=(0.11, 1.05), xytext=(0.28, 1.09), fontsize=14,
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.8),
                    bbox=dict(boxstyle='round,pad=0.3', fc='#fff9db', alpha=0.9, ec='grey'))

    # Panel B: EOL Lifespan Distributions
    ax_life = axes[1]
    ax_life.set_title('(b) Target EOL Cycle Life Distributions', fontweight='bold', pad=12, fontsize=20)
    ax_life.set_xlabel('Battery Chemistry', fontsize=18)
    ax_life.set_ylabel('End-of-Life (Cycles)', fontsize=18)
    
    sns.violinplot(data=df_life, x='Chemistry', y='Cycle_Life', ax=ax_life,
                   hue='Chemistry', palette=CHEM_COLORS, inner=None, linewidth=2.0, alpha=0.85,
                   order=['Li-ion', 'Zn-ion', 'Na-ion', 'CALB'], legend=False,
                   log_scale=True)
    
    sns.stripplot(data=df_life, x='Chemistry', y='Cycle_Life', ax=ax_life,
                  hue='Chemistry', palette=CHEM_COLORS, size=5, alpha=0.25, jitter=0.20, edgecolor='black',
                  linewidth=0.6, order=['Li-ion', 'Zn-ion', 'Na-ion', 'CALB'], legend=False,
                  log_scale=True)
    
    ax_life.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax_life.grid(True, linestyle=':', alpha=0.7, which='both')
    ax_life.tick_params(axis='both', which='major', labelsize=16)
    for s in ['top', 'right']: ax_life.spines[s].set_visible(False)
    
    # ax_life.annotate('Severe label imbalance\n(Li-ion vs Na/Zn-ion)\n— justifies KDE loss',
    #                  xy=(2.0, 1000), xytext=(0.6, 170), fontsize=14,
    #                  arrowprops=dict(arrowstyle='->', color='#333', lw=1.8),
    #                  bbox=dict(boxstyle='round,pad=0.3', fc='#ebf5fb', alpha=0.9, ec='grey'))

    # fig.suptitle('Multi-Chemistry SOH Trajectories Envelopes and EOL Target Distributions',
    #              fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, 'chemistry_degradation_envelope.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'chemistry_degradation_envelope.jpg'), bbox_inches='tight', dpi=600)
    plt.close()
    print("✓ Figure 2 plotted.")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Thermodynamic Plateau Preservation (real Zn-ion data)
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure_3():
    path = check_data_file('fig_thermodynamic_plateaus.csv')
    if not path:
        return
    print("Plotting Figure 3: Thermodynamic Plateau Preservation...")
    df = pd.read_csv(path)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    COLORS = ['#2ca02c', '#d62728']
    
    df_e = df[df['cycle_type'] == 'Early'].sort_values('point_idx')
    df_a = df[df['cycle_type'] == 'Aged'].sort_values('point_idx')
    
    ci_e = df_e['cycle_num'].iloc[0]
    ci_a = df_a['cycle_num'].iloc[0]
    el = f'Cycle {ci_e} (Early)'
    al = f'Cycle {ci_a} (Aged)'
    
    v_max_global = df_e['v_max_global'].iloc[0]
    c_nom = df_e['c_nom'].iloc[0]
    
    # Panel 1 — Raw physical scale
    axes[0].plot(df_e['capacity_raw'].values, df_e['voltage_raw'].values, color=COLORS[0], lw=2, label=el)
    axes[0].plot(df_a['capacity_raw'].values, df_a['voltage_raw'].values, color=COLORS[1], lw=2, label=al)
    axes[0].set_title('(a) Raw Discharge Profiles', fontweight='bold')
    axes[0].set(xlabel='Capacity (mAh)', ylabel='Voltage (V)')
    axes[0].legend(fontsize=14); axes[0].grid(ls='--', alpha=0.5)
    for s in ['top', 'right']: axes[0].spines[s].set_visible(False)
    
    # Panel 2 — Aligned (min-max per cycle)
    axes[1].plot(df_e['capacity_aligned'].values, df_e['voltage_aligned'].values, color=COLORS[0], lw=2, label=el)
    axes[1].plot(df_a['capacity_aligned'].values, df_a['voltage_aligned'].values, color=COLORS[1], lw=2, label=al)
    axes[1].set_title('(b) Aligned Scaling', fontweight='bold')
    axes[1].set(xlabel='Normalized Capacity', ylabel='Normalized Voltage [0–1]')
    axes[1].legend(fontsize=14); axes[1].grid(ls='--', alpha=0.5)
    for s in ['top', 'right']: axes[1].spines[s].set_visible(False)
    
    mid = (df_e['voltage_aligned'].mean() + df_a['voltage_aligned'].mean()) / 2
    axes[1].annotate('Both curves forced to [0,1]\n— plateau position erased',
                     xy=(0.5, mid), xytext=(0.05, 0.12), fontsize=14,
                     arrowprops=dict(arrowstyle='->', color='#555', lw=1.3),
                     bbox=dict(boxstyle='round,pad=0.3', fc='#ffeaea', alpha=0.85))
    
    # Panel 3 — Unaligned (ours)
    axes[2].plot(df_e['capacity_unaligned'].values, df_e['voltage_unaligned'].values, color=COLORS[0], lw=2, label=el)
    axes[2].plot(df_a['capacity_unaligned'].values, df_a['voltage_unaligned'].values, color=COLORS[1], lw=2, label=al)
    axes[2].set_title('(c) Unaligned Scaling (Ours)', fontweight='bold')
    axes[2].set(xlabel='Normalized Capacity Q/Q_nom', ylabel='Voltage / V_max')
    axes[2].legend(fontsize=14); axes[2].grid(ls='--', alpha=0.5)
    for s in ['top', 'right']: axes[2].spines[s].set_visible(False)
    
    dy = abs(df_e['voltage_unaligned'].mean() - df_a['voltage_unaligned'].mean())
    if dy > 0.003:
        x0 = 0.55
        y1 = float(np.interp(x0, df_e['capacity_unaligned'].values, df_e['voltage_unaligned'].values))
        y2 = float(np.interp(x0, df_a['capacity_unaligned'].values, df_a['voltage_unaligned'].values))
        axes[2].annotate('', xy=(x0, y2), xytext=(x0, y1), fontsize=14,
                         arrowprops=dict(arrowstyle='<->', color='#333', lw=1.8))
        axes[2].text(x0 + 0.03, (y1 + y2) / 2,
                     f'Plateau shift\npreserved\n({abs(y1-y2)*v_max_global*1000:.0f} mV)',
                     fontsize=14, va='center', color='#333',
                     bbox=dict(boxstyle='round,pad=0.2', fc='#e8f5e9', alpha=0.85))
                     
    # fig.suptitle('Thermodynamic Plateau Preservation: Aligned vs. Unaligned Feature Scaling (Zn-ion)',
    #              fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_thermodynamic_plateaus.jpg'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_thermodynamic_plateaus.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 3 plotted.")

# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Latent Aging Trajectories (Aligned vs. Unaligned Zn-ion)
# ──────────────────────────────────────────────────────────────────────────────
def add_confidence_ellipse(x, y, ax, n_std=1.5, edgecolor='#d32f2f', facecolor='none', **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-8)
    ell_radius_x = np.sqrt(max(0, 1 + pearson))
    ell_radius_y = np.sqrt(max(0, 1 - pearson))
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def add_trajectory_arrows(x, y, ax, num_arrows=3, color='#424242', lw=1.5):
    window = max(5, len(x) // 10)
    x_smooth = pd.Series(x).rolling(window, min_periods=1, center=True).mean().values
    y_smooth = pd.Series(y).rolling(window, min_periods=1, center=True).mean().values
    indices = np.linspace(len(x_smooth) * 0.15, len(x_smooth) * 0.85, num_arrows, dtype=int)
    for idx in indices:
        if idx + 8 < len(x_smooth):
            dx = x_smooth[idx + 8] - x_smooth[idx]
            dy = y_smooth[idx + 8] - y_smooth[idx]
            length = np.hypot(dx, dy)
            if length > 1e-5:
                dx_norm = dx / length
                dy_norm = dy / length
                ax.annotate('', xy=(x_smooth[idx] + dx_norm * 0.05, y_smooth[idx] + dy_norm * 0.05),
                            xytext=(x_smooth[idx], y_smooth[idx]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                            shrinkA=0, shrinkB=0, mutation_scale=15),
                            zorder=4)

def plot_figure_4():
    path = check_data_file('fig_latent_trajectories.csv')
    if not path:
        return
    print("Plotting Figure 4: Latent Aging Trajectories...")
    df = pd.read_csv(path)
    
    # Check UMAP or t-SNE columns
    x_col = 'umap_dim1' if 'umap_dim1' in df.columns else 'tsne_dim1'
    y_col = 'umap_dim2' if 'umap_dim2' in df.columns else 'tsne_dim2'
    
    df_al = df[df['model_type'] == 'Aligned'].sort_values('sample_idx')
    df_un = df[df['model_type'] == 'Unaligned'].sort_values('sample_idx')
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # Quantiles for cohorts (confidence ellipses)
    q_low = df['true_life'].quantile(0.25)
    q_high = df['true_life'].quantile(0.75)
    
    # Panel A — Aligned
    sc1 = axes[0].scatter(df_al[x_col].values, df_al[y_col].values, c=df_al['true_life'].values,
                          cmap='viridis', s=22, alpha=0.8, edgecolors='none', zorder=3)
    sns.kdeplot(x=df_al[x_col].values, y=df_al[y_col].values, ax=axes[0], colors='#9e9e9e', alpha=0.35, levels=5, linewidths=0.8, zorder=2)
    al_short = df_al[df_al['true_life'] <= q_low]
    al_long = df_al[df_al['true_life'] >= q_high]
    if len(al_short) > 2:
        add_confidence_ellipse(al_short[x_col].values, al_short[y_col].values, axes[0], n_std=1.5, edgecolor='#d32f2f', lw=2.0, ls='--', label='Short-Lived (<=25%)', zorder=5)
    if len(al_long) > 2:
        add_confidence_ellipse(al_long[x_col].values, al_long[y_col].values, axes[0], n_std=1.5, edgecolor='#1976d2', lw=2.0, ls='--', label='Long-Lived (>=75%)', zorder=5)
    add_trajectory_arrows(df_al[x_col].values, df_al[y_col].values, axes[0], num_arrows=3, color='#424242')
    
    axes[0].set_title('(a) Aligned Latent Space UMAP', fontsize=18, fontweight='bold')
    axes[0].set_xlabel('UMAP Dimension 1', fontsize=14); axes[0].set_ylabel('UMAP Dimension 2', fontsize=14)
    axes[0].grid(True, ls='--', alpha=0.3)
    for s in ['top', 'right']: axes[0].spines[s].set_visible(False)
    axes[0].legend(loc='best', fontsize=10, framealpha=0.85)
    cb1 = fig.colorbar(sc1, ax=axes[0], pad=0.02)
    cb1.set_label('True Cycle Life (Cycles)', fontsize=14)
    cb1.ax.tick_params(labelsize=11)
    
    # Panel B — Unaligned
    sc2 = axes[1].scatter(df_un[x_col].values, df_un[y_col].values, c=df_un['true_life'].values,
                          cmap='viridis', s=22, alpha=0.8, edgecolors='none', zorder=3)
    sns.kdeplot(x=df_un[x_col].values, y=df_un[y_col].values, ax=axes[1], colors='#9e9e9e', alpha=0.35, levels=5, linewidths=0.8, zorder=2)
    un_short = df_un[df_un['true_life'] <= q_low]
    un_long = df_un[df_un['true_life'] >= q_high]
    if len(un_short) > 2:
        add_confidence_ellipse(un_short[x_col].values, un_short[y_col].values, axes[1], n_std=1.5, edgecolor='#d32f2f', lw=2.0, ls='--', label='Short-Lived (<=25%)', zorder=5)
    if len(un_long) > 2:
        add_confidence_ellipse(un_long[x_col].values, un_long[y_col].values, axes[1], n_std=1.5, edgecolor='#1976d2', lw=2.0, ls='--', label='Long-Lived (>=75%)', zorder=5)
    add_trajectory_arrows(df_un[x_col].values, df_un[y_col].values, axes[1], num_arrows=3, color='#424242')
    
    axes[1].set_title('(b) Unaligned Latent Space UMAP (Ours)', fontsize=18, fontweight='bold')
    axes[1].set_xlabel('UMAP Dimension 1', fontsize=14); axes[1].set_ylabel('UMAP Dimension 2', fontsize=14)
    axes[1].grid(True, ls='--', alpha=0.3)
    for s in ['top', 'right']: axes[1].spines[s].set_visible(False)
    axes[1].legend(loc='best', fontsize=10, framealpha=0.85)
    cb2 = fig.colorbar(sc2, ax=axes[1], pad=0.02)
    cb2.set_label('True Cycle Life (Cycles)', fontsize=14)
    cb2.ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_latent_trajectories.jpg'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_latent_trajectories.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 4 plotted.")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Multi-chemistry Plateau Comparison
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure_5():
    path = check_data_file('fig_multi_chem_plateaus.csv')
    if not path:
        return
    print("Plotting Figure 5: Multi-chemistry Plateau Comparison...")
    df = pd.read_csv(path)
    
    chemistries = ['Li-ion', 'Zn-ion', 'Na-ion', 'CALB']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    fig.suptitle('Discharge Voltage Profiles Across Battery Chemistries',
                 fontsize=22, fontweight='bold', y=0.98)
                 
    for idx, chem in enumerate(chemistries):
        ax = axes[idx // 2, idx % 2]
        chem_df = df[df['chemistry'] == chem]
        if len(chem_df) == 0:
            ax.set_title(f'{chem}\n(no data)', fontsize=22, fontweight='bold')
            continue
            
        df_e = chem_df[chem_df['cycle_type'] == 'Early'].sort_values('point_idx')
        df_a = chem_df[chem_df['cycle_type'] == 'Aged'].sort_values('point_idx')
        
        ci_e = df_e['cycle_num'].iloc[0]
        ci_a = df_a['cycle_num'].iloc[0]
        
        ax.plot(df_e['q_over_qnom'].values, df_e['voltage'].values, color='#2ca02c', lw=3.5, label=f'Early (# {ci_e})')
        ax.plot(df_a['q_over_qnom'].values, df_a['voltage'].values, color='#d62728', lw=3.5, label=f'Aged  (# {ci_a})')
        ax.set_title(f'({chr(97+idx)}) {chem}', fontsize=22, fontweight='bold')
        ax.set_xlabel('Q / Q_nom', fontsize=18)
        ax.set_ylabel('Voltage (V)', fontsize=18)
        ax.legend(fontsize=16, loc='upper right')
        ax.grid(ls='--', alpha=0.5)
        ax.tick_params(labelsize=18)
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
        
        # Add inset zoom on flat discharge plateau region
        axins = ax.inset_axes([0.15, 0.15, 0.45, 0.4])
        axins.plot(df_e['q_over_qnom'].values, df_e['voltage'].values, color='#2ca02c', lw=2.0)
        axins.plot(df_a['q_over_qnom'].values, df_a['voltage'].values, color='#d62728', lw=2.0)
        
        if chem == 'Li-ion':
            axins.set_xlim(0.3, 0.7); axins.set_ylim(3.10, 3.35)
        elif chem == 'Zn-ion':
            axins.set_xlim(0.2, 0.6); axins.set_ylim(1.25, 1.48)
        elif chem == 'Na-ion':
            axins.set_xlim(0.3, 0.7); axins.set_ylim(2.90, 3.50)
        elif chem == 'CALB':
            axins.set_xlim(0.3, 0.7); axins.set_ylim(3.40, 3.80)
            
        axins.tick_params(labelsize=10)
        axins.grid(True, ls='--', alpha=0.3)
        ax.indicate_inset_zoom(axins, edgecolor="black")
        
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_multi_chem_plateaus.jpg'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_multi_chem_plateaus.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 5 plotted.")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Multi-chemistry t-SNE Grid
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure_6():
    path = check_data_file('fig_tsne_grid.csv')
    if not path:
        return
    print("Plotting Figure 6: Multi-chemistry UMAP Grid...")
    df = pd.read_csv(path)
    
    # Check UMAP or t-SNE columns
    x_col = 'umap_dim1' if 'umap_dim1' in df.columns else 'tsne_dim1'
    y_col = 'umap_dim2' if 'umap_dim2' in df.columns else 'tsne_dim2'
    
    chemistries = [
        ('MIX_large', 'Li-ion (MIX_large)'),
        ('ZN-coin', 'Zn-coin (ZN-coin)'),
        ('NAion', 'Na-ion (NAion)'),
        ('CALB', 'CALB (CALB)')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=300)
    axes = axes.flatten()
    
    for idx, (chem_id, title) in enumerate(chemistries):
        chem_df = df[df['chemistry'] == chem_id].sort_values('sample_idx')
        ax = axes[idx]
        if len(chem_df) == 0:
            ax.set_title(f'{title}\n(no data)', fontsize=16, fontweight='bold')
            continue
            
        x = chem_df[x_col].values
        y = chem_df[y_col].values
        life = chem_df['true_life'].values
        
        sc = ax.scatter(x, y, c=life, cmap='viridis', edgecolors='none', alpha=0.75, s=22, zorder=3)
        sns.kdeplot(x=x, y=y, ax=ax, colors='#9e9e9e', alpha=0.35, levels=5, linewidths=0.8, zorder=2)
        
        # Confidence ellipses for short-lived vs long-lived cohorts
        q_low = chem_df['true_life'].quantile(0.25)
        q_high = chem_df['true_life'].quantile(0.75)
        short_mask = chem_df['true_life'] <= q_low
        long_mask = chem_df['true_life'] >= q_high
        
        if short_mask.sum() > 2:
            add_confidence_ellipse(x[short_mask], y[short_mask], ax, n_std=1.5, edgecolor='#d32f2f', lw=1.8, ls='--', label='Short-Lived (<=25%)', zorder=5)
        if long_mask.sum() > 2:
            add_confidence_ellipse(x[long_mask], y[long_mask], ax, n_std=1.5, edgecolor='#1976d2', lw=1.8, ls='--', label='Long-Lived (>=75%)', zorder=5)
            
        # Trajectory arrows
        add_trajectory_arrows(x, y, ax, num_arrows=3, color='#424242')
        
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('True Cycle Life (Cycles)', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('UMAP Component 1', fontsize=12)
        ax.set_ylabel('UMAP Component 2', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
        ax.legend(loc='best', fontsize=9, framealpha=0.85)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_tsne_grid.jpg'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_tsne_grid.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 6 plotted.")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 7: Latent Feature Correlation Heatmap (CALB)
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure_7():
    path = check_data_file('fig_latent_correlation.csv')
    if not path:
        return
    print("Plotting Figure 7: Latent Correlation Heatmap Grid...")
    df = pd.read_csv(path)
    
    chemistries = [
        ('Li-ion', 'Li-ion (MIX_large)'),
        ('Zn-ion', 'Zn-coin (ZN-coin)'),
        ('Na-ion', 'Na-ion (NAion)'),
        ('CALB', 'CALB (CALB)')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    axes = axes.flatten()
    
    for idx, (chem_id, title) in enumerate(chemistries):
        chem_df = df[df['chemistry'] == chem_id]
        ax = axes[idx]
        if len(chem_df) == 0:
            ax.set_title(f'{title}\n(no data)', fontsize=12)
            continue
            
        heatmap_data = chem_df[['soh_corr', 'cycle_corr', 'voltage_corr']].values
        ylabels = chem_df['dimension'].values
        
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            annot_kws={'size': 9, 'weight': 'bold'},
            fmt=".3f", 
            cmap="RdBu_r", 
            center=0,
            vmin=-1, vmax=1,
            xticklabels=['SOH', 'Cycle', 'Voltage'],
            yticklabels=[str(y).replace('Latent Dim', 'Dim') for y in ylabels],
            ax=ax,
            cbar_kws={'label': 'Pearson Correlation Coefficient (r)'}
        )
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        cbar = ax.collections[0].colorbar
        cbar.set_label('Pearson Correlation Coefficient (r)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        ax.set_title(f'({chr(97+idx)}) {title}', fontweight='bold', fontsize=14, pad=10)
        ax.set_ylabel('Latent Dimension', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        
    fig.suptitle('Latent Feature–Physics Correlation Heatmaps\n'
                 '(Top 20 Correlated Features vs Physical Metrics Across All 4 Chemistries)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_latent_correlation.jpg'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_latent_correlation.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 7 plotted.")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 8: Latent Physical Correlation Heatmap Grid
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure_8():
    path = check_data_file('fig_latent_physical_correlation.csv')
    if not path:
        return
    print("Plotting Figure 8: Latent Physical Correlation Heatmap Grid...")
    df = pd.read_csv(path)
    
    chemistries = [
        ('MIX_large', 'Li-ion (MIX_large)'),
        ('ZN-coin', 'Zn-coin (ZN-coin)'),
        ('NAion', 'Na-ion (NAion)'),
        ('CALB', 'CALB (CALB)')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    axes = axes.flatten()
    
    for idx, (chem_id, title) in enumerate(chemistries):
        chem_df = df[df['chemistry'] == chem_id]
        ax = axes[idx]
        if len(chem_df) == 0:
            ax.set_title(f'{title}\n(no data)', fontsize=12)
            continue
            
        pivot_df = chem_df.pivot(index='dimension', columns='metric', values='correlation')
        metrics_order = ['Capacity', 'Plateau Voltage', 'dQ/dV Peak', 'Resistance']
        max_abs = pivot_df.abs().max(axis=1)
        sorted_dims = max_abs.sort_values(ascending=False).index
        pivot_df = pivot_df.loc[sorted_dims, metrics_order]
        
        sns.heatmap(pivot_df, annot=True, annot_kws={'size': 11, 'weight': 'bold'}, cmap='RdBu_r', vmin=-1.0, vmax=1.0, 
                    fmt=".2f", ax=ax, cbar_kws={'label': 'Pearson Correlation Coefficient (r)'})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        
        cbar = ax.collections[0].colorbar
        cbar.set_label('Pearson Correlation Coefficient (r)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        ax.set_title(f'({chr(97+idx)}) {title}', fontweight='bold', fontsize=16, pad=10)
        ax.set_ylabel('Latent Dimension', fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        
    fig.suptitle('Battery-JEPA Latent Space Physical Interpretation\n'
                  '(Top 12 Correlated Dimensions Across All 4 Chemistries)', 
                  fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_latent_physical_correlation.jpg'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_latent_physical_correlation.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 8 plotted.")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 9: Latent Correlation Distributions and Cumulative Explained Curves
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure_9():
    path = check_data_file('fig_latent_correlation_distributions.csv')
    if not path:
        return
    print("Plotting Figure 9: Correlation Distributions and Cumulative Explained Curves...")
    df = pd.read_csv(path)
    
    chemistries = [
        ('MIX_large', 'Li-ion (MIX_large)'),
        ('ZN-coin', 'Zn-coin (ZN-coin)'),
        ('NAion', 'Na-ion (NAion)'),
        ('CALB', 'CALB (CALB)')
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300)
    metrics = ['Capacity', 'Plateau Voltage', 'dQ/dV Peak', 'Resistance']
    colors = ['#0288d1', '#2e7d32', '#f57c00', '#d32f2f']
    
    for idx, (chem_id, title) in enumerate(chemistries):
        chem_df = df[df['chemistry'] == chem_id]
        if len(chem_df) == 0:
            continue
            
        # Panel Row 0: Histogram (KDE distributions)
        ax_hist = axes[0, idx]
        for m_idx, metric in enumerate(metrics):
            m_df = chem_df[chem_df['metric'] == metric].sort_values('dimension_idx')
            vals = m_df['pearson_r'].values
            sns.kdeplot(vals, ax=ax_hist, label=metric, color=colors[m_idx], lw=2.2, fill=True, alpha=0.08)
            
        ax_hist.set_title(f'({chr(97+idx)}) {title}', fontweight='bold', fontsize=18, pad=10)
        ax_hist.set_xlabel('Pearson Correlation (r)', fontsize=13)
        ax_hist.set_ylabel('Density', fontsize=13)
        ax_hist.set_xlim([-1.05, 1.05])
        ax_hist.grid(True, ls='--', alpha=0.3)
        for s in ['top', 'right']: ax_hist.spines[s].set_visible(False)
        if idx == 0:
            ax_hist.legend(loc='upper right', fontsize=11)
            
        # Panel Row 1: Cumulative Explained
        ax_cum = axes[1, idx]
        for m_idx, metric in enumerate(metrics):
            m_df = chem_df[chem_df['metric'] == metric].sort_values('dimension_idx')
            vals = m_df['pearson_r'].values
            sorted_abs_r = np.sort(np.abs(vals))[::-1]
            cum_sum = np.cumsum(sorted_abs_r)
            cum_pct = cum_sum / cum_sum[-1] * 100
            ax_cum.plot(np.arange(1, 65), cum_pct, label=metric, color=colors[m_idx], lw=2.5)
            
        ax_cum.set_title(f'({chr(101+idx)}) {title}', fontweight='bold', fontsize=18, pad=10)
        ax_cum.set_xlabel('Sorted Latent Dimensions (1-64)', fontsize=13)
        ax_cum.set_ylabel('Cumulative % of Absolute Correlation', fontsize=13)
        ax_cum.set_xlim([1, 64])
        ax_cum.set_ylim([0, 102])
        ax_cum.grid(True, ls='--', alpha=0.3)
        for s in ['top', 'right']: ax_cum.spines[s].set_visible(False)
        if idx == 0:
            ax_cum.legend(loc='lower right', fontsize=11)
            
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_latent_correlation_distributions.jpg'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_latent_correlation_distributions.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 9 plotted.")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 10: Latent Physical Scatter Plots (Zn-coin)
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure_10():
    path = check_data_file('fig_latent_physical_scatter.csv')
    if not path:
        return
    print("Plotting Figure 10: Latent Physical Scatter Plots...")
    df = pd.read_csv(path)
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=300)
    axes = axes.flatten()
    
    configs = [
        ('MIX_large', 'dim_38', 'plateau_voltage', 'Plateau Voltage (V)', '(a) Li-ion (MIX_large)', 'upper left'),
        ('ZN-coin', 'dim_47', 'resistance', 'Resistance Proxy (Ohm)', '(b) Zn-coin (ZN-coin)', 'upper right'),
        ('NAion', 'dim_60', 'dq_dv_peak', 'dQ/dV Peak Height (Ah/V)', '(c) Na-ion (NAion)', 'upper left'),
        ('CALB', 'dim_35', 'resistance', 'Resistance Proxy (Ohm)', '(d) CALB', 'upper left')
    ]
    
    for idx, (chem_id, x_col, y_col, y_lbl, title, loc) in enumerate(configs):
        ax = axes[idx]
        chem_df = df[df['chemistry'] == chem_id]
        if len(chem_df) == 0:
            ax.set_title(f'{title}\n(no data)', fontsize=16)
            continue
            
        x = chem_df[x_col].values
        y = chem_df[y_col].values
        
        color_key = 'Li-ion' if chem_id == 'MIX_large' else ('Zn-ion' if chem_id == 'ZN-coin' else ('Na-ion' if chem_id == 'NAion' else 'CALB'))
        color = CHEM_COLORS[color_key]
        
        from scipy.stats import pearsonr
        r_val, p_val = pearsonr(x, y)
        
        sns.regplot(x=x, y=y, ax=ax, color=color, 
                    scatter_kws={'s': 18, 'alpha': 0.35, 'edgecolors': 'none', 'zorder': 2},
                    line_kws={'color': '#d32f2f', 'lw': 2.0, 'ls': '--', 'zorder': 4})
                    
        sns.kdeplot(x=x, y=y, ax=ax, color=color, fill=True, alpha=0.22, levels=6, thresh=0.03, zorder=3)
                    
        ax.set_title(title, fontweight='bold', fontsize=18, pad=10)
        ax.set_xlabel(f'Latent Feature Value ({x_col.upper().replace("_", " ")})', fontsize=13)
        ax.set_ylabel(y_lbl, fontsize=13)
        ax.grid(True, ls='--', alpha=0.3)
        ax.tick_params(labelsize=11)
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
        
        p_text = f"p < 10^{{-4}}" if p_val < 1e-4 else f"p = {p_val:.4f}"
        ax.text(0.05 if 'left' in loc else 0.95, 0.95 if 'upper' in loc else 0.05,
                f'$r = {r_val:+.4f}$\n${p_text}$', fontsize=13, fontweight='bold',
                transform=ax.transAxes,
                verticalalignment='top' if 'upper' in loc else 'bottom',
                horizontalalignment='left' if 'left' in loc else 'right',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.9, ec='#bdbdbd', lw=1.2))
                
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_latent_physical_scatter.jpg'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_latent_physical_scatter.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 10 plotted.")

def main():
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()
    plot_figure_7()
    plot_figure_8()
    plot_figure_9()
    plot_figure_10()
    print("\nALL MANUSCRIPT FIGURES REPRODUCED SUCCESSFULLY IN ./figures/")

if __name__ == '__main__':
    main()
