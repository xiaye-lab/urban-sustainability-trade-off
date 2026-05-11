"""
step_8_Pareto_Frontier_Simulation.py
===========================================
Advanced Pareto Simulation with Capacity Filtering:
1. Filters lagging nodes to find those with actual physical space for interventions.
2. Applies constraint-aware interventions to ALL capable lagging nodes.
3. Simulates the outcomes through the trained GNN.
4. Uses a strict "Win-Win" sorting algorithm to find the absolute best 6 nodes.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore')

from build_graph import load_graph
from gnn_model import UrbanGNN 

print("="*70)
print("  SMART PARETO TRAJECTORY SIMULATION (CAPACITY FILTERED)")
print("="*70)

# ── 1. Load Data, Graph, and Archetypes ──────────────────────────────────────
output = load_graph()
data, df_nodes, feature_cols, target_cols = output[0], output[1], output[2], output[3]

df_arch = pd.read_csv("archetypes.csv")
if 'OBJECTID' in df_nodes.columns and 'OBJECTID' in df_arch.columns:
    df_full = pd.merge(df_nodes, df_arch[['OBJECTID', 'Archetype']], on='OBJECTID', how='inner')
else:
    df_full = df_nodes.copy()
    df_full['Archetype'] = df_arch['Archetype'].values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UrbanGNN(num_features=15, hidden_dim=64, num_targets=2, heads=4).to(device)
model.load_state_dict(torch.load("urban_gnn_model.pth", map_location=device))
model.eval()

# ── 2. Find "Capable" Lagging Nodes and Calculate Interventions ──────────────
print("Scanning city for lagging nodes with physical optimization capacity...")
target_benchmarks = {}
capable_ids = []

for arch in df_full['Archetype'].unique():
    df_arch_sub = df_full[df_full['Archetype'] == arch]
    
    mean_ecsi = df_arch_sub['ECSI'].mean()
    mean_sevi = df_arch_sub['SEVI'].mean()
    
    # Establish Targets based on top performers
    top_perf = df_arch_sub[(df_arch_sub['ECSI'] < df_arch_sub['ECSI'].quantile(0.25)) & 
                           (df_arch_sub['SEVI'] > df_arch_sub['SEVI'].quantile(0.75))]
    if len(top_perf) < 5: top_perf = df_arch_sub
        
    target_benchmarks[arch] = {
        'NDVI': top_perf['NDVI'].quantile(0.85),
        'LSI':  top_perf['LSI'].quantile(0.85)
    }
    
    # Filter 1: Must be lagging
    lagging = df_arch_sub[(df_arch_sub['ECSI'] > mean_ecsi) | (df_arch_sub['SEVI'] < mean_sevi)].copy()
    
    # Filter 2: Must have physical capacity (Max possible NDVI must be higher than current)
    lagging['Max_NDVI'] = 0.8 * (1.0 - lagging['BCR'])
    lagging['Capacity'] = lagging['Max_NDVI'] - lagging['NDVI']
    
    # Only keep nodes that have at least 5% physical space to add greenery or space to add LSI
    capable = lagging[(lagging['Capacity'] > 0.05) | (target_benchmarks[arch]['LSI'] - lagging['LSI'] > 0.1)]
    capable_ids.extend(capable['OBJECTID'].tolist())

df_candidates = df_full[df_full['OBJECTID'].isin(capable_ids)].copy()
print(f"Found {len(df_candidates)} capable candidate nodes.")

# ── 3. Apply Interventions to Candidates ─────────────────────────────────────
X_sim = df_full[feature_cols].copy()

for idx, row in df_candidates.iterrows():
    arch = row['Archetype']
    target_ndvi = target_benchmarks[arch]['NDVI']
    target_lsi  = target_benchmarks[arch]['LSI']
    
    max_physical_ndvi = 0.8 * (1.0 - row['BCR'])
    max_physical_ndvi = max(0, min(1.0, max_physical_ndvi))
    
    proposed_ndvi = min(target_ndvi, max_physical_ndvi)
    delta_ndvi = max(0, proposed_ndvi - row['NDVI'])
    delta_lsi = max(0, target_lsi - row['LSI'])
    
    X_sim.loc[idx, 'NDVI'] += delta_ndvi
    X_sim.loc[idx, 'LSI']  += delta_lsi
    
    df_candidates.at[idx, 'Applied_dNDVI'] = delta_ndvi
    df_candidates.at[idx, 'Applied_dLSI'] = delta_lsi
    df_candidates.at[idx, 'Blocked_by_BCR'] = "Yes" if (proposed_ndvi == max_physical_ndvi and delta_ndvi > 0) else "No"

# ── 4. Run GNN Simulation ────────────────────────────────────────────────────
print("Simulating optimization trajectories through the Graph Neural Network...")
X_sim_scaled = (X_sim - df_full[feature_cols].mean()) / df_full[feature_cols].std()
X_tensor = torch.tensor(X_sim_scaled.values, dtype=torch.float).to(device)
edge_index = data.edge_index.to(device)

with torch.no_grad():
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        pred_scaled = model(X_tensor, edge_index, data.edge_attr.to(device))
    elif hasattr(data, 'edge_weight') and data.edge_weight is not None:
        pred_scaled = model(X_tensor, edge_index, data.edge_weight.to(device))
    else:
        pred_scaled = model(X_tensor, edge_index)

pred_real = pred_scaled.cpu().numpy()
pred_real[:, 0] = (pred_real[:, 0] * df_full['ECSI'].std()) + df_full['ECSI'].mean()
pred_real[:, 1] = (pred_real[:, 1] * df_full['SEVI'].std()) + df_full['SEVI'].mean()

df_candidates['ECSI_Simulated'] = pred_real[df_candidates.index, 0]
df_candidates['SEVI_Simulated'] = pred_real[df_candidates.index, 1]

# ── 5. Strict Selection of the Top 6 Performers ──────────────────────────────
print("Extracting the absolute best trajectories per archetype...")

# Calculate deltas
df_candidates['d_ECSI'] = df_candidates['ECSI_Simulated'] - df_candidates['ECSI'] # We want Negative
df_candidates['d_SEVI'] = df_candidates['SEVI_Simulated'] - df_candidates['SEVI'] # We want Positive

# Strict Pareto Score: Heavily reward Win-Wins (Stress goes down, Vitality goes up)
# If stress goes up (d_ECSI > 0), we penalize the score by subtracting it.
df_candidates['Pareto_Score'] = df_candidates['d_SEVI'] - (df_candidates['d_ECSI'] * 2.0)

# Sort by Archetype, then by the highest Strict Pareto Score
df_best_6 = df_candidates.sort_values(['Archetype', 'Pareto_Score'], ascending=[True, False]).groupby('Archetype').head(2)

df_best_6.to_csv("pareto_best_6_nodes.csv", index=False)

# ── 6. Console Report ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  TOP 6 OPTIMIZED PARETO NODES (HIGH SIGNIFICANCE)")
print("="*70)

for _, row in df_best_6.iterrows():
    print(f"Node ID: {row['OBJECTID']} | Archetype: {row['Archetype']}")
    print(f"  Physical Interventions Applied:")
    print(f"    +Δ NDVI: {row['Applied_dNDVI']:.4f} (Hit BCR ceiling? {row['Blocked_by_BCR']})")
    print(f"    +Δ LSI : {row['Applied_dLSI']:.4f}")
    print(f"  Predicted Shifts:")
    print(f"    Stress (ECSI)  : {row['ECSI']:.4f} -> {row['ECSI_Simulated']:.4f}  (Δ {row['d_ECSI']:+.4f})")
    print(f"    Vitality (SEVI): {row['SEVI']:.4f} -> {row['SEVI_Simulated']:.4f}  (Δ {row['d_SEVI']:+.4f})\n")

# ── 7. Compute Pareto Frontier Lines ──────────────────────────────────────────
print("Computing Pareto frontier lines...")

def compute_pareto_frontier(ecsi_vals, sevi_vals, n_bins=100):
    """
    Compute two complementary Pareto frontier representations.

    True Pareto frontier
    --------------------
    A node is Pareto optimal if no other node has BOTH lower ECSI AND higher
    SEVI simultaneously. Sorting by ECSI ascending and scanning for
    monotone-increasing SEVI gives the exact Pareto-optimal set — the
    upper-left boundary of the scatter.

    Smoothed upper envelope
    -----------------------
    Divides the ECSI range into n_bins equal intervals and takes the
    maximum SEVI in each bin. A monotone filter is then applied to enforce
    the Pareto property. The result is a smooth continuous line suitable
    for publication figures.

    Parameters
    ----------
    ecsi_vals : array-like  (lower = better environment)
    sevi_vals : array-like  (higher = better vitality)
    n_bins    : int  resolution of the smoothed envelope

    Returns
    -------
    frontier_e, frontier_s : arrays  — true Pareto-optimal points (scatter)
    envelope_e, envelope_s : arrays  — smoothed upper envelope (line)
    """
    ecsi_arr = np.asarray(ecsi_vals)
    sevi_arr = np.asarray(sevi_vals)

    # ── True Pareto frontier ─────────────────────────────────────────────
    order    = np.argsort(ecsi_arr)
    e_sorted = ecsi_arr[order]
    s_sorted = sevi_arr[order]

    pareto_idx = []
    best_sevi  = -np.inf
    for i in range(len(e_sorted)):
        if s_sorted[i] > best_sevi:
            pareto_idx.append(i)
            best_sevi = s_sorted[i]

    frontier_e = e_sorted[pareto_idx]
    frontier_s = s_sorted[pareto_idx]

    # ── Smoothed upper envelope ──────────────────────────────────────────
    bins = np.linspace(ecsi_arr.min(), ecsi_arr.max(), n_bins + 1)
    bin_centers, bin_max = [], []
    for i in range(len(bins) - 1):
        mask = (ecsi_arr >= bins[i]) & (ecsi_arr < bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append(0.5 * (bins[i] + bins[i + 1]))
            bin_max.append(float(sevi_arr[mask].max()))

    # Monotone filter: scan right-to-left, keep only points with max SEVI seen
    env_e, env_s = [], []
    best = -np.inf
    for e, s in zip(reversed(bin_centers), reversed(bin_max)):
        if s > best:
            best = s
            env_e.insert(0, e)
            env_s.insert(0, s)

    # Optional: extend frontier to ECSI=0 (theoretical ideal)
    if env_e and env_e[0] > ecsi_arr.min():
        env_e.insert(0, ecsi_arr.min())
        env_s.insert(0, env_s[0])

    return (np.array(frontier_e), np.array(frontier_s),
            np.array(env_e),      np.array(env_s))


# Compute on all 9,130 nodes using observed ECSI/SEVI
frontier_e, frontier_s, envelope_e, envelope_s = compute_pareto_frontier(
    df_full['ECSI'].values,
    df_full['SEVI'].values,
    n_bins=120,
)

# Also compute the simulated frontier (after interventions)
sim_e_all = pred_real[:, 0]
sim_s_all = pred_real[:, 1]
_, _, sim_env_e, sim_env_s = compute_pareto_frontier(
    sim_e_all, sim_s_all, n_bins=120
)

print(f"  Observed Pareto frontier: {len(frontier_e)} points")
print(f"  Smoothed envelope:        {len(envelope_e)} bins")

# ── 8. Plot the Pareto Frontier ──────────────────────────────────────────────
print("Generating Pareto Frontier visualization...")
fig, ax = plt.subplots(figsize=(11, 9))

# ── Background scatter ───────────────────────────────────────────────────────
ax.scatter(df_full['ECSI'], df_full['SEVI'],
           color='lightgray', alpha=0.25, s=8,
           label='All urban nodes (9,130)', zorder=1)

# ── Observed Pareto frontier lines ───────────────────────────────────────────
# Smoothed envelope — main frontier line
ax.plot(envelope_e, envelope_s,
        color='#2563EB', linewidth=2.2, linestyle='-',
        label='Observed Pareto frontier (smoothed)', zorder=3)

# True Pareto-optimal points — show as small markers on the line
ax.scatter(frontier_e, frontier_s,
           color='#2563EB', s=30, zorder=4, alpha=0.7,
           label=f'Pareto-optimal nodes (n={len(frontier_e)})')

# ── Simulated frontier after interventions ────────────────────────────────────
ax.plot(sim_env_e, sim_env_s,
        color='#16A34A', linewidth=2.0, linestyle='--',
        label='Simulated frontier (post-intervention)', zorder=3)

# ── Archetype colours ─────────────────────────────────────────────────────────
colors = {
    'Economic Anchor':    '#E24B4A',
    'Balanced Transition':'#F1A226',
    'Ecological Buffer':  '#1D9E75',
}

for idx, row in df_best_6.iterrows():
    arch = row['Archetype']
    c    = colors[arch]

    # Original state
    ax.scatter(row['ECSI'], row['SEVI'],
               color=c, edgecolor='black', linewidth=0.8,
               s=90, marker='o', zorder=6)

    # Simulated (optimised) state
    ax.scatter(row['ECSI_Simulated'], row['SEVI_Simulated'],
               color=c, edgecolor='black', linewidth=0.8,
               s=180, marker='*', zorder=6)

    # Intervention arrow
    ax.annotate('',
                xy=(row['ECSI_Simulated'], row['SEVI_Simulated']),
                xytext=(row['ECSI'], row['SEVI']),
                arrowprops=dict(
                    facecolor=c, edgecolor=c,
                    alpha=0.90, width=1.8, headwidth=9, shrink=0.04,
                ),
                zorder=5)

# ── Pareto zone annotation ────────────────────────────────────────────────────
ax.text(0.04, 0.96,
        'Pareto Optimal Zone\n(High Vitality, Low Stress)',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1D9E75', alpha=0.18,
                  edgecolor='#1D9E75', linewidth=1.0))

# Shade the upper-left region (near-Pareto zone)
ax.fill_betweenx([envelope_s[0], 1.0],
                 [0.0, 0.0],
                 [envelope_e[0], envelope_e[0]],
                 alpha=0.06, color='#2563EB', zorder=0)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mlines.Line2D([0], [0], color='#2563EB', linewidth=2.2,
                  label='Observed Pareto frontier'),
    mlines.Line2D([0], [0], color='#16A34A', linewidth=2.0, linestyle='--',
                  label='Simulated frontier (post-intervention)'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markersize=8, label='Original state (top nodes)'),
    mlines.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                  markersize=12, label='Optimised state (simulated)'),
    mlines.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor='#E24B4A', markersize=9,
                  label='Economic Anchor'),
    mlines.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor='#F1A226', markersize=9,
                  label='Balanced Transition'),
    mlines.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor='#1D9E75', markersize=9,
                  label='Ecological Buffer'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9.5,
          framealpha=0.92, edgecolor='#cccccc')

# ── Labels and formatting ─────────────────────────────────────────────────────
ax.set_title('Simulated Optimal Interventions on the ECSI–SEVI Pareto Frontier',
             fontsize=13, pad=14)
ax.set_xlabel('Environmental Composite Stress Index (ECSI)  ← lower is better',
              fontsize=11)
ax.set_ylabel('Socioeconomic Vitality Index (SEVI)  → higher is better',
              fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, linestyle='--', alpha=0.35)

plt.tight_layout()
plt.savefig("pareto_frontier_simulation.png", dpi=300, bbox_inches='tight')
print("Saved visualization -> pareto_frontier_simulation.png")
print("="*70)
