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

# ── 7. Plot the Pareto Frontier ──────────────────────────────────────────────
print("Generating Pareto Frontier visualization...")
plt.figure(figsize=(10, 8))

# Plot background
plt.scatter(df_full['ECSI'], df_full['SEVI'], color='lightgray', alpha=0.3, s=10, label='All Urban Nodes')

colors = {'Economic Anchor': '#E24B4A', 'Balanced Transition': '#F1A226', 'Ecological Buffer': '#1D9E75'}

for idx, row in df_best_6.iterrows():
    arch = row['Archetype']
    c = colors[arch]
    
    # Original
    plt.scatter(row['ECSI'], row['SEVI'], color=c, edgecolor='black', s=80, marker='o', zorder=5)
    # Simulated
    plt.scatter(row['ECSI_Simulated'], row['SEVI_Simulated'], color=c, edgecolor='black', s=150, marker='*', zorder=5)
    # Arrow
    plt.annotate('', xy=(row['ECSI_Simulated'], row['SEVI_Simulated']), 
                 xytext=(row['ECSI'], row['SEVI']),
                 arrowprops=dict(facecolor=c, edgecolor=c, alpha=0.9, width=2, headwidth=9, shrink=0.03),
                 zorder=4)

legend_elements = [
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Original State'),
    mlines.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=12, label='Optimized State'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E24B4A', markersize=8, label='Economic Anchor'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F1A226', markersize=8, label='Balanced Transition'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1D9E75', markersize=8, label='Ecological Buffer')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.title('Simulated Optimal Interventions Towards the Pareto Frontier', fontsize=14, pad=15)
plt.xlabel('Environmental Stress Index (ECSI) → Lower is Better', fontsize=12)
plt.ylabel('Socioeconomic Vitality Index (SEVI) → Higher is Better', fontsize=12)

plt.text(0.05, 0.95, 'Pareto Optimal Zone\n(High Vitality, Low Stress)', 
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#1D9E75', alpha=0.2))

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("pareto_frontier_simulation.png", dpi=300)
print("Saved visualization -> pareto_frontier_simulation.png")
print("="*70)