"""
step_8_Pareto_Frontier_Simulation.py
===========================================
Selects 6 specific lagging nodes across the 3 archetypes, applies site-specific 
constraint-aware morphological interventions, and simulates their trajectory 
toward the Pareto Frontier (Low ECSI, High SEVI) using the trained GNN.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from build_graph import load_graph
from gnn_model import UrbanGNN 

print("="*70)
print("  PARETO FRONTIER TRAJECTORY SIMULATION (6 SPECIFIC NODES)")
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

# ── 2. Load the Trained GNN Model ────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UrbanGNN(num_features=15, hidden_dim=64, num_targets=2, heads=4).to(device)
model.load_state_dict(torch.load("urban_gnn_model.pth", map_location=device))
model.eval()

# ── 3. Select 6 Specific Lagging Nodes (2 per Archetype) ─────────────────────
print("Selecting 6 representative lagging nodes...")
selected_node_ids = []
target_benchmarks = {}

np.random.seed(42) # For reproducible node selection

for arch in df_full['Archetype'].unique():
    df_arch_sub = df_full[df_full['Archetype'] == arch]
    
    # Calculate baseline and top performers for target setting
    mean_ecsi = df_arch_sub['ECSI'].mean()
    mean_sevi = df_arch_sub['SEVI'].mean()
    
    top_performers = df_arch_sub[(df_arch_sub['ECSI'] < df_arch_sub['ECSI'].quantile(0.25)) & 
                                 (df_arch_sub['SEVI'] > df_arch_sub['SEVI'].quantile(0.75))]
    if len(top_performers) < 5: top_performers = df_arch_sub
        
    target_benchmarks[arch] = {
        'NDVI': top_performers['NDVI'].quantile(0.85),
        'LSI':  top_performers['LSI'].quantile(0.85)
    }
    
    # Find lagging nodes
    lagging = df_arch_sub[(df_arch_sub['ECSI'] > mean_ecsi) & (df_arch_sub['SEVI'] < mean_sevi)]
    
    # Randomly select 2 nodes
    sampled = lagging.sample(n=2, random_state=42)
    selected_node_ids.extend(sampled['OBJECTID'].tolist())

# Filter our tracking dataframe to just these 6 nodes
df_6_nodes = df_full[df_full['OBJECTID'].isin(selected_node_ids)].copy()

# ── 4. Calculate Constraints and Apply Interventions ─────────────────────────
print("Applying physical constraints and maximizing feasible interventions...")

X_sim = df_full[feature_cols].copy()

for idx, row in df_6_nodes.iterrows():
    arch = row['Archetype']
    target_ndvi = target_benchmarks[arch]['NDVI']
    target_lsi  = target_benchmarks[arch]['LSI']
    
    # Physical NDVI Constraint
    max_physical_ndvi = 0.8 * (1.0 - row['BCR'])
    max_physical_ndvi = max(0, min(1.0, max_physical_ndvi))
    
    proposed_ndvi = min(target_ndvi, max_physical_ndvi)
    delta_ndvi = max(0, proposed_ndvi - row['NDVI'])
    
    # LSI Constraint
    delta_lsi = max(0, target_lsi - row['LSI'])
    
    # Apply to the simulation matrix
    X_sim.loc[idx, 'NDVI'] += delta_ndvi
    X_sim.loc[idx, 'LSI']  += delta_lsi
    
    # Save parameters for printing
    df_6_nodes.at[idx, 'Applied_dNDVI'] = delta_ndvi
    df_6_nodes.at[idx, 'Applied_dLSI'] = delta_lsi
    df_6_nodes.at[idx, 'Blocked_by_BCR'] = "Yes" if (proposed_ndvi == max_physical_ndvi and delta_ndvi > 0) else "No"

# ── 5. Run GNN Forward Pass ──────────────────────────────────────────────────
print("Simulating new outcomes through the Graph Neural Network...")

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

# Extract the new simulated values for our 6 nodes
df_6_nodes['ECSI_Simulated'] = pred_real[df_6_nodes.index, 0]
df_6_nodes['SEVI_Simulated'] = pred_real[df_6_nodes.index, 1]

# ── 6. Console Report ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  6-NODE PARETO TRAJECTORY REPORT")
print("="*70)

for _, row in df_6_nodes.iterrows():
    print(f"Node ID: {row['OBJECTID']} | Archetype: {row['Archetype']}")
    print(f"  Physical Interventions Applied:")
    print(f"    +Δ NDVI: {row['Applied_dNDVI']:.4f} (Hit BCR ceiling? {row['Blocked_by_BCR']})")
    print(f"    +Δ LSI : {row['Applied_dLSI']:.4f}")
    print(f"  Predicted Shifts:")
    print(f"    Stress (ECSI)  : {row['ECSI']:.4f} -> {row['ECSI_Simulated']:.4f}  (Δ {row['ECSI_Simulated'] - row['ECSI']:+.4f})")
    print(f"    Vitality (SEVI): {row['SEVI']:.4f} -> {row['SEVI_Simulated']:.4f}  (Δ {row['SEVI_Simulated'] - row['SEVI']:+.4f})\n")

# ── 7. Plot the Pareto Frontier ──────────────────────────────────────────────
print("Generating Pareto Frontier visualization...")

plt.figure(figsize=(10, 8))

# Plot all background nodes in light gray
plt.scatter(df_full['ECSI'], df_full['SEVI'], color='lightgray', alpha=0.3, s=10, label='All Urban Nodes (Background)')

colors = {'Economic Anchor': '#E24B4A', 'Balanced Transition': '#F1A226', 'Ecological Buffer': '#1D9E75'}

# Plot the 6 nodes and draw arrows
for idx, row in df_6_nodes.iterrows():
    arch = row['Archetype']
    c = colors[arch]
    
    # Original Point
    plt.scatter(row['ECSI'], row['SEVI'], color=c, edgecolor='black', s=80, marker='o', zorder=5)
    
    # Simulated Point
    plt.scatter(row['ECSI_Simulated'], row['SEVI_Simulated'], color=c, edgecolor='black', s=150, marker='*', zorder=5)
    
    # Draw Arrow
    plt.annotate('', xy=(row['ECSI_Simulated'], row['SEVI_Simulated']), 
                 xytext=(row['ECSI'], row['SEVI']),
                 arrowprops=dict(facecolor=c, edgecolor=c, alpha=0.8, width=1.5, headwidth=8, shrink=0.05),
                 zorder=4)

# Create custom legend
import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Original State'),
    mlines.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=12, label='Optimized State'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E24B4A', markersize=8, label='Economic Anchor'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F1A226', markersize=8, label='Balanced Transition'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1D9E75', markersize=8, label='Ecological Buffer')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.title('Simulated Interventions Moving Towards the Pareto Frontier', fontsize=14, pad=15)
plt.xlabel('Environmental Stress Index (ECSI) → Lower is Better', fontsize=12)
plt.ylabel('Socioeconomic Vitality Index (SEVI) → Higher is Better', fontsize=12)

# Add a text box indicating the "Ideal Pareto Zone"
plt.text(0.05, 0.95, 'Pareto Optimal Zone\n(High Vitality, Low Stress)', 
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#1D9E75', alpha=0.2))

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("pareto_frontier_simulation.png", dpi=300)
print("Saved visualization -> pareto_frontier_simulation.png")
print("="*70)