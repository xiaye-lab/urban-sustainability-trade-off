"""
Simulate_interventions.py
===========================================
Applies the constraint-aware optimization strategies to the physical features 
of lagging urban nodes, and re-runs the trained GATv2 model to predict the 
final impact on ECSI and SEVI.
"""

import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from build_graph import load_graph

print("="*70)
print("  GNN OPTIMIZATION SIMULATION (CONSTRAINT-AWARE)")
print("="*70)

# ── 1. Load Data and Graph ───────────────────────────────────────────────────
print("Loading original graph data and scaler...")
output = load_graph()
data, df_nodes, feature_cols, target_cols = output[0], output[1], output[2], output[3]

import joblib
try:
    # Load the actual target scaler saved during training
    target_scaler = joblib.load("target_scaler.pkl")
except FileNotFoundError:
    print("target_scaler.pkl not found. Assuming targets were not scaled.")
    target_scaler = None

# ── 2. Load the Trained Model (FIXED) ────────────────────────────────────────
print("Loading trained GNN model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # 1. Import your exact model class from gnn_model.py
    from gnn_model import UrbanGNN 

    # 2. Rebuild the empty model architecture using your exact training parameters
    model = UrbanGNN(
        num_features=15, 
        hidden_dim=64, 
        num_targets=2, 
        heads=4
    ).to(device)
    
    # 3. Load the saved weights into the architecture
    model.load_state_dict(torch.load("urban_gnn_model.pth", map_location=device))
    model.eval()
    print("Model weights successfully loaded!")
    
except Exception as e:
    print(f"\nCRITICAL ERROR loading model: {e}")
    print("-> Make sure 'gnn_model.py' is in this folder and contains the 'UrbanGNN' class.")
    exit()

# ── 3. Load Strategies and Archetypes ────────────────────────────────────────
print("Loading constraint-aware strategies...")
df_strat = pd.read_csv("constraint_aware_strategies.csv")
df_arch = pd.read_csv("archetypes.csv")

# Merge archetypes into the working dataset
if 'OBJECTID' in df_nodes.columns and 'OBJECTID' in df_arch.columns:
    df_sim = pd.merge(df_nodes, df_arch[['OBJECTID', 'Archetype']], on='OBJECTID', how='inner')
else:
    df_sim = df_nodes.copy()
    df_sim['Archetype'] = df_arch['Archetype'].values

# Create a copy of the feature matrix to modify
X_sim = df_sim[feature_cols].copy()

# ── 4. Apply Interventions to Lagging Nodes ──────────────────────────────────
print("Applying physical interventions to the landscape...")

modified_count = 0
for _, strat in df_strat.iterrows():
    arch = strat['Archetype']
    d_ndvi = strat['Avg_Delta_NDVI']
    d_lsi = strat['Avg_Delta_LSI']
    
    # Archetype Baseline
    mean_ecsi = df_sim[df_sim['Archetype'] == arch]['ECSI'].mean()
    mean_sevi = df_sim[df_sim['Archetype'] == arch]['SEVI'].mean()
    
    # Identify Lagging Nodes
    mask = (df_sim['Archetype'] == arch) & ((df_sim['ECSI'] > mean_ecsi) | (df_sim['SEVI'] < mean_sevi))
    modified_count += mask.sum()
    
    # Apply LSI Shift (Functional edge density)
    X_sim.loc[mask, 'LSI'] += d_lsi
    
    # Apply NDVI Shift with Strict Physical Boundary (1 - BCR)
    max_physical_ndvi = 0.8 * (1.0 - X_sim.loc[mask, 'BCR'])
    proposed_ndvi = X_sim.loc[mask, 'NDVI'] + d_ndvi
    X_sim.loc[mask, 'NDVI'] = np.minimum(proposed_ndvi, max_physical_ndvi)

print(f" -> Interventions physically applied to {modified_count} lagging nodes.")

# ── 5. Run GNN Forward Pass ──────────────────────────────────────────────────
print("Standardizing features and running simulated city through GNN...")

# CRITICAL FIX: Standardize the modified features using the original dataset's mean and std!
X_sim_scaled = (X_sim - df_nodes[feature_cols].mean()) / df_nodes[feature_cols].std()

# Convert the scaled features to a tensor
X_tensor = torch.tensor(X_sim_scaled.values, dtype=torch.float).to(device)
edge_index = data.edge_index.to(device)

with torch.no_grad():
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        pred_scaled = model(X_tensor, edge_index, data.edge_attr.to(device))
    elif hasattr(data, 'edge_weight') and data.edge_weight is not None:
        pred_scaled = model(X_tensor, edge_index, data.edge_weight.to(device))
    else:
        pred_scaled = model(X_tensor, edge_index)

# CRITICAL FIX: Manually reverse the target scaling to get back to the [0, 1] index range!
pred_real = pred_scaled.cpu().numpy()
pred_real[:, 0] = (pred_real[:, 0] * df_nodes['ECSI'].std()) + df_nodes['ECSI'].mean()
pred_real[:, 1] = (pred_real[:, 1] * df_nodes['SEVI'].std()) + df_nodes['SEVI'].mean()

df_sim['ECSI_Simulated'] = pred_real[:, 0]
df_sim['SEVI_Simulated'] = pred_real[:, 1]

# ── 6. Print Paper-Ready Analysis ────────────────────────────────────────────
print("\n" + "="*70)
print("  SIMULATION RESULTS: IMPACT ON URBAN PERFORMANCE")
print("="*70)

delta_ecsi_total = df_sim['ECSI_Simulated'].mean() - df_sim['ECSI'].mean()
delta_sevi_total = df_sim['SEVI_Simulated'].mean() - df_sim['SEVI'].mean()

print("CITY-WIDE IMPACT (All Nodes):")
print(f"  Overall ECSI (Stress)   : {df_sim['ECSI'].mean():.4f} -> {df_sim['ECSI_Simulated'].mean():.4f}  (Δ {delta_ecsi_total:+.4f})")
print(f"  Overall SEVI (Vitality) : {df_sim['SEVI'].mean():.4f} -> {df_sim['SEVI_Simulated'].mean():.4f}  (Δ {delta_sevi_total:+.4f})")

print("\nARCHETYPE IMPACT (Lagging Nodes Only):")
for arch in df_strat['Archetype']:
    mean_ecsi = df_sim[df_sim['Archetype'] == arch]['ECSI'].mean()
    mean_sevi = df_sim[df_sim['Archetype'] == arch]['SEVI'].mean()
    mask = (df_sim['Archetype'] == arch) & ((df_sim['ECSI'] > mean_ecsi) | (df_sim['SEVI'] < mean_sevi))
    
    d_ecsi = df_sim.loc[mask, 'ECSI_Simulated'].mean() - df_sim.loc[mask, 'ECSI'].mean()
    d_sevi = df_sim.loc[mask, 'SEVI_Simulated'].mean() - df_sim.loc[mask, 'SEVI'].mean()
    
    print(f"  [{arch}] ({mask.sum()} nodes optimized)")
    print(f"    Stress Relief (ΔECSI)  : {d_ecsi:+.4f}")
    print(f"    Vitality Boost (ΔSEVI) : {d_sevi:+.4f}")

df_sim.to_csv("simulated_optimizations_final.csv", index=False)
print("\nSaved full simulation results -> simulated_optimizations_final.csv")
print("="*70)

# ── 6. Print Paper-Ready Analysis ────────────────────────────────────────────
print("\n" + "="*70)
print("  SIMULATION RESULTS: IMPACT ON URBAN PERFORMANCE")
print("="*70)

delta_ecsi_total = df_sim['ECSI_Simulated'].mean() - df_sim['ECSI'].mean()
delta_sevi_total = df_sim['SEVI_Simulated'].mean() - df_sim['SEVI'].mean()

print("CITY-WIDE IMPACT (All Nodes):")
print(f"  Overall ECSI (Stress)   : {df_sim['ECSI'].mean():.4f} -> {df_sim['ECSI_Simulated'].mean():.4f}  (Δ {delta_ecsi_total:+.4f})")
print(f"  Overall SEVI (Vitality) : {df_sim['SEVI'].mean():.4f} -> {df_sim['SEVI_Simulated'].mean():.4f}  (Δ {delta_sevi_total:+.4f})")

print("\nARCHETYPE IMPACT (Lagging Nodes Only):")
for arch in df_strat['Archetype']:
    mean_ecsi = df_sim[df_sim['Archetype'] == arch]['ECSI'].mean()
    mean_sevi = df_sim[df_sim['Archetype'] == arch]['SEVI'].mean()
    mask = (df_sim['Archetype'] == arch) & ((df_sim['ECSI'] > mean_ecsi) | (df_sim['SEVI'] < mean_sevi))
    
    d_ecsi = df_sim.loc[mask, 'ECSI_Simulated'].mean() - df_sim.loc[mask, 'ECSI'].mean()
    d_sevi = df_sim.loc[mask, 'SEVI_Simulated'].mean() - df_sim.loc[mask, 'SEVI'].mean()
    
    print(f"  [{arch}] ({mask.sum()} nodes optimized)")
    print(f"    Stress Relief (ΔECSI)  : {d_ecsi:+.4f}")
    print(f"    Vitality Boost (ΔSEVI) : {d_sevi:+.4f}")

df_sim.to_csv("simulated_optimizations_final.csv", index=False)
print("\nSaved full simulation results -> simulated_optimizations_final.csv")
print("="*70)