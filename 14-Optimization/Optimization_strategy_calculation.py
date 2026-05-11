"""
Optimization_strategy_calculation.py
===========================================
Calculates "Constraint-Aware" optimization strategies for urban planning.
Replaces flat standard-deviation shifts with physically bounded, archetype-specific 
interventions to address Reviewer critiques regarding real-world feasibility.

Rules:
1. Targeting: Only targets nodes underperforming their archetype's average.
2. Empirical Feasibility: Shifts features to the 85th percentile of top-performers 
   in that specific archetype, rather than an arbitrary +1.0 sigma.
3. Physical Bounding: NDVI (vegetation) increases are strictly capped by 
   the available unbuilt land: Max_NDVI = 0.8 * (1 - BCR).
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import your graph loader
from build_graph import load_graph

print("="*70)
print("  CONSTRAINT-AWARE URBAN OPTIMIZATION STRATEGIES")
print("="*70)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("Loading graph data...")
# The *_ catches any extra variables returned by load_graph to prevent unpacking errors
data, df_nodes, feature_cols, target_cols, *_ = load_graph()

# Load Archetypes (Assuming compute_archetypes.py saved this)
try:
    df_arch = pd.read_csv("archetypes.csv")
    # Merge on OBJECTID to ensure perfect alignment
    if 'OBJECTID' in df_nodes.columns and 'OBJECTID' in df_arch.columns:
        df = pd.merge(df_nodes, df_arch[['OBJECTID', 'Archetype']], on='OBJECTID', how='inner')
    else:
        # Fallback: assume row order is perfectly preserved
        df_nodes['Archetype'] = df_arch['Archetype'].values
        df = df_nodes
except FileNotFoundError:
    print("WARNING: archetypes.csv not found! Simulating archetypes based on SEVI for demonstration...")
    df_nodes['Archetype'] = pd.qcut(df_nodes['SEVI'], q=3, labels=['Ecological Buffer', 'Balanced Transition', 'Economic Anchor'])
    df = df_nodes

# Define the targets we want to optimize (based on SHAP findings)
# ECSI Reducers: Increase NDVI, Decrease/Optimize SVF
# SEVI Enhancers: Increase LSI (Landscape configuration), Increase FAR (Vertical volume)
features_to_optimize = ['NDVI', 'LSI', 'FAR', 'BCR']

# ── 2. Archetype-Specific Processing ─────────────────────────────────────────
results = []

for arch in df['Archetype'].unique():
    print(f"\n[{arch.upper()}]")
    df_sub = df[df['Archetype'] == arch].copy()
    
    # Calculate Archetype Baselines
    mean_ecsi = df_sub['ECSI'].mean()
    mean_sevi = df_sub['SEVI'].mean()
    
    # Identify the Top 15% "Best Practice" nodes in this archetype (Low ECSI, High SEVI)
    # We use these to set realistic target bounds instead of arbitrary shifts
    top_performers = df_sub[(df_sub['ECSI'] < df_sub['ECSI'].quantile(0.25)) & 
                            (df_sub['SEVI'] > df_sub['SEVI'].quantile(0.75))]
    
    if len(top_performers) < 5:
        top_performers = df_sub # Fallback if too strict
        
    target_ndvi = top_performers['NDVI'].quantile(0.85)
    target_lsi  = top_performers['LSI'].quantile(0.85)
    target_far  = top_performers['FAR'].quantile(0.85)

    # ── 3. Isolate Underperforming Nodes (The "Lagging" areas) ───────────────
    # Nodes that have worse stress or worse vitality than the archetype average
    lagging_mask = (df_sub['ECSI'] > mean_ecsi) | (df_sub['SEVI'] < mean_sevi)
    df_lag = df_sub[lagging_mask].copy()
    
    print(f"  Total Nodes: {len(df_sub)} | Lagging Nodes targeted: {len(df_lag)}")
    
    # ── 4. Calculate Physically Bounded Interventions ────────────────────────
    
    # INTERVENTION A: Greening (NDVI)
    # CONSTRAINT: You cannot plant trees on top of a building footprint.
    # Max possible NDVI is roughly 80% of the non-built area (1 - BCR).
    df_lag['Max_Physical_NDVI'] = 0.8 * (1.0 - df_lag['BCR'])
    df_lag['Max_Physical_NDVI'] = df_lag['Max_Physical_NDVI'].clip(lower=0, upper=1.0)
    
    # The proposed target is the archetype "Best Practice", but CAPPED by physical reality
    df_lag['Proposed_NDVI'] = np.minimum(target_ndvi, df_lag['Max_Physical_NDVI'])
    
    # Calculate the actual feasible shift
    df_lag['Delta_NDVI'] = df_lag['Proposed_NDVI'] - df_lag['NDVI']
    # Filter out nodes that are already above the target
    df_lag.loc[df_lag['Delta_NDVI'] < 0, 'Delta_NDVI'] = 0 
    
    # INTERVENTION B: Landscape Configuration (LSI)
    # CONSTRAINT: LSI shifts must remain within the empirical distribution of the archetype
    df_lag['Proposed_LSI'] = target_lsi
    df_lag['Delta_LSI'] = df_lag['Proposed_LSI'] - df_lag['LSI']
    df_lag.loc[df_lag['Delta_LSI'] < 0, 'Delta_LSI'] = 0

    # ── 5. Summarize Feasible Shifts ─────────────────────────────────────────
    avg_feasible_ndvi = df_lag['Delta_NDVI'].mean()
    max_feasible_ndvi = df_lag['Delta_NDVI'].max()
    blocked_by_bcr = len(df_lag[df_lag['NDVI'] + df_lag['Delta_NDVI'] >= df_lag['Max_Physical_NDVI']])
    
    avg_feasible_lsi = df_lag['Delta_LSI'].mean()
    
    print(f"  -> Target Best-Practice NDVI for archetype: {target_ndvi:.4f}")
    print(f"  -> FEASIBLE NDVI SHIFT : +{avg_feasible_ndvi:.4f} on average (Max: +{max_feasible_ndvi:.4f})")
    print(f"     *Note: {blocked_by_bcr} lagging nodes hit their maximum physical BCR constraint.")
    print(f"  -> FEASIBLE LSI SHIFT  : +{avg_feasible_lsi:.4f} (Increasing functional edge density)")
    
    results.append({
        'Archetype': arch,
        'Lagging_Nodes': len(df_lag),
        'Avg_Delta_NDVI': round(avg_feasible_ndvi, 4),
        'Blocked_by_BCR': blocked_by_bcr,
        'Avg_Delta_LSI': round(avg_feasible_lsi, 4)
    })

# Save the realistic strategies
df_results = pd.DataFrame(results)
df_results.to_csv("constraint_aware_strategies.csv", index=False)
print("\n" + "="*70)
print("Saved feasible optimization parameters -> constraint_aware_strategies.csv")
print("="*70)