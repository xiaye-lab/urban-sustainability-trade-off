import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import joblib

from build_graph import load_graph
from gnn_model import UrbanGNN

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
# Path to the folder containing urban_gnn_model.pth and target_scaler.pkl
# Adjust this to match your directory structure
MODEL_DIR = r"D:\04-SG_GNN_ESPI\0-major revision (6.9)\1_GNN\12-GNN"

NUM_SAMPLES = 30       
SEED = 42

STRATEGIES = {
    "Increase Density (BCR)": {
        "feature": "BCR",     
        "delta": 1.0,         
        "desc": "Increase BCR"
    },
    "Increase Height (AHI)": {
        "feature": "AHI",     
        "delta": 1.0,        
        "desc": "Increase AHI"
    },
    "Increase Greenery (NDVI)": {
        "feature": "NDVI",    
        "delta": 1.0,         
        "desc": "Increase NDVI"
    }
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

# ==============================================================================
# 1. CORE CALCULATION: BENEFIT
# ==============================================================================
def calculate_benefit(model, data, target_node_idx, feat_idx, delta, target_scaler):
    edge_index = data.edge_index.cpu().numpy()
    neighbors = edge_index[1][edge_index[0] == target_node_idx]

    model.eval()
    with torch.no_grad():
        pred_old_raw = model(data.x, data.edge_index, data.edge_weight).cpu().numpy()
    pred_old = target_scaler.inverse_transform(pred_old_raw)

    x_mod = data.x.clone()
    x_mod[target_node_idx, feat_idx] += delta

    with torch.no_grad():
        pred_new_raw = model(x_mod, data.edge_index, data.edge_weight).cpu().numpy()
    pred_new = target_scaler.inverse_transform(pred_new_raw)

    diff = pred_new - pred_old

    benefit = np.zeros_like(diff)
    benefit[:, 0] = -1 * diff[:, 0]  # ECSI: lower stress = positive benefit
    benefit[:, 1] =      diff[:, 1]  # SEVI: higher vitality = positive benefit

    local_gain = benefit[target_node_idx]
    neighbor_gain = benefit[neighbors].sum(axis=0) if len(neighbors) > 0 \
                    else np.zeros_like(local_gain)

    return local_gain, neighbor_gain

# ==============================================================================
# 2. MAIN ANALYSIS LOOP
# ==============================================================================
def run_comparison():
    print("[1/5] Loading Data...")
    data, df_nodes, feature_cols, target_cols, feat_scaler = load_graph(
        shp_file="fishnet_final_composite.shp", k_sim=8
    )

    print(f"[2/5] Loading Model...")
    model = UrbanGNN(num_features=data.x.shape[1], hidden_dim=64,
                     num_targets=len(target_cols), heads=4)
    try:
        model.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, "urban_gnn_model.pth"), weights_only=True))
    except FileNotFoundError:
        print(f"Error: 'urban_gnn_model.pth' not found in {MODEL_DIR}")
        return
    model.eval()

    try:
        target_scaler = joblib.load(os.path.join(MODEL_DIR, "target_scaler.pkl"))
        print("   Target scaler loaded.")
    except FileNotFoundError:
        print(f"Error: 'target_scaler.pkl' not found in {MODEL_DIR}. Run train_gnn.py first.")
        return

    print(f"[3/5] Categorizing Nodes (Hubs vs Normal)...")
    row, col = data.edge_index
    deg = torch.bincount(row, minlength=data.num_nodes).numpy()

    with torch.no_grad():
        preds_raw = model(data.x, data.edge_index, data.edge_weight).cpu().numpy()
    preds = target_scaler.inverse_transform(preds_raw)

    # Inefficiency = high ECSI stress AND low SEVI vitality
    inefficiency_score = preds[:, 0] - preds[:, 1]

    deg_top_10 = np.percentile(deg, 90)
    deg_median = np.median(deg)

    hub_candidates    = [i for i in range(data.num_nodes) if deg[i] >= deg_top_10]
    normal_candidates = [i for i in range(data.num_nodes)
                         if deg_median < deg[i] < deg_top_10]

    hub_indices  = sorted(hub_candidates,    key=lambda i: inefficiency_score[i], reverse=True)[:NUM_SAMPLES]
    norm_indices = sorted(normal_candidates, key=lambda i: inefficiency_score[i], reverse=True)[:NUM_SAMPLES]

    print(f"   > Selected {len(hub_indices)} Hubs and {len(norm_indices)} Normal Nodes.")

    print(f"[4/5] Simulating Interventions...")
    plot_data = {
        'Strategy': [],
        'Hub_ECSI_Mult': [],  'Hub_SEVI_Mult': [],
        'Norm_ECSI_Mult': [], 'Norm_SEVI_Mult': []
    }

    for strat_name, config in STRATEGIES.items():
        if config['feature'] not in feature_cols:
            continue

        feat_idx = feature_cols.index(config['feature'])
        delta    = config['delta']
        print(f"   > Testing {strat_name}...")

        def analyze_group(indices):
            total_loc = np.zeros(2)
            total_nei = np.zeros(2)
            for idx in indices:
                local, neighbor = calculate_benefit(
                    model, data, idx, feat_idx, delta, target_scaler)
                total_loc += local
                total_nei += neighbor
            return total_loc, total_nei

        h_loc, h_nei = analyze_group(hub_indices)
        n_loc, n_nei = analyze_group(norm_indices)

        h_ecsi_r = h_nei[0] / h_loc[0] if abs(h_loc[0]) > 1e-4 else 0
        h_sevi_r = h_nei[1] / h_loc[1] if abs(h_loc[1]) > 1e-4 else 0
        n_ecsi_r = n_nei[0] / n_loc[0] if abs(n_loc[0]) > 1e-4 else 0
        n_sevi_r = n_nei[1] / n_loc[1] if abs(n_loc[1]) > 1e-4 else 0

        plot_data['Strategy'].append(strat_name)
        plot_data['Hub_ECSI_Mult'].append(h_ecsi_r)
        plot_data['Hub_SEVI_Mult'].append(h_sevi_r)
        plot_data['Norm_ECSI_Mult'].append(n_ecsi_r)
        plot_data['Norm_SEVI_Mult'].append(n_sevi_r)

    print("[5/5] Generating Plots...")

    strategies = plot_data['Strategy']
    x     = np.arange(len(strategies))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- PLOT 1: ECSI (Environmental Stress) BENEFIT ---
    rects1_e = ax1.bar(x - width/2, plot_data['Hub_ECSI_Mult'],  width,
                       label='Inefficient Hubs',        color='#4257CA', alpha=0.9)
    rects2_e = ax1.bar(x + width/2, plot_data['Norm_ECSI_Mult'], width,
                       label='Inefficient Normal Nodes', color='#E2E2E2', alpha=0.9)

    ax1.set_ylabel('Benefit Multiplier')
    ax1.set_title('Environmental Spillover (ECSI Reduction)\n(Hubs vs Normal Nodes)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=15, ha='right')
    max_ecsi = max(max(plot_data['Hub_ECSI_Mult']), max(plot_data['Norm_ECSI_Mult']))
    ax1.set_ylim(0, max_ecsi * 1.3)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.axhline(0, color='black', linewidth=0.8)

    # --- PLOT 2: SEVI (Socioeconomic Vitality) BENEFIT ---
    rects1_s = ax2.bar(x - width/2, plot_data['Hub_SEVI_Mult'],  width,
                       label='Inefficient Hubs',        color='#CD5659', alpha=0.9)
    rects2_s = ax2.bar(x + width/2, plot_data['Norm_SEVI_Mult'], width,
                       label='Inefficient Normal Nodes', color='#E2E2E2', alpha=0.9)

    ax2.set_ylabel('Benefit Multiplier')
    ax2.set_title('Socioeconomic Spillover (SEVI Increase)\n(Hubs vs Normal Nodes)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15, ha='right')
    max_sevi = max(max(plot_data['Hub_SEVI_Mult']), max(plot_data['Norm_SEVI_Mult']))
    ax2.set_ylim(0, max_sevi * 1.3)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.8)

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}x',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1_e, ax1); autolabel(rects2_e, ax1)
    autolabel(rects1_s, ax2); autolabel(rects2_s, ax2)

    plt.tight_layout()
    plt.savefig("plot_spillover_hubs_vs_normal.png", dpi=300)
    print("Saved plot_spillover_hubs_vs_normal.png")
    plt.close()

if __name__ == "__main__":
    run_comparison()