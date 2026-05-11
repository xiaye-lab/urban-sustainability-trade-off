"""
explainability.py
=================
Post-hoc analysis for the GATv2 model predicting ECSI and SEVI.

Analysis pipeline (recommended order for paper reporting)
----------------------------------------------------------
  1. Grouped ablation study   <- PRIMARY feature importance ranking
       Zeros correlated feature groups simultaneously so they cannot
       compensate for each other. Robust to multicollinearity.
       Saves -> grouped_ablation.csv

  2. Individual ablation study  <- secondary / supplementary
       Zeros one feature at a time. Still affected by multicollinearity
       for correlated pairs (NB_* / base features, NDVI/NDWI).
       Saves -> ablation_study.csv

  3. Permutation feature importance  <- supplementary cross-check
       Permutes one feature at a time. Most affected by multicollinearity
       (correlated partners compensate). Use only alongside grouped ablation.
       Saves -> feature_importance.csv

  4. True PDP (model-direct)   <- PRIMARY nonlinear shape analysis
       Directly varies each feature over a grid and averages predictions.
       Does NOT use SHAP attribution -> not affected by multicollinearity.
       Saves -> pdp_results.csv

  5. SHAP values (KernelExplainer)  <- directional / global overview
       Beeswarm and dependence plots. Direction of effect is trustworthy
       for all features. MAGNITUDE is unreliable for correlated pairs
       (NB_AHI/AHI, NB_BCR/BCR). Use grouped ablation for ranking.
       Saves -> shap_values_ecsi.npy, shap_values_sevi.npy

  6. SHAP grouped importance  <- companion to beeswarm
       Sums SHAP magnitude across correlated pairs to recover true importance.
       Saves -> shap_grouped_importance.csv

  7. GNNExplainer  <- edge-level attribution for individual nodes
       Saves -> gnn_explain_edges_node{idx}.csv

Multicollinearity note
----------------------
Final 8-feature set after VIF-driven drops (NDWI, ABV, NB_ABV, NB_FAR).
Remaining correlated pairs:
  NB_AHI / AHI   -> grouped as "AHI (local+nbr)"
  NB_BCR / BCR   -> grouped as "BCR (local+nbr)"
  BCR / FAR      -> moderate cross-type correlation (~0.79)
  NDVI           -> independent after NDWI removal (VIF ~1.6)
  DIS_CBD        -> independent (VIF ~1.4)
  DIS_MRT        -> independent (VIF ~1.9)
Note: FAR is kept without its NB_ counterpart. Its neighbourhood signal
is partially captured by NB_BCR and NB_AHI through their shared
high-density footprint. Cite grouped ablation as primary importance.
"""

import torch
import numpy as np
import pandas as pd
import shap
import joblib
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.explain import Explainer, GNNExplainer


# ── Correlated feature groups (defined once, used by multiple functions) ────

CORRELATED_GROUPS = {
    "AHI (local+nbr)"  : ["AHI", "NB_AHI"],
    "BCR (local+nbr)"  : ["BCR", "NB_BCR"],
    "FAR"              : ["FAR"],           # NB_FAR dropped; FAR kept standalone
    "NDVI"             : ["NDVI"],          # NDWI dropped; NDVI independent
    "DIS_CBD"          : ["DIS_CBD"],
    "DIS_MRT"          : ["DIS_MRT"],
}

# VIF flag — features whose individual SHAP magnitude may be less reliable
# Only the NB_/local pairs remain after the four drops.
HIGH_VIF_FEATURES = {
    "NB_AHI", "AHI",
    "NB_BCR", "BCR",
}


# ── Shared helpers ───────────────────────────────────────────────────────────

def _get_true_and_pred(model, data, mask_np, target_scaler):
    """Return (y_true, pred_subset, pred_all) in original ECSI/SEVI units."""
    with torch.no_grad():
        pred_all = model(data.x, data.edge_index, data.edge_weight).cpu().numpy()

    if target_scaler is not None:
        y_true      = target_scaler.inverse_transform(data.y[mask_np].cpu().numpy())
        pred_subset = target_scaler.inverse_transform(pred_all[mask_np])
    else:
        y_true      = data.y[mask_np].cpu().numpy()
        pred_subset = pred_all[mask_np]

    return y_true, pred_subset, pred_all


def _base_r2(y_true, pred_subset):
    r2 = r2_score(y_true, pred_subset, multioutput="raw_values")
    return r2[0], r2[1]


def _mask_np(mask, n):
    return np.ones(n, dtype=bool) if mask is None else mask.cpu().numpy()


# ── 1. Grouped Ablation Study  (PRIMARY importance ranking) ─────────────────

def grouped_ablation_study(model, data, feature_cols,
                           mask=None, target_scaler=None,
                           groups=None):
    """
    PRIMARY feature importance ranking -- robust to multicollinearity.

    Zeros ALL members of a correlated group simultaneously so no partner
    feature can compensate. For independent features (DIS_CBD, DIS_MRT)
    the group contains only one member, equivalent to individual ablation.

    Saves -> grouped_ablation.csv
    """
    print("\n" + "=" * 60)
    print("  GROUPED ABLATION STUDY  <- primary importance ranking")
    print("  (correlated pairs zeroed together -- no compensation)")
    print("=" * 60)
    model.eval()

    if groups is None:
        groups = CORRELATED_GROUPS

    mp = _mask_np(mask, len(data.y))
    y_true, base_subset, _ = _get_true_and_pred(model, data, mp, target_scaler)
    base_ecsi, base_sevi   = _base_r2(y_true, base_subset)

    print(f"  Baseline R2 | ECSI: {base_ecsi:.4f} | SEVI: {base_sevi:.4f}\n")

    feat_idx = {f: i for i, f in enumerate(feature_cols)}
    results  = []

    for group_name, members in groups.items():
        idxs = [feat_idx[f] for f in members if f in feat_idx]
        if not idxs:
            print(f"  [skip] {group_name} -- no matching features in model")
            continue

        X_ab = data.x.clone()
        for i in idxs:
            X_ab[:, i] = 0              # zero ALL members at once

        with torch.no_grad():
            pred_ab = model(X_ab, data.edge_index, data.edge_weight).cpu().numpy()

        p_sub  = target_scaler.inverse_transform(pred_ab[mp]) \
                 if target_scaler is not None else pred_ab[mp]
        new_r2 = r2_score(y_true, p_sub, multioutput="raw_values")

        drop_ecsi  = base_ecsi - new_r2[0]
        drop_sevi  = base_sevi - new_r2[1]
        total_drop = abs(drop_ecsi) + abs(drop_sevi)

        results.append({
            "Group"        : group_name,
            "Members"      : ", ".join(members),
            "N_features"   : len(idxs),
            "R2_drop_ECSI" : round(drop_ecsi,  4),
            "R2_drop_SEVI" : round(drop_sevi,  4),
            "Total_Drop"   : round(total_drop, 4),
        })
        print(f"  {group_name:<26} | "
              f"ECSI drop: {drop_ecsi:+.4f} | "
              f"SEVI drop: {drop_sevi:+.4f} | "
              f"Total: {total_drop:.4f}")

    df = pd.DataFrame(results).sort_values("Total_Drop", ascending=False)
    df.to_csv("grouped_ablation.csv", index=False)
    print("\n  Saved -> grouped_ablation.csv")
    return df


# ── 2. Individual Ablation Study  (supplementary) ───────────────────────────

def ablation_study(model, data, feature_cols, mask=None, target_scaler=None):
    """
    Supplementary individual ablation -- zeros one feature at a time.

    NOTE: For high-VIF feature pairs the R2 drop is understated because the
    correlated partner compensates. Use grouped_ablation_study() for the
    primary importance ranking. This function is retained for supplementary
    reporting and comparison with the grouped result.

    Saves -> ablation_study.csv
    """
    print("\n" + "=" * 60)
    print("  INDIVIDUAL ABLATION STUDY  (supplementary)")
    print("  NOTE: correlated pairs understate individual importance")
    print("=" * 60)
    model.eval()

    mp = _mask_np(mask, len(data.y))
    y_true, base_subset, _ = _get_true_and_pred(model, data, mp, target_scaler)
    base_ecsi, base_sevi   = _base_r2(y_true, base_subset)

    print(f"  Baseline R2 | ECSI: {base_ecsi:.4f} | SEVI: {base_sevi:.4f}\n")

    results = []
    for i, col in enumerate(feature_cols):
        X_ab = data.x.clone()
        X_ab[:, i] = 0

        with torch.no_grad():
            pred_ab = model(X_ab, data.edge_index, data.edge_weight).cpu().numpy()

        p_sub  = target_scaler.inverse_transform(pred_ab[mp]) \
                 if target_scaler is not None else pred_ab[mp]
        new_r2 = r2_score(y_true, p_sub, multioutput="raw_values")

        drop_ecsi = base_ecsi - new_r2[0]
        drop_sevi = base_sevi - new_r2[1]
        high_vif  = col in HIGH_VIF_FEATURES

        results.append({
            "Feature"      : col,
            "R2_drop_ECSI" : round(drop_ecsi,  4),
            "R2_drop_SEVI" : round(drop_sevi,  4),
            "Total_Drop"   : round(abs(drop_ecsi) + abs(drop_sevi), 4),
            "High_VIF"     : high_vif,
        })
        flag = " [high VIF -- see grouped ablation]" if high_vif else ""
        print(f"  {col:<15} | ECSI: {drop_ecsi:+.4f} | "
              f"SEVI: {drop_sevi:+.4f}{flag}")

    df = pd.DataFrame(results).sort_values("Total_Drop", ascending=False)
    df.to_csv("ablation_study.csv", index=False)
    print("\n  Saved -> ablation_study.csv")
    return df


# ── 3. Permutation Feature Importance  (supplementary) ──────────────────────

def permutation_feature_importance(model, data, df_nodes, feature_cols,
                                   mask=None, n_repeats=3, target_scaler=None):
    """
    Supplementary permutation importance -- permutes one feature at a time.

    NOTE: Most affected by multicollinearity. Correlated partners compensate
    during permutation, understating individual importance for high-VIF pairs.
    Use for cross-validation alongside grouped ablation, not as primary ranking.

    Saves -> feature_importance.csv
    """
    print("\n" + "=" * 60)
    print("  PERMUTATION FEATURE IMPORTANCE  (supplementary)")
    print("  NOTE: high-VIF pairs most affected -- use grouped ablation")
    print("        for primary importance ranking")
    print("=" * 60)
    model.eval()

    mp = _mask_np(mask, len(data.y))
    y_true, pred_subset, _ = _get_true_and_pred(model, data, mp, target_scaler)
    base_ecsi, base_sevi   = _base_r2(y_true, pred_subset)

    print(f"  Baseline R2 | ECSI: {base_ecsi:.4f} | SEVI: {base_sevi:.4f}\n")

    results = []
    for i, col in enumerate(feature_cols):
        imp_ecsi_list, imp_sevi_list = [], []
        for _ in range(n_repeats):
            X_perm        = data.x.clone()
            perm          = np.random.permutation(len(X_perm))
            X_perm[:, i]  = X_perm[perm, i]

            with torch.no_grad():
                pred_perm = model(X_perm, data.edge_index, data.edge_weight).cpu().numpy()

            p_sub = target_scaler.inverse_transform(pred_perm[mp]) \
                    if target_scaler is not None else pred_perm[mp]
            r2p   = r2_score(y_true, p_sub, multioutput="raw_values")
            imp_ecsi_list.append(base_ecsi - r2p[0])
            imp_sevi_list.append(base_sevi - r2p[1])

        imp_ecsi = float(np.mean(imp_ecsi_list))
        imp_sevi = float(np.mean(imp_sevi_list))
        high_vif = col in HIGH_VIF_FEATURES

        results.append({
            "Feature"        : col,
            "Importance_ECSI": round(imp_ecsi, 4),
            "Importance_SEVI": round(imp_sevi, 4),
            "Total_Imp"      : round(abs(imp_ecsi) + abs(imp_sevi), 4),
            "High_VIF"       : high_vif,
        })
        flag = " [high VIF]" if high_vif else ""
        print(f"  {col:<15} | ECSI: {imp_ecsi:+.4f} | "
              f"SEVI: {imp_sevi:+.4f}{flag}")

    df = pd.DataFrame(results).sort_values("Total_Imp", ascending=False)
    df.to_csv("feature_importance.csv", index=False)
    print("\n  Saved -> feature_importance.csv")
    return df


# ── 4. True PDP  (PRIMARY nonlinear shape analysis) ─────────────────────────

def compute_pdp(model, data, feature_names, df_nodes,
                target_features=None, grid_resolution=50):
    """
    PRIMARY nonlinear relationship analysis -- direct model queries.

    Sets each feature to a grid value and averages predictions across all nodes.
    Does NOT use SHAP attribution -> completely unaffected by multicollinearity.

    The shape of each PDP curve is trustworthy for ALL features including
    high-VIF pairs because the question is:
      'What does the model predict on average when BCR = x?'
    not:
      'How much credit does BCR get?'

    The correlated partner (NB_BCR) remains at its real observed values
    across all nodes, so its average effect is baked into the baseline.
    The PDP curve shows only the marginal contribution of moving one feature.

    grid_resolution=50 gives smooth curves; reduce to 20 for faster runs.
    Saves -> pdp_results.csv
    """
    if target_features is None:
        target_features = list(feature_names)      # run all features by default

    print("\n" + "=" * 60)
    print("  TRUE PDP  <- primary nonlinear shape analysis")
    print("  (direct model queries -- not affected by multicollinearity)")
    print("=" * 60)
    model.eval()

    feat_map = {name: i for i, name in enumerate(feature_names)}
    results  = []

    for feat_name in target_features:
        if feat_name not in feat_map:
            print(f"  [skip] {feat_name} not in feature list")
            continue

        feat_idx  = feat_map[feat_name]
        orig_vals = df_nodes[feat_name].values \
                    if feat_name in df_nodes.columns else None
        mu    = float(orig_vals.mean()) if orig_vals is not None else 0.0
        sigma = float(orig_vals.std())  if orig_vals is not None else 1.0
        if sigma == 0:
            sigma = 1.0

        scaled_vals = data.x[:, feat_idx].cpu().numpy()
        grid        = np.linspace(scaled_vals.min(), scaled_vals.max(),
                                  grid_resolution)

        print(f"  Scanning {feat_name} ({grid_resolution} points) ...")

        for val in grid:
            x_tmp = data.x.clone()
            x_tmp[:, feat_idx] = float(val)

            with torch.no_grad():
                preds = model(x_tmp, data.edge_index, data.edge_weight).cpu().numpy()

            results.append({
                "Feature"      : feat_name,
                "Value_scaled" : round(float(val), 6),
                "Value_true"   : round(float(val) * sigma + mu, 6),
                "Average_ECSI" : round(float(preds[:, 0].mean()), 6),
                "Average_SEVI" : round(float(preds[:, 1].mean()), 6),
                "Std_ECSI"     : round(float(preds[:, 0].std()),  6),
                "Std_SEVI"     : round(float(preds[:, 1].std()),  6),
            })

    df_pdp = pd.DataFrame(results)
    df_pdp.to_csv("pdp_results.csv", index=False)
    print(f"\n  Saved -> pdp_results.csv  "
          f"({len(target_features)} features x {grid_resolution} points)")
    return df_pdp


# ── 5. SHAP Values  (directional / global overview) ─────────────────────────

def compute_shap_values(model, data, feature_cols, sample_size=None):
    """
    SHAP KernelExplainer -- global feature attribution overview.

    USE FOR:
      - Beeswarm: direction of effect (positive/negative) for each feature
      - Dependence plots: nonlinear shape confirmation (use alongside PDP)
      - Global importance for independent features (DIS_CBD, DIS_MRT reliable)

    DO NOT USE AS PRIMARY RANKING for high-VIF pairs:
      NB_ABV/ABV, NB_AHI/AHI, NB_FAR/FAR, NB_BCR/BCR, NDVI/NDWI
    For those, use grouped_ablation_study() + shap_grouped_importance().

    Saves -> shap_values_ecsi.npy, shap_values_sevi.npy, shap_X_explain.npy
    """
    print("\n" + "=" * 60)
    print("  SHAP VALUES  (directional overview)")
    print("  Primary use: beeswarm direction + dependence plot shape")
    print("  NOTE: magnitude unreliable for high-VIF pairs")
    print("=" * 60)
    model.eval()

    X_all      = data.x.cpu().numpy()
    background = shap.kmeans(X_all, 50)

    def model_predict(x_numpy):
        device     = next(model.parameters()).device
        x_tensor   = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        N          = x_tensor.shape[0]
        self_loops = torch.stack([torch.arange(N, device=device),
                                  torch.arange(N, device=device)])
        with torch.no_grad():
            out = model(x_tensor, self_loops, torch.ones(x_tensor.shape[0]))
        return out.cpu().numpy()

    explainer = shap.KernelExplainer(model_predict, background)

    if sample_size is None:
        print(f"  Using full dataset ({len(X_all):,} nodes). This may take time.")
        X_explain = X_all
    else:
        print(f"  Sampling {sample_size:,} nodes.")
        idx       = np.random.choice(len(X_all), sample_size, replace=False)
        X_explain = X_all[idx]

    print(f"  Explaining {len(X_explain):,} nodes ...")
    shap_values = explainer.shap_values(X_explain)

    if isinstance(shap_values, list):
        vals_ecsi, vals_sevi = shap_values[0], shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        vals_ecsi, vals_sevi = shap_values[:, :, 0], shap_values[:, :, 1]
    else:
        vals_ecsi = vals_sevi = shap_values

    np.save("shap_values_ecsi.npy", vals_ecsi)
    np.save("shap_values_sevi.npy", vals_sevi)
    np.save("shap_X_explain.npy",   X_explain)
    pd.DataFrame(feature_cols, columns=["Feature"]).to_csv(
        "shap_feature_names.csv", index=False
    )
    print("  Saved -> shap_values_ecsi.npy, shap_values_sevi.npy, "
          "shap_X_explain.npy")
    return shap_values


# ── 6. SHAP Grouped Importance  (companion to beeswarm) ─────────────────────

def shap_grouped_importance(groups=None):
    """
    Sums absolute SHAP values across correlated groups to recover true importance.

    Corrects for the attribution-splitting problem: when NB_ABV and ABV share
    credit, neither looks fully important individually. Summing their absolute
    SHAP values gives the combined importance, comparable to grouped ablation.

    Requires shap_values_*.npy (run compute_shap_values first).
    Saves -> shap_grouped_importance.csv
    """
    print("\n" + "=" * 60)
    print("  SHAP GROUPED IMPORTANCE  (corrects attribution splitting)")
    print("=" * 60)

    if groups is None:
        groups = CORRELATED_GROUPS

    try:
        shap_ecsi     = np.load("shap_values_ecsi.npy")
        shap_sevi     = np.load("shap_values_sevi.npy")
        feature_names = pd.read_csv("shap_feature_names.csv")["Feature"].tolist()
    except FileNotFoundError as e:
        print(f"  Error: {e}\n  Run compute_shap_values() first.")
        return None

    feat_idx = {f: i for i, f in enumerate(feature_names)}
    results  = []

    for group_name, members in groups.items():
        idxs = [feat_idx[f] for f in members if f in feat_idx]
        if not idxs:
            continue

        # Sum absolute SHAP across all members of the group
        ecsi_combined = np.abs(shap_ecsi[:, idxs]).sum(axis=1).mean()
        sevi_combined = np.abs(shap_sevi[:, idxs]).sum(axis=1).mean()

        results.append({
            "Group"             : group_name,
            "Members"           : ", ".join(members),
            "Mean_AbsSHAP_ECSI" : round(ecsi_combined, 6),
            "Mean_AbsSHAP_SEVI" : round(sevi_combined, 6),
            "Total_AbsSHAP"     : round(ecsi_combined + sevi_combined, 6),
        })
        print(f"  {group_name:<26} | "
              f"ECSI: {ecsi_combined:.4f} | SEVI: {sevi_combined:.4f}")

    df = pd.DataFrame(results).sort_values("Total_AbsSHAP", ascending=False)
    df.to_csv("shap_grouped_importance.csv", index=False)
    print("\n  Saved -> shap_grouped_importance.csv")
    return df


# ── 7. GNN Explainer  (edge-level attribution) ───────────────────────────────

class _EdgeWeightWrapper(torch.nn.Module):
    """
    Thin wrapper so GNNExplainer can call model(x, edge_index) with two
    arguments.  GNNExplainer's internal forward pass does not pass edge_weight,
    but UrbanGNN.forward() requires it as a positional argument.  This wrapper
    captures the fixed edge_weight tensor from the original graph and injects
    it automatically, keeping the signature that Explainer expects.
    """
    def __init__(self, model, edge_weight):
        super().__init__()
        self.model       = model
        self.edge_weight = edge_weight

    def forward(self, x, edge_index):
        # GNNExplainer may pass a subgraph whose edge count differs from the
        # full graph, so we cannot reuse self.edge_weight directly.  Instead
        # we fall back to uniform weights (all-ones) for the subgraph edges,
        # which preserves the topology mask that GNNExplainer is optimising.
        ew = x.new_ones(edge_index.size(1))
        return self.model(x, edge_index, ew)


def run_gnn_explainer(model, data, node_idx=0):
    """
    GNNExplainer -- identifies which edges matter most for one node's prediction.
    Not affected by multicollinearity (operates on graph topology, not features).
    Saves -> gnn_explain_edges_node{node_idx}.csv
    """
    print(f"\n===== GNNExplainer (node {node_idx}) =====")

    # Wrap model so GNNExplainer can call it with (x, edge_index) only
    wrapped = _EdgeWeightWrapper(model, data.edge_weight)
    wrapped.eval()

    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="regression", task_level="node", return_type="raw"),
    )
    explanation = explainer(data.x, data.edge_index, index=node_idx)
    edge_mask   = explanation.edge_mask.cpu().numpy()
    top_idx     = edge_mask.argsort()[-10:][::-1]
    pd.DataFrame({
        "edge_index": top_idx,
        "importance": edge_mask[top_idx],
    }).to_csv(f"gnn_explain_edges_node{node_idx}.csv", index=False)
    print(f"  Saved -> gnn_explain_edges_node{node_idx}.csv")
    return explanation


# ── Standalone entry-point ───────────────────────────────────────────────────

if __name__ == "__main__":
    from build_graph import load_graph
    from gnn_model import UrbanGNN

    # ── Load graph and model ─────────────────────────────────────────────────
    data, df_nodes, feature_cols, target_cols, _ = load_graph(
        shp_file="fishnet.shp",
    )

    model = UrbanGNN(num_features=data.x.shape[1], hidden_dim=64, num_targets=2)
    model.load_state_dict(torch.load("urban_gnn_model.pth", weights_only=True))
    model.eval()
    print("Model loaded <- urban_gnn_model.pth")

    try:
        target_scaler = joblib.load("target_scaler.pkl")
        print("Target scaler loaded <- target_scaler.pkl")
    except FileNotFoundError:
        print("Warning: target_scaler.pkl not found -- metrics in normalised space.")
        target_scaler = None

    # ── Recreate test mask (same seed as train_gnn.py) ───────────────────────
    num_nodes = data.x.shape[0]
    rng       = np.random.default_rng(42)
    indices   = np.arange(num_nodes)
    rng.shuffle(indices)
    n_train  = int(0.70 * num_nodes)
    n_val    = int(0.15 * num_nodes)
    test_idx = indices[n_train + n_val:]
    mask     = torch.zeros(num_nodes, dtype=torch.bool)
    mask[test_idx] = True

    # ── Normalise targets to match training space ────────────────────────────
    if target_scaler is not None:
        y_all     = data.y.numpy()
        train_idx = indices[:n_train]
        _sc       = StandardScaler().fit(y_all[train_idx])
        data.y    = torch.tensor(_sc.transform(y_all), dtype=torch.float)

    # ── Run pipeline ─────────────────────────────────────────────────────────

    # 1. PRIMARY: grouped ablation -> feature importance ranking
    grouped_ablation_study(
        model, data, feature_cols,
        mask=mask, target_scaler=target_scaler,
    )

    # 2. Supplementary: individual ablation (for comparison / appendix)
    ablation_study(
        model, data, feature_cols,
        mask=mask, target_scaler=target_scaler,
    )

    # 3. Supplementary: permutation importance (cross-check)
    permutation_feature_importance(
        model, data, df_nodes, feature_cols,
        mask=mask, n_repeats=3, target_scaler=target_scaler,
    )

    # 4. PRIMARY: true PDP -> nonlinear relationship shapes (all features)
    compute_pdp(
        model, data, feature_cols, df_nodes,
        target_features=feature_cols,
        grid_resolution=50,
    )

    # 5. SHAP: directional overview + dependence plot support
    compute_shap_values(model, data, feature_cols, sample_size=None)

    # 6. SHAP grouped importance: corrects attribution splitting for high-VIF pairs
    shap_grouped_importance()

    # 7. GNNExplainer: edge attribution for one test node
    node_idx = int(test_idx[0]) if len(test_idx) > 0 else 0
    run_gnn_explainer(model, data, node_idx=node_idx)

    # ── Summary of output files ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  OUTPUT FILES")
    print("=" * 60)
    print("  PRIMARY (cite these in paper)")
    print("    grouped_ablation.csv          feature importance ranking")
    print("    pdp_results.csv               nonlinear relationship shapes")
    print()
    print("  SUPPLEMENTARY")
    print("    ablation_study.csv            individual ablation")
    print("    feature_importance.csv        permutation importance")
    print("    shap_values_ecsi/sevi.npy     beeswarm + dependence plots")
    print("    shap_grouped_importance.csv   SHAP with VIF correction")
    print(f"   gnn_explain_edges_node{node_idx}.csv")
