"""
build_graph.py
==============
Builds the urban graph from fishnet_final_composite.shp.

Key changes from previous version
----------------------------------
  - Input: fishnet_final_composite.shp (columns already clean, no rename needed)
  - Features: 15 (expanded from 8 — adds LSI, LPI, MENN, SHDI, SVF, BHD, Albedo)
  - Targets: ECSI, SEVI read directly from shapefile (pre-computed via
             Entropy weighting and Entropy-TOPSIS in compute_composite.py)
             CRITIC computation removed.

Edge construction (two complementary, non-overlapping types)
-------------------------------------------------------------
1. Geometric edges  — rook contiguity from shared polygon boundaries.
   Weights: row-standardised  w_ij = 1 / degree(i).

2. Functional similarity edges — mutual k-nearest-neighbour in
   z-score-normalised feature space (cosine similarity).
   Weights: cosine similarity score in (0, 1].

Merge: w_ij = max(w_geo, w_sim) per node pair.
"""

import numpy as np
import pandas as pd
import torch
import geopandas as gpd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ── Feature and target definitions ──────────────────────────────────────────

FEATURE_COLS = [
    "BCR",    "NB_BCR",  "FAR",
    "AHI",    "NB_AHI",
    "LSI",    "LPI",     "MENN",   "SHDI",
    "SVF",    "BHD",     "Albedo",
    "NDVI",   "DIS_CBD", "DIS_MRT",
]

TARGET_COLS = ["ECSI", "SEVI"]


# ── Outlier clipping ─────────────────────────────────────────────────────────

def _clip_outliers(df, cols, q_low=0.01, q_high=0.99):
    for col in cols:
        if col in df.columns:
            lo, hi = df[col].quantile([q_low, q_high])
            df[col] = df[col].clip(lo, hi)
    return df


# ── VIF multicollinearity check ──────────────────────────────────────────────

def compute_vif(X_scaled, feature_names):
    if not HAS_STATSMODELS:
        print("  statsmodels not installed — skipping VIF. pip install statsmodels")
        return None
    vif = pd.DataFrame({
        "Feature": feature_names,
        "VIF": [variance_inflation_factor(X_scaled, i)
                for i in range(X_scaled.shape[1])]
    }).sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif


# ── Spatial Distance Decay Edges ─────────────────────────────

def _build_spatial_edges(gdf, radius=600.0, sigma=200.0):
    from scipy.spatial import cKDTree
    print(f"  Building spatial edges (Gaussian decay, radius={radius}m)...")
    
    # Extract projected centroids (units in meters based on SVY21)
    coords = np.column_stack((gdf["X"].values, gdf["Y"].values))
    tree = cKDTree(coords)

    # Query all pairs within the specified radius
    pairs = tree.query_pairs(r=radius)

    src_list, dst_list, w_list = [], [], []
    for i, j in pairs:
        # Calculate Euclidean distance between centroids
        dist = np.linalg.norm(coords[i] - coords[j])

        # Gaussian distance decay: w = 1.0 at d=0, decays smoothly as d increases
        w = np.exp(-(dist**2) / (2 * sigma**2))

        # Add both directions for a symmetric undirected graph
        src_list.extend([i, j])
        dst_list.extend([j, i])
        w_list.extend([w, w])

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    w_raw = np.array(w_list, dtype=np.float32)

    # Row-standardisation (maintains consistency with spatial lag principles)
    # Sum of spatial weights for each node will equal 1.0
    node_weight_sum = np.zeros(len(gdf), dtype=np.float32)
    np.add.at(node_weight_sum, src, w_raw)
    
    # Avoid division by zero for isolated nodes (if any exist)
    node_weight_sum[node_weight_sum == 0] = 1.0
    
    w_std = w_raw / node_weight_sum[src]

    deg = len(src) / len(gdf)
    print(f"  Distance edges: {len(src):,}  |  mean neighbors/node: {deg:.2f}")
    return src, dst, w_std

# ── Mutual KNN similarity edges ──────────────────────────────────────────────

def _build_similarity_edges(X_scaled, k=8):
    print(f"  Building mutual KNN similarity edges (k={k})...")
    sim   = cosine_similarity(X_scaled)
    top_k = {i: set(int(j) for j in np.argsort(-sim[i])[1: k + 1])
             for i in range(len(X_scaled))}

    src_list, dst_list, w_list = [], [], []
    for i in range(len(X_scaled)):
        for j in top_k[i]:
            if i in top_k[j]:
                src_list.append(i)
                dst_list.append(j)
                w_list.append(float(sim[i, j]))

    print(f"  Similarity edges: {len(src_list):,}")
    return (np.array(src_list, dtype=np.int64),
            np.array(dst_list, dtype=np.int64),
            np.array(w_list,   dtype=np.float32))


# ── Main loader ──────────────────────────────────────────────────────────────

def load_graph(shp_file="fishnet_final_composite.shp", k_sim=8):
    """
    Returns
    -------
    data         : torch_geometric.data.Data
                   .x [N,15], .y [N,2], .edge_index [2,E], .edge_weight [E]
    gdf          : GeoDataFrame with ECSI, SEVI, node_idx, X, Y
    feature_cols : list[str]  (15 features)
    target_cols  : list[str]  ['ECSI', 'SEVI']
    scaler       : fitted StandardScaler
    """

    # 1. Load ─────────────────────────────────────────────────────────────────
    print(f"Loading {shp_file}...")
    gdf = gpd.read_file(shp_file)
    gdf = gdf.sort_values("OBJECTID").reset_index(drop=True)
    gdf["node_idx"] = gdf.index
    gdf["X"] = gdf.geometry.centroid.x
    gdf["Y"] = gdf.geometry.centroid.y
    print(f"  {len(gdf):,} nodes  |  CRS: {gdf.crs}")

    # 2. Validate required columns ────────────────────────────────────────────
    all_required = FEATURE_COLS + TARGET_COLS
    missing = [c for c in all_required if c not in gdf.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available: {gdf.columns.tolist()}\n"
            f"Ensure fishnet_final_composite.shp was produced by "
            f"compute_composite.py."
        )

    # 3. Clean ────────────────────────────────────────────────────────────────
    gdf = gdf.replace([np.inf, -np.inf], np.nan)
    num_cols = gdf.select_dtypes(include=[np.number]).columns
    gdf[num_cols] = gdf[num_cols].fillna(gdf[num_cols].mean())

    # 4. Report ECSI / SEVI (pre-computed by compute_composite.py) ───────────
    print(f"  ECSI [{gdf['ECSI'].min():.4f}, {gdf['ECSI'].max():.4f}] "
          f"mean={gdf['ECSI'].mean():.4f}  "
          f"(Entropy-weighted, 0=low stress, 1=high stress)")
    print(f"  SEVI [{gdf['SEVI'].min():.4f}, {gdf['SEVI'].max():.4f}] "
          f"mean={gdf['SEVI'].mean():.4f}  "
          f"(Entropy-TOPSIS, 0=low vitality, 1=high vitality)")

    # 5. Features & scaling ───────────────────────────────────────────────────
    # 15-feature set after VIF / correlation analysis (see correlation_analysis_full.py)
    # All VIF <= 5.47; no cross-group redundancy (new x orig pairs all < 0.60)
    #
    # Groups with internal correlation (handled by grouped ablation):
    #   AHI / NB_AHI       r = 0.81
    #   BCR / NB_BCR / FAR r = 0.76 – 0.79 (within-group)
    #   LSI / LPI / SHDI   r = 0.62 – 0.73 (landscape metrics)
    feature_cols = FEATURE_COLS
    target_cols  = TARGET_COLS

    gdf = _clip_outliers(gdf, feature_cols)
    X   = gdf[feature_cols].values.astype(float)
    Y   = gdf[target_cols].values.astype(float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. VIF check ────────────────────────────────────────────────────────────
    print("\n  VIF multicollinearity check (15 features):")
    vif_df = compute_vif(X_scaled, feature_cols)
    if vif_df is not None:
        print(vif_df.to_string(index=False))
        high = vif_df[vif_df["VIF"] > 10]
        if not high.empty:
            print(f"  Warning: High VIF features (>10): {high['Feature'].tolist()}")
        else:
            print("  All VIF <= 10 — no severe multicollinearity")
        vif_df.to_csv("vif_report.csv", index=False)
        print("  Saved -> vif_report.csv")

    # 7. Spatial distance-decay edges ─────────────────────────────────────────
    src_geo, dst_geo, w_geo = _build_spatial_edges(gdf, radius=600.0, sigma=200.0)

    # 8. Similarity edges ─────────────────────────────────────────────────────
    src_sim, dst_sim, w_sim = _build_similarity_edges(X_scaled, k=k_sim)

    # 9. Merge ────────────────────────────────────────────────────────────────
    all_src = np.concatenate([src_geo, dst_geo, src_sim, dst_sim])
    all_dst = np.concatenate([dst_geo, src_geo, dst_sim, src_sim])
    all_w   = np.concatenate([w_geo,   w_geo,   w_sim,   w_sim  ])

    edge_dict = {}
    for u, v, w in zip(all_src, all_dst, all_w):
        if u != v:
            k_ = (int(u), int(v))
            edge_dict[k_] = max(edge_dict.get(k_, 0.0), float(w))

    fs = np.array([k[0] for k in edge_dict], dtype=np.int64)
    fd = np.array([k[1] for k in edge_dict], dtype=np.int64)
    fw = np.array([edge_dict[k] for k in edge_dict], dtype=np.float32)

    edge_index  = torch.tensor([fs, fd], dtype=torch.long)
    edge_weight = torch.tensor(fw,       dtype=torch.float)

    # 10. Sanity checks ───────────────────────────────────────────────────────
    assert (edge_index[0] == edge_index[1]).sum() == 0, "Self-loops found!"
    assert float(edge_weight.min()) > 0,                "Zero weights found!"
    fwd_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    bwd_set = set(zip(edge_index[1].tolist(), edge_index[0].tolist()))
    assert fwd_set == bwd_set, "Graph is not symmetric!"
    print(f"\nGraph: {len(gdf):,} nodes | {edge_index.shape[1]:,} edges")
    print("  Validation: no self-loops, symmetric, all weights in (0,1]")

    data = Data(
        x=torch.tensor(X_scaled, dtype=torch.float),
        y=torch.tensor(Y,        dtype=torch.float),
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    return data, gdf, feature_cols, target_cols, scaler


if __name__ == "__main__":
    data, gdf, feature_cols, target_cols, _ = load_graph()
    print("\nfeature_cols :", feature_cols)
    print(f"num features  : {len(feature_cols)}")
    print("x shape      :", data.x.shape)
    print("y shape      :", data.y.shape)
    print("edge_index   :", data.edge_index.shape)
    print("edge_weight  :", data.edge_weight.shape)
