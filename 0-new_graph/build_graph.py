"""
build_graph.py
==============
Builds the urban graph directly from fishnet.shp — no CSV files needed.

Edge construction (two complementary, non-overlapping types)
-------------------------------------------------------------
1. Geometric edges  — rook contiguity from shared polygon boundaries.
   Pairs whose intersection has length > 0 (shared LINE) are rook neighbours.
   Corner-touching pairs (shared POINT, length = 0) are excluded.
   Weights: row-standardised  w_ij = 1 / degree(i).

2. Functional similarity edges — mutual k-nearest-neighbour in
   z-score-normalised feature space (cosine similarity only; no geography,
   since spatial proximity is already captured by geometric edges).
   Edge (i,j) kept only if j ∈ top-k(i) AND i ∈ top-k(j).
   Weights: cosine similarity score in (0, 1].

Merge: w_ij = max(w_geo, w_sim) per node pair.
Edge weights passed to GATv2Conv as edge_attr (1-D scalar).

Target construction (CRITIC method)
------------------------------------
  ECSI — cost criteria: LST, PM25, CO2
  SEVI — benefit criteria: NTL, POI, RND, POPU
  CRITIC weight: C_j = sigma_j * sum(1 - r_ij);  w_j = C_j / sum(C)
  Features: 8 (BCR, NB_BCR, FAR, AHI, NB_AHI, NDVI, DIS_CBD, DIS_MRT)
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


# ── Column mapping (ArcGIS-truncated → readable) ───────────────────────────
SHP_COL_MAP = {
    "SG_200m_Fe": "OBJECTID",
    "SG_200m__1": "BCR",
    "SG_200m__2": "FAR",
    "SG_200m__3": "AHI",
    "SG_200m__4": "ABV",
    "SG_200m__5": "NB_BCR",
    "SG_200m__6": "NB_FAR",
    "SG_200m__7": "NB_AHI",
    "SG_200m__8": "NB_ABV",
    "SG_200m__9": "NDVI",
    "SG_200m_10": "NDWI",
    "SG_200m_11": "DIS_MRT",
    "SG_200m_12": "DIS_CBD",
    "SG_200m_13": "LST",
    "SG_200m_14": "NTL",
    # POI, RND, PM25, CO2, POPU already correctly named in shapefile
}


# ── CRITIC composite index ──────────────────────────────────────────────────

def _critic_index(df, cols, benefit):
    arr = df[cols].values.astype(float)
    mn, mx = arr.min(0), arr.max(0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    norm = (arr - mn) / rng if benefit else (mx - arr) / rng
    std = norm.std(0, ddof=1)
    corr = np.corrcoef(norm.T)
    C = std * np.sum(1 - corr, axis=0)
    w = C / C.sum()
    return (norm * w).sum(1)


# ── Outlier clipping ────────────────────────────────────────────────────────

def _clip_outliers(df, cols, q_low=0.01, q_high=0.99):
    for col in cols:
        if col in df.columns:
            lo, hi = df[col].quantile([q_low, q_high])
            df[col] = df[col].clip(lo, hi)
    return df


# ── VIF multicollinearity check ─────────────────────────────────────────────

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


# ── Rook contiguity edges ───────────────────────────────────────────────────

def _build_rook_edges(gdf):
    from shapely.strtree import STRtree
    print("  Building rook contiguity edges...")
    geoms = gdf.geometry.values
    tree  = STRtree(geoms)

    src_list, dst_list = [], []
    for i, geom in enumerate(geoms):
        for j in tree.query(geom):
            if i == j:
                continue
            if geom.intersection(geoms[j]).length > 0:
                src_list.append(i)
                dst_list.append(j)

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    deg = Counter(src_list)
    w   = np.array([1.0 / deg[s] for s in src_list], dtype=np.float32)

    print(f"  Rook edges: {len(src):,}  |  mean degree: "
          f"{np.mean(list(deg.values())):.2f}")
    return src, dst, w


# ── Mutual KNN similarity edges ─────────────────────────────────────────────

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


# ── Main loader ─────────────────────────────────────────────────────────────

def load_graph(shp_file="fishnet.shp", k_sim=8):
    """
    Returns
    -------
    data         : torch_geometric.data.Data
                   .x [N,F], .y [N,2], .edge_index [2,E], .edge_weight [E]
    gdf          : GeoDataFrame with ECSI, SEVI, node_idx, X, Y
    feature_cols : list[str]
    target_cols  : list[str]  ['ECSI','SEVI']
    scaler       : fitted StandardScaler
    """

    # 1. Load & rename ────────────────────────────────────────────────────────
    print(f"Loading {shp_file}...")
    gdf = gpd.read_file(shp_file)
    gdf = gdf.rename(columns=SHP_COL_MAP)
    gdf = gdf.sort_values("OBJECTID").reset_index(drop=True)
    gdf["node_idx"] = gdf.index
    gdf["X"] = gdf.geometry.centroid.x
    gdf["Y"] = gdf.geometry.centroid.y
    print(f"  {len(gdf):,} nodes  |  CRS: {gdf.crs}")

    # 2. Clean ────────────────────────────────────────────────────────────────
    gdf = gdf.replace([np.inf, -np.inf], np.nan)
    num_cols = gdf.select_dtypes(include=[np.number]).columns
    gdf[num_cols] = gdf[num_cols].fillna(gdf[num_cols].mean())

    # 3. ECSI & SEVI via CRITIC ───────────────────────────────────────────────
    ecsi_cols = ["LST", "PM25", "CO2"]
    sevi_cols = ["NTL", "POI", "RND", "POPU"]
    for col in ecsi_cols + sevi_cols:
        if col not in gdf.columns:
            raise ValueError(f"Required column '{col}' missing in {shp_file}.")

    # Clip POPU before CRITIC: population density has a heavy right tail
    # (dense HDB estates vs. parks/water cells near zero), which would compress
    # the normalised range and inflate POPU's CRITIC weight artificially.
    gdf = _clip_outliers(gdf, ["POPU"])

    gdf["ECSI"] = _critic_index(gdf, ecsi_cols, benefit=False)
    gdf["SEVI"] = _critic_index(gdf, sevi_cols, benefit=True)
    print(f"  ECSI [{gdf['ECSI'].min():.4f}, {gdf['ECSI'].max():.4f}] "
          f"mean={gdf['ECSI'].mean():.4f}")
    print(f"  SEVI [{gdf['SEVI'].min():.4f}, {gdf['SEVI'].max():.4f}] "
          f"mean={gdf['SEVI'].mean():.4f}")

    # 4. Features & scaling ───────────────────────────────────────────────────
    # Dropped from original 12 after correlation / VIF analysis:
    #   NDWI   — near-duplicate of NDVI (same NIR band); VIF 13.6 -> NDVI VIF 1.6
    #   ABV    — weakest individual ablation (total 0.099); near-linear FAR x AHI
    #   NB_ABV — VIF 25.9; signal already covered by NB_AHI + NB_FAR
    #   NB_FAR — VIF 12.8; hub correlated with NB_BCR(0.82), FAR(0.78), NB_AHI(0.72)
    #            FAR (local) retained as more informative (ablation 0.277 vs 0.168)
    feature_cols = [
        "BCR", "NB_BCR", "FAR",
        "AHI", "NB_AHI",
        "NDVI", "DIS_CBD", "DIS_MRT",
    ]
    target_cols = ["ECSI", "SEVI"]

    gdf = _clip_outliers(gdf, feature_cols)
    X = gdf[feature_cols].values.astype(float)
    Y = gdf[target_cols].values.astype(float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. VIF check ────────────────────────────────────────────────────────────
    print("\n  VIF multicollinearity check:")
    vif_df = compute_vif(X_scaled, feature_cols)
    if vif_df is not None:
        print(vif_df.to_string(index=False))
        high = vif_df[vif_df["VIF"] > 10]
        if not high.empty:
            print(f"  Warning: High VIF features (>10): {high['Feature'].tolist()}")
        else:
            print("  All VIF <= 10 — no severe multicollinearity")
        vif_df.to_csv("vif_report.csv", index=False)
        print("  Saved → vif_report.csv")

    # 6. Geometric edges ──────────────────────────────────────────────────────
    src_geo, dst_geo, w_geo = _build_rook_edges(gdf)

    # 7. Similarity edges ─────────────────────────────────────────────────────
    src_sim, dst_sim, w_sim = _build_similarity_edges(X_scaled, k=k_sim)

    # 8. Merge ────────────────────────────────────────────────────────────────
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

    # 9. Sanity checks ────────────────────────────────────────────────────────
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
    print("x shape      :", data.x.shape)
    print("y shape      :", data.y.shape)
    print("edge_index   :", data.edge_index.shape)
    print("edge_weight  :", data.edge_weight.shape)
