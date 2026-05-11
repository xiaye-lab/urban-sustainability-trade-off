"""
correlation_analysis.py
=======================
Pearson correlation matrix and VIF for the final 8-feature set after dropping
NDWI, ABV, NB_ABV, and NB_FAR from the original 12.

Outputs
-------
  correlation_matrix_8feat.png   — annotated heatmap
  correlation_matrix_8feat.csv   — full r matrix
  vif_8feat.csv                  — VIF per feature
  high_pairs_8feat.csv           — pairs |r| >= 0.60, sorted descending
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler

from build_graph import SHP_COL_MAP, _clip_outliers, compute_vif

DROPPED      = ["NDWI", "ABV", "NB_ABV", "NB_FAR"]
FEATURE_COLS = [
    "BCR", "NB_BCR", "FAR",
    "AHI", "NB_AHI",
    "NDVI", "DIS_CBD", "DIS_MRT",
]


def load_features(shp_file="fishnet.shp"):
    print(f"Loading {shp_file}...")
    gdf = gpd.read_file(shp_file)
    gdf = gdf.rename(columns=SHP_COL_MAP)
    gdf = gdf.replace([np.inf, -np.inf], np.nan)
    num_cols = gdf.select_dtypes(include=[np.number]).columns
    gdf[num_cols] = gdf[num_cols].fillna(gdf[num_cols].mean())
    gdf = _clip_outliers(gdf, FEATURE_COLS)
    missing = [c for c in FEATURE_COLS if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print(f"  {len(gdf):,} nodes | {len(FEATURE_COLS)} features "
          f"(dropped: {DROPPED})")
    return gdf[FEATURE_COLS]


def compute_correlation(df):
    corr = df.corr(method="pearson")
    corr.to_csv("correlation_matrix_8feat.csv")
    print("Saved -> correlation_matrix_8feat.csv")
    return corr


def plot_heatmap(corr):
    n     = len(corr)
    names = corr.columns.tolist()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            val    = corr.values[i, j]
            txtcol = "white" if abs(val) > 0.50 else "#222222"
            weight = "bold"  if abs(val) >= 0.70 and i != j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9.5, color=txtcol, fontweight=weight)
            if i != j and abs(val) >= 0.70:
                ax.add_patch(FancyBboxPatch(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle="square,pad=0",
                    linewidth=1.8, edgecolor="#111111", facecolor="none"
                ))

    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03).set_label("Pearson r", fontsize=10)
    ax.set_title(
        "Pearson correlation — final 8-feature set\n"
        "(dropped: NDWI, ABV, NB_ABV, NB_FAR  |  bold border = |r| >= 0.70)",
        fontsize=12, pad=14
    )
    plt.tight_layout()
    plt.savefig("correlation_matrix_8feat.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved -> correlation_matrix_8feat.png")


def print_high_pairs(corr, thresh=0.60):
    names = corr.columns.tolist()
    rows  = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = corr.values[i, j]
            if abs(r) >= thresh:
                rows.append({
                    "Feature_A": names[i],
                    "Feature_B": names[j],
                    "Pearson_r": round(r, 4),
                    "Abs_r":     round(abs(r), 4),
                    "Level":     ("very high" if abs(r) >= 0.85
                                  else "high" if abs(r) >= 0.70
                                  else "moderate"),
                })
    rows = sorted(rows, key=lambda x: -x["Abs_r"])
    df   = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"  HIGH-CORRELATION PAIRS  (|r| >= {thresh})")
    print(f"{'='*60}")
    if df.empty:
        print("  None above threshold.")
    else:
        for _, row in df.iterrows():
            tag = {"very high": "***", "high": "** ", "moderate": "*  "}[row["Level"]]
            print(f"  {tag} {row['Feature_A']:<10} <-> {row['Feature_B']:<10}"
                  f"  r = {row['Pearson_r']:+.3f}  ({row['Level']})")
    df.to_csv("high_pairs_8feat.csv", index=False)
    print("Saved -> high_pairs_8feat.csv")
    return df


def run_vif(df):
    X_scaled = StandardScaler().fit_transform(df.values)
    vif_df   = compute_vif(X_scaled, FEATURE_COLS)
    if vif_df is None:
        return None
    print(f"\n{'='*60}")
    print("  VIF -- 8-feature set")
    print(f"{'='*60}")
    print(vif_df.to_string(index=False))
    high = vif_df[vif_df["VIF"] > 10]
    if high.empty:
        print("\n  All VIF <= 10 -- safe to update build_graph.py.")
    else:
        print(f"\n  Still above VIF 10: {high['Feature'].tolist()}")
    vif_df.to_csv("vif_8feat.csv", index=False)
    print("Saved -> vif_8feat.csv")
    return vif_df


def summarise(corr, vif_df, high_pairs):
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    max_off = high_pairs["Abs_r"].max() if not high_pairs.empty else 0.0
    n_high  = len(high_pairs[high_pairs["Level"].isin(["very high", "high"])])
    n_mod   = len(high_pairs[high_pairs["Level"] == "moderate"])
    print(f"  Features          : {len(FEATURE_COLS)}")
    print(f"  Pairs |r|>=0.85   : {len(high_pairs[high_pairs['Level']=='very high'])}")
    print(f"  Pairs |r|>=0.70   : {n_high}")
    print(f"  Pairs |r|>=0.60   : {n_high + n_mod}")
    print(f"  Max off-diag |r|  : {max_off:.3f}")
    if vif_df is not None:
        print(f"  Max VIF           : {vif_df['VIF'].max():.2f}")
        print(f"  Features VIF > 10 : {(vif_df['VIF'] > 10).sum()}")
        if (vif_df["VIF"] > 10).sum() == 0:
            print("\n  Multicollinearity resolved. Update build_graph.py now.")
        else:
            still = vif_df[vif_df["VIF"] > 10]["Feature"].tolist()
            print(f"\n  Residual concern: {still}")


if __name__ == "__main__":
    df         = load_features("fishnet.shp")
    corr       = compute_correlation(df)
    plot_heatmap(corr)
    high_pairs = print_high_pairs(corr, thresh=0.60)
    vif_df     = run_vif(df)
    summarise(corr, vif_df, high_pairs)
