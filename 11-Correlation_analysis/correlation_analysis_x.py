"""
correlation_analysis_full.py
=============================
Pearson correlation matrix and VIF for the full 15-feature set:

  Original 8   : BCR, NB_BCR, FAR, AHI, NB_AHI, NDVI, DIS_CBD, DIS_MRT
  New 7        : LSI, LPI, MENN, SHDI, SVF, BHD, Albedo

Loads from fishnet_final_price.shp (all columns already clean — no rename needed).

Outputs
-------
  correlation_matrix_full.png    annotated heatmap (15 x 15)
  correlation_matrix_full.csv    full r matrix
  vif_full.csv                   VIF per feature
  high_pairs_full.csv            pairs |r| >= 0.60, sorted descending
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import os

warnings.filterwarnings("ignore")

# PROJ conflict fix
try:
    import pyproj as _pp
    _pd = os.path.join(os.path.dirname(_pp.__file__), "proj_dir", "share", "proj")
    if os.path.isdir(_pd):
        os.environ["PROJ_DATA"] = _pd
        os.environ["PROJ_LIB"]  = _pd
except Exception:
    pass

SHP_FILE = "fishnet_final_composite.shp"   # columns already clean, no rename needed

ORIGINAL_8 = ["BCR", "NB_BCR", "FAR", "AHI", "NB_AHI", "NDVI", "DIS_CBD", "DIS_MRT"]
NEW_7      = ["LSI", "LPI", "MENN", "SHDI", "SVF", "BHD", "Albedo"]
ALL_FEAT   = ORIGINAL_8 + NEW_7   # 15 features total


def _clip_outliers(df, cols, n_std=3):
    df = df.copy()
    for col in cols:
        mean, std = df[col].mean(), df[col].std()
        df[col] = df[col].clip(lower=mean - n_std * std, upper=mean + n_std * std)
    return df


def compute_vif(X_scaled, feature_names):
    try:
        vif_values = [variance_inflation_factor(X_scaled, i)
                      for i in range(X_scaled.shape[1])]
        return pd.DataFrame({
            "Feature": feature_names,
            "VIF":     [round(v, 2) for v in vif_values],
        }).sort_values("VIF", ascending=False).reset_index(drop=True)
    except Exception as e:
        print(f"  VIF calculation failed: {e}")
        return None


def load_features():
    print(f"Loading {SHP_FILE}...")
    gdf = gpd.read_file(SHP_FILE)
    missing = [c for c in ALL_FEAT if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {gdf.columns.tolist()}")
    gdf = gdf.replace([np.inf, -np.inf], np.nan)
    num_cols = gdf.select_dtypes(include=[np.number]).columns
    gdf[num_cols] = gdf[num_cols].fillna(gdf[num_cols].mean())
    gdf = _clip_outliers(gdf, ALL_FEAT)
    print(f"  {len(gdf):,} nodes  |  {len(ALL_FEAT)} features")
    return gdf[ALL_FEAT]


def compute_correlation(df):
    corr = df.corr(method="pearson")
    corr.to_csv("correlation_matrix_full.csv")
    print("Saved -> correlation_matrix_full.csv")
    return corr


def plot_heatmap(corr):
    n, names = len(corr), corr.columns.tolist()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            val    = corr.values[i, j]
            txtcol = "white" if abs(val) > 0.50 else "#222222"
            weight = "bold"  if abs(val) >= 0.70 and i != j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=txtcol, fontweight=weight)
            if i != j and abs(val) >= 0.70:
                ax.add_patch(FancyBboxPatch(
                    (j-0.48, i-0.48), 0.96, 0.96, boxstyle="square,pad=0",
                    linewidth=1.8, edgecolor="#111111", facecolor="none"))

    div = len(ORIGINAL_8) - 0.5
    ax.axhline(div, color="#333333", lw=1.5, linestyle="--", alpha=0.6)
    ax.axvline(div, color="#333333", lw=1.5, linestyle="--", alpha=0.6)

    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=9.5)

    ax.text(-2.0, len(ORIGINAL_8)/2-0.5, "Original\n8",
            fontsize=9, va="center", ha="center", rotation=90, color="#333333", style="italic")
    ax.text(-2.0, len(ORIGINAL_8)+len(NEW_7)/2-0.5, "New\n7",
            fontsize=9, va="center", ha="center", rotation=90, color="#333333", style="italic")

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03).set_label("Pearson r", fontsize=10)
    ax.set_title(
        "Pearson correlation — full 15-feature set\n"
        "(dashed line separates original 8 from new 7  |  bold border = |r| >= 0.70)",
        fontsize=11, pad=14)
    plt.tight_layout()
    plt.savefig("correlation_matrix_full.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved -> correlation_matrix_full.png")


def print_high_pairs(corr, thresh=0.60):
    names = corr.columns.tolist()
    rows  = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            r = corr.values[i, j]
            if abs(r) >= thresh:
                a_new, b_new = names[i] in NEW_7, names[j] in NEW_7
                pair_type = ("new x new"  if a_new and b_new else
                             "orig x orig" if not a_new and not b_new else
                             "new x orig")
                rows.append({
                    "Feature_A": names[i], "Feature_B": names[j],
                    "Pearson_r": round(r, 4), "Abs_r": round(abs(r), 4),
                    "Level": ("very high" if abs(r) >= 0.85 else
                              "high"      if abs(r) >= 0.70 else "moderate"),
                    "Pair_type": pair_type,
                })
    rows = sorted(rows, key=lambda x: -x["Abs_r"])
    df   = pd.DataFrame(rows)
    print(f"\n{'='*65}")
    print(f"  HIGH-CORRELATION PAIRS  (|r| >= {thresh})")
    print(f"{'='*65}")
    if df.empty:
        print("  None above threshold.")
    else:
        for _, row in df.iterrows():
            tag = {"very high":"***","high":"** ","moderate":"*  "}[row["Level"]]
            print(f"  {tag} {row['Feature_A']:<10} <-> {row['Feature_B']:<10}"
                  f"  r={row['Pearson_r']:+.3f}  ({row['Level']})  [{row['Pair_type']}]")
    df.to_csv("high_pairs_full.csv", index=False)
    print("Saved -> high_pairs_full.csv")
    return df


def run_vif(df):
    X_scaled = StandardScaler().fit_transform(df.values)
    vif_df   = compute_vif(X_scaled, ALL_FEAT)
    if vif_df is None:
        return None
    print(f"\n{'='*65}")
    print("  VIF -- full 15-feature set")
    print(f"{'='*65}")
    for _, row in vif_df.iterrows():
        tag   = "  <-- ABOVE 10" if row["VIF"] > 10 else ""
        group = "(new)" if row["Feature"] in NEW_7 else "(orig)"
        print(f"  {row['Feature']:<12} {group}  VIF = {row['VIF']:.2f}{tag}")
    high = vif_df[vif_df["VIF"] > 10]
    if high.empty:
        print("\n  All VIF <= 10 -- no multicollinearity issues.")
    else:
        print(f"\n  Features above VIF 10: {high['Feature'].tolist()}")
    vif_df.to_csv("vif_full.csv", index=False)
    print("Saved -> vif_full.csv")
    return vif_df


def summarise(corr, vif_df, high_pairs):
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    max_off = high_pairs["Abs_r"].max() if not high_pairs.empty else 0.0
    n_vh = len(high_pairs[high_pairs["Level"] == "very high"])
    n_h  = len(high_pairs[high_pairs["Level"] == "high"])
    n_m  = len(high_pairs[high_pairs["Level"] == "moderate"])
    print(f"  Total features        : {len(ALL_FEAT)}")
    print(f"  Pairs |r| >= 0.85    : {n_vh}")
    print(f"  Pairs |r| >= 0.70    : {n_vh + n_h}")
    print(f"  Pairs |r| >= 0.60    : {n_vh + n_h + n_m}")
    print(f"  Max off-diag |r|     : {max_off:.3f}")
    if vif_df is not None:
        print(f"  Max VIF              : {vif_df['VIF'].max():.2f}")
        print(f"  Features VIF > 10   : {(vif_df['VIF'] > 10).sum()}")
    if not high_pairs.empty:
        print(f"\n  Cross-group pairs (new x orig, |r| >= 0.60):")
        cross = high_pairs[high_pairs["Pair_type"] == "new x orig"]
        if cross.empty:
            print("    None -- all new features add independent information.")
        else:
            for _, row in cross.iterrows():
                print(f"    {row['Feature_A']:<10} <-> {row['Feature_B']:<10}"
                      f"  r={row['Pearson_r']:+.3f}  ({row['Level']})")


if __name__ == "__main__":
    df         = load_features()
    corr       = compute_correlation(df)
    plot_heatmap(corr)
    high_pairs = print_high_pairs(corr, thresh=0.60)
    vif_df     = run_vif(df)
    summarise(corr, vif_df, high_pairs)
