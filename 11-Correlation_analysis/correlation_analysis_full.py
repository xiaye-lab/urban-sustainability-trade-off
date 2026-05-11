"""
correlation_analysis_full.py
=============================
1. Pearson correlation matrix and VIF for the full 15-feature set
2. Spearman correlation of all 15 features vs. dependent variables (ECSI, SEVI)

  Original 8   : BCR, NB_BCR, FAR, AHI, NB_AHI, NDVI, DIS_CBD, DIS_MRT
  New 7        : LSI, LPI, MENN, SHDI, SVF, BHD, Albedo
  Dependents   : ECSI, SEVI

Loads from fishnet_final_composite.shp (all columns already clean).

Outputs
-------
  correlation_matrix_full.png         annotated heatmap (15 x 15, independent features)
  correlation_matrix_full.csv         full r matrix (independent features)
  vif_full.csv                        VIF per feature
  high_pairs_full.csv                 pairs |r| >= 0.60, sorted descending
  dep_correlation_spearman.csv        Spearman r of each feature vs ECSI and SEVI
  dep_correlation_spearman.png        bar chart: feature vs ECSI / SEVI
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr
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

SHP_FILE = "fishnet_final_composite.shp"

ORIGINAL_8 = ["BCR", "NB_BCR", "FAR", "AHI", "NB_AHI", "NDVI", "DIS_CBD", "DIS_MRT"]
NEW_7      = ["LSI", "LPI", "MENN", "SHDI", "SVF", "BHD", "Albedo"]
ALL_FEAT   = ORIGINAL_8 + NEW_7          # 15 features
DEPENDENTS = ["ECSI", "SEVI"]


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


def load_data():
    print(f"Loading {SHP_FILE}...")
    gdf = gpd.read_file(SHP_FILE)

    all_cols = ALL_FEAT + DEPENDENTS
    missing = [c for c in all_cols if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {gdf.columns.tolist()}")

    gdf = gdf.replace([np.inf, -np.inf], np.nan)
    num_cols = gdf.select_dtypes(include=[np.number]).columns
    gdf[num_cols] = gdf[num_cols].fillna(gdf[num_cols].mean())
    gdf = _clip_outliers(gdf, ALL_FEAT)   # only clip features, not composites

    print(f"  {len(gdf):,} nodes  |  {len(ALL_FEAT)} features  |  {len(DEPENDENTS)} dependents")
    return gdf[ALL_FEAT], gdf[DEPENDENTS]


# ─────────────────────────────────────────────
#  SECTION 1 : independent feature correlation
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  SECTION 2 : feature vs. dependent (Spearman)
# ─────────────────────────────────────────────
def compute_dep_correlation(feat_df, dep_df):
    """
    Spearman r between each of the 15 features and each dependent (ECSI, SEVI).
    Also flags NDVI and NDWI explicitly for the reviewer's concern.
    """
    print(f"\n{'='*65}")
    print("  SPEARMAN CORRELATION  --  Features vs. ECSI / SEVI")
    print(f"{'='*65}")
    print(f"  {'Feature':<12}  {'Group':<6}  {'r_ECSI':>8}  {'p_ECSI':>8}  "
          f"{'r_SEVI':>8}  {'p_SEVI':>8}")
    print(f"  {'-'*60}")

    rows = []
    for feat in ALL_FEAT:
        group = "new" if feat in NEW_7 else "orig"
        r_ecsi, p_ecsi = spearmanr(feat_df[feat], dep_df["ECSI"])
        r_sevi, p_sevi = spearmanr(feat_df[feat], dep_df["SEVI"])

        sig_ecsi = "***" if p_ecsi < 0.001 else ("**" if p_ecsi < 0.01 else
                   ("*"  if p_ecsi < 0.05  else "ns"))
        sig_sevi = "***" if p_sevi < 0.001 else ("**" if p_sevi < 0.01 else
                   ("*"  if p_sevi < 0.05  else "ns"))

        flag = "  <-- NDVI/NDWI (reviewer concern)" if feat in ("NDVI", "NDWI") else ""
        print(f"  {feat:<12}  {group:<6}  {r_ecsi:>+8.4f}  {sig_ecsi:>8}  "
              f"  {r_sevi:>+8.4f}  {sig_sevi:>8}{flag}")

        rows.append({
            "Feature": feat, "Group": group,
            "r_ECSI": round(r_ecsi, 4), "p_ECSI": round(p_ecsi, 6), "sig_ECSI": sig_ecsi,
            "r_SEVI": round(r_sevi, 4), "p_SEVI": round(p_sevi, 6), "sig_SEVI": sig_sevi,
        })

    # ECSI <-> SEVI inter-index correlation
    r_idx, p_idx = spearmanr(dep_df["ECSI"], dep_df["SEVI"])
    print(f"\n  ECSI <-> SEVI  :  Spearman r = {r_idx:+.4f}  (p = {p_idx:.4e})")
    print(f"  {'='*60}")

    df = pd.DataFrame(rows)
    df.to_csv("dep_correlation_spearman.csv", index=False)
    print("Saved -> dep_correlation_spearman.csv")
    return df, (r_idx, p_idx)


def plot_dep_correlation(dep_corr_df):
    """
    Grouped horizontal bar chart: Spearman r for each feature vs ECSI and SEVI.
    Features sorted by |r_ECSI| descending.
    """
    df = dep_corr_df.sort_values("r_ECSI", key=abs, ascending=True)   # ascending for horizontal

    features = df["Feature"].tolist()
    r_ecsi   = df["r_ECSI"].tolist()
    r_sevi   = df["r_SEVI"].tolist()
    n        = len(features)

    y      = np.arange(n)
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))
    bars1 = ax.barh(y + height/2, r_ecsi, height, label="vs ECSI",
                    color="#E05C5C", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.barh(y - height/2, r_sevi, height, label="vs SEVI",
                    color="#4A90D9", alpha=0.85, edgecolor="white", linewidth=0.5)

    # threshold lines
    for thresh, ls in [(0.6, "--"), (0.3, ":")]:
        ax.axvline( thresh, color="#555", lw=0.8, linestyle=ls, alpha=0.5)
        ax.axvline(-thresh, color="#555", lw=0.8, linestyle=ls, alpha=0.5)

    ax.axvline(0, color="#333", lw=1.0)

    # colour the y-tick labels by group
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=9)
    for tick, feat in zip(ax.get_yticklabels(), features):
        tick.set_color("#1a5276" if feat in NEW_7 else "#784212")

    # annotate bar values
    for bar in bars1:
        w = bar.get_width()
        ax.text(w + (0.01 if w >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f"{w:+.2f}", va="center", ha="left" if w >= 0 else "right", fontsize=7)
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + (0.01 if w >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f"{w:+.2f}", va="center", ha="left" if w >= 0 else "right", fontsize=7)

    ax.set_xlabel("Spearman r", fontsize=10)
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(
        "Spearman correlation — GNN features vs. ECSI and SEVI\n"
        "(blue labels = new features  |  brown labels = original features  |"
        "  dashed = |r|=0.60  dotted = |r|=0.30)",
        fontsize=10, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig("dep_correlation_spearman.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved -> dep_correlation_spearman.png")


# ─────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────
def summarise(corr, vif_df, high_pairs, dep_corr_df, idx_corr):
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

    print(f"\n  Cross-group pairs (new x orig, |r| >= 0.60):")
    if not high_pairs.empty:
        cross = high_pairs[high_pairs["Pair_type"] == "new x orig"]
        if cross.empty:
            print("    None -- all new features add independent information.")
        else:
            for _, row in cross.iterrows():
                print(f"    {row['Feature_A']:<10} <-> {row['Feature_B']:<10}"
                      f"  r={row['Pearson_r']:+.3f}  ({row['Level']})")

    # highlight NDVI vs ECSI for the reviewer
    print(f"\n  NDVI / NDWI vs. ECSI (reviewer concern):")
    for feat in ("NDVI", "NDWI"):
        row = dep_corr_df[dep_corr_df["Feature"] == feat]
        if not row.empty:
            r = row.iloc[0]["r_ECSI"]
            s = row.iloc[0]["sig_ECSI"]
            print(f"    {feat:<6}  r_ECSI = {r:+.4f}  ({s})")
        else:
            print(f"    {feat:<6}  not found in feature set")

    r_idx, p_idx = idx_corr
    print(f"\n  ECSI <-> SEVI inter-index Spearman r = {r_idx:+.4f}  (p = {p_idx:.4e})")


if __name__ == "__main__":
    feat_df, dep_df = load_data()

    # Section 1 -- independent feature multicollinearity
    corr       = compute_correlation(feat_df)
    plot_heatmap(corr)
    high_pairs = print_high_pairs(corr, thresh=0.60)
    vif_df     = run_vif(feat_df)

    # Section 2 -- feature vs. dependent variable correlation
    dep_corr_df, idx_corr = compute_dep_correlation(feat_df, dep_df)
    plot_dep_correlation(dep_corr_df)

    summarise(corr, vif_df, high_pairs, dep_corr_df, idx_corr)
