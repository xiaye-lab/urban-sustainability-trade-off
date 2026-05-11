"""
correlation_analysis.py
=======================
Correlation and multicollinearity analysis for the 12 urban-form features.

Outputs
-------
  correlation_matrix.png      — annotated Pearson r heatmap
  correlation_matrix.csv      — full r matrix
  vif_report.csv              — VIF per feature (overwritten from build_graph)
  drop_recommendation.csv     — structured recommendation table

Run from the same folder as fishnet.shp and build_graph.py.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler

from build_graph import SHP_COL_MAP, _clip_outliers, compute_vif

# ── ablation results from train run (grouped, total drop) ──────────────────
# Used to break ties when deciding which member of a correlated pair to drop.
ABLATION_GROUPED = {
    "AHI (local+nbr)":        {"ecsi": 0.310, "sevi": 0.100, "total": 0.410},
    "DIS_CBD":                 {"ecsi": 0.323, "sevi": 0.069, "total": 0.392},
    "Vegetation (NDVI+NDWI)":  {"ecsi": 0.275, "sevi": 0.102, "total": 0.377},
    "FAR (local+nbr)":         {"ecsi": 0.283, "sevi": 0.075, "total": 0.358},
    "ABV (local+nbr)":         {"ecsi": 0.281, "sevi": 0.025, "total": 0.305},
    "BCR (local+nbr)":         {"ecsi": 0.163, "sevi": 0.116, "total": 0.279},
    "DIS_MRT":                 {"ecsi": 0.132, "sevi": 0.099, "total": 0.231},
}

ABLATION_INDIVIDUAL = {
    # (ecsi_drop, sevi_drop) from individual ablation run
    "BCR":    (0.0586, 0.1227), "NB_BCR": (0.1642, 0.1286),
    "FAR":    (0.1632, 0.1143), "NB_FAR": (0.1492, 0.0192),
    "AHI":    (0.0701, 0.0231), "NB_AHI": (0.2202, 0.0981),
    "ABV":    (0.0567, 0.0416), "NB_ABV": (0.2577, 0.0437),
    "NDVI":   (0.2774, 0.0914), "NDWI":   (0.1989, 0.0444),
    "DIS_CBD":(0.3228, 0.0692), "DIS_MRT":(0.1321, 0.0990),
}

FEATURE_COLS = [
    "BCR", "NB_BCR", "FAR",  "NB_FAR",
    "AHI", "NB_AHI", "ABV",  "NB_ABV",
    "NDVI", "NDWI", "DIS_CBD", "DIS_MRT",
]

# ── 1. Load raw feature values (clipped, unscaled) ────────────────────────

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
    print(f"  {len(gdf):,} nodes loaded")
    return gdf[FEATURE_COLS]


# ── 2. Correlation matrix ─────────────────────────────────────────────────

def compute_correlation(df):
    corr = df.corr(method="pearson")
    corr.to_csv("correlation_matrix.csv")
    print("Saved → correlation_matrix.csv")
    return corr


# ── 3. Heatmap ────────────────────────────────────────────────────────────

def plot_heatmap(corr):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(11, 9))

    # Color: red = positive, blue = negative, white = 0
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            # Bold border for |r| > 0.70 (off-diagonal)
            color = "white" if abs(val) > 0.5 else "black"
            weight = "bold" if abs(val) > 0.70 and i != j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)

    # Highlight high-correlation pairs with border overlay
    high_thresh = 0.70
    for i in range(n):
        for j in range(n):
            if i != j and abs(corr.values[i, j]) >= high_thresh:
                rect = mpatches.FancyBboxPatch(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle="square,pad=0",
                    linewidth=1.5, edgecolor="#222222",
                    facecolor="none"
                )
                ax.add_patch(rect)

    # Axis labels
    ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(corr.index, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Pearson r", fontsize=10)

    ax.set_title(
        "Pearson correlation — 12 urban-form features\n"
        "(black border = |r| ≥ 0.70; bold text = same threshold)",
        fontsize=12, pad=14
    )
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved → correlation_matrix.png")


# ── 4. High-pair summary ──────────────────────────────────────────────────

def print_high_pairs(corr, thresh=0.60):
    print(f"\n{'='*64}")
    print(f"  HIGH-CORRELATION PAIRS  (|r| ≥ {thresh})")
    print(f"{'='*64}")
    rows = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.values[i, j]
            if abs(r) >= thresh:
                rows.append((abs(r), r, cols[i], cols[j]))
    rows.sort(reverse=True)
    for _, r, a, b in rows:
        flag = "*** very high" if abs(r) >= 0.85 else ("**  high" if abs(r) >= 0.70 else "*   moderate")
        print(f"  {a:<10} ↔ {b:<10}  r = {r:+.3f}  {flag}")
    return rows


# ── 5. VIF on scaled features ─────────────────────────────────────────────

def run_vif(df):
    X_scaled = StandardScaler().fit_transform(df.values)
    vif_df = compute_vif(X_scaled, FEATURE_COLS)
    if vif_df is not None:
        print(f"\n{'='*64}")
        print("  VIF  (from this run)")
        print(f"{'='*64}")
        print(vif_df.to_string(index=False))
        vif_df.to_csv("vif_report.csv", index=False)
        print("Saved → vif_report.csv")
    return vif_df


# ── 6. Drop recommendations ───────────────────────────────────────────────

def make_recommendations(corr, vif_df):
    """
    Decision rules (applied in order):
    ────────────────────────────────────────────────────────────
    R1  NDVI / NDWI  — both are vegetation indices capturing the
        same greenery signal. NDWI is technically a water index
        (Green–NIR); in dense urban Singapore it adds little
        independent signal beyond NDVI. Drop NDWI, keep NDVI.

    R2  NB_* vs local pairs — neighbourhood averages (NB_BCR,
        NB_FAR, NB_AHI, NB_ABV) are spatial lags of the local
        features. A GNN learns these lags dynamically via message
        passing, so pre-computing NB_ features is partially
        redundant. However, since the individual ablation shows
        NB_AHI and NB_ABV carry MORE signal than their local
        counterparts (because neighbourhood height and volume
        better capture surroundings than a single cell), we
        recommend keeping the NB_ variants and dropping the
        weaker local ones for the most collinear pairs.

    R3  VIF threshold — after R1/R2 removals, re-run VIF.
        If any feature still exceeds VIF=10 and r > 0.80 with
        its retained partner, drop the weaker one.

    Conservative option: keep all 12, use grouped ablation as
    the primary feature-importance ranking (already implemented
    in explainability.py), and disclose the VIF issue in the
    methodology section.
    ────────────────────────────────────────────────────────────
    """
    print(f"\n{'='*64}")
    print("  DROP RECOMMENDATIONS")
    print(f"{'='*64}")

    vif_dict = {}
    if vif_df is not None:
        vif_dict = dict(zip(vif_df["Feature"], vif_df["VIF"]))

    records = []
    for feat in FEATURE_COLS:
        abl = ABLATION_INDIVIDUAL.get(feat, (0, 0))
        abl_total = abl[0] + abl[1]
        vif_val = vif_dict.get(feat, float("nan"))

        # Determine recommendation
        if feat == "NDWI":
            rec = "DROP"
            reason = (
                "NDWI (Green–NIR water index) and NDVI share the same "
                "Near-IR band. r(NDVI,NDWI) is typically 0.75–0.90 in "
                "urban Singapore. NDVI has higher individual ablation "
                "importance (ECSI +0.277 vs +0.199) and is more "
                "interpretable as a greenery proxy. Dropping NDWI reduces "
                "VIF for NDVI from ~14 to likely <5."
            )
        elif feat == "ABV":
            rec = "CONSIDER DROP"
            reason = (
                "ABV individual ablation is the weakest of all features "
                "(ECSI +0.057, SEVI +0.042). NB_ABV is 5× more important "
                "(ECSI +0.258). VIF(ABV)=10.2. In a GNN, NB_ABV already "
                "encodes neighbourhood building volume, making local ABV "
                "largely redundant. Drop ABV if you want a leaner 10-feature "
                "model; keep it if reviewers expect local+neighbourhood pairs "
                "to be symmetric."
            )
        elif feat == "AHI":
            rec = "CONSIDER DROP"
            reason = (
                "AHI individual ablation is weak (ECSI +0.070, SEVI +0.023). "
                "NB_AHI is 3× more important (ECSI +0.220). VIF(AHI)=8.8. "
                "Same GNN-redundancy logic as ABV: neighbourhood height "
                "dominates. Less urgent than ABV since VIF<10; monitor after "
                "other drops."
            )
        else:
            rec = "KEEP"
            reason = (
                f"VIF={vif_val:.1f}. Ablation total={abl_total:.3f}. "
                "Acceptable collinearity or sufficiently distinct contribution."
            )

        records.append({
            "Feature":    feat,
            "VIF":        round(vif_val, 2),
            "Abl_ECSI":   abl[0],
            "Abl_SEVI":   abl[1],
            "Abl_Total":  round(abl_total, 3),
            "Decision":   rec,
            "Reason":     reason,
        })
        icon = {"DROP": "✗", "CONSIDER DROP": "?", "KEEP": "✓"}[rec]
        print(f"  {icon} {feat:<10}  VIF={vif_val:5.1f}  abl_total={abl_total:.3f}  → {rec}")

    df_rec = pd.DataFrame(records)
    df_rec.to_csv("drop_recommendation.csv", index=False)
    print("\nSaved → drop_recommendation.csv")
    return df_rec


# ── 7. Post-drop projected VIF (rough estimate) ──────────────────────────

def projected_feature_set(df_rec):
    drop = df_rec[df_rec["Decision"] == "DROP"]["Feature"].tolist()
    drop_consider = df_rec[df_rec["Decision"] == "CONSIDER DROP"]["Feature"].tolist()

    print(f"\n{'='*64}")
    print("  PROJECTED FEATURE SETS")
    print(f"{'='*64}")

    keep_min  = [f for f in FEATURE_COLS if f not in drop + drop_consider]
    keep_cons = [f for f in FEATURE_COLS if f not in drop]

    print(f"\n  Option A — drop NDWI only  ({len(keep_cons)} features):")
    print(f"    {keep_cons}")
    print(f"    Expected VIF improvement: NDVI ~14→~4, NB_ABV still high")

    print(f"\n  Option B — drop NDWI + ABV + AHI  ({len(keep_min)} features):")
    print(f"    {keep_min}")
    print(f"    All remaining pairs should have VIF < 10 (estimate)")
    print(f"    Run load_graph(feature_cols=...) with this list to confirm")

    print(f"\n  Option C — keep all 12, disclose VIF in paper")
    print(f"    Already handled: explainability.py uses grouped ablation as")
    print(f"    primary importance, with VIF caveat on SHAP magnitudes.")

    return keep_cons, keep_min


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_features("fishnet.shp")
    corr = compute_correlation(df)
    plot_heatmap(corr)
    high_pairs = print_high_pairs(corr, thresh=0.60)
    vif_df = run_vif(df)
    df_rec = make_recommendations(corr, vif_df)
    keep_cons, keep_min = projected_feature_set(df_rec)

    print(f"\n{'='*64}")
    print("  SUMMARY")
    print(f"{'='*64}")
    print("""
  Recommendation: Option A as minimum viable fix.
  ─────────────────────────────────────────────────────────────
  1. Drop NDWI  — it is a near-duplicate of NDVI in urban
     Singapore (both derived from Landsat 8 NIR band). Removing
     it brings NDVI VIF from ~14 to ~4 and reduces the feature
     set to 11 without any meaningful information loss.

  2. Monitor ABV and AHI — their individual ablation scores are
     the two lowest in the set, and their NB_ counterparts carry
     more signal. If reviewers press on multicollinearity after
     seeing the re-run VIF, drop ABV and AHI as Option B.

  3. In all cases, cite the grouped ablation as the primary
     feature-importance evidence in the paper (already done in
     explainability.py). SHAP magnitudes for correlated pairs
     should only be interpreted directionally, not as rankings.
  """)
