"""
compute_composite.py
=====================
Computes ECSI and SEVI composite indicators using
Combined CRITIC-Entropy weighting + TOPSIS.

Method: Combined CRITIC-Entropy (geometric mean of both objective weights)
  - Entropy weight : rewards indicators with high spatial discriminating power
  - CRITIC weight  : rewards indicators with high variance AND low inter-correlation
  - Geometric mean : balances both without either dominating
  - Min weight floor (5%): prevents theoretically important indicators
    from being suppressed to near-zero by uniform spatial distribution

References:
  Basilio et al. (2024) EC-TOPSIS. arXiv:2504.04169
  Wang et al. (2024) Sci Rep. doi:10.1038/s41598-024-55554-z
  Liu et al. (2025) Int J Environ Res. doi:10.1007/s41742-025-00850-3
  Saisana et al. (2005) JRC-OECD Handbook on Composite Indicators

Inputs  : fishnet_final_price.shp
Outputs : fishnet_final_composite.shp, composite_values.csv,
          composite_weights.csv, composite_sensitivity.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr

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

# Configuration
SHP_FILE   = "fishnet_final_POI.shp"
OUT_SHP    = "fishnet_final_composite.shp"
ECSI_COLS  = ["LST", "PM25", "CO2"]
SEVI_COLS  = ["NTL", "POI_COUNT", "POI_DIVERS", "RND", "POPU", "PRICE_PSM"]
MIN_WEIGHT = 0.05


def minmax(x):
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x, dtype=float)


def minmax_df(df, cols):
    return np.column_stack([minmax(df[c].values.astype(float)) for c in cols])


def entropy_weights(X):
    n = X.shape[0]
    P = X / (X.sum(axis=0, keepdims=True) + 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    H = -(1.0 / np.log(n)) * (P * logP).sum(axis=0)
    d = 1.0 - H
    return d / d.sum()


def critic_weights(X):
    std      = X.std(axis=0)
    corr_mat = np.corrcoef(X.T)
    conflict = np.sum(1 - corr_mat, axis=0)
    info     = std * conflict
    return info / info.sum()


def combined_weights(X, min_w=MIN_WEIGHT):
    w_ent  = entropy_weights(X)
    w_crit = critic_weights(X)
    w_comb = np.sqrt(w_ent * w_crit)
    w_comb = w_comb / w_comb.sum()
    m = len(w_comb)
    if min_w * m < 1.0:
        w_comb = np.maximum(w_comb, min_w)
        w_comb = w_comb / w_comb.sum()
    return w_ent, w_crit, w_comb


def topsis(Xw, cost=False):
    A_pos = Xw.min(axis=0) if cost else Xw.max(axis=0)
    A_neg = Xw.max(axis=0) if cost else Xw.min(axis=0)
    D_pos = np.sqrt(((Xw - A_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((Xw - A_neg) ** 2).sum(axis=1))
    denom = D_pos + D_neg
    return np.where(denom > 0, D_neg / denom, 0.5)


def compute(df, cols, label, is_cost):
    print("\n" + "-" * 60)
    print("  " + label)
    print("  Indicators: " + ", ".join(cols))

    df_c = df[cols].copy()
    for c in cols:
        df_c[c] = pd.to_numeric(df_c[c], errors="coerce")
        df_c[c] = df_c[c].fillna(df_c[c].median())
    X = minmax_df(df_c, cols)

    w_ent, w_crit, w_comb = combined_weights(X)
    w_eq = np.ones(len(cols)) / len(cols)

    print("")
    print("  Indicator      Entropy    CRITIC   Combined")
    print("  " + "-" * 44)
    for c, we, wc, wb in zip(cols, w_ent, w_crit, w_comb):
        print("  {:<14} {:>8.4f}  {:>8.4f}  {:>8.4f}".format(c, we, wc, wb))
    print("  " + "-" * 44)
    print("  {:<14} {:>8.4f}  {:>8.4f}  {:>8.4f}".format(
        "SUM", w_ent.sum(), w_crit.sum(), w_comb.sum()))

    if is_cost:
        primary    = (X * w_comb).sum(axis=1)
        eq_score   = X.mean(axis=1)
        crit_score = (X * w_crit).sum(axis=1)
        ent_score  = (X * w_ent).sum(axis=1)
    else:
        primary    = topsis(X * w_comb, cost=False)
        eq_score   = topsis(X * w_eq,   cost=False)
        crit_score = topsis(X * w_crit, cost=False)
        ent_score  = topsis(X * w_ent,  cost=False)

    r_eq,   _ = spearmanr(primary, eq_score)
    r_crit, _ = spearmanr(primary, crit_score)
    r_ent,  _ = spearmanr(primary, ent_score)

    print("")
    print("  Sensitivity (Spearman r vs combined weight primary):")
    for lbl, r in [("vs Equal Weights", r_eq),
                   ("vs Pure CRITIC",   r_crit),
                   ("vs Pure Entropy",  r_ent)]:
        flag = "robust" if abs(r) >= 0.90 else "check"
        print("    {:<22} r = {:.4f}  {}".format(lbl, r, flag))

    return {
        "primary":  minmax(primary),
        "w_ent":    w_ent,
        "w_crit":   w_crit,
        "w_comb":   w_comb,
        "sensitivity": {
            "combined_vs_equal":   round(r_eq,   4),
            "combined_vs_critic":  round(r_crit, 4),
            "combined_vs_entropy": round(r_ent,  4),
        },
    }


def run():
    print("=" * 60)
    print("  COMPOSITE INDICATOR COMPUTATION")
    print("  Method : Combined CRITIC-Entropy weighting")
    print("  ECSI   : weighted linear aggregation")
    print("  SEVI   : TOPSIS with combined weights")
    print("  Min weight floor per indicator: {}%".format(int(MIN_WEIGHT * 100)))
    print("=" * 60)

    print("\n[1/3] Loading " + SHP_FILE + " ...")
    gdf = gpd.read_file(SHP_FILE)
    print("      {:,} nodes  |  {} columns".format(len(gdf), len(gdf.columns)))

    missing = [c for c in ECSI_COLS + SEVI_COLS if c not in gdf.columns]
    if missing:
        raise ValueError("Missing columns: {}\nAvailable: {}".format(
            missing, gdf.columns.tolist()))

    print("\n[2/3] Computing composites...")
    ecsi = compute(gdf, ECSI_COLS, "ECSI  (Environmental Composite Stress Index)", is_cost=True)
    sevi = compute(gdf, SEVI_COLS, "SEVI  (Socioeconomic Vitality Index)", is_cost=False)

    gdf["ECSI"] = ecsi["primary"].round(6)
    gdf["SEVI"] = sevi["primary"].round(6)

    print("\n" + "=" * 60)
    print("  COMPOSITE SUMMARY")
    print("=" * 60)
    for name, col in [("ECSI  (0=low stress,   1=high stress)",   "ECSI"),
                      ("SEVI  (0=low vitality, 1=high vitality)", "SEVI")]:
        s = gdf[col]
        print("\n  " + name)
        print("    mean={:.4f}  std={:.4f}  min={:.4f}  max={:.4f}".format(
            s.mean(), s.std(), s.min(), s.max()))

    print("\n[3/3] Saving...")
    gdf.to_file(OUT_SHP)
    print("  Saved -> " + OUT_SHP)

    gdf[["OBJECTID", "ECSI", "SEVI"]].to_csv("composite_values.csv", index=False)
    print("  Saved -> composite_values.csv")

    rows = []
    for comp, res, cols in [("ECSI", ecsi, ECSI_COLS), ("SEVI", sevi, SEVI_COLS)]:
        for c, we, wc, wb in zip(cols, res["w_ent"], res["w_crit"], res["w_comb"]):
            rows.append({
                "Composite":  comp,
                "Indicator":  c,
                "Entropy_w":  round(float(we), 4),
                "CRITIC_w":   round(float(wc), 4),
                "Combined_w": round(float(wb), 4),
            })
    pd.DataFrame(rows).to_csv("composite_weights.csv", index=False)
    print("  Saved -> composite_weights.csv")

    sens = []
    for comp, res in [("ECSI", ecsi), ("SEVI", sevi)]:
        for pair, r in res["sensitivity"].items():
            sens.append({"Composite": comp, "Comparison": pair, "Spearman_r": r})
    pd.DataFrame(sens).to_csv("composite_sensitivity.csv", index=False)
    print("  Saved -> composite_sensitivity.csv")

    print("\n  Done. New columns added: ECSI, SEVI")
    return gdf


if __name__ == "__main__":
    run()
