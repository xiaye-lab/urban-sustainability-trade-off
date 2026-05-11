"""
morans_i_analysis.py
====================
Moran's I spatial autocorrelation analysis — defends the random 70/15/15 split.

Four sequential tests (R3.1.4 response):
  Step 1 — Observed ECSI/SEVI        (expected HIGH: urban data is clustered)
  Step 2 — Residuals, all nodes      (overall unexplained spatial signal)
  Step 3 — Residuals, test nodes     (KEY: split bias check)
  Step 4 — Predictions, test nodes   (model captures spatial structure?)

Run AFTER train_gnn.py has produced gnn_prediction.csv.
Requires: pip install esda libpysal
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from libpysal.weights import KNN
from esda.moran import Moran

PRED_CSV    = "gnn_prediction.csv"
K           = 8
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(PRED_CSV).reset_index(drop=True)
print(f"Loaded {len(df):,} nodes")

required = ["X", "Y", "ECSI", "SEVI", "ECSI_pred", "SEVI_pred"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}. Re-run train_gnn.py first.")

# ── Recreate split (same seed as train_gnn.py) ────────────────────────────────
N   = len(df)
rng = np.random.default_rng(SEED)
idx = np.arange(N); rng.shuffle(idx)
n_tr = int(TRAIN_RATIO * N)
n_va = int(VAL_RATIO   * N)
split = np.full(N, "train", dtype=object)
split[idx[n_tr: n_tr + n_va]] = "val"
split[idx[n_tr + n_va:]]      = "test"
df["split"] = split
print(f"Split → train:{(split=='train').sum():,} | "
      f"val:{(split=='val').sum():,} | test:{(split=='test').sum():,}\n")

df["ECSI_resid"] = df["ECSI"] - df["ECSI_pred"]
df["SEVI_resid"] = df["SEVI"] - df["SEVI_pred"]

# ── Spatial weights ───────────────────────────────────────────────────────────
w_all  = KNN.from_array(df[["X","Y"]].values.tolist(), k=K)
w_all.transform = "r"

df_test = df[df["split"] == "test"].reset_index(drop=True)
w_test  = KNN.from_array(df_test[["X","Y"]].values.tolist(), k=K)
w_test.transform = "r"

# ── Moran helper ──────────────────────────────────────────────────────────────
def moran_test(vals, w, label):
    mi  = Moran(vals, w)
    sig = "p<0.05 *" if mi.p_sim < 0.05 else "p>=0.05 (ns)"
    print(f"  {label:<48}  I={mi.I:>7.4f}  {sig}")
    return mi

# ── Tests ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 1 — Observed values (expected HIGH)")
print("=" * 70)
mi_e_obs = moran_test(df["ECSI"].values,    w_all, "ECSI observed (all)")
mi_s_obs = moran_test(df["SEVI"].values,    w_all, "SEVI observed (all)")

print("\n" + "=" * 70)
print("  STEP 2 — Residuals, all nodes")
print("=" * 70)
mi_e_ra  = moran_test(df["ECSI_resid"].values, w_all, "ECSI residuals (all)")
mi_s_ra  = moran_test(df["SEVI_resid"].values, w_all, "SEVI residuals (all)")

print("\n" + "=" * 70)
print("  STEP 3 — Residuals, test nodes  ← KEY RESULT")
print("=" * 70)
mi_e_rt  = moran_test(df_test["ECSI_resid"].values, w_test, "ECSI residuals (test)")
mi_s_rt  = moran_test(df_test["SEVI_resid"].values, w_test, "SEVI residuals (test)")

print("\n" + "=" * 70)
print("  STEP 4 — Predictions, test nodes")
print("=" * 70)
mi_e_pt  = moran_test(df_test["ECSI_pred"].values, w_test, "ECSI predictions (test)")
mi_s_pt  = moran_test(df_test["SEVI_pred"].values, w_test, "SEVI predictions (test)")

# ── Summary ───────────────────────────────────────────────────────────────────
def drop_pct(obs, res):
    return (obs - res) / abs(obs) * 100 if obs != 0 else 0

print("\n" + "=" * 70)
print(f"  {'Metric':<46} {'ECSI':>8} {'SEVI':>8}")
print(f"  {'-'*62}")
rows = [
    ("Observed Moran I (all)",           mi_e_obs.I, mi_s_obs.I),
    ("Residual Moran I (all)",           mi_e_ra.I,  mi_s_ra.I),
    ("Residual Moran I (test)",          mi_e_rt.I,  mi_s_rt.I),
    ("p-value residuals (test)",         mi_e_rt.p_sim, mi_s_rt.p_sim),
    ("Prediction Moran I (test)",        mi_e_pt.I,  mi_s_pt.I),
    ("Residual reduction from observed", drop_pct(mi_e_obs.I, mi_e_rt.I),
                                         drop_pct(mi_s_obs.I, mi_s_rt.I)),
]
for label, ve, vs in rows:
    print(f"  {label:<46} {ve:>8.4f} {vs:>8.4f}")

# ── Scatter plots ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Moran's I — GATv2 Residuals and Predictions (Test Set)",
             fontsize=12, y=0.98)

panels = [
    (axes[0,0], "ECSI_resid", w_test, mi_e_rt, "ECSI residuals (test)"),
    (axes[0,1], "SEVI_resid", w_test, mi_s_rt, "SEVI residuals (test)"),
    (axes[1,0], "ECSI_pred",  w_test, mi_e_pt, "ECSI predictions (test)"),
    (axes[1,1], "SEVI_pred",  w_test, mi_s_pt, "SEVI predictions (test)"),
]
for ax, col, w_mat, mi, title in panels:
    vals = df_test[col].values
    lag  = w_mat.sparse.toarray() @ vals
    ax.scatter(vals, lag, alpha=0.2, s=4, color="#378ADD", rasterized=True)
    ax.axhline(0, color="#888780", lw=0.6, ls="--")
    ax.axvline(0, color="#888780", lw=0.6, ls="--")
    m, b = np.polyfit(vals, lag, 1)
    xl = np.linspace(vals.min(), vals.max(), 200)
    ax.plot(xl, m*xl+b, color="#A32D2D", lw=1.5)
    sig = "p<0.05" if mi.p_sim < 0.05 else "p>=0.05"
    ax.set_title(f"Moran's I={mi.I:.4f} ({sig})", fontsize=10)
    ax.set_xlabel(title, fontsize=9)
    ax.set_ylabel("Spatial lag", fontsize=9)

plt.tight_layout()
plt.savefig("morans_scatter_residuals.png", dpi=200, bbox_inches="tight")
print("\nSaved → morans_scatter_residuals.png")
