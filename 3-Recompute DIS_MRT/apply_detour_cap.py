"""
apply_detour_cap.py
====================
Produces the final DIS_MRT column by using network distance where reliable
and falling back to Euclidean where the detour factor exceeds a cap.

Finding from investigate_detour_outliers.py:
  - 28% of nodes have detour > 3x, all with very small Euclidean distances
    (4–117 m). These are nodes whose centroid sits inside an MRT station
    building — the OSM walk network has no path through the station footprint,
    so Dijkstra routes all the way around. This is a data artefact, not a
    real-world barrier.
  - Detour <= 3x nodes (n=6,552, 72%) represent genuine pedestrian routing.

Cap rule:
  If network_dist / euclidean_dist > CAP  →  use euclidean_dist
  Otherwise                               →  use network_dist

Outputs
-------
  fishnet_subset_netdist.shp     — final shapefile with capped DIS_MRT
  dis_mrt_final_comparison.csv   — per-node values and method used
  dis_mrt_final_summary.png      — distribution + scatter
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

CAP = 3.0    # detour factor threshold; above this → Euclidean fallback

# ── Load ─────────────────────────────────────────────────────────────────────
df  = pd.read_csv("dis_mrt_comparison.csv")
gdf = gpd.read_file("fishnet_subset_netdist.shp")

# Rename OBJECTID if still truncated
if "SG_200m_Fe" in gdf.columns:
    gdf = gdf.rename(columns={"SG_200m_Fe": "OBJECTID"})

df["detour"] = df["DIS_MRT_NET"] / df["DIS_MRT_EUC"]

# ── Apply cap ─────────────────────────────────────────────────────────────────
use_network  = df["detour"] <= CAP
df["DIS_MRT_FINAL"]  = np.where(use_network, df["DIS_MRT_NET"], df["DIS_MRT_EUC"])
df["method_used"]    = np.where(use_network, "network", "euclidean_fallback")

n_net = use_network.sum()
n_euc = (~use_network).sum()

# ── Statistics ────────────────────────────────────────────────────────────────
r_all, _ = pearsonr(df["DIS_MRT_EUC"], df["DIS_MRT_FINAL"])
r_net, _ = pearsonr(
    df.loc[use_network, "DIS_MRT_EUC"],
    df.loc[use_network, "DIS_MRT_NET"]
)
detour_valid = df.loc[use_network, "detour"]

print("=" * 60)
print("  FINAL DIS_MRT — CAPPED NETWORK DISTANCE")
print("=" * 60)
print(f"  Detour cap applied     : {CAP}x")
print(f"  Network distance used  : {n_net:,} nodes ({n_net/len(df)*100:.1f}%)")
print(f"  Euclidean fallback     : {n_euc:,} nodes ({n_euc/len(df)*100:.1f}%)")
print(f"  (fallback = node centroid inside station building — OSM routing artefact)")
print()
print(f"  Network-only subset:")
print(f"    Mean detour          : {detour_valid.mean():.3f}x")
print(f"    Median detour        : {detour_valid.median():.3f}x")
print(f"    Pearson r (euc vs net): {r_net:.4f}")
print()
print(f"  Final DIS_MRT (all nodes):")
print(f"    Pearson r vs Euclidean: {r_all:.4f}")
print(f"    Range                 : {df['DIS_MRT_FINAL'].min():.1f} – "
      f"{df['DIS_MRT_FINAL'].max():.1f} m")

# ── Write final DIS_MRT into shapefile ───────────────────────────────────────
id_to_final = df.set_index("OBJECTID")["DIS_MRT_FINAL"].to_dict()
gdf["DIS_MRT"] = gdf["OBJECTID"].map(id_to_final).fillna(gdf["DIS_MRT"]).round(2)
gdf.to_file("fishnet_subset_netdist.shp")
print(f"\nSaved → fishnet_subset_netdist.shp  (DIS_MRT overwritten)")

# ── Save CSV ─────────────────────────────────────────────────────────────────
df[["OBJECTID", "DIS_MRT_EUC", "DIS_MRT_NET", "DIS_MRT_FINAL",
    "detour", "method_used"]].round(2).to_csv(
    "dis_mrt_final_comparison.csv", index=False
)
print("Saved → dis_mrt_final_comparison.csv")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(df.loc[use_network,  "DIS_MRT_EUC"],
           df.loc[use_network,  "DIS_MRT_FINAL"],
           alpha=0.25, s=6, color="#2E75B6", label=f"Network ({n_net:,})")
ax.scatter(df.loc[~use_network, "DIS_MRT_EUC"],
           df.loc[~use_network, "DIS_MRT_FINAL"],
           alpha=0.4,  s=6, color="#ED7D31",
           label=f"Euclidean fallback ({n_euc:,})")
lim = [0, df["DIS_MRT_EUC"].max()]
ax.plot(lim, lim, "k--", lw=1, label="1:1")
ax.set_xlabel("Euclidean DIS_MRT (m)", fontsize=11)
ax.set_ylabel("Final DIS_MRT (m)", fontsize=11)
ax.set_title(f"Euclidean vs Final DIS_MRT\nr = {r_all:.4f}", fontsize=12)
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(df["DIS_MRT_EUC"],   bins=60, alpha=0.5,
        color="#2E75B6", label="Euclidean (original)")
ax.hist(df["DIS_MRT_FINAL"], bins=60, alpha=0.5,
        color="#ED7D31", label=f"Final (network + fallback, cap={CAP}x)")
ax.set_xlabel("DIS_MRT (m)", fontsize=11)
ax.set_ylabel("Node count", fontsize=11)
ax.set_title("Distribution comparison", fontsize=12)
ax.legend(fontsize=9)

fig.suptitle(
    f"Final DIS_MRT: network distance (detour ≤ {CAP}x) + Euclidean fallback\n"
    f"Network: {n_net:,} nodes ({n_net/len(df)*100:.1f}%)  |  "
    f"Fallback: {n_euc:,} nodes ({n_euc/len(df)*100:.1f}%)",
    fontsize=11, y=1.02
)
plt.tight_layout()
plt.savefig("dis_mrt_final_summary.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved → dis_mrt_final_summary.png")

print("\n" + "=" * 60)
print("  REVIEWER RESPONSE — KEY NUMBERS")
print("=" * 60)
print(f"  DIS_CBD  Pearson r (euc vs network) : 0.9691  → Euclidean retained")
print(f"  DIS_MRT  Pearson r (euc vs network) : {r_net:.4f}  → Network distance adopted")
print(f"  Method   : network distance for {n_net/len(df)*100:.1f}% of nodes; Euclidean")
print(f"             fallback for {n_euc/len(df)*100:.1f}% where OSM walk network has gaps")
print(f"             inside MRT station footprints (detour > {CAP}x).")
