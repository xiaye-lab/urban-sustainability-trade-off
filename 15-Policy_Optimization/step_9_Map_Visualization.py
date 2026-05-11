"""
step_9_Map_Visualization.py
===========================================
Generates an unscaled spatial map of Singapore based on the original shapefile.
Highlights the 6 specific lagging nodes used in the Pareto Frontier simulation,
color-coded by their urban archetype.
"""

import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("  SPATIAL MAPPING OF PARETO SIMULATION NODES")
print("="*70)

# ── 1. Load Spatial Data and Archetypes ──────────────────────────────────────
print("Loading unscaled shapefile and archetypes...")
# Read the true geographic boundaries (EPSG:3414)
gdf = gpd.read_file("fishnet_final_composite.shp")
df_arch = pd.read_csv("archetypes.csv")

# Merge the archetype labels onto the spatial data
gdf = gdf.merge(df_arch[['OBJECTID', 'Archetype']], on='OBJECTID', how='left')

# ── 2. Identify the 6 Specific Nodes ─────────────────────────────────────────
# Automatically load the best 6 nodes from the simulation step
try:
    df_best = pd.read_csv("pareto_best_6_nodes.csv")
    selected_nodes = df_best['OBJECTID'].tolist()
except FileNotFoundError:
    print("Error: Could not find 'pareto_best_6_nodes.csv'. Run step 8 first!")
    exit()

# Filter the GeoDataFrame to isolate just these 6 nodes
gdf_selected = gdf[gdf['OBJECTID'].isin(selected_nodes)]
print(f"Located {len(gdf_selected)} target nodes on the map.")

# ── 3. Generate the Map Visualization ────────────────────────────────────────
print("Plotting geographic distribution...")

# Create a large, high-resolution figure
fig, ax = plt.subplots(figsize=(14, 9))

# Plot the entire unscaled fishnet as a light gray background
gdf.plot(ax=ax, color='lightgray', edgecolor='none', alpha=0.4)

# Define the exact same colors used in the Pareto scatter plot
colors = {
    'Economic Anchor': '#E24B4A', 
    'Balanced Transition': '#F1A226', 
    'Ecological Buffer': '#1D9E75'
}

# Plot the 6 specific nodes
for arch, color in colors.items():
    subset = gdf_selected[gdf_selected['Archetype'] == arch]
    if not subset.empty:
        # 1. Color the grid cell itself
        subset.plot(ax=ax, color=color, edgecolor='black', linewidth=1.5, zorder=4)
        
        # 2. Add a large star at the centroid so it is highly visible
        subset.centroid.plot(ax=ax, marker='*', color=color, edgecolor='black', markersize=400, zorder=5)

# Add clear labels (OBJECTID) next to each star
for idx, row in gdf_selected.iterrows():
    # Calculate the centroid of the specific grid cell for the label
    x = row.geometry.centroid.x
    y = row.geometry.centroid.y
    
    # Add a white box with black text for readability
    ax.annotate(text=f"Node {row['OBJECTID']}",
                xy=(x, y),
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=11, 
                fontweight='bold',
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9))

# ── 4. Formatting and Legend ─────────────────────────────────────────────────
plt.title('Spatial Distribution of Target Nodes for Pareto Optimization', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Easting (EPSG:3414)', fontsize=12)
plt.ylabel('Northing (EPSG:3414)', fontsize=12)

# Create custom legend to match the map markers
legend_elements = [
    mlines.Line2D([0], [0], marker='*', color='w', markerfacecolor='#E24B4A', markeredgecolor='black', markersize=16, label='Economic Anchor'),
    mlines.Line2D([0], [0], marker='*', color='w', markerfacecolor='#F1A226', markeredgecolor='black', markersize=16, label='Balanced Transition'),
    mlines.Line2D([0], [0], marker='*', color='w', markerfacecolor='#1D9E75', markeredgecolor='black', markersize=16, label='Ecological Buffer'),
    mlines.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', markersize=14, label='Urban Baseline Grid')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.9, edgecolor='black')

# Clean up the map area
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# Save the final map
plt.savefig("pareto_nodes_map.png", dpi=300, bbox_inches='tight')
print("Saved map visualization -> pareto_nodes_map.png")
print("="*70)