import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. Load Data
# Ensure these files are in your directory
try:
    nodes_df = pd.read_csv('nodes_with_metrics.csv')
    pareto_df = pd.read_csv('classified_pareto_nodes.csv')
except FileNotFoundError:
    print("Files not found. Please ensure data files are in the working directory.")
    exit()

# 2. Filter Data
# We filter only for the 3 requested categories
target_classes = ['Economic Anchor', 'Balanced Transition', 'Ecological Buffer']
pareto_df_filtered = pareto_df[pareto_df['Node_Classification'].isin(target_classes)].copy()

# 3. Define Indicators
indicators = [
    'BCR', 'NB_BCR', 'FAR', 'NB_FAR', 'AHI', 'NB_AHI', 
    'ABV', 'NB_ABV', 'NDVI', 'NDWI', 'DIS_MRT', 'DIS_CBD',
    'Degree Centrality', 'Betweenness Centrality', 'Clustering Coefficient'
]

# 4. Color Map
color_map = {
    'Economic Anchor': "#CD5659",      # Purple
    'Balanced Transition': "#F7B69B",  # Red
    'Ecological Buffer': "#4257CA"     # Green
}

# 5. Setup Figure
# "whitegrid" style matches the vertical lines seen in your reference image
sns.set_style("whitegrid") 

n_cols = 4
n_rows = (len(indicators) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 6 * n_rows))
axes = axes.flatten()

# 6. Plotting Loop (Swarm Plot)
for i, indicator in enumerate(indicators):
    ax = axes[i]
    
    if indicator in pareto_df_filtered.columns:
        # SWARMPLOT: This creates the "beeswarm" style where points don't overlap
        sns.swarmplot(
            data=pareto_df_filtered,
            x='Node_Classification',  # Categories on X axis
            y=indicator,              # Values on Y axis
            hue='Node_Classification',
            palette=color_map,
            ax=ax,
            size=15,                   # Adjust dot size 
            edgecolor='gray',         # Slight border for definition
            linewidth=0.5
        )
    
    ax.set_title(indicator, fontsize=11, pad=10, fontweight='bold')
    ax.set_ylabel('Value', fontsize=10)
    ax.set_xlabel('')
    
    # Remove individual legends to keep it clean
    if ax.get_legend():
        ax.legend_.remove()

# 7. Layout Adjustment
plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.92)

# Remove empty subplots
for i in range(len(indicators), len(axes)):
    fig.delaxes(axes[i])

# 8. Global Legend
unique_classes = [cls for cls in target_classes if cls in pareto_df_filtered['Node_Classification'].unique()]
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=cls, 
           markerfacecolor=color_map.get(cls, '#333333'), 
           markersize=10, markeredgecolor='k')
    for cls in unique_classes
]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
       ncol=len(unique_classes), fontsize=12, title="Node Classification")

plt.savefig('swarmplot_analysis.png', dpi=300, bbox_inches='tight')
plt.show()