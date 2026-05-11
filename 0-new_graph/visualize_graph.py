"""
visualize_graph.py
==================
Renders graph topology coloured by ECSI or SEVI.
Only edges with weight > 0.80 are drawn for visual clarity.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from build_graph import load_graph


def plot_graph_structure(shp_file="fishnet.shp", color_by="ECSI",
                         max_nodes=None, out_png=None):
    if out_png is None:
        out_png = f"graph_structure_{color_by}.png"

    print(f"Generating graph visualisation — colour by {color_by}...")
    data, gdf, _, _, _ = load_graph(shp_file=shp_file)

    strong     = data.edge_weight > 0.80
    edge_index = data.edge_index[:, strong]

    G = nx.Graph()
    G.add_edges_from(edge_index.cpu().numpy().T)
    G.add_nodes_from(gdf["node_idx"].tolist())

    if max_nodes and len(G.nodes()) > max_nodes:
        keep = np.random.choice(list(G.nodes()), max_nodes, replace=False)
        G    = G.subgraph(keep)
        gdf  = gdf[gdf["node_idx"].isin(keep)]

    pos       = {int(r.node_idx): (r.X, r.Y) for _, r in gdf.iterrows()}
    df_idx    = gdf.set_index("node_idx")
    node_colors = ([df_idx.loc[n, color_by] for n in G.nodes()]
                   if color_by in df_idx.columns else "gray")

    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=node_colors,
                           cmap="coolwarm", ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.1, alpha=0.2, ax=ax)

    labels = {"ECSI": "Environmental Condition Stress Index (ECSI)",
              "SEVI": "Socioeconomic Vitality Index (SEVI)"}
    ax.set_title(labels.get(color_by, color_by), fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    plot_graph_structure(color_by="ECSI")
    plot_graph_structure(color_by="SEVI")
