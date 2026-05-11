"""
gnn_model.py
============
GATv2-based graph neural network predicting ECSI and SEVI simultaneously.

Architecture
------------
Layer 1 : GATv2Conv(F → 64, heads=4, concat=True, edge_dim=1)  → 256-dim
          ELU + Dropout(0.2)
Layer 2 : GATv2Conv(256 → 64, heads=1, concat=False, edge_dim=1) → 64-dim
          ELU
Output  : Linear(64→1) for ECSI  +  Linear(64→1) for SEVI  → [N,2]

Edge weights (row-standardised rook + cosine similarity, merged by max)
are passed as 1-D edge_attr to both GATv2Conv layers.  The GATv2 attention
mechanism then learns dynamic, query-dependent re-weighting ON TOP of these
structural priors.  This resolves Reviewer 3.1.1: manual weights initialise
structural strength; attention refines importance conditioned on node features.

Separate output heads prevent competing gradients between ECSI and SEVI,
which have partially overlapping but distinct dominant drivers.

Total parameters: ~42K (well-regularised for ~7,300 training nodes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class UrbanGNN(torch.nn.Module):

    def __init__(self, num_features, hidden_dim=64, num_targets=2, heads=4):
        super().__init__()
        concat_dim = hidden_dim * heads          # 64 × 4 = 256

        # edge_dim=1: accept scalar edge weight as edge_attr
        self.conv1 = GATv2Conv(
            num_features, hidden_dim,
            heads=heads, concat=True, edge_dim=1,
        )
        self.conv2 = GATv2Conv(
            concat_dim, hidden_dim,
            heads=1, concat=False, edge_dim=1,
        )

        # Separate heads — one per target (avoids competing gradients)
        self.out_ecsi = nn.Linear(hidden_dim, 1)
        self.out_sevi = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight):
        """
        Parameters
        ----------
        x           : [N, F]   node features (z-score normalised)
        edge_index  : [2, E]   graph connectivity
        edge_weight : [E]      scalar edge weights in (0, 1]
        """
        ew = edge_weight.unsqueeze(-1)          # [E, 1] for edge_dim=1

        x = self.conv1(x, edge_index, edge_attr=ew)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index, edge_attr=ew)
        x = F.elu(x)

        return torch.cat([self.out_ecsi(x), self.out_sevi(x)], dim=1)
