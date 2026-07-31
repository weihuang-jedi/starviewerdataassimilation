#!/usr/bin/env python3
"""
hierarchical_decoder.py
-----------------------
Multi-mesh GNN Decoder for Observation Ingestion and Analysis Feedback.
Ingests observations at lower mesh resolution (e.g., Mesh 3), computes increments,
and upsamples/decodes back to high-resolution model forecast grid (e.g., Mesh 4+).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class IcosahedralGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(IcosahedralGraphConv, self).__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=self.lin(x))

    def message(self, x_j):
        return x_j


class MultiMeshHierarchicalDecoder(nn.Module):
    """
    Hierarchical Decoder propagating observation increments from Mesh 3 -> Mesh 4.
    """
    def __init__(self, edge_index_m3, edge_index_m4, num_m3_nodes=642, num_m4_nodes=2562, in_vars=6, levels=32, hidden_dim=128):
        super(MultiMeshHierarchicalDecoder, self).__init__()
        
        self.num_m3_nodes = num_m3_nodes
        self.num_m4_nodes = num_m4_nodes
        self.in_features = in_vars * levels

        self.register_buffer('edge_index_m3', edge_index_m3)
        self.register_buffer('edge_index_m4', edge_index_m4)

        # 1. Mesh 3 Innovation Encoder
        self.m3_conv = IcosahedralGraphConv(self.in_features, hidden_dim)
        self.m3_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

        # 2. Upsampling Interpolator Matrix (Mesh 3 -> Mesh 4 Nearest Neighbor Projection)
        # Upsamples node features from 642 nodes to 2562 nodes
        upsample_idx = torch.linspace(0, num_m3_nodes - 1, steps=num_m4_nodes).long()
        self.register_buffer('upsample_idx', upsample_idx)

        # 3. Mesh 4 Decoder & Residual Integrator
        self.m4_conv1 = IcosahedralGraphConv(hidden_dim, hidden_dim)
        self.m4_norm1 = nn.LayerNorm(hidden_dim)
        self.m4_conv2 = IcosahedralGraphConv(hidden_dim, self.in_features)

    def forward(self, bg_state_m4, obs_innovations_m3):
        """
        bg_state_m4        : High-res background forecast at Mesh 4 (Batch, Vars, Levels, Nodes_M4)
        obs_innovations_m3 : Observational innovations ingested at Mesh 3 (Batch, Vars, Levels, Nodes_M3)
        """
        batch_size, num_vars, levels, _ = bg_state_m4.shape

        # Flatten Mesh 3 Innovations -> Shape: (Batch * Nodes_M3, Vars * Levels)
        m3_flat = obs_innovations_m3.permute(0, 3, 1, 2).reshape(batch_size * self.num_m3_nodes, self.in_features)

        if batch_size > 1:
            e3_list = [self.edge_index_m3 + (b * self.num_m3_nodes) for b in range(batch_size)]
            batched_e3 = torch.cat(e3_list, dim=1)
        else:
            batched_e3 = self.edge_index_m3

        # Encode innovation at Mesh 3
        h_m3 = self.act(self.m3_norm(self.m3_conv(m3_flat, batched_e3)))

        # Upsample Mesh 3 representations -> Mesh 4
        # Shape transition: (Batch, Nodes_M3, Dim) -> (Batch, Nodes_M4, Dim)
        h_m3_reshaped = h_m3.view(batch_size, self.num_m3_nodes, -1)
        h_m4_upsampled = h_m3_reshaped[:, self.upsample_idx, :].reshape(batch_size * self.num_m4_nodes, -1)

        # Decode & Refine on Mesh 4
        if batch_size > 1:
            e4_list = [self.edge_index_m4 + (b * self.num_m4_nodes) for b in range(batch_size)]
            batched_e4 = torch.cat(e4_list, dim=1)
        else:
            batched_e4 = self.edge_index_m4

        h_m4 = self.act(self.m4_norm1(self.m4_conv1(h_m4_upsampled, batched_e4)))
        delta_m4 = self.m4_conv2(h_m4, batched_e4)

        # Reshape delta to matches Mesh 4 shape: (Batch, Vars, Levels, Nodes_M4)
        analysis_increment = delta_m4.view(batch_size, self.num_m4_nodes, num_vars, levels).permute(0, 2, 3, 1)

        # Final Analysis = Background Forecast + Decoded Hierarchical Increment
        analysis_state_m4 = bg_state_m4 + analysis_increment
        return analysis_state_m4
