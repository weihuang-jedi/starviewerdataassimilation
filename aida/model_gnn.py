# model_gnn.py
import torch
import torch.nn as nn


class MeshGraphConv(nn.Module):
    """Message passing layer on icosahedral mesh nodes."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index):
        # x shape: (Batch, Num_Nodes, Channels)
        src, dst = edge_index[0], edge_index[1]

        # Gather node features for edge endpoints
        x_src = x[:, src, :]
        x_dst = x[:, dst, :]

        # Message passing over graph edges
        edge_features = torch.cat([x_src, x_dst], dim=-1)
        messages = self.mlp(edge_features)

        # Aggregate messages back to destination nodes
        out = torch.zeros_like(x)
        out.index_add_(1, dst, messages)
        return out


class IcosahedralAIDA_GNN(nn.Module):
    """Icosahedral Mesh AI Data Assimilation Model."""

    def __init__(self, num_nodes=2562, num_levels=32, in_vars=6, hidden_dim=128):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_levels = num_levels

        # Input channels = (Variables * Height Levels) for x_b + Observation Innovations
        in_channels = (in_vars * num_levels) * 2

        # 1. Node Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Graph Processor Layers (Message Passing across Icosahedral Faces)
        self.processor1 = MeshGraphConv(hidden_dim, hidden_dim)
        self.processor2 = MeshGraphConv(hidden_dim, hidden_dim)

        # 3. Output Decoder -> Predicts Increments dx
        out_channels = in_vars * num_levels
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, x_b, obs_innovations, edge_index):
        # x_b shape: (Batch, Num_Nodes, Height * Vars)
        # Combine background state and observation innovations
        x_in = torch.cat([x_b, obs_innovations], dim=-1)

        # Encode
        feat = self.encoder(x_in)

        # Graph Message Passing
        feat = feat + self.processor1(feat, edge_index)
        feat = feat + self.processor2(feat, edge_index)

        # Predict Analysis Increment dx
        dx = self.decoder(feat)

        # Calculate Analysis State x_a
        x_a = x_b + dx

        return x_a, dx
