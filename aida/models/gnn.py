import torch
import torch.nn as nn


class GraphConvBlock(nn.Module):
    """Graph Convolution Message Passing Block with LayerNorm for numerical stability."""
    def __init__(self, channels: int):
        super().__init__()
        self.fc_msg = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        self.fc_update = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, C, L, N = x.shape
        x_perm = x.permute(0, 2, 3, 1).reshape(B * L * N, C)

        src, dst = edge_index[0], edge_index[1]

        shift = torch.arange(B * L, device=x.device).unsqueeze(1) * N
        src_expanded = (src.unsqueeze(0) + shift).reshape(-1)
        dst_expanded = (dst.unsqueeze(0) + shift).reshape(-1)

        msg_in = torch.cat([x_perm[src_expanded], x_perm[dst_expanded]], dim=-1)
        messages = self.fc_msg(msg_in)

        aggr_msg = torch.zeros_like(x_perm)
        aggr_msg.index_add_(0, dst_expanded, messages)

        # Compute degree average for aggregation
        deg = torch.zeros(B * L * N, 1, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst_expanded, torch.ones((dst_expanded.shape[0], 1), device=x.device, dtype=x.dtype))
        aggr_msg = aggr_msg / torch.clamp(deg, min=1.0)

        updated = self.fc_update(torch.cat([x_perm, aggr_msg], dim=-1))
        updated = self.norm(updated)

        out = updated.reshape(B, L, N, C).permute(0, 3, 1, 2)
        return x + out


class IcosahedralGNNSurrogate(nn.Module):
    """GNN Atmospheric Surrogate Model for icosahedral mesh fields."""
    def __init__(self, in_vars: int = 7, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.encoder = nn.Conv2d(in_vars, hidden_dim, kernel_size=1)
        self.gnn_layers = nn.ModuleList([GraphConvBlock(hidden_dim) for _ in range(num_layers)])
        self.decoder = nn.Conv2d(hidden_dim, in_vars, kernel_size=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for layer in self.gnn_layers:
            h = layer(h, edge_index)
        out = self.decoder(h)
        return out
