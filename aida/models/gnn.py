import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class EdgeNodeGNNBlock(nn.Module):
    """
    GraphCast-style Edge-Node Interaction Block.
    Processes node features and edge attributes with 3-layer MLPs.
    Uses sub-chunking (chunk_size=100000) to keep peak VRAM under control.
    """
    def __init__(self, channels: int, edge_dim: int = 16, chunk_size: int = 100000):
        super().__init__()
        self.chunk_size = chunk_size

        # Edge update MLP: inputs [src_node, dst_node, prev_edge] -> (channels * 2) -> hidden
        self.edge_mlp = nn.Sequential(
            nn.Linear(channels * 2 + edge_dim, channels * 2),
            nn.LayerNorm(channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, edge_dim)
        )

        # Node update MLP: inputs [node_feat, aggregated_edges] -> (channels * 2) -> hidden
        self.node_mlp = nn.Sequential(
            nn.Linear(channels + edge_dim, channels * 2),
            nn.LayerNorm(channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

        self.node_norm = nn.LayerNorm(channels)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def _forward_edge_mlp_chunked(self, edge_in: torch.Tensor) -> torch.Tensor:
        """Processes edge_in through edge_mlp in small 100k sub-chunks to avoid VRAM spikes."""
        num_edges = edge_in.shape[0]
        if num_edges <= self.chunk_size:
            return self.edge_mlp(edge_in)

        outputs = []
        for i in range(0, num_edges, self.chunk_size):
            outputs.append(self.edge_mlp(edge_in[i:i + self.chunk_size]))
        return torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, L, N = x.shape
        x_perm = x.permute(0, 2, 3, 1).reshape(B * L * N, C)

        src, dst = edge_index[0], edge_index[1]
        shift = torch.arange(B * L, device=x.device).unsqueeze(1) * N
        src_expanded = (src.unsqueeze(0) + shift).reshape(-1)
        dst_expanded = (dst.unsqueeze(0) + shift).reshape(-1)

        # 1. Update Edge Features
        edge_attr_expanded = edge_attr.repeat(B * L, 1)

        if edge_attr_expanded.shape[0] != src_expanded.shape[0]:
            num_repeat = src_expanded.shape[0] // edge_attr.shape[0]
            edge_attr_expanded = edge_attr.repeat(num_repeat, 1)

        edge_in = torch.cat([x_perm[src_expanded], x_perm[dst_expanded], edge_attr_expanded], dim=-1)

        # Sub-chunked pass
        updated_edge_attr = edge_attr_expanded + self._forward_edge_mlp_chunked(edge_in)
        updated_edge_attr = self.edge_norm(updated_edge_attr)

        # 2. Aggregate Messages to Target Nodes
        aggr_msg = torch.zeros_like(x_perm[:, :updated_edge_attr.shape[-1]], dtype=updated_edge_attr.dtype)
        aggr_msg.index_add_(0, dst_expanded, updated_edge_attr)

        deg = torch.zeros(B * L * N, 1, device=x.device, dtype=updated_edge_attr.dtype)
        deg.index_add_(0, dst_expanded, torch.ones((dst_expanded.shape[0], 1), device=x.device, dtype=updated_edge_attr.dtype))
        aggr_msg = aggr_msg / torch.clamp(deg, min=1.0)

        # 3. Update Node Features
        node_in = torch.cat([x_perm, aggr_msg], dim=-1)
        updated_nodes = x_perm + self.node_mlp(node_in)
        updated_nodes = self.node_norm(updated_nodes)

        out = updated_nodes.reshape(B, L, N, C).permute(0, 3, 1, 2)
        return out, updated_edge_attr[:edge_attr.shape[0]]


class ScalableAIDAProcessor(nn.Module):
    """
    Scalable High-Capacity GNN Atmospheric Surrogate Model.
    Uses dynamic edge embedding initialization, chunked MLPs, and gradient checkpointing.
    Supports 25-channel AI-Data Assimilation input.
    """
    def __init__(
        self,
        in_vars: int = 25,
        out_vars: int = 7,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        num_levels: int = 32,
        num_layers: int = 4,
        max_edges: int = 300000,
        q_idx: int = 4,
        q_floor: float = 1e-7
    ):
        super().__init__()
        self.q_idx = q_idx
        self.q_floor = q_floor

        # Pre-encoder MLP (projects in_vars -> hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_vars, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=1)
        )

        # Dynamically sized edge embedding tensor
        self.edge_embedding = nn.Parameter(torch.randn(max_edges, edge_dim) * 0.02)

        # Deep Processor Backbone
        self.gnn_layers = nn.ModuleList([
            EdgeNodeGNNBlock(channels=hidden_dim, edge_dim=edge_dim, chunk_size=100000) for _ in range(num_layers)
        ])

        # Post-decoder MLP (projects hidden_dim -> out_vars)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, out_vars, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        num_edges = edge_index.shape[1]
        edge_attr = self.edge_embedding[:num_edges]

        for layer in self.gnn_layers:
            if self.training and x.is_cuda:
                h, edge_attr = checkpoint.checkpoint(layer, h, edge_index, edge_attr, use_reentrant=False)
            else:
                h, edge_attr = layer(h, edge_index, edge_attr)

        out = self.decoder(h)

        # Humidity softplus constraint
        q_constrained = F.softplus(out[:, self.q_idx:self.q_idx+1, :, :]) + self.q_floor
        out = torch.cat([out[:, :self.q_idx, :, :], q_constrained, out[:, self.q_idx+1:, :, :]], dim=1)

        return out


# Alias ScalableAIDAProcessor to IcosahedralGNNSurrogate for backward compatibility
IcosahedralGNNSurrogate = ScalableAIDAProcessor
