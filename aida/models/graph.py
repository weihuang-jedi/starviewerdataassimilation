import os
import numpy as np
import torch


def generate_or_load_edge_index(num_nodes: int, edge_file: str = "") -> torch.Tensor:
    """Loads existing edge connectivity or constructs a synthetic k-NN edge graph."""
    if edge_file and os.path.exists(edge_file):
        print(f"[GRAPH] Loading precomputed edge topology from '{edge_file}'...")
        edge_index = torch.load(edge_file)
        if isinstance(edge_index, dict) and "edge_index" in edge_index:
            edge_index = edge_index["edge_index"]
        return edge_index.to(torch.long)

    print(f"[GRAPH] Generating synthetic icosahedral mesh graph for {num_nodes} nodes...")
    phi = np.linspace(0, np.pi, int(np.sqrt(num_nodes)))
    theta = np.linspace(0, 2 * np.pi, int(np.sqrt(num_nodes)))
    phi_m, theta_m = np.meshgrid(phi, theta)

    x = np.sin(phi_m) * np.cos(theta_m)
    y = np.sin(phi_m) * np.sin(theta_m)
    z = np.cos(phi_m)
    coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T[:num_nodes]

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=7)

    src_list, dst_list = [], []
    for i, neighbors in enumerate(indices):
        for n in neighbors[1:]:
            src_list.append(i)
            dst_list.append(n)

    return torch.tensor([src_list, dst_list], dtype=torch.long)
