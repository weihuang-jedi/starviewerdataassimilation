import argparse
import os
import torch

def inspect_graph_pt(pt_path: str):
    """
    Loads and runs diagnostic assertions on a PyTorch Geometric edge_index file.
    """
    if not os.path.exists(pt_path):
        print(f"CRITICAL ERROR: File '{pt_path}' does not exist.")
        return

    print(f"Loading PyTorch Tensor file: {pt_path}")
    try:
        tensor = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read PyTorch file. Is it corrupted? Error: {e}")
        return

    # Basic Type & Shape Checks
    print("-" * 55)
    print("                      DIAGNOSTIC REPORT")
    print("-" * 55)
    print(f"Tensor Data Type        : {tensor.dtype}")
    print(f"Tensor Class Type       : {type(tensor)}")
    print(f"Tensor Matrix Shape     : {list(tensor.shape)}")
    
    # Assert standard edge_index format: [2, num_edges]
    if len(tensor.shape) != 2 or tensor.shape[0] != 2:
        print("WARNING: This tensor shape deviates from standard PyTorch Geometric layouts (expected shape: [2, num_edges]).")
        if tensor.shape[1] == 2:
            print(" -> Detected transposition: Tensor is [num_edges, 2]. Fixing orientation dynamically for diagnostics...")
            tensor = tensor.t()

    num_edges = tensor.shape[1]
    print(f"Total Directed Edges    : {num_edges:,}")

    # Check for NaN / Infinite values
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    print(f"NaN Values Detected     : {nan_count}")
    print(f"Inf Values Detected     : {inf_count}")

    # Range and Index Bounds Validation
    min_idx = tensor.min().item()
    max_idx = tensor.max().item()
    print(f"Minimum Node ID         : {min_idx}")
    print(f"Maximum Node ID         : {max_idx}")
    print(f"Total Unique Node IDs   : {len(torch.unique(tensor)):,}")

    # Topological Validation
    src, dst = tensor[0], tensor[1]
    self_loops = (src == dst).sum().item()
    print(f"Self-Loops Found        : {self_loops} (Nodes connected to themselves)")

    # Check for Bidirectional Symmetry (Undirected validation)
    # Stacking src/dst vs dst/src to check symmetry
    print("Checking for graph symmetry...")
    forward_tuples = set(zip(src.tolist(), dst.tolist()))
    backward_tuples = set(zip(dst.tolist(), src.tolist()))
    is_undirected = forward_tuples == backward_tuples
    print(f"Is Graph Fully Symmetric: {is_undirected} (Every edge A->B has a corresponding B->A)")

    # Compute node degrees (number of neighbors)
    unique_nodes, degrees = torch.unique(tensor[0], return_counts=True)
    print(f"Mean Node Degree (Avg)  : {degrees.float().mean().item():.2f} connections")
    print(f"Max Node Degree         : {degrees.max().item()} connections")
    print(f"Min Node Degree         : {degrees.min().item()} connections")
    print("-" * 55 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic inspector for PyTorch GNN geometry (.pt) files.")
    parser.add_argument("-f", "--file", required=True, help="Path to your .pt tensor edge file")
    args = parser.parse_args()
    inspect_graph_pt(args.file)

