#!/usr/bin/env python3
"""
inspect_aida_inputs.py
----------------------
Inspects icosahedral Zarr background and NetCDF graph mesh coordinates.
"""

import torch
import xarray as xr
import zarr

# 1. Inspect Zarr Background Data
zarr_path = "../data/icosahedral_2023.zarr"
print(f"=== Inspecting Zarr Background: {zarr_path} ===")
z_root = zarr.open(zarr_path, mode="r")
print(z_root.tree())

# 2. Inspect Mesh NetCDF File (e.g., m4 mesh level)
mesh_nc_path = "../data/graph/global_icosahedral_mesh_m4.nc"
print(f"\n=== Inspecting Graph Mesh: {mesh_nc_path} ===")
ds_mesh = xr.open_dataset(mesh_nc_path)
print(ds_mesh)

# 3. Inspect Edge Index Tensor (PyTorch)
edge_pt_path = "../data/graph/edge_index_m4.pt"
print(f"\n=== Inspecting Graph Edges: {edge_pt_path} ===")
edge_index = torch.load(edge_pt_path)
print(
    f"Edge Index Shape: {edge_index.shape}, Min Node: {edge_index.min()}, Max Node: {edge_index.max()}"
)
