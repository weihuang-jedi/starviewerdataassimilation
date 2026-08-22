#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def verify_boundary_edges(edge_index_path: str, mesh_nc_path: str, lon_threshold: float = 300.0):
    """
    Identifies and plots edges that cross between the easternmost (e.g., >300°) 
    and westernmost (e.g., <60°) longitudes on the icosahedral grid.
    """
    if not os.path.exists(edge_index_path) or not os.path.exists(mesh_nc_path):
        raise FileNotFoundError(f"Missing input files:\n  - {edge_index_path}\n  - {mesh_nc_path}")

    # 1. Load graph edges and coordinate positions
    print(f"[LOADING] Graph topology: {edge_index_path}")
    edge_index = torch.load(edge_index_path, map_location="cpu").numpy()

    print(f"[LOADING] Mesh geometry: {mesh_nc_path}")
    ds_mesh = xr.open_dataset(mesh_nc_path)
    lats = ds_mesh["latitude"].values
    lons = ds_mesh["longitude"].values  # Expected in [0, 360] range
    
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]

    # 2. Extract edge endpoint longitudes and latitudes
    src_lons, dst_lons = lons[src_nodes], lons[dst_nodes]
    src_lats, dst_lats = lats[src_nodes], lats[dst_nodes]

    # 3. Identify boundary-crossing edges
    # An edge crosses the 0/360 meridian if one node is near 360° and the other is near 0°
    lon_diff = np.abs(src_lons - dst_lons)
    crossing_mask = lon_diff > lon_threshold

    crossing_count = np.sum(crossing_mask)
    print(f"\n==================================================")
    print(f" Total Nodes             : {len(lats):,}")
    print(f" Total Directed Edges    : {edge_index.shape[1]:,}")
    print(f" Boundary-Crossing Edges : {crossing_count:,}")
    print(f"==================================================\n")

    if crossing_count == 0:
        print("❌ WARNING: No boundary-crossing edges detected! The 0/360 seam is unconnected.")
    else:
        print("✅ SUCCESS: Boundary connections found bridging 0° and 360°!")

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': None})

    # --- Plot 1: Global View (Highlighting Crossing Edges in Red) ---
    ax1.scatter(lons, lats, c="gray", s=8, alpha=0.5, label="Nodes")

    # Draw regular intra-domain edges (subsampled for readability)
    non_crossing_idx = np.where(~crossing_mask)[0]
    subsample = non_crossing_idx[::max(1, len(non_crossing_idx) // 3000)]
    for idx in subsample:
        ax1.plot([src_lons[idx], dst_lons[idx]], [src_lats[idx], dst_lats[idx]], 
                 c="lightgray", lw=0.5, alpha=0.4)

    # Draw all seam-crossing edges in bold red (wrapping longitudes for visual continuity)
    crossing_indices = np.where(crossing_mask)[0]
    for idx in crossing_indices:
        x0, x1 = src_lons[idx], dst_lons[idx]
        y0, y1 = src_lats[idx], dst_lats[idx]
        
        # Shift longitudes > 180 to negative range for seamless visual rendering on the plot
        x0_plot = x0 - 360 if x0 > 180 else x0
        x1_plot = x1 - 360 if x1 > 180 else x1
        ax1.plot([x0_plot, x1_plot], [y0, y1], c="crimson", lw=1.5, alpha=0.8)

    ax1.set_title(f"Global Node Topology ({crossing_count} Seam Edges in Red)")
    ax1.set_xlabel("Longitude (deg)")
    ax1.set_ylabel("Latitude (deg)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Plot 2: Polar Orthographic Projection View ---
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    
    # Convert lat/lon to polar radians for viewing top-down over North Pole
    lons_rad = np.radians(lons)
    r_coords = 90 - lats  # Co-latitude (0 at pole, 90 at equator)

    ax2.scatter(lons_rad, r_coords, c="steelblue", s=10, alpha=0.6)

    # Draw seam edges across the meridian on polar plot
    for idx in crossing_indices:
        r0, r1 = 90 - src_lats[idx], 90 - dst_lats[idx]
        t0, t1 = np.radians(src_lons[idx]), np.radians(dst_lons[idx])
        ax2.plot([t0, t1], [r0, r1], c="crimson", lw=1.5, alpha=0.9)

    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_title("North Polar Co-Latitude View (Seamless Wrap)")

    plt.tight_layout()
    output_png = "boundary_verification.png"
    plt.savefig(output_png, dpi=200)
    print(f"[EXPORT] Plot saved to: {output_png}\n")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Verify 0/360 boundary connectivity on icosahedral PyG edge_index tensors.")
    parser.add_argument("-e", "--edge_file", default="graph-grid/edge_index_m6.pt", help="Path to edge_index_m*.pt")
    parser.add_argument("-m", "--mesh_file", default="graph-grid/global_icosahedral_mesh_m6.nc", help="Path to global_icosahedral_mesh_m*.nc")
    args = parser.parse_args()

    verify_boundary_edges(args.edge_file, args.mesh_file)


if __name__ == "__main__":
    main()
