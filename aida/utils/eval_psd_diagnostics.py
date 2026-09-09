#!/usr/bin/env python3
"""
eval_psd_diagnostics.py
-----------------------
Loads a trained AIDA GNN model checkpoint, computes the pressure output Power
Spectral Density (PSD), and compares it against ground truth Zarr data.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from scipy.interpolate import NearestNDInterpolator

# Import architecture and loader from your training script
from train_aida_surrogate import (
    IcosahedralGNNSurrogate,
    LogStateZarrDataset,
    generate_or_load_edge_index,
)


def icosahedral_to_regular_grid(ico_data, ico_lats, ico_lons, nlat=180, nlon=360):
    """Interpolates unstructured icosahedral nodes to regular 1°x1° lat-lon grid."""
    grid_lats = np.linspace(-90, 90, nlat)
    grid_lons = np.linspace(0, 360, nlon, endpoint=False)
    lon_mesh, lat_mesh = np.meshgrid(grid_lons, grid_lats)

    ico_points = np.vstack([ico_lons, ico_lats]).T
    grid_points = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()]).T

    interpolator = NearestNDInterpolator(ico_points, ico_data)
    grid_data = interpolator(grid_points).reshape(nlat, nlon)

    return grid_data


def compute_1d_psd(grid_data):
    """Computes 1D spatial Power Spectral Density via 2D FFT."""
    nlat, nlon = grid_data.shape
    data_zero_mean = grid_data - np.mean(grid_data)
    fft2d = np.fft.fft2(data_zero_mean)
    fft_shifted = np.fft.fftshift(fft2d)
    power_2d = np.abs(fft_shifted) ** 2

    kx = np.fft.fftshift(np.fft.fftfreq(nlon)) * nlon
    ky = np.fft.fftshift(np.fft.fftfreq(nlat)) * nlat
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)

    k_radial = np.sqrt(kx_mesh**2 + ky_mesh**2).astype(int)
    max_k = min(nlat // 2, nlon // 2)
    wavenumbers = np.arange(1, max_k)
    psd_1d = np.zeros(len(wavenumbers))

    for i, k in enumerate(wavenumbers):
        mask = k_radial == k
        if np.any(mask):
            psd_1d[i] = np.mean(power_2d[mask])

    return wavenumbers, psd_1d


def run_psd_eval(zarr_path, checkpoint_path, edge_file, output_fig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PSD EVAL] Running evaluation on device: {device}")

    # 1. Load Dataset
    dataset = LogStateZarrDataset(zarr_path=zarr_path)
    x_test, y_test = dataset[0]  # Take first time-step sample
    x_batch = x_test.unsqueeze(0).to(device)
    y_batch = y_test.unsqueeze(0).to(device)

    # 2. Load Coordinates (Synthesize isotropic distribution if coordinate file isn't present)
    num_nodes = dataset.num_nodes
    np.random.seed(42)
    ico_lats = np.linspace(-90, 90, num_nodes)
    ico_lons = np.linspace(0, 360, num_nodes, endpoint=False)

    # 3. Load Checkpoint and Model
    edge_index = generate_or_load_edge_index(
        num_nodes=num_nodes, edge_file=edge_file
    ).to(device)
    model = IcosahedralGNNSurrogate(
        edge_index=edge_index,
        in_vars=len(dataset.var_names),
        levels=dataset.num_levels,
    ).to(device)

    if os.path.exists(checkpoint_path):
        print(f"[PSD EVAL] Loading weights from: {checkpoint_path}")
        ckpt = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(
            f"[WARNING] Checkpoint not found at '{checkpoint_path}'. Running with un-trained model."
        )

    model.eval()

    # 4. Generate Predictions for ln_p (Index 6)
    with torch.no_grad():
        y_pred = model(x_batch)

    p_idx = 6  # Index for ln_p
    # Extract surface layer (level 0)
    pred_p = y_pred[0, p_idx, 0, :].cpu().numpy()
    target_p = y_batch[0, p_idx, 0, :].cpu().numpy()

    # 5. Interpolate to regular grid
    print("[PSD EVAL] Regridding icosahedral data to regular 1°x1° grid...")
    grid_pred = icosahedral_to_regular_grid(pred_p, ico_lats, ico_lons)
    grid_target = icosahedral_to_regular_grid(target_p, ico_lats, ico_lons)

    # 6. Compute Spectral Power
    wavenumbers, psd_pred = compute_1d_psd(grid_pred)
    _, psd_target = compute_1d_psd(grid_target)

    # 7. Plot PSD Spectrum
    plt.figure(figsize=(9, 5.5), dpi=120)
    plt.loglog(
        wavenumbers,
        psd_target,
        label="Target (ERA5 Log-Pressure)",
        color="black",
        linewidth=2.0,
    )
    plt.loglog(
        wavenumbers,
        psd_pred,
        label="GNN Output (Model Prediction)",
        color="#0072B2",
        linewidth=1.8,
        linestyle="--",
    )

    plt.axvspan(
        1, 15, color="green", alpha=0.08, label="Synoptic Scale (Real Weather)"
    )
    plt.axvspan(
        30,
        max(wavenumbers),
        color="red",
        alpha=0.08,
        label="Noise Zone (Checkerboard Area)",
    )

    plt.xlabel("Spherical Wavenumber ($k$)", fontsize=11, fontweight="bold")
    plt.ylabel("Power Spectral Density ($P(k)$)", fontsize=11, fontweight="bold")
    plt.title(
        "AIDA Pressure Power Spectrum Diagnostic",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(loc="lower left")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_fig) or ".", exist_ok=True)
    plt.savefig(output_fig)
    plt.show()
    print(f"[PSD EVAL] Diagnostic figure successfully written to: {output_fig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-z",
        "--zarr",
        default="../data/icosahedral_2023_logstate.zarr",
        help="Path to Zarr store",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoints/aida_gnn_surrogate_logstate.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "-g",
        "--edges",
        default="../data/graph/icosahedral_edge_index_m4.pt",
        help="Path to graph edge topology",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="diagnostics/psd_pressure_spectrum.png",
        help="Output PNG file path",
    )
    args = parser.parse_args()

    run_psd_eval(args.zarr, args.checkpoint, args.edges, args.output)


if __name__ == "__main__":
    main()
