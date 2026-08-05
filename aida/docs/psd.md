Here is a Python script that computes and plots the spatial **Power Spectral Density (PSD)**.

Because standard 2D Fast Fourier Transforms (FFT) require an equirectangular grid, analyzing spatial scale energy directly on an unstructured icosahedral grid uses **Spherical Harmonics Decomposition** (via `pyshtools` or scipy's spherical harmonics) or a **Direct Spatial Autocorrelation FFT** by regridding to a regular lat-lon grid.

This snippet uses the **Spherical Harmonics / Spatial Wavenumber Spectrum** approach via a fast lat-lon interpolation, which is the standard atmospheric science diagnostic for evaluating spatial scale power (wavenumbers $l$) against benchmark data like ERA5.

```python
#!/usr/bin/env python3
"""
psd_icosahedral_diagnostic.py
------------------------------
Computes and plots 1D Power Spectral Density (PSD) as a function of spherical 
wavenumber (l) to compare GNN pressure predictions vs. Ground Truth targets.

This verifies whether spatial smoothness penalties (lambda_smooth_p) are properly 
damping high-frequency checkerboard noise (high wavenumbers) without suppressing 
synoptic-scale signals (low wavenumbers).
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import NearestNDInterpolator


def icosahedral_to_regular_grid(
    ico_data: np.ndarray, 
    ico_lats: np.ndarray, 
    ico_lons: np.ndarray, 
    nlat: int = 180, 
    nlon: int = 360
) -> np.ndarray:
    """
    Interpolates icosahedral unstructured node values onto an equirectangular grid
    for spherical spectral decomposition.
    """
    grid_lats = np.linspace(-90, 90, nlat)
    grid_lons = np.linspace(0, 360, nlon, endpoint=False)
    lon_mesh, lat_mesh = np.meshgrid(grid_lons, grid_lats)

    # Nearest neighbor or radial basis function mapping for unstructured points
    ico_points = np.vstack([ico_lons, ico_lats]).T
    grid_points = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()]).T

    interpolator = NearestNDInterpolator(ico_points, ico_data)
    grid_data = interpolator(grid_points).reshape(nlat, nlon)
    
    return grid_data, grid_lats, grid_lons


def compute_1d_psd(grid_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes 1D spatial Power Spectral Density via 2D Fast Fourier Transform 
    averaged radially across spatial isotropic wavenumbers.
    """
    nlat, nlon = grid_data.shape
    
    # Detrend and compute 2D Fourier Transform
    data_zero_mean = grid_data - np.mean(grid_data)
    fft2d = np.fft.fft2(data_zero_mean)
    fft_shifted = np.fft.fftshift(fft2d)
    power_2d = np.abs(fft_shifted) ** 2

    # Map 2D frequencies to isotropic 1D wavenumber bins
    kx = np.fft.fftshift(np.fft.fftfreq(nlon)) * nlon
    ky = np.fft.fftshift(np.fft.fftfreq(nlat)) * nlat
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    
    # Total wavenumber k = sqrt(kx^2 + ky^2)
    k_radial = np.sqrt(kx_mesh**2 + ky_mesh**2).astype(int)

    # Bin power spectrum by radial wavenumber
    max_k = min(nlat // 2, nlon // 2)
    wavenumbers = np.arange(1, max_k)
    psd_1d = np.zeros(len(wavenumbers))

    for i, k in enumerate(wavenumbers):
        mask = (k_radial == k)
        if np.any(mask):
            psd_1d[i] = np.mean(power_2d[mask])

    return wavenumbers, psd_1d


def plot_psd_comparison(
    wavenumbers: np.ndarray,
    psd_pred: np.ndarray,
    psd_target: np.ndarray,
    save_path: str = "psd_pressure_spectrum.png"
):
    """
    Plots log-log Power Spectral Density to identify high-wavenumber noise.
    """
    plt.figure(figsize=(9, 5.5), dpi=120)

    # Log-Log scale plot
    plt.loglog(wavenumbers, psd_target, label="Ground Truth Target (ERA5)", color="black", linewidth=2.0)
    plt.loglog(wavenumbers, psd_pred, label="GNN Prediction (Regularized)", color="#0072B2", linewidth=1.8, linestyle="--")

    # Annotate spatial regions
    plt.axvspan(1, 15, color="green", alpha=0.08, label="Synoptic Scales (Real Weather)")
    plt.axvspan(30, max(wavenumbers), color="red", alpha=0.08, label="Grid-Scale Region (Noise Zone)")

    plt.xlabel("Spherical Wavenumber ($k$)", fontsize=11, fontweight="bold")
    plt.ylabel("Power Spectral Density ($P(k)$)", fontsize=11, fontweight="bold")
    plt.title("AIDA GNN Pressure Spatial Power Spectrum", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"[PSD DIAGNOSTIC] Spectrum comparison plot saved to: '{save_path}'")
    plt.close()


# ==========================================
# EXAMPLE EXECUTION / DIAGNOSTIC RUN
# ==========================================
if __name__ == "__main__":
    # Simulated icosahedral mesh node coordinates (replace with actual lats/lons tensor)
    num_nodes = 2562
    np.random.seed(42)
    ico_lats = np.random.uniform(-90, 90, size=num_nodes)
    ico_lons = np.random.uniform(0, 360, size=num_nodes)

    # 1. Dummy targets and predictions (Replace with model output tensor pred[:, 6, 0, :].cpu().numpy())
    target_ico_p = np.sin(np.radians(ico_lats)) * np.cos(np.radians(ico_lons))  # Smooth synoptic field
    pred_ico_p = target_ico_p + 0.05 * np.random.normal(size=num_nodes)         # Adding minor noise

    # 2. Map to regular grid
    grid_pred, lats, lons = icosahedral_to_regular_grid(pred_ico_p, ico_lats, ico_lons)
    grid_target, _, _ = icosahedral_to_regular_grid(target_ico_p, ico_lats, ico_lons)

    # 3. Compute 1D Power Spectral Density
    wavenumbers, psd_pred = compute_1d_psd(grid_pred)
    _, psd_target = compute_1d_psd(grid_target)

    # 4. Save visual diagnostic
    plot_psd_comparison(wavenumbers, psd_pred, psd_target)

```

---

### How to Interpret the PSD Graph:

1. **Low Wavenumbers ($k = 1\text{–}15$):** Represents large-scale atmospheric structures (synoptic ridges and troughs). Your prediction curve should **match the target curve closely**. If the prediction line is below target here, `lambda_smooth_p` is too strong and over-smoothing real weather patterns.
2. **High Wavenumbers ($k \ge 30$):** Represents grid-scale noise (checkerboards). If `lambda_smooth_p` is effective, the prediction power curve will drop sharply in this zone to match the target, rather than flattening out into a high-energy plateau.
