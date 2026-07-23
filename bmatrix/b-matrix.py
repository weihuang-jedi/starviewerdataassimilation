import numpy as np
import scipy.optimize as opt
from scipy.spatial.distance import cdist

def soar_correlation(r, L_h):
    """Second-Order Autoregressive (SOAR) correlation function."""
    return (1.0 + r / L_h) * np.exp(-r / L_h)

def estimate_nmc_length_scale(forecast_diffs, coords_2d, max_dist_km=1000.0, bin_size_km=50.0):
    """
    Estimates horizontal length scale L_h using the NMC method on forecast differences.
    
    Parameters:
    -----------
    forecast_diffs : np.ndarray
        Array of shape (n_samples, n_spatial_points) containing (T48 - T24) differences.
    coords_2d : np.ndarray
        Array of shape (n_spatial_points, 2) with [latitude, longitude] in degrees.
        
    Returns:
    --------
    L_h_est : float
        Estimated horizontal length scale in kilometers.
    """
    n_samples, n_pts = forecast_diffs.shape
    
    # 1. Compute spatial pairwise distance matrix (in km)
    # Approx 1 deg lat ~ 111 km
    dists = cdist(coords_2d, coords_2d) * 111.0  
    
    # 2. Compute sample correlation matrix across grid points
    cov = np.cov(forecast_diffs, rowvar=False)
    stds = np.sqrt(np.diag(cov))
    stds[stds == 0] = 1.0  # Avoid zero division
    corr = cov / np.outer(stds, stds)
    
    # 3. Bin correlations by pairwise distance
    bins = np.arange(0.0, max_dist_km, bin_size_km)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_corrs = []
    
    for i in range(len(bins) - 1):
        mask = (dists >= bins[i]) & (dists < bins[i+1])
        if np.any(mask):
            mean_corrs.append(np.mean(corr[mask]))
        else:
            mean_corrs.append(np.nan)
            
    bin_centers = np.array(bin_centers)
    mean_corrs = np.array(mean_corrs)
    
    # Filter out NaNs for curve fitting
    valid = ~np.isnan(mean_corrs)
    
    # 4. Fit SOAR curve to find optimal L_h
    popt, _ = opt.curve_fit(soar_correlation, bin_centers[valid], mean_corrs[valid], p0=[200.0])
    L_h_est = popt[0]
    
    return L_h_est

# ------------------------------------------------------------------------------
# Example Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 60  # 60 sample pairs (e.g., 2 months of daily 00Z forecasts)
    n_lats, n_lons = 10, 10
    n_pts = n_lats * n_lons
    
    # Create 2D coordinates
    lats = np.linspace(30, 45, n_lats)
    lons = np.linspace(-100, -85, n_lons)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    coords_2d = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
    
    # Generate synthetic forecast differences (T48 - T24) with true L_h = 250 km
    true_L_h = 250.0
    dists = cdist(coords_2d, coords_2d) * 111.0
    true_B_H = (1.5**2) * soar_correlation(dists, true_L_h) + 1e-4 * np.eye(n_pts)
    L_true = np.linalg.cholesky(true_B_H)
    
    # Simulated forecast difference ensemble
    forecast_diffs = (L_true @ np.random.randn(n_pts, n_samples)).T
    
    # Estimate L_h using NMC function
    L_h_estimated = estimate_nmc_length_scale(forecast_diffs, coords_2d)
    
    # Scale factor (alpha) to account for error growth over 24h forecast window
    alpha = 0.2  # Tuning parameter calibrated against innovation statistics
    variance_profile = alpha * np.var(forecast_diffs, axis=0)
    
    print(f"True Length Scale L_h:      {true_L_h:.2f} km")
    print(f"NMC Estimated L_h:          {L_h_estimated:.2f} km")
    print(f"Mean Background Variance:  {np.mean(variance_profile):.4f} K^2")
