#!/usr/bin/env python3
"""
verification_metrics.py
------------------------
Modular metric calculation engine for atmospheric and earth system datasets.
Supports 2D (lat, lon), 3D (height, lat, lon), and 4D (time, height, lat, lon) datasets.
"""

import numpy as np
import xarray as xr


def compute_latitude_weights(lats: np.ndarray) -> np.ndarray:
    """Computes normalized cosine latitude weights for area-weighted spatial metrics."""
    rad_lats = np.radians(lats)
    weights = np.cos(rad_lats)
    return weights / np.mean(weights)


def calc_rmse(fcst: np.ndarray, ref: np.ndarray, weights: np.ndarray = None) -> float:
    """Root Mean Square Error (RMSE)."""
    sq_err = (fcst - ref) ** 2
    if weights is not None:
        # Apply spatial weighting along latitude axis (axis -2)
        sq_err = sq_err * weights[:, None]
    return float(np.sqrt(np.nanmean(sq_err)))


def calc_mae(fcst: np.ndarray, ref: np.ndarray, weights: np.ndarray = None) -> float:
    """Mean Absolute Error (MAE)."""
    abs_err = np.abs(fcst - ref)
    if weights is not None:
        abs_err = abs_err * weights[:, None]
    return float(np.nanmean(abs_err))


def calc_bias(fcst: np.ndarray, ref: np.ndarray, weights: np.ndarray = None) -> float:
    """Mean Bias Error (FCST - REF)."""
    bias = fcst - ref
    if weights is not None:
        bias = bias * weights[:, None]
    return float(np.nanmean(bias))


def calc_acc(fcst: np.ndarray, ref: np.ndarray, clim: np.ndarray = None, weights: np.ndarray = None) -> float:
    """
    Anomaly Correlation Coefficient (ACC).
    If climatology 'clim' is None, anomalies are calculated relative to the field spatial mean.
    """
    if clim is None:
        f_prime = fcst - np.nanmean(fcst)
        r_prime = ref - np.nanmean(ref)
    else:
        f_prime = fcst - clim
        r_prime = ref - clim

    if weights is not None:
        w = weights[:, None]
        covariance = np.nansum(w * f_prime * r_prime)
        variance_f = np.nansum(w * (f_prime ** 2))
        variance_r = np.nansum(w * (r_prime ** 2))
    else:
        covariance = np.nansum(f_prime * r_prime)
        variance_f = np.nansum(f_prime ** 2)
        variance_r = np.nansum(r_prime ** 2)

    denom = np.sqrt(variance_f * variance_r)
    if denom == 0:
        return 0.0

    return float(covariance / denom)


def calc_relative_diff(fcst: np.ndarray, ref: np.ndarray) -> float:
    """Relative Mean Error Percentage (%)."""
    denom = np.nanmean(np.abs(ref))
    if denom == 0:
        return 0.0
    return float((np.nanmean(np.abs(fcst - ref)) / denom) * 100.0)


def evaluate_variable(fcst_arr: xr.DataArray, ref_arr: xr.DataArray, lats: np.ndarray) -> dict:
    """
    Evaluates a single variable layer-by-layer or across the entire array.
    Returns a dictionary of overall and vertical level metrics.
    """
    weights = compute_latitude_weights(lats) if lats is not None else None

    fcst_vals = fcst_arr.values
    ref_vals = ref_arr.values

    # Overall field evaluation
    overall_results = {
        "RMSE": calc_rmse(fcst_vals, ref_vals, weights),
        "MAE": calc_mae(fcst_vals, ref_vals, weights),
        "BIAS": calc_bias(fcst_vals, ref_vals, weights),
        "ACC": calc_acc(fcst_vals, ref_vals, weights=weights),
        "RelDiff_pct": calc_relative_diff(fcst_vals, ref_vals),
    }

    # Vertical profile breakdown (if 3D or 4D with 'height' or 'level')
    level_results = []
    if "height" in fcst_arr.dims or "level" in fcst_arr.dims:
        height_dim = "height" if "height" in fcst_arr.dims else "level"
        num_levels = len(fcst_arr[height_dim])

        for h_idx in range(num_levels):
            f_slice = fcst_arr.isel({height_dim: h_idx}).values
            r_slice = ref_arr.isel({height_dim: h_idx}).values

            level_val = float(fcst_arr[height_dim].values[h_idx])
            level_results.append({
                "level": level_val,
                "RMSE": calc_rmse(f_slice, r_slice, weights),
                "MAE": calc_mae(f_slice, r_slice, weights),
                "BIAS": calc_bias(f_slice, r_slice, weights),
                "ACC": calc_acc(f_slice, r_slice, weights=weights),
            })

    return {"overall": overall_results, "levels": level_results}

