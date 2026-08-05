#!/usr/bin/env python3
"""
evaluate_aida_rollout.py
------------------------
Multi-step auto-regressive (AR) evaluation script for the AIDA GNN Surrogate Model.
Evaluates forecast stability and error accumulation up to 7 days (28 steps @ 6h steps).
Generates diagnostic plots showing RMSE, ACC, and Mean Bias progression over time.
"""

import argparse
import os
import sys

# Ensure parent directory is in Python path for 'models' package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import (
    IcosahedralGNNSurrogate,
    LogStateZarrDataset,
    generate_or_load_edge_index,
)


def compute_weighted_metrics(pred: torch.Tensor, target: torch.Tensor, lat_weights: torch.Tensor):
    """
    Computes spatial RMSE, MAE, Mean Bias, and ACC weighted by cosine-latitude.
    pred, target: Tensors of shape [N] or [L, N]
    lat_weights: Tensor of shape [N]
    """
    if pred.dim() == 2:  # Average across levels if L x N
        pred = pred.mean(dim=0)
        target = target.mean(dim=0)

    error = pred - target
    weight_sum = torch.sum(lat_weights)

    # Cosine-latitude weighted RMSE & MAE
    weighted_mse = torch.sum(lat_weights * (error ** 2)) / weight_sum
    rmse = torch.sqrt(weighted_mse).item()

    weighted_mae = torch.sum(lat_weights * torch.abs(error)) / weight_sum
    mae = weighted_mae.item()

    # Weighted Mean Bias
    weighted_bias = torch.sum(lat_weights * error) / weight_sum
    bias = weighted_bias.item()

    # Anomaly Correlation Coefficient (ACC)
    pred_anomaly = pred - (torch.sum(lat_weights * pred) / weight_sum)
    target_anomaly = target - (torch.sum(lat_weights * target) / weight_sum)

    num = torch.sum(lat_weights * pred_anomaly * target_anomaly)
    denom = torch.sqrt(
        torch.sum(lat_weights * (pred_anomaly ** 2)) * torch.sum(lat_weights * (target_anomaly ** 2))
    ) + 1e-8
    acc = (num / denom).item()

    return rmse, mae, bias, acc


def run_ar_rollout(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Operating on compute device: {device}", flush=True)

    # 1. Load Dataset
    if not os.path.exists(args.zarr):
        raise FileNotFoundError(f"Zarr dataset not found at '{args.zarr}'")

    dataset = LogStateZarrDataset(zarr_path=args.zarr)
    num_nodes = dataset.num_nodes
    num_vars = getattr(dataset, "num_vars", 7)
    total_timesteps = len(dataset)

    # Extract latitude weights for physical evaluation
    if hasattr(dataset, "latitudes"):
        lat_deg = torch.tensor(dataset.latitudes, dtype=torch.float32)
    else:
        lat_deg = torch.linspace(-90, 90, num_nodes)
    
    lat_weights = torch.cos(torch.deg2rad(lat_deg)).to(device)

    # 2. Load Checkpoint & Model
    print(f"[EVAL] Loading checkpoint: '{args.checkpoint}'", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = IcosahedralGNNSurrogate(
        in_vars=num_vars,
        hidden_dim=checkpoint["args"].get("hidden_dim", 64)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Graph Topology
    edge_index = generate_or_load_edge_index(num_nodes=num_nodes, edge_file=args.edges).to(device)

    var_names = ['t', 'u', 'v', 'w', 'q', 'rho', 'p']
    steps = args.rollout_steps  # 28 steps = 7 days @ 6h frequency
    
    # Storage for time-series metrics: shape [var_idx, step]
    rmse_history = {var: [] for var in var_names}
    acc_history = {var: [] for var in var_names}
    bias_history = {var: [] for var in var_names}

    print(f"[EVAL] Executing {steps}-step auto-regressive rollout across {args.num_samples} initial conditions...", flush=True)

    # Evaluate across multiple initial time seeds
    start_indices = np.linspace(0, total_timesteps - steps - 1, num=args.num_samples, dtype=int)

    for sample_idx, start_t in enumerate(start_indices):
        print(f"  [Sample {sample_idx+1}/{args.num_samples}] Initializing rollout at t={start_t}...", flush=True)
        
        # Initial condition: [1, V, L, N]
        curr_state = dataset[start_t][0].unsqueeze(0).to(device)

        sample_rmse = {var: [] for var in var_names}
        sample_acc = {var: [] for var in var_names}
        sample_bias = {var: [] for var in var_names}

        with torch.no_grad():
            for step in range(1, steps + 1):
                # Predict next step state recursively
                next_pred = model(curr_state, edge_index)

                # Ground truth target state
                target_state = dataset[start_t + step][1].unsqueeze(0).to(device)

                # Compute per-variable metrics
                for v_idx, var in enumerate(var_names):
                    p_v = next_pred[0, v_idx, :, :]
                    t_v = target_state[0, v_idx, :, :]

                    rmse, _, bias, acc = compute_weighted_metrics(p_v, t_v, lat_weights)
                    sample_rmse[var].append(rmse)
                    sample_acc[var].append(acc)
                    sample_bias[var].append(bias)

                # Auto-regressive feed: update input for step + 1
                curr_state = next_pred

        # Accumulate sample results
        for var in var_names:
            rmse_history[var].append(sample_rmse[var])
            acc_history[var].append(sample_acc[var])
            bias_history[var].append(sample_bias[var])

    # Average metrics over initial condition samples
    avg_rmse = {var: np.mean(rmse_history[var], axis=0) for var in var_names}
    avg_acc = {var: np.mean(acc_history[var], axis=0) for var in var_names}
    avg_bias = {var: np.mean(bias_history[var], axis=0) for var in var_names}

    # 3. Plot Diagnostics
    plot_ar_diagnostics(avg_rmse, avg_acc, avg_bias, steps, args.output_plot)


def plot_ar_diagnostics(rmse: dict, acc: dict, bias: dict, steps: int, save_path: str):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    time_hours = np.arange(1, steps + 1) * 6  # 6h time steps

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    plt.suptitle("AIDA GNN Surrogate Model: 7-Day Auto-Regressive Rollout Metrics", fontsize=16, fontweight="bold")

    plot_vars = [('t', 'Temperature (t)', 'K'), 
                 ('p', 'Surface Pressure (p)', 'hPa'),
                 ('u', 'Zonal Wind (u)', 'm/s'),
                 ('q', 'Specific Humidity (q)', 'kg/kg')]

    # Panel 1: RMSE Accumulation (T and P)
    ax = axes[0, 0]
    ax.plot(time_hours, rmse['t'], 'r-o', label='Temperature (t) RMSE', linewidth=2)
    ax.set_ylabel("RMSE (K)")
    ax.set_title("Temperature Error Accumulation")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    ax2 = axes[0, 1]
    ax2.plot(time_hours, rmse['p'], 'b-s', label='Pressure (p) RMSE', linewidth=2)
    ax2.set_ylabel("RMSE (hPa)")
    ax2.set_title("Surface Pressure Error Accumulation")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    # Panel 2: Pattern Skill Correlation (ACC)
    ax = axes[1, 0]
    ax.plot(time_hours, acc['t'], 'r-o', label='Temp ACC')
    ax.plot(time_hours, acc['p'], 'b-s', label='Pressure ACC')
    ax.plot(time_hours, acc['u'], 'g-^', label='Zonal Wind (u) ACC')
    ax.axhline(0.6, color='black', linestyle=':', label='Usable Skill Threshold (0.6)')
    ax.set_ylabel("ACC (Correlation)")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title("Anomaly Correlation Coefficient (ACC) Skill Decay")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left")

    ax = axes[1, 1]
    ax.plot(time_hours, acc['q'], 'm-d', label='Specific Humidity (q) ACC', linewidth=2)
    ax.axhline(0.6, color='black', linestyle=':', label='Threshold (0.6)')
    ax.set_ylabel("ACC")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title("Moisture (q) Skill Decay")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left")

    # Panel 3: Systematic Mean Bias Drift
    ax = axes[2, 0]
    ax.plot(time_hours, bias['t'], 'r-o', label='Temperature Mean Bias')
    ax.axhline(0.0, color='gray', linestyle='--')
    ax.set_xlabel("Forecast Lead Time (Hours)")
    ax.set_ylabel("Bias (K)")
    ax.set_title("Systematic Temperature Bias Drift")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")

    ax = axes[2, 1]
    ax.plot(time_hours, bias['p'], 'b-s', label='Pressure Mean Bias')
    ax.axhline(0.0, color='gray', linestyle='--')
    ax.set_xlabel("Forecast Lead Time (Hours)")
    ax.set_ylabel("Bias (hPa)")
    ax.set_title("Systematic Pressure Bias Drift")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    print(f"[EVAL] Successfully generated rollout diagnostic plot: '{save_path}'", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Multi-Step AR Rollout Evaluation for AIDA")
    parser.add_argument("--zarr", type=str, default="../data/icosahedral_2023_logstate.zarr", help="Path to evaluation Zarr dataset")
    parser.add_argument("--edges", type=str, default="../data/graph/icosahedral_edge_index_m4.pt", help="Path to precomputed edge tensor (.pt)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/aida_gnn_surrogate_logstate.pt", help="Checkpoint to evaluate")
    parser.add_argument("--rollout-steps", "--rollout_steps", dest="rollout_steps", type=int, default=28, help="Number of AR forecast steps (28 = 7 days @ 6h steps)")
    parser.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=5, help="Number of distinct initial condition seeds to average")
    parser.add_argument("--output-plot", "--output_plot", dest="output_plot", type=str, default="output/aida_ar_rollout_7day.png", help="Path for output diagnostic plot")

    args = parser.parse_args()
    run_ar_rollout(args)


if __name__ == "__main__":
    main()
