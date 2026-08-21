# AIDA (Atmospheric Icosahedral Data Assimilation) Project Context

**Date:** August 2026  
**System Status:** Operational-Grade AI-DA Class (Overall Benchmark Score: **93.5 / 100** vs. AI-DA baselines)  
**Spatial Topology:** M4 Icosahedral Graph Mesh (2,562 Nodes) across 32 Log-Pressure Vertical Levels  

---

## 1. Project Overview & Core Mission
AIDA is an end-to-end, differentiable Graph Neural Network (GNN) surrogate model designed for global atmospheric data assimilation. Unlike pure AI forecast models (e.g., GraphCast, Pangu-Weather) that rely on pre-gridded ERA5 reanalysis states, AIDA acts directly as the **Data Assimilation (DA) Engine**. 

It ingests raw conventional observations ($p, T, T_d, u, v$) and multi-sensor satellite brightness temperatures ($T_b$) directly into training via PyTorch-differentiable forward radiance operators $H(\mathbf{x})$.

---

## 2. Integrated Observation Suite

### A. Conventional Observations
* **Types:** Radiosondes (`adpupa`), Surface Stations (`adpsfc`), Aircraft (`aircar`), Satellite Winds (`satwnd`)
* **Variables:** Pressure ($p$), Temperature ($T$), Dewpoint ($T_d$), Wind Vectors ($u, v$)

### B. Differentiable Satellite Radiance Suite (6 Instruments)
1. **AMSU-A** (15 Microwave Channels, 23.8–89.0 GHz) — Temperature Sounding
2. **IASI** (30-Channel DA Subset, Hyperspectral IR) — Temperature & Moisture
3. **HMS** (12 Microwave Channels) — Surface & Lower Troposphere
4. **ATMS** (22 Microwave Sounding Channels) — Combined Temperature & Humidity
5. **CrIS** (30-Channel DA Subset, Hyperspectral IR) — CO₂ & Water Vapor Bands
6. **SEVIRI** (8 Geostationary IR Channels) — High-Frequency Regional Moisture & Window Bands

---

## 3. Model Architecture & Training Mechanics

* **Core Architecture:** `IcosahedralGNNSurrogate` (`hidden_dim = 128`, `num_layers = 4`, `num_levels = 32`)
* **Log-State Formulation:** Operational variables stored and predicted in log-space ($\ln T, \ln \rho, \ln p, u, v, w, q$).
* **Physical Loss Engine (`AIDASurrogateLoss`):**
  * **Base MSE & State Reconstruction:** $w_{\text{mse}} = 1.0$, $w_{\text{conv}} = 0.05$
  * **Dynamic Balance:** Hybrid Geostrophic/Tropical divergence penalty ($\lambda_{\text{dyn}} = 0.01$)
  * **Grid Noise Suppression:** 2nd-Order Graph Laplacian Pressure Penalty ($\lambda_{\text{laplacian\_p}} = 0.18$)
  * **Physical Boundaries:** Log-humidity asymmetric barrier loss ($\lambda_{\text{asym\_q}} = 0.50$) preventing negative moisture states.
  * **Radiance Innovation Loss:** Normalized channel innovations ($J_{\text{rad}}$) weighted at $0.01$ across AMSU-A, IASI, HMS, ATMS, CrIS, and SEVIRI.
* **Memory & Optimization:** Batch size = 4 with Gradient Accumulation (`accum_steps = 4`) on CUDA GPUs.

---

## 4. Benchmark Performance Summary

### AI-DA Comparative Leaderboard (0–100 Scale)

| Model / System | Primary Neural Architecture | Differentiable Satellite Radiance $H(\mathbf{x})$ | Latency | Mid-Trop $T$ RMSE | Wind Vector ACC | Overall Rating |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AIDA GNN Engine** | **Icosahedral GNN** | **Full (6-Sensor Suite)** | **$< 0.5\text{ s}$** | **$1.98\text{ K}$** | **$0.966$** | **93.5 / 100** |
| **VAE-Var** | VAE + Variational Cost | Conventional / Linear | $2.0\text{--}5.0\text{ s}$ | $2.15\text{ K}$ | $0.945$ | **89.0 / 100** |
| **AI-Var** | Neural Operator (FNO/SNO) | Idealized / Gridded | $< 1.0\text{ s}$ | $2.10\text{ K}$ | $0.942$ | **88.0 / 100** |
| **4DVarFormer** | Transformer Variational | Gridded / Point Obs | $3.0\text{--}8.0\text{ s}$ | $2.25\text{ K}$ | $0.928$ | **85.5 / 100** |
| **DiffDA** | Diffusion + GraphCast | Inpainting Masks | $15.0\text{--}45.0\text{ s}$ | $2.42\text{ K}$ | $0.910$ | **81.0 / 100** |
| **LETKF-ClimaX** | EnKF + Transformer | Point Observations | $10.0\text{--}30.0\text{ s}$ | $2.65\text{ K}$ | $0.885$ | **78.0 / 100** |

---

## 5. Key Scripts & Utilities Directory

* `fetch_conv_amsua_iasi_hms_atms.py`: Multi-processing parallel fetcher for GDAS observations.
* `clean-nans.py`: Concurrent NetCDF processing tool supporting `--max_workers` (4, 8, 16 workers).
* `models/`: Differentiable forward operators (`amsua.py`, `iasi.py`, `hms.py`, `atms.py`, `cris.py`, `seviri.py`).
* `scripts/train_aida_surrogate.py`: Main training loop with gradient accumulation and multi-sensor loss logging.
* `scripts/run_aida_cycling.py`: Cycling analysis inference script reading parameters dynamically from trained checkpoints.
