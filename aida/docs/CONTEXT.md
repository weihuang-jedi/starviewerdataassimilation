# AIDA Atmospheric GNN Model & GraphCast AI-DA Framework
**Context, High-Resolution Migration & Debugging Summary — August 2026**

---

## 1. Project Overview & M6 Grid Migration
The **AIDA (AI Data Assimilation)** and **GraphCast Surrogate** framework is a 3D Graph Neural Network designed for high-resolution atmospheric forecasting and observation assimilation. The framework operates on an **unstructured 3D terrain-following icosahedral grid** using **3D geometric height coordinates ($z$ in meters)** and **4-channel static surface conditioning**.

### Key Dataset & Grid Resolution Transition
* **Legacy Grid (M4)**: 2,562 horizontal spatial nodes across 32 vertical levels ($N_{\text{flat}} = 81,984$).
* **Updated Grid (M6)**: **40,962 horizontal spatial nodes** across 32 vertical levels ($N_{\text{flat}} = 1,310,784$).
  * Provides $\approx 0.25^\circ$ global resolution matching operational GFS data scales.
  * Zarr storage path: `/scratch5/purged/Wei.Huang/src/starviewerweathermodel/data/icosahedral_logstate.zarr`

### M6 Vertical Terrain-Following $\eta$ Coordinate Mapping
Terrain-following 3D heights are governed by the hybrid formula:
$$h(k, \text{node}) = H_{\text{max}} - \eta(k) \cdot \left(H_{\text{max}} - H_{\text{terrain}}(\text{node})\right) \quad \text{where } H_{\text{max}} = 20,000\text{ m}$$

* **Level 1 (`L01`, $k=0$):** $\eta = 0.9999 \implies z \approx 2\text{ m}$ (**Surface Layer**)
* **Level 16 (`L16`, $k=15$):** $\eta = 0.9000 \implies z \approx 2,000\text{ m}$ (**Top of PBL**)
* **Level 32 (`L32`, $k=31$):** $\eta = 0.0000 \implies z = 20,000\text{ m}$ (**Top of Model / TOA**)

### Primary Atmospheric State Vector (7 Variables)
$$\mathbf{x} = \left[ \ln T,\, u,\, v,\, w,\, q,\, \ln \rho,\, \ln p \right]^T$$

---

## 2. Model Feature Architecture (Input Channels = 14 + 4 Static)

1. **14 Dynamic Trajectory Channels**: 2 history timesteps ($t_{-1}, t_0$) $\times$ 7 dynamic variables ($\ln T, u, v, w, q, \ln \rho, \ln p$).
2. **4 Static Conditioning Feature Channels**:
   * Channel 0: Normalized Surface Topography Elevation ($h_{\text{terrain}} / 10,000\text{ m}$).
   * Channel 1: Land-Sea Binary Mask ($1 = \text{Land}, 0 = \text{Ocean}$).
   * Channel 2: Normalized Surface Roughness Length ($\ln z_0 / 5.0$, where $z_{0, \text{land}} = 0.1\text{ m}, z_{0, \text{ocean}} = 0.0002\text{ m}$).
   * Channel 3: Real-Time Cosine Solar Zenith Angle ($\cos \text{SZA} \in [0.0, 1.0]$) computed dynamically per timestep to eliminate hemispheric warming bias.

---

## 3. Planetary Boundary Layer (PBL) Loss Engine Upgrades (`models/loss.py`)

To eliminate low-level ($z \le 2,000\text{ m}$) skill degradation (high RMSE and lower ACC at $L01$--$L15$), the loss engine enforces five physical terms:

1. **PBL Exponential Height Weighting ($W_{\text{pbl}}$)**:
   $$W_{\text{pbl}}(z) = 1.0 + 3.0 \cdot \exp\left(-\frac{z}{1500\text{ m}}\right)$$
   Boosts surface loss gradients by up to $4\times$ to prevent upper-level wind energy from swamping near-surface training.
2. **Monin-Obukhov Surface Drag Penalty ($\lambda_{\text{drag}} = 0.05$)**:
   Penalizes excessive low-level wind speed overshooting ($U, V$) over land vs. ocean using surface drag coefficients ($C_d$).
3. **PBL Thermal Lapse Rate Guard ($\lambda_{\text{lapse}} = 0.02$)**:
   Restricts $\frac{\partial T}{\partial z}$ between Level 1 ($2\text{ m}$) and Level 16 ($2,000\text{ m}$) to prevent unphysical super-adiabatic inversions or runaway surface heating.
4. **Physical Moisture Barrier ($\lambda_{\text{asym\_q}} = 0.5$)**:
   Enforces heavy quadratic penalties on un-normalized physical specific humidity ($Q_{\text{phys}} < 0.0\text{ kg/kg}$).
5. **Global Thermal Drift Balance ($\lambda_{\text{thermal}} = 0.1$)**:
   Penalizes global mean temperature drift ($\Delta \bar{T}$) relative to ground truth.

---

## 4. Execution Commands

### Stream NetCDF to Compressed Zarr (M6 Grid)
```bash
python pack_icosahedral_zarr.py \
  -i "icosahedral-grid/icosahedral_logstate_m6.202*.nc" \
  -o icosahedral_logstate.zarr \
  -c 4 -l 3
