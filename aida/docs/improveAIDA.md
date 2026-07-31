Your plot diagnostic gives us a huge clue about **where** the failure is happening:

Look closely at the variable breakdown:

* **`u`, `v`, `q**`: 0% NaNs (Valid fields coming from input/background)
* **`w`**: 7.6% NaNs (Partial failure)
* **`ln_t`, `ln_p`, `ln_rho**`: **100% NaNs**

Because `u` and `v` are completely valid while the log-thermodynamic variables are 100% NaN, **the model training weights or initial input data pipeline is fine for horizontal winds, but dying during log-transformations or density derivations.**

Here is the step-by-step diagnostic roadmap to locate and fix the exact point of failure:

---

## 1. Check `convert_netcdf_to_logstate.py` (Log Transformation & $\rho$ Derivation)

The log-state transformation calculates:


$$\text{ln\_t} = \log(T)$$

$$\text{ln\_p} = \log(p)$$

$$\text{ln\_rho} = \log(\rho) = \log\left(\frac{p}{R_d \cdot T}\right)$$

### Where it fails:

If temperature $T \le 0$ (e.g. uninitialized levels or missing pressure levels) or pressure $p \le 0$ (e.g. top of atmosphere levels or zero-padding), **`np.log()` returns `NaN` or `-inf**`.

* **Action**: Check if $p$ or $T$ in the raw NetCDF have zero or negative values before taking the log.
* **Fix**: Ensure standard unit conversion (Kelvin for $T$, Pascals for $p$) and apply a tiny epsilon threshold before applying $\log$:

```python
# In convert_netcdf_to_logstate.py or log-transform step:
eps = 1e-6
p_safe = np.maximum(p_pa, eps)
T_safe = np.maximum(t_kelvin, 180.0) # Atmosphere floor in Kelvin

rho = p_safe / (287.058 * T_safe)
rho_safe = np.maximum(rho, eps)

ln_p = np.log(p_safe)
ln_t = np.log(T_safe)
ln_rho = np.log(rho_safe)

```

---

## 2. Check Input Feature Normalization/Scaling in Model

In `IcosahedralGNNSurrogate`, if features are standardized using mean and standard deviation:


$$x_{norm} = \frac{x - \mu}{\sigma}$$

If `std` ($\sigma$) for any level/channel was computed as `0.0` or `NaN` (due to zero-padded unpopulated levels), every single state variable passed to the GNN immediately becomes `NaN` during the forward pass!

* **Action**: Print `mean` and `std` vectors used by the normalization layer.
* **Fix**: Replace any `0.0` std values with `1.0` or `1e-6`:

```python
# In model feature scaler/normalizer:
std = torch.where(std == 0.0, torch.tensor(1.0), std)
std = torch.nan_to_num(std, nan=1.0)

```

---

## 3. Check `NonHydrostaticIcosahedralLoss` Residuals

Your custom loss function evaluates the Ideal Gas Law residual:


$$\text{Residual} = \exp(\text{ln\_p}) - \exp(\text{ln\_rho}) \cdot R_d \cdot \exp(\text{ln\_t})$$

### Where it fails during training:

During early training iterations, if model log-state predictions output large deltas (e.g., $\text{ln\_p} > 80$), `torch.exp(80)` causes **numerical overflow (`Inf`)**. When subtracted inside the loss function, `Inf - Inf` produces **`NaN` gradients**, permanently corrupting model weights via backpropagation!

* **Action**: Check if loss function output became `NaN` during training.
* **Fix**: Clamp predicted log-states or use log-residual space directly:

```python
# Inside NonHydrostaticIcosahedralLoss forward pass:
# Clamp log predictions to physically realistic ranges before exponentiation
ln_p_pred = torch.clamp(ln_p_pred, min=-5.0, max=13.0)   # ~0.006 Pa to 440 kPa
ln_t_pred = torch.clamp(ln_t_pred, min=5.0, max=6.0)     # ~148 K to 403 K
ln_rho_pred = torch.clamp(ln_rho_pred, min=-15.0, max=2.0)

# Ideal Gas Law in pure log space avoids exp() overflow:
# ln(p) = ln(rho) + ln(R_d) + ln(T)
R_d_log = np.log(287.058)
gas_residual = ln_p_pred - (ln_rho_pred + R_d_log + ln_t_pred)
loss_thermo = torch.mean(gas_residual ** 2)

```

---

## 4. Check Cycling Predictor (`run_aida_cycling.py`)

In `run_aida_cycling.py`, check how the analysis state is updated:


$$\text{Analysis} = \text{Background} + \Delta_{\text{GNN}}$$

If the model outputs un-normalized deltas ($\Delta_{\text{GNN}}$), but the background state is in real units (or vice versa), adding them blows up the state.

* **Diagnostic Script**: Run this snippet on your GNN checkpoint to verify whether model weights themselves contain `NaN`s:

```python
import torch

checkpoint = torch.load("path/to/your/gnn_model.pt", map_location="cpu")
has_nan_weights = False

for name, param in checkpoint.items():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"[WEIGHT ERROR] Parameter {name} contains NaNs or Infs!")
        has_nan_weights = True

if not has_nan_weights:
    print("[WEIGHT OK] All model weights are finite and clean.")

```

---

## Summary Strategy

1. **Run the weight check above.**
* If weights have NaNs $\rightarrow$ Fix loss function `exp()` overflow in `NonHydrostaticIcosahedralLoss` (Point 3) and retrain.
* If weights are clean $\rightarrow$ Fix log-transformation in `convert_netcdf_to_logstate.py` (Point 1) and scaling/normalization tensors (Point 2).
