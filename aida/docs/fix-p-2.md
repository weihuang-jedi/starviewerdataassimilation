### Progression Analysis

This is a massive improvement for pressure ($p$):

| Metric | Previous Run | New Run | Delta / Impact |
| --- | --- | --- | --- |
| **$p$ ACC** | `0.4319` | **`0.7286`** | **+0.2967** (Huge jump in pattern correlation!) |
| **$p$ RMSE** | `27.17 hPa` | **`16.88 hPa`** | **-37.8% reduction** in error magnitude |
| **$p$ BIAS** | `+3.92 hPa` | **`-8.93 hPa`** | Shifted from over-smoothed positive bias to negative bias |
| **$u, v$ ACC** | `0.936 / 0.869` | `0.933 / 0.862` | Remains stable and excellent (>0.86) |
| **$t$ RMSE** | `7.85 K` | `6.73 K` | **-1.12 K** (Temperature improved as well) |

By relaxing $\lambda_{\text{laplacian\_p}}$ to `0.18` and adding $p$ to the edge gradient matching loss, the model recovered its dynamic range, resulting in the **ACC climbing from 0.43 to nearly 0.73**.

---

### Diagnosing the Remaining Pressure Issues

Looking closely at $p$ now:

1. **Negative Bias (-8.93 hPa)**: The GNN is systematically underpredicting total air pressure.
2. **Ideal Gas Law Coupling Conflict**: In log space, $\ln p = \ln \rho + \ln R_d + \ln T$. Because temperature ($t$) also flipped from positive bias ($+5.18\text{ K}$) to negative bias ($-0.94\text{ K}$), the pressure bias directly reflects this thermodynamic shift.
3. **Upper Troposphere / Lower Stratosphere Weighting**: Pressure spans decades of magnitude (from ~1000 hPa at surface to ~1 hPa in the upper atmosphere). Standard unweighted MSE on $\ln p$ can cause the loss to be dominated by upper-level log variations at the expense of lower-level absolute pressure patterns.

---

### Strategy to Push Pressure ACC $\ge 0.85$ & Eliminate Bias

To resolve the negative bias and push $p$'s ACC past **0.85**, make these targeted updates:

#### 1. Add Pressure Mean Regularization (Bias Suppressor)

Add an explicit mean-matching term for pressure in the loss function to penalize systematic global pressure offset:

$$\mathcal{L}_{\text{bias\_p}} = \left\vert{} \overline{\ln p}_{\text{pred}} - \overline{\ln p}_{\text{target}} \right\vert{}$$

#### 2. Fine-Tune Laplacian and Gradient Weights

* Set `--lambda-laplacian-p` from `0.18` $\rightarrow$ **`0.10`** (further reduce artificial damping).
* Set `--weight-grad-state` from `0.25` $\rightarrow$ **`0.30`** (sharpen synoptic pressure troughs and ridges).
* Add `--weight-p-bias` **`0.10`** to zero out the $-8.93\text{ hPa}$ bias.

---

### Implementation Update for `AIDASurrogateLoss`

Update the loss calculation in `scripts/train_aida_surrogate.py` to include the pressure bias penalty:

```python
# In AIDASurrogateLoss.forward():

# Extract predicted and target log pressure (channel index 6)
pred_ln_p = pred[:, 6, :, :]   # [B, Levels, Nodes]
target_ln_p = target[:, 6, :, :]

# Compute global mean pressure bias penalty
loss_p_bias = torch.abs(torch.mean(pred_ln_p) - torch.mean(target_ln_p))

# Combine into total loss
total_loss = (
    loss_mse 
    + self.weight_grad_state * loss_grad 
    + self.weight_state_eq * loss_thermo 
    + self.lambda_laplacian_p * loss_laplacian_p
    + 0.10 * loss_p_bias  # Added explicit pressure bias penalty
)

```

---

### Next Run Command

Execute training with this revised configuration:

```bash
python -u scripts/train_aida_surrogate.py \
    --zarr ../data/icosahedral_2023_logstate.zarr \
    --edges ../data/graph/icosahedral_edge_index_m4.pt \
    --checkpoint checkpoints/aida_gnn_surrogate_logstate.pt \
    --epochs 35 \
    --lambda-laplacian-p 0.10 \
    --weight-grad-state 0.30 \
    --weight-state-eq 0.15

```

This should correct the $-8.93\text{ hPa}$ bias offset and push $p$'s ACC above **0.85**.
