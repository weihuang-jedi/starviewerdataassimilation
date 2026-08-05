That is a **huge win** on the wind vectors—an Anomaly Correlation Coefficient (ACC) of **0.935 for $u$** and **0.868 for $v$** means the GNN has genuinely captured the synoptic atmospheric dynamics!

However, looking at the pressure field ($p$) and specific humidity ($q$), we have a classic surrogate modeling trade-off:

### Diagnostic Breakdown

1. **Pressure ($p$) ACC @ 0.432 | RMSE ~27 hPa**:
Increasing $\lambda_{\text{laplacian\_p}}$ to `0.30` successfully suppressed local spatial checkerboards, but it introduced **excessive spatial damping**. The GNN is now over-smoothing synoptic high/low pressure systems and introducing a positive bias (+3.92 hPa), dragging down the pattern correlation.
2. **Specific Humidity ($q$) RelDiff @ 17,785% | ACC @ 0.151**:
$q$ varies across orders of magnitude from the surface to the stratosphere. Small absolute errors at upper levels trigger massive percentage relative differences. $q$ needs a scale-aware or log-space loss treatment.

---

### Strategy to Fix Pressure ($p$) Without Bringing Back Checkerboards

To boost $p$'s ACC back above **0.85+** and lower the RMSE, we need to balance **sharpness** and **smoothness**:

1. **Slightly Relax $\lambda_{\text{laplacian\_p}}$ (0.30 $\rightarrow$ 0.18)**:
0.30 was a bit too aggressive, dampening valid pressure gradients. 0.18 keeps high-frequency noise in check while restoring dynamic contrast.
2. **Add $L_1$ Edge Gradient Matching for $p$**:
Right now, $p$ is excluded from the edge gradient loss (`sharp_var_indices = [0, 1, 2, 4]`). Adding $p$ (index `6`) into `sharp_var_indices` forces the model to reconstruct real meteorological fronts and tight pressure gradients.
3. **Tighten Thermodynamic State Equation Coupling**:
Increase `weight_state_eq` slightly so $T$, $\rho$, and $p$ stay physically consistent via $p = \rho R_d T$.

---

### Step-by-Step Implementation

Modify `AIDASurrogateLoss` in **`scripts/train_aida_surrogate.py`**:

#### 1. Include Pressure in Gradient Matching

In `__init__`, update `sharp_var_indices` to include index `6` ($p$):

```python
sharp_var_indices: list[int] = [0, 1, 2, 4, 6]  # Added 6 (ln_p)

```

#### 2. Tune Default Hyperparameters in `train_aida_surrogate.py`

In `main()`, adjust the default CLI arguments:

```python
parser.add_argument(
    "--lambda-laplacian-p", "--lambda_laplacian_p",
    dest="lambda_laplacian_p", type=float, default=0.18,  # Tuned down from 0.30
    help="2nd-order graph Laplacian weight for pressure"
)
parser.add_argument(
    "--weight-grad-state", "--weight_grad_state",
    dest="weight_grad_state", type=float, default=0.25,   # Boosted from 0.20 to sharpen fronts
    help="State gradient matching weight"
)
parser.add_argument(
    "--weight-state-eq", "--weight_state_eq",
    dest="weight_state_eq", type=float, default=0.15,     # Increased from 0.10 for physical consistency
    help="Ideal gas residual weight"
)

```

---

### Next Execution Command

Run the training pipeline with these updated weights to focus heavily on $p$'s spatial pattern accuracy:

```bash
python -u scripts/train_aida_surrogate.py \
    --zarr ../data/icosahedral_2023_logstate.zarr \
    --edges ../data/graph/icosahedral_edge_index_m4.pt \
    --checkpoint checkpoints/aida_gnn_surrogate_logstate.pt \
    --epochs 30 \
    --lambda-laplacian-p 0.18 \
    --weight-grad-state 0.25 \
    --weight-state-eq 0.15

```

Once trained, re-run cycling and validation. We should see $p$'s ACC climb from **0.43** toward **0.80+**!
