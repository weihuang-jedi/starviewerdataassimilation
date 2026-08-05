#!/usr/bin/env python3
"""
Calculate tau_min_p threshold for AIDAPressureRegularizedLoss based on 
dataset statistics and physical lower bound for surface pressure.
"""

import numpy as np

# Physical absolute lower bound for atmospheric pressure (e.g., 500 hPa = 50000 Pa for deep low / upper level)
# Or for sea level pressure / lowest layer, set to ~87000 Pa (870 hPa - severe typhoon)
P_PHYSICAL_MIN_PA = 50000.0  # 500 hPa in Pascals

# Step 1: Physical log transform
LN_P_PHYSICAL_MIN = np.log(P_PHYSICAL_MIN_PA)  # ~10.8197

# --- REPLACE THESE VALUES WITH YOUR DATASET'S LN_P NORMALIZATION STATS ---
# Example dataset stats for ln_p across all vertical levels:
LN_P_MEAN = 10.95    # Mean of ln(p) in raw data
LN_P_STD = 0.45      # Standard deviation of ln(p) in raw data
EPSILON = 1e-6       # Safety epsilon used during scaling
# ------------------------------------------------------------------------

def compute_tau_min_p(p_phys_min, mean, std, eps=1e-6):
    ln_p_target = np.log(p_phys_min)
    # Z-score standardization formula used in dataset prep
    tau_min = (ln_p_target - mean) / (std + eps)
    return tau_min

tau_calculated = compute_tau_min_p(P_PHYSICAL_MIN_PA, LN_P_MEAN, LN_P_STD, EPSILON)

print(f"="*60)
print(f"CALIBRATION REPORT FOR tau_min_p")
print(f"="*60)
print(f"Physical Pressure Lower Bound : {P_PHYSICAL_MIN_PA / 100:.1f} hPa ({P_PHYSICAL_MIN_PA:.1f} Pa)")
print(f"Physical Log Space Value      : {LN_P_PHYSICAL_MIN:.4f}")
print(f"Dataset Mean / Std for ln_p   : Mean={LN_P_MEAN:.4f}, Std={LN_P_STD:.4f}")
print(f"-"*60)
print(f"RECOMMENDED tau_min_p         : {tau_calculated:.4f}")
print(f"="*60)
print(f"Pass this value to --tau_min_p when running train_aida_mesh.py")
