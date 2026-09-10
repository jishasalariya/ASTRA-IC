import numpy as np
import pandas as pd
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

def create_lot(
    filename,
    n_chips=500,
    n_static_outliers=5,
    n_latent_defects=5,
    random_seed=42,
    base_iddq=10.0,
    noise_level=0.15
):
    """
    Synthesizes multi-parametric ATE Burn-In test telemetry for semiconductor lots.
    Physics model: Coupled Arrhenius logarithmic leakage drift + hot-carrier propagation delay progression.
    """
    np.random.seed(random_seed)
    chip_ids = [f"IC_{i:04d}" for i in range(n_chips)]

    # Baseline nominal parameters (Hour 0)
    iddq_0h = np.random.normal(loc=base_iddq, scale=1.2, size=n_chips)
    tprop_0h = np.random.normal(loc=12.0, scale=0.4, size=n_chips)

    # Physical drift coefficients (Nominal Arrhenius progression)
    alpha = np.random.uniform(0.35, 0.55, size=n_chips)
    beta = 0.05
    noise = lambda: np.random.normal(0, noise_level, size=n_chips)
    noise_t = lambda: np.random.normal(0, 0.04, size=n_chips)

    iddq_24h = iddq_0h + alpha * np.log(1 + beta * 24) + noise()
    iddq_96h = iddq_0h + alpha * np.log(1 + beta * 96) + noise()
    iddq_168h = iddq_0h + alpha * np.log(1 + beta * 168) + noise()

    tprop_24h = tprop_0h + 0.08 * alpha * np.log(1 + beta * 24) + noise_t()
    tprop_96h = tprop_0h + 0.12 * alpha * np.log(1 + beta * 96) + noise_t()
    tprop_168h = tprop_0h + 0.15 * alpha * np.log(1 + beta * 168) + noise_t()

    labels = ["NOMINAL"] * n_chips

    # 1. Inject Static Outliers (Start high at ~42-45 µA, below static 50 µA limit)
    for i in range(n_static_outliers):
        iddq_0h[i] = np.random.uniform(41.5, 44.5)
        iddq_24h[i] = iddq_0h[i] + 0.45 + np.random.normal(0, 0.1)
        iddq_96h[i] = iddq_0h[i] + 1.10 + np.random.normal(0, 0.1)
        iddq_168h[i] = iddq_0h[i] + 1.75 + np.random.normal(0, 0.1)

        tprop_24h[i] = tprop_0h[i] + 0.14 + np.random.normal(0, 0.03)
        tprop_96h[i] = tprop_0h[i] + 0.26 + np.random.normal(0, 0.03)
        tprop_168h[i] = tprop_0h[i] + 0.38 + np.random.normal(0, 0.03)
        labels[i] = "STATIC_OUTLIER"

    # 2. Inject Latent Drift Defects (Start normal ~10 µA, fail severely by 168h)
    start_idx = n_static_outliers
    for i in range(start_idx, start_idx + n_latent_defects):
        iddq_0h[i] = np.random.uniform(9.8, 11.2)
        iddq_24h[i] = iddq_0h[i] + np.random.uniform(3.8, 4.6)
        iddq_96h[i] = iddq_0h[i] + np.random.uniform(16.0, 19.5)
        iddq_168h[i] = iddq_0h[i] + np.random.uniform(34.0, 38.0)

        # Coupled delay degradation due to hot carrier injection & electromigration
        tprop_24h[i] = tprop_0h[i] + np.random.uniform(0.70, 0.95)
        tprop_96h[i] = tprop_0h[i] + np.random.uniform(1.80, 2.30)
        tprop_168h[i] = tprop_0h[i] + np.random.uniform(3.80, 4.70)
        labels[i] = "LATENT_DRIFT_DEFECT"

    df = pd.DataFrame({
        "chip_id": chip_ids,
        "true_label": labels,
        "tprop_0h": np.round(tprop_0h, 3),
        "tprop_24h": np.round(tprop_24h, 3),
        "tprop_96h": np.round(tprop_96h, 3),
        "tprop_168h": np.round(tprop_168h, 3),
        "iddq_0h": np.round(iddq_0h, 3),
        "iddq_24h": np.round(iddq_24h, 3),
        "iddq_96h": np.round(iddq_96h, 3),
        "iddq_168h": np.round(iddq_168h, 3)
    })

    df.to_csv(filename, index=False)
    print(f"✓ Generated '{filename}': {n_chips} chips ({n_static_outliers} Static Outliers, {n_latent_defects} Latent Defects)")

# Generate 4 distinct test lots for live presentation/demo
if __name__ == "__main__":
    os.makedirs("test_lots", exist_ok=True)

    # Lot 1: Standard Flight Lot (500 chips, balanced 10 defects)
    create_lot("test_lots/lot_flight_qualified_500.csv", n_chips=500, n_static_outliers=5, n_latent_defects=5, random_seed=42)

    # Lot 2: High-Defect Stress Batch (250 chips, 15 defects to show heavy screening)
    create_lot("test_lots/lot_high_stress_250.csv", n_chips=250, n_static_outliers=8, n_latent_defects=7, random_seed=101)

    # Lot 3: Pure Clean Nominal Batch (300 chips, 0 defects -> all PASS)
    create_lot("test_lots/lot_pure_nominal_300.csv", n_chips=300, n_static_outliers=0, n_latent_defects=0, random_seed=202)

    # Lot 4: Large Production Lot (1000 chips, 20 defects)
    create_lot("test_lots/lot_production_run_1000.csv", n_chips=1000, n_static_outliers=10, n_latent_defects=10, random_seed=303)