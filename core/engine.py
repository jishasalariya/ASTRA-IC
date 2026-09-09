"""
ASTRA-IC Core Mathematical Engine
Aerospace Semiconductor Telemetry & Reliability Analytics for Integrated Circuits

Modules:
- Module A: Dynamic Part Average Testing (PAT) using Modified Z-score (MAD) per AEC-Q001
- Module B: Feature Engineering & Gaussian Process Regression (GPR) Drift Forecaster (24h -> 168h)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from typing import Tuple, Dict, Any, Optional

# Standard Aerospace / Burn-In Constants
STATIC_DATASHEET_LIMIT = 50.0   # µA - Static ATE screening absolute ceiling
CRITICAL_THRESHOLD = 30.0       # µA - 168h latent failure threshold
AEC_Q001_Z_THRESHOLD = 3.5      # Modified Z-score outlier cutoff
BURN_IN_TOTAL_HOURS = 168       # Standard military burn-in test duration
SCREEN_HOURS = 24               # ASTRA-IC early screening timestamp


def generate_ate_lot(n_chips: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """
    Simulates ATE test data for space-grade ICs at 0h, 24h, 96h, and 168h.
    - 98% Nominal parts (stable, logarithmic Arrhenius drift)
    - 1% Static Outliers (high initial leakage, but below 50µA absolute limit)
    - 1% Latent Drift Defects (starts normal, fails catastrophically by 168h)
    """
    np.random.seed(random_seed)
    chip_ids = [f"IC_{i:04d}" for i in range(n_chips)]

    # Baseline nominal parameters
    iddq_0h = np.random.normal(loc=10.0, scale=1.5, size=n_chips)
    tprop_0h = np.random.normal(loc=12.0, scale=0.5, size=n_chips)

    alpha = np.random.uniform(0.3, 0.6, size=n_chips)
    beta = 0.05
    noise = lambda: np.random.normal(0, 0.15, size=n_chips)

    iddq_24h = iddq_0h + alpha * np.log(1 + beta * 24) + noise()
    iddq_96h = iddq_0h + alpha * np.log(1 + beta * 96) + noise()
    iddq_168h = iddq_0h + alpha * np.log(1 + beta * 168) + noise()

    labels = ["NOMINAL"] * n_chips

    # Inject Static Outliers (starts high at ~43µA, absolute limit is 50µA)
    n_static = max(1, int(n_chips * 0.01))
    for i in range(n_static):
        iddq_0h[i] = np.random.uniform(42.0, 45.0)
        iddq_24h[i] = iddq_0h[i] + 0.5 + np.random.normal(0, 0.1)
        iddq_96h[i] = iddq_0h[i] + 1.2 + np.random.normal(0, 0.1)
        iddq_168h[i] = iddq_0h[i] + 1.8 + np.random.normal(0, 0.1)
        labels[i] = "STATIC_OUTLIER"

    # Inject Latent Drift Defects (starts nominal ~10µA, fails by 168h)
    n_latent = max(1, int(n_chips * 0.01))
    for i in range(n_static, n_static + n_latent):
        iddq_0h[i] = np.random.uniform(9.5, 11.0)
        iddq_24h[i] = iddq_0h[i] + 4.2
        iddq_96h[i] = iddq_0h[i] + 18.5
        iddq_168h[i] = iddq_0h[i] + 36.0
        labels[i] = "LATENT_DRIFT_DEFECT"

    return pd.DataFrame({
        "chip_id": chip_ids,
        "true_label": labels,
        "tprop_0h": np.round(tprop_0h, 3),
        "iddq_0h": np.round(iddq_0h, 3),
        "iddq_24h": np.round(iddq_24h, 3),
        "iddq_96h": np.round(iddq_96h, 3),
        "iddq_168h": np.round(iddq_168h, 3)
    })


def run_dynamic_pat(df: pd.DataFrame, z_thresh: float = AEC_Q001_Z_THRESHOLD) -> pd.DataFrame:
    """
    Module A: AEC-Q001 Dynamic Part Average Testing (PAT) at Hour 0.
    Uses Modified Z-score based on Median Absolute Deviation (MAD):
    M_i = 0.6745 * |x_i - Median| / MAD
    Flags outliers where M_i > z_thresh.
    """
    data = df.copy()
    med = np.median(data["iddq_0h"])
    mad = np.median(np.abs(data["iddq_0h"] - med))
    mad = mad if mad != 0 else 1e-6
    data["mod_z_score"] = 0.6745 * np.abs(data["iddq_0h"] - med) / mad
    data["pat_flagged"] = data["mod_z_score"] > z_thresh
    return data


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for the 24h GPR drift forecasting model:
    - iddq_0h: Initial standby leakage current (µA)
    - iddq_24h: Standby leakage current at 24h burn-in (µA)
    - delta_24_0: Raw drift velocity (µA/24h)
    - drift_ratio: Proportional leakage amplification
    - lot_relative_slope: Standardized drift acceleration relative to lot baseline
    """
    X = pd.DataFrame(index=df.index)
    X["iddq_0h"] = df["iddq_0h"]
    X["iddq_24h"] = df["iddq_24h"]
    X["delta_24_0"] = df["iddq_24h"] - df["iddq_0h"]
    X["drift_ratio"] = df["iddq_24h"] / (df["iddq_0h"] + 1e-6)
    lot_mean_slope = np.mean(X["delta_24_0"])
    X["lot_relative_slope"] = X["delta_24_0"] / (lot_mean_slope + 1e-6)
    return X


class AstraGPREngine:
    """
    Gaussian Process Regression (GPR) engine for 24h -> 168h trajectory forecasting.
    Provides calibrated uncertainty bounds (sigma) and triggers early abort recommendations.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=5,
            random_state=self.random_state
        )
        self.is_trained = False
        self.train_df: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None

    def train_baseline_model(self, train_df: Optional[pd.DataFrame] = None) -> None:
        """Trains the GPR model on reference ATE training lot (default seed 101, 300 chips)."""
        if train_df is None:
            train_df = generate_ate_lot(n_chips=300, random_seed=101)
        self.train_df = train_df
        self.X_train = engineer_features(train_df)
        y_train = train_df["iddq_168h"].values
        self.model.fit(self.X_train, y_train)
        self.is_trained = True

    def predict_lot(
        self,
        df: pd.DataFrame,
        critical_threshold: float = CRITICAL_THRESHOLD
    ) -> pd.DataFrame:
        """
        Runs GPR trajectory prediction on lot data.
        Calculates:
        - pred_168h: Mean predicted 168h leakage current (µA)
        - pred_std: Epistemic & aleatoric uncertainty sigma (µA)
        - upper_3sigma: Upper 99.73% confidence ceiling = pred_168h + 3*pred_std
        - early_abort: Flag if upper_3sigma >= critical_threshold
        """
        if not self.is_trained:
            self.train_baseline_model()

        X_test = engineer_features(df)
        pred_mean, pred_std = self.model.predict(X_test, return_std=True)

        res_df = df.copy()
        res_df["pred_168h"] = np.round(pred_mean, 2)
        res_df["pred_std"] = np.round(pred_std, 2)
        res_df["upper_3sigma"] = np.round(pred_mean + 3 * pred_std, 2)
        res_df["early_abort"] = res_df["upper_3sigma"] >= critical_threshold

        return res_df


def process_lot_data(
    df: pd.DataFrame,
    gpr_engine: Optional[AstraGPREngine] = None,
    z_thresh: float = AEC_Q001_Z_THRESHOLD,
    critical_threshold: float = CRITICAL_THRESHOLD,
    force_recompute: bool = False
) -> Tuple[pd.DataFrame, AstraGPREngine]:
    """
    Unified processing function for ingested ATE lot CSVs:
    1. Validates or computes Module A Dynamic PAT (Hour 0)
    2. Validates or computes Module B GPR 168h predictions & early aborts
    """
    if gpr_engine is None:
        gpr_engine = AstraGPREngine()
        gpr_engine.train_baseline_model()

    # Check if df already contains full pre-computed columns and recompute is not forced
    has_pat = "mod_z_score" in df.columns and "pat_flagged" in df.columns
    has_gpr = "pred_168h" in df.columns and "upper_3sigma" in df.columns and "early_abort" in df.columns

    if has_pat and has_gpr and not force_recompute:
        # Re-evaluate boolean flags if thresholds changed from defaults
        processed_df = df.copy()
        processed_df["pat_flagged"] = processed_df["mod_z_score"] > z_thresh
        processed_df["early_abort"] = processed_df["upper_3sigma"] >= critical_threshold
        return processed_df, gpr_engine

    # Step 1: Run Dynamic PAT
    processed_df = run_dynamic_pat(df, z_thresh=z_thresh)

    # Step 2: Run GPR Prediction
    processed_df = gpr_engine.predict_lot(processed_df, critical_threshold=critical_threshold)

    return processed_df, gpr_engine


def calculate_lot_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes the 4 high-impact mission control summary KPIs:
    1. Total Screened ICs
    2. Static ATE Pass Rate (100% passes 50µA datasheet ceiling)
    3. ASTRA-IC Early Aborts Flagged
    4. Chamber Testing Time Saved (144h saved per aborted component)
    """
    total_screened = len(df)
    
    # Static ATE Pass Rate (checks if 0h or 24h exceeded 50µA datasheet ceiling)
    static_ate_failures = (df["iddq_0h"] > STATIC_DATASHEET_LIMIT).sum()
    static_pass_rate = 100.0 if total_screened == 0 else ((total_screened - static_ate_failures) / total_screened) * 100.0

    # Early Aborts: chips flagged for 24h shutdown (Upper 3-sigma >= 30µA OR PAT flagged)
    early_aborts = df["early_abort"].sum() if "early_abort" in df.columns else 0
    pat_outliers = df["pat_flagged"].sum() if "pat_flagged" in df.columns else 0
    
    # Latent defect capture
    if "true_label" in df.columns:
        latent_total = (df["true_label"] == "LATENT_DRIFT_DEFECT").sum()
        latent_caught = ((df["true_label"] == "LATENT_DRIFT_DEFECT") & df["early_abort"]).sum()
        escape_rate = 0.0 if latent_total == 0 else ((latent_total - latent_caught) / latent_total) * 100.0
    else:
        escape_rate = 0.0

    # Testing Chamber Time Saved:
    # Baseline burn-in: 168 hours for all chips
    # ASTRA-IC burn-in: 24 hours for early aborts, saving (168 - 24) = 144 hours each
    time_saved_percent = ((BURN_IN_TOTAL_HOURS - SCREEN_HOURS) / BURN_IN_TOTAL_HOURS) * 100.0
    hours_saved_per_aborted = (BURN_IN_TOTAL_HOURS - SCREEN_HOURS)
    total_hours_saved = early_aborts * hours_saved_per_aborted

    return {
        "total_screened": total_screened,
        "static_ate_pass_rate": static_pass_rate,
        "early_aborts": int(early_aborts),
        "pat_outliers": int(pat_outliers),
        "time_saved_percent": round(time_saved_percent, 1),
        "total_hours_saved": int(total_hours_saved),
        "hours_saved_per_chip": hours_saved_per_aborted,
        "escape_rate": escape_rate,
        "chamber_hours_saved_display": f"{round(time_saved_percent, 1)}%",
        "hours_saved_badge": f"{hours_saved_per_aborted} Hours / Lot Saved"
    }
