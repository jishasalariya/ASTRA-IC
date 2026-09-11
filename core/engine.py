"""
ASTRA-IC Core Mathematical Engine
Aerospace Semiconductor Telemetry & Reliability Analytics for Integrated Circuits

Modules:
- Module A: Dynamic Part Average Testing (PAT) using Modified Z-score (MAD) per AEC-Q001
- Module B: Multi-Parametric Feature Engineering (Leakage + Propagation Delay)
- Module B: Gaussian Process Regression (GPR) Drift Forecaster & Continuous Active Learning Adaptation
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from typing import Tuple, Dict, Any, Optional, List

# Standard Aerospace / Burn-In Constants
STATIC_DATASHEET_LIMIT = 50.0   # µA - Static ATE screening absolute ceiling
CRITICAL_THRESHOLD = 30.0       # µA - 168h latent failure threshold
AEC_Q001_Z_THRESHOLD = 3.5      # Modified Z-score outlier cutoff
BURN_IN_TOTAL_HOURS = 168       # Standard military burn-in test duration
SCREEN_HOURS = 24               # ASTRA-IC early screening timestamp


def generate_ate_lot(n_chips: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """
    Simulates multi-parametric ATE test data for space-grade ICs at 0h, 24h, 96h, and 168h.
    Synthesizes both Standby Current (Iddq) and Propagation Delay (tprop).
    - 98% Nominal parts (stable logarithmic Arrhenius drift)
    - 1% Static Outliers (high initial leakage, below 50µA absolute limit)
    - 1% Latent Drift Defects (normal at 0h, fails catastrophically by 168h)
    """
    np.random.seed(random_seed)
    chip_ids = [f"IC_{i:04d}" for i in range(n_chips)]

    # Baseline nominal parameters (Hour 0)
    iddq_0h = np.random.normal(loc=10.0, scale=1.5, size=n_chips)
    tprop_0h = np.random.normal(loc=12.0, scale=0.5, size=n_chips)

    alpha = np.random.uniform(0.3, 0.6, size=n_chips)
    beta = 0.05
    noise = lambda: np.random.normal(0, 0.15, size=n_chips)
    noise_t = lambda: np.random.normal(0, 0.04, size=n_chips)

    # Progression for iddq (µA)
    iddq_24h = iddq_0h + alpha * np.log(1 + beta * 24) + noise()
    iddq_96h = iddq_0h + alpha * np.log(1 + beta * 96) + noise()
    iddq_168h = iddq_0h + alpha * np.log(1 + beta * 168) + noise()

    # Progression for propagation delay tprop (ns)
    tprop_24h = tprop_0h + 0.10 * alpha * np.log(1 + beta * 24) + noise_t()
    tprop_96h = tprop_0h + 0.12 * alpha * np.log(1 + beta * 96) + noise_t()
    tprop_168h = tprop_0h + 0.15 * alpha * np.log(1 + beta * 168) + noise_t()

    labels = ["NOMINAL"] * n_chips

    # Inject Static Outliers (starts high at ~43µA, absolute limit is 50µA)
    n_static = max(1, int(n_chips * 0.01))
    for i in range(n_static):
        iddq_0h[i] = np.random.uniform(42.0, 45.0)
        iddq_24h[i] = iddq_0h[i] + 0.5 + np.random.normal(0, 0.1)
        iddq_96h[i] = iddq_0h[i] + 1.2 + np.random.normal(0, 0.1)
        iddq_168h[i] = iddq_0h[i] + 1.8 + np.random.normal(0, 0.1)

        tprop_24h[i] = tprop_0h[i] + 0.15 + np.random.normal(0, 0.05)
        tprop_96h[i] = tprop_0h[i] + 0.28 + np.random.normal(0, 0.05)
        tprop_168h[i] = tprop_0h[i] + 0.40 + np.random.normal(0, 0.05)
        labels[i] = "STATIC_OUTLIER"

    # Inject Latent Drift Defects (starts nominal ~10µA, fails by 168h)
    n_latent = max(1, int(n_chips * 0.01))
    for i in range(n_static, n_static + n_latent):
        iddq_0h[i] = np.random.uniform(9.5, 11.0)
        iddq_24h[i] = iddq_0h[i] + 4.2
        iddq_96h[i] = iddq_0h[i] + 18.5
        iddq_168h[i] = iddq_0h[i] + 36.0

        # Coupled electro-thermal delay degradation (hot carrier injection)
        tprop_24h[i] = tprop_0h[i] + 0.85 + np.random.normal(0, 0.05)
        tprop_96h[i] = tprop_0h[i] + 2.10 + np.random.normal(0, 0.05)
        tprop_168h[i] = tprop_0h[i] + 4.50 + np.random.normal(0, 0.05)
        labels[i] = "LATENT_DRIFT_DEFECT"

    return pd.DataFrame({
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


def normalize_ate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes and validates an ingested ATE CSV DataFrame:
    1. Trims whitespace from column headers
    2. Case-insensitively maps common column aliases:
       - chip_id: chip_id, chip, dut, dut_id, part_id, device_id, serial, id
       - iddq_0h: iddq_0h, iddq0h, iddq_0, current_0h, i_0h, 0h
       - iddq_24h: iddq_24h, iddq24h, iddq_24, current_24h, i_24h, 24h
       - tprop_0h: tprop_0h, tprop0h, tprop_0, delay_0h, tp_0h
       - tprop_24h: tprop_24h, tprop24h, tprop_24, delay_24h, tp_24h
    3. Auto-generates chip_id if missing (e.g. IC_0000, IC_0001, ...)
    4. Validates presence of essential measurements (iddq_0h and iddq_24h)
    5. Coerces measurements to numeric floats
    """
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    lower_cols = {c.lower(): c for c in data.columns}
    col_mapping = {}

    aliases = {
        "chip_id": ["chip_id", "chipid", "chip", "dut", "dut_id", "part_id", "device_id", "serial", "serial_number", "id"],
        "iddq_0h": ["iddq_0h", "iddq0h", "iddq_0", "iddq0", "current_0h", "current0h", "i_0h", "0h", "iddq_hour_0"],
        "iddq_24h": ["iddq_24h", "iddq24h", "iddq_24", "iddq24", "current_24h", "current24h", "i_24h", "24h", "iddq_hour_24"],
        "iddq_96h": ["iddq_96h", "iddq96h", "iddq_96", "iddq96", "current_96h", "96h"],
        "iddq_168h": ["iddq_168h", "iddq168h", "iddq_168", "iddq168", "current_168h", "168h"],
        "tprop_0h": ["tprop_0h", "tprop0h", "tprop_0", "tprop0", "delay_0h", "tp_0h"],
        "tprop_24h": ["tprop_24h", "tprop24h", "tprop_24", "tprop24", "delay_24h", "tp_24h"],
        "tprop_96h": ["tprop_96h", "tprop96h", "tprop_96", "tprop96", "delay_96h"],
        "tprop_168h": ["tprop_168h", "tprop168h", "tprop_168", "tprop168", "delay_168h"],
        "true_label": ["true_label", "label", "target", "ground_truth", "status", "class"]
    }

    for target, pattern_list in aliases.items():
        if target not in data.columns:
            for pat in pattern_list:
                if pat in lower_cols:
                    col_mapping[lower_cols[pat]] = target
                    break

    if col_mapping:
        data.rename(columns=col_mapping, inplace=True)

    # Validate essential minimum columns
    missing_req = [c for c in ["iddq_0h", "iddq_24h"] if c not in data.columns]
    if missing_req:
        found_cols = ", ".join([f"'{c}'" for c in data.columns])
        raise ValueError(
            f"Missing required burn-in current telemetry column(s): {', '.join(missing_req)}. "
            f"Found columns: [{found_cols}]. "
            "Please ensure your CSV contains at least 0-hour and 24-hour leakage readings."
        )

    # Auto-generate chip_id if missing
    if "chip_id" not in data.columns:
        data["chip_id"] = [f"IC_{i:04d}" for i in range(len(data))]
    else:
        data["chip_id"] = data["chip_id"].astype(str)

    # Coerce numerical telemetry columns
    num_cols = ["iddq_0h", "iddq_24h", "iddq_96h", "iddq_168h", "tprop_0h", "tprop_24h", "tprop_96h", "tprop_168h"]
    for nc in num_cols:
        if nc in data.columns:
            data[nc] = pd.to_numeric(data[nc], errors="coerce")

    # Drop rows where essential measurements are NaN
    data = data.dropna(subset=["iddq_0h", "iddq_24h"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("Uploaded CSV contains no valid numerical rows for iddq_0h and iddq_24h.")

    return data


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
    Multi-Parametric Feature Engineering for GPR Drift Forecasting:
    - iddq_0h: Initial standby leakage current (µA)
    - iddq_24h: Standby leakage current at 24h burn-in (µA)
    - delta_24_0: Raw leakage drift velocity (µA/24h)
    - drift_ratio: Proportional leakage amplification
    - lot_relative_slope: Standardized drift acceleration relative to lot baseline
    
    Multi-parametric correlation terms (with graceful fallback if tprop is missing):
    - delta_tprop: Propagation delay drift velocity (ns/24h)
    - tprop_ratio: Relative timing delay amplification
    - cross_leakage_delay_drift: Cross-coupled electro-thermal degradation (delta_iddq * delta_tprop)
    """
    X = pd.DataFrame(index=df.index)
    X["iddq_0h"] = df["iddq_0h"]
    X["iddq_24h"] = df["iddq_24h"]
    X["delta_24_0"] = df["iddq_24h"] - df["iddq_0h"]
    X["drift_ratio"] = df["iddq_24h"] / (df["iddq_0h"] + 1e-6)
    lot_mean_slope = np.mean(X["delta_24_0"])
    X["lot_relative_slope"] = X["delta_24_0"] / (lot_mean_slope + 1e-6)

    # Multi-parametric delay terms
    if "tprop_0h" in df.columns and "tprop_24h" in df.columns:
        X["delta_tprop"] = df["tprop_24h"] - df["tprop_0h"]
        X["tprop_ratio"] = df["tprop_24h"] / (df["tprop_0h"] + 1e-6)
        X["cross_leakage_delay_drift"] = X["delta_24_0"] * X["delta_tprop"]
    else:
        # Neutral imputation fallback: no delay drift observed
        X["delta_tprop"] = 0.0
        X["tprop_ratio"] = 1.0
        X["cross_leakage_delay_drift"] = 0.0

    return X


class AstraGPREngine:
    """
    Gaussian Process Regression (GPR) engine for 24h -> 168h trajectory forecasting.
    Includes Bayesian Uncertainty Sampling for Continuous Active Learning Adaptation.
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
        self.X_train_pool: Optional[pd.DataFrame] = None
        self.y_train_pool: Optional[np.ndarray] = None
        self.adaptation_history: List[Dict[str, Any]] = []

    def train_baseline_model(self, train_df: Optional[pd.DataFrame] = None) -> None:
        """Trains the GPR model on reference ATE training lot (default seed 101, 300 chips)."""
        if train_df is None:
            train_df = generate_ate_lot(n_chips=300, random_seed=101)
        self.train_df = train_df
        self.X_train_pool = engineer_features(train_df)
        self.y_train_pool = train_df["iddq_168h"].values
        self.model.fit(self.X_train_pool, self.y_train_pool)
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

    def active_adapt_lot(
        self,
        new_lot_df: pd.DataFrame,
        uncertainty_percentile: float = 90.0,
        max_samples: int = 15
    ) -> Dict[str, Any]:
        """
        Continuous Bayesian Active Learning:
        1. Computes posterior predictive variance sigma(x) on new lot.
        2. Identifies boundary candidates with highest epistemic uncertainty.
        3. Updates memory pool and re-tunes kernel hyperparameters online.
        """
        if not self.is_trained:
            self.train_baseline_model()

        X_new = engineer_features(new_lot_df)
        prior_mean, prior_std = self.model.predict(X_new, return_std=True)

        # Identify boundary samples with highest uncertainty
        unc_cutoff = np.percentile(prior_std, uncertainty_percentile)
        high_unc_mask = prior_std >= unc_cutoff
        candidate_indices = np.where(high_unc_mask)[0]

        if len(candidate_indices) > max_samples:
            # Pick the top max_samples most uncertain
            sorted_by_unc = candidate_indices[np.argsort(prior_std[candidate_indices])[::-1]]
            selected_indices = sorted_by_unc[:max_samples]
        else:
            selected_indices = candidate_indices

        if len(selected_indices) == 0:
            return {
                "adapted": False,
                "reason": "Uncertainty below adaptation threshold",
                "samples_adapted": 0,
                "mean_prior_std": float(np.mean(prior_std)),
                "mean_updated_std": float(np.mean(prior_std))
            }

        X_adapt = X_new.iloc[selected_indices]
        
        # If true labels exist in new_lot_df, use ground truth; otherwise condition on posterior mean
        if "iddq_168h" in new_lot_df.columns:
            y_adapt = new_lot_df["iddq_168h"].iloc[selected_indices].values
        else:
            y_adapt = prior_mean[selected_indices]

        # Augment training memory pool
        self.X_train_pool = pd.concat([self.X_train_pool, X_adapt], ignore_index=True)
        self.y_train_pool = np.concatenate([self.y_train_pool, y_adapt])

        # Re-fit GPR online with warm start
        self.model.fit(self.X_train_pool, self.y_train_pool)

        # Re-evaluate uncertainty to measure variance reduction
        _, updated_std = self.model.predict(X_new, return_std=True)

        adapt_record = {
            "adapted": True,
            "samples_adapted": int(len(selected_indices)),
            "mean_prior_std": round(float(np.mean(prior_std)), 3),
            "mean_updated_std": round(float(np.mean(updated_std)), 3),
            "variance_reduction_pct": round(float((np.mean(prior_std) - np.mean(updated_std)) / (np.mean(prior_std) + 1e-6) * 100), 1),
            "adapted_chip_ids": list(new_lot_df["chip_id"].iloc[selected_indices].values if "chip_id" in new_lot_df.columns else [f"IC_{idx:04d}" for idx in selected_indices])
        }
        self.adaptation_history.append(adapt_record)
        return adapt_record


def process_lot_data(
    df: pd.DataFrame,
    gpr_engine: Optional[AstraGPREngine] = None,
    z_thresh: float = AEC_Q001_Z_THRESHOLD,
    critical_threshold: float = CRITICAL_THRESHOLD,
    force_recompute: bool = False,
    run_active_adaptation: bool = True
) -> Tuple[pd.DataFrame, AstraGPREngine]:
    """
    Unified processing function for ingested ATE lot CSVs:
    0. Normalizes column headers, aliases, and generates chip_id if missing
    1. Validates or computes Module A Dynamic PAT (Hour 0)
    2. Runs continuous Bayesian Active Learning adaptation
    3. Runs Module B GPR 168h trajectory predictions & early abort triage
    """
    if gpr_engine is None:
        gpr_engine = AstraGPREngine()
        gpr_engine.train_baseline_model()

    # Step 0: Normalize and sanitize input DataFrame
    clean_df = normalize_ate_dataframe(df)

    # Step 1: Run Dynamic PAT
    processed_df = run_dynamic_pat(clean_df, z_thresh=z_thresh)

    # Step 2: Continuous Bayesian Active Learning
    if run_active_adaptation:
        gpr_engine.active_adapt_lot(processed_df)

    # Step 3: Run GPR Prediction
    processed_df = gpr_engine.predict_lot(processed_df, critical_threshold=critical_threshold)

    return processed_df, gpr_engine


def calculate_lot_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes high-impact mission control summary KPIs:
    1. Total Screened ICs
    2. Static ATE Pass Rate
    3. ASTRA-IC Early Aborts Flagged
    4. Chamber Testing Time Saved (144h saved per aborted component)
    """
    total_screened = len(df)
    
    static_ate_failures = (df["iddq_0h"] > STATIC_DATASHEET_LIMIT).sum()
    static_pass_rate = 100.0 if total_screened == 0 else ((total_screened - static_ate_failures) / total_screened) * 100.0

    early_aborts = df["early_abort"].sum() if "early_abort" in df.columns else 0
    pat_outliers = df["pat_flagged"].sum() if "pat_flagged" in df.columns else 0
    
    if "true_label" in df.columns:
        latent_total = (df["true_label"] == "LATENT_DRIFT_DEFECT").sum()
        latent_caught = ((df["true_label"] == "LATENT_DRIFT_DEFECT") & df["early_abort"]).sum()
        escape_rate = 0.0 if latent_total == 0 else ((latent_total - latent_caught) / latent_total) * 100.0
    else:
        escape_rate = 0.0

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
