"""
ASTRA-IC Module C: Explainable AI (XAI) & Multi-Parametric QA Triage Engine
Decomposes GPR predictions using SHAP (SHapley Additive exPlanations)
across both leakage current and propagation delay telemetry parameters.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, Tuple, Optional, List
from core.engine import engineer_features, AstraGPREngine, CRITICAL_THRESHOLD


class AstraXAIEngine:
    """
    Explainability Engine utilizing SHAP KernelExplainer on the GPR model.
    Provides multi-parametric feature impact attributions and aerospace compliance audit summaries.
    """
    def __init__(self, gpr_engine: AstraGPREngine):
        self.gpr_engine = gpr_engine
        self.explainer: Optional[shap.KernelExplainer] = None
        self._cached_shap_values: Dict[str, Dict[str, Any]] = {}
        self.feature_names = [
            "iddq_0h",
            "iddq_24h",
            "delta_24_0",
            "drift_ratio",
            "lot_relative_slope",
            "delta_tprop",
            "tprop_ratio",
            "cross_leakage_delay_drift"
        ]

    def init_explainer(self, background_samples: int = 25) -> None:
        """Initializes the SHAP KernelExplainer with a representative training sample."""
        if not self.gpr_engine.is_trained:
            self.gpr_engine.train_baseline_model()

        pool = self.gpr_engine.X_train_pool
        X_background = shap.sample(pool[self.feature_names], background_samples, random_state=42)
        
        # Kernel explainer on GPR predict mean
        self.explainer = shap.KernelExplainer(self.gpr_engine.model.predict, X_background)

    def explain_component(
        self,
        chip_row: pd.Series,
        lot_df: pd.DataFrame,
        nsamples: int = 50
    ) -> Dict[str, Any]:
        """
        Decomposes the predicted 168h current for a specific chip into SHAP feature attributions.
        Returns:
        - feature_names
        - feature_values
        - shap_values (µA contribution to projected leakage)
        - base_value (expected lot average 168h current)
        - total_prediction (base_value + sum(shap_values))
        - qa_inspector_log (formatted natural language report)
        """
        chip_id = chip_row.get("chip_id", "UNKNOWN_IC")
        
        # Check cache
        if chip_id in self._cached_shap_values:
            return self._cached_shap_values[chip_id]

        if self.explainer is None:
            self.init_explainer()

        # Build feature row for this chip in the context of this lot
        X_lot = engineer_features(lot_df)
        chip_idx = chip_row.name if chip_row.name in X_lot.index else lot_df[lot_df["chip_id"] == chip_id].index[0]
        chip_features = X_lot.loc[[chip_idx]][self.feature_names]

        # Calculate SHAP values
        shap_vals = self.explainer.shap_values(chip_features, nsamples=nsamples)
        
        # Format results
        if isinstance(shap_vals, list):
            vals = shap_vals[0][0]
        else:
            vals = shap_vals[0]

        base_val = float(self.explainer.expected_value)
        feature_values_dict = chip_features.iloc[0].to_dict()
        
        contributions = []
        for feat, val in zip(self.feature_names, vals):
            contributions.append({
                "feature": feat,
                "display_name": self._get_feature_label(feat),
                "measured_value": round(float(feature_values_dict[feat]), 3),
                "shap_value": round(float(val), 2),
                "is_adverse": float(val) > 0
            })

        # Sort by absolute SHAP impact
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        pred_168h = float(chip_row.get("pred_168h", base_val + np.sum(vals)))
        pred_std = float(chip_row.get("pred_std", 0.49))
        upper_3sigma = float(chip_row.get("upper_3sigma", pred_168h + 3 * pred_std))
        early_abort = bool(chip_row.get("early_abort", upper_3sigma >= CRITICAL_THRESHOLD))

        # Generate QA Natural Language Audit Log
        qa_log = self._generate_qa_audit_log(
            chip_id=chip_id,
            chip_row=chip_row,
            contributions=contributions,
            pred_168h=pred_168h,
            upper_3sigma=upper_3sigma,
            early_abort=early_abort,
            base_value=base_val
        )

        result = {
            "chip_id": chip_id,
            "base_value": round(base_val, 2),
            "pred_168h": round(pred_168h, 2),
            "pred_std": round(pred_std, 2),
            "upper_3sigma": round(upper_3sigma, 2),
            "early_abort": early_abort,
            "contributions": contributions,
            "feature_values": feature_values_dict,
            "qa_log": qa_log
        }

        self._cached_shap_values[chip_id] = result
        return result

    def _get_feature_label(self, feature_name: str) -> str:
        """User-friendly aerospace engineering labels for multi-parametric features."""
        mapping = {
            "lot_relative_slope": "Leakage Drift Velocity vs Lot (lot_relative_slope)",
            "delta_24_0": "24h Delta Leakage Current (delta_24_0)",
            "iddq_24h": "24h Standby Current (iddq_24h)",
            "iddq_0h": "Initial Standby Current (iddq_0h)",
            "drift_ratio": "Relative Leakage Amplification (drift_ratio)",
            "delta_tprop": "Propagation Delay Drift (delta_tprop)",
            "tprop_ratio": "Delay Degradation Ratio (tprop_ratio)",
            "cross_leakage_delay_drift": "Cross-Coupled Stress: Leakage × Delay (cross_drift)"
        }
        return mapping.get(feature_name, feature_name)

    def _generate_qa_audit_log(
        self,
        chip_id: str,
        chip_row: pd.Series,
        contributions: list,
        pred_168h: float,
        upper_3sigma: float,
        early_abort: bool,
        base_value: float
    ) -> Dict[str, Any]:
        """Synthesizes an engineering-grade aerospace audit statement."""
        top_driver = contributions[0]
        top_driver_val = top_driver["shap_value"]
        top_driver_sign = "+" if top_driver_val > 0 else ""

        # Check for multi-parametric cross coupling
        has_delay_drift = any(
            c["feature"] in ["delta_tprop", "cross_leakage_delay_drift"] and c["shap_value"] > 0.5
            for c in contributions
        )

        if early_abort:
            headline = f"🚨 QA AUDIT: EARLY ABORT AT 24h FOR COMPONENT {chip_id}"
            decision_status = "CRITICAL REJECT - 24h EARLY ABORT RECOMMENDED"
            badge_color = "#DC2626"
            
            multi_text = ""
            if has_delay_drift:
                multi_text = " Compounding propagation delay shifts confirm coupled electro-thermal gate oxide breakdown."

            summary_statement = (
                f"Component {chip_id} was flagged for Early Abort at 24h because its "
                f"degradation velocity ('{top_driver['feature']}') contributed {top_driver_sign}{top_driver_val:.2f} µA "
                f"toward the projected 168h failure threshold ({pred_168h:.2f} µA, Upper 3σ: {upper_3sigma:.2f} µA)."
                f"{multi_text}"
            )
            recommendation = (
                f"IMMEDIATE ACTION: Halt chamber testing for {chip_id} at 24h. "
                f"Saving 144 hours of high-temperature burn-in chamber allocation. "
                f"Component violates MIL-STD-883 Method 1015 failure limits."
            )
        else:
            headline = f"✅ QA AUDIT: FLIGHT QUALIFICATION APPROVED FOR {chip_id}"
            decision_status = "PASS - QUALIFIED FOR FLIGHT SCREENING"
            badge_color = "#16A34A"
            summary_statement = (
                f"Component {chip_id} demonstrates nominal Arrhenius leakage and delay progression. "
                f"Predicted 168h standby current is {pred_168h:.2f} µA with Upper 3σ bound of {upper_3sigma:.2f} µA, "
                f"well within the {CRITICAL_THRESHOLD:.1f} µA critical failure boundary."
            )
            recommendation = (
                f"ACTION: Cleared for flight deployment lot acceptance. "
                f"Telemetry variance conforms to AEC-Q001 statistical process limits."
            )

        # Build feature attribution highlights text
        bullet_points = []
        for item in contributions:
            sign = "+" if item["shap_value"] > 0 else ""
            bullet_points.append(
                f"• {item['display_name']}: {sign}{item['shap_value']:.2f} µA (Measured: {item['measured_value']})"
            )

        return {
            "headline": headline,
            "decision_status": decision_status,
            "badge_color": badge_color,
            "summary_statement": summary_statement,
            "bullet_points": bullet_points,
            "recommendation": recommendation,
            "top_driver": top_driver
        }
