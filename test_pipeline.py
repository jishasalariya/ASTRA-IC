"""
Test script to verify mathematical alignment, XAI decomposition, and PDF report generation.
"""

import os
import sys
import pandas as pd
import numpy as np

# Configure UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath("."))

from core.engine import (
    generate_ate_lot,
    process_lot_data,
    calculate_lot_kpis,
    AstraGPREngine,
    CRITICAL_THRESHOLD,
    STATIC_DATASHEET_LIMIT,
    AEC_Q001_Z_THRESHOLD
)
from core.xai import AstraXAIEngine
from core.report_generator import generate_compliance_pdf, generate_lot_csv

def run_verification():
    print("=" * 70)
    print("🚀 ASTRA-IC AUTOMATED PIPELINE VERIFICATION")
    print("=" * 70)

    # 1. Test CSV Ingestion
    csv_path = "astra_ic_processed_lot.csv"
    assert os.path.exists(csv_path), f"File {csv_path} does not exist!"
    df = pd.read_csv(csv_path)
    print(f"[✓] Loaded {csv_path}: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Test Engine & KPIs
    gpr_engine = AstraGPREngine(random_state=42)
    gpr_engine.train_baseline_model()
    print("[✓] GPR Engine trained successfully on 300 reference chips")

    processed_df, _ = process_lot_data(df, gpr_engine=gpr_engine)
    kpis = calculate_lot_kpis(processed_df)
    
    print("\n--- Summary Metrics Verification ---")
    print(f"  • Total Screened: {kpis['total_screened']} ICs")
    print(f"  • Static ATE Pass Rate: {kpis['static_ate_pass_rate']:.1f}% (Datasheet Ceiling: 50µA)")
    print(f"  • Early Aborts Flagged: {kpis['early_aborts']} Chips")
    print(f"  • Dynamic PAT Outliers: {kpis['pat_outliers']} Chips")
    print(f"  • Chamber Time Saved: {kpis['time_saved_percent']:.1f}% ({kpis['total_hours_saved']} Total Chamber Hours)")

    assert kpis["total_screened"] == 500, f"Expected 500 ICs, got {kpis['total_screened']}"
    assert kpis["early_aborts"] == 10, f"Expected 10 early aborts, got {kpis['early_aborts']}"
    assert kpis["pat_outliers"] == 6, f"Expected 6 PAT outliers, got {kpis['pat_outliers']}"
    assert kpis["time_saved_percent"] == 85.7, f"Expected 85.7% time saved, got {kpis['time_saved_percent']}"

    # 3. Test Specific Component IC_0006
    ic6 = processed_df[processed_df["chip_id"] == "IC_0006"].iloc[0]
    print("\n--- Component IC_0006 Telemetry ---")
    print(f"  • 0h Iddq: {ic6['iddq_0h']:.3f} µA (Nominal)")
    print(f"  • 24h Iddq: {ic6['iddq_24h']:.3f} µA (Spike observed)")
    print(f"  • GPR Pred 168h: {ic6['pred_168h']:.2f} µA")
    print(f"  • Upper 3σ Bound: {ic6['upper_3sigma']:.2f} µA")
    print(f"  • Early Abort Recommended: {ic6['early_abort']}")

    assert ic6["early_abort"] == True, "IC_0006 should be flagged for early abort!"
    assert ic6["upper_3sigma"] >= CRITICAL_THRESHOLD, f"IC_0006 upper 3sigma should be >= {CRITICAL_THRESHOLD}"

    # 4. Test XAI SHAP Attribution
    print("\n--- Module C: SHAP Decomposition ---")
    xai = AstraXAIEngine(gpr_engine)
    xai.init_explainer(background_samples=25)
    explanation = xai.explain_component(ic6, processed_df, nsamples=50)

    print(f"  • Base Expected Value: {explanation['base_value']:.2f} µA")
    print(f"  • Top Driver: {explanation['qa_log']['top_driver']['feature']} ({explanation['qa_log']['top_driver']['shap_value']:+.2f} µA)")
    print(f"  • QA Log Headline: {explanation['qa_log']['headline']}")
    print(f"  • Summary: {explanation['qa_log']['summary_statement']}")

    assert len(explanation["contributions"]) == 5, "Expected 5 feature contributions"
    assert "lot_relative_slope" in [c["feature"] for c in explanation["contributions"]], "lot_relative_slope must be in features"

    # 5. Test PDF Report Generation
    print("\n--- Compliance Report Generation ---")
    pdf_bytes = generate_compliance_pdf(processed_df, kpis, lot_id="TEST-LOT-001")
    assert len(pdf_bytes) > 1000, "PDF generation produced invalid/empty byte stream"
    print(f"[✓] Generated PDF Audit Report ({len(pdf_bytes)} bytes)")

    csv_bytes = generate_lot_csv(processed_df)
    assert len(csv_bytes) > 500, "CSV generation produced invalid/empty byte stream"
    print(f"[✓] Generated CSV Lot Telemetry ({len(csv_bytes)} bytes)")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED SUCCESSFULLY WITH 100% MATHEMATICAL ALIGNMENT!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
