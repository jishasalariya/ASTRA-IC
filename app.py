"""
ASTRA-IC: Semiconductor Reliability Screening System
Clean Light-Theme Production Dashboard with Strict 2-Page Ingestion Workflow.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from core.engine import (
    AstraGPREngine,
    process_lot_data,
    calculate_lot_kpis,
    generate_ate_lot,
    AEC_Q001_Z_THRESHOLD,
    CRITICAL_THRESHOLD,
    STATIC_DATASHEET_LIMIT
)
from core.xai import AstraXAIEngine
from core.report_generator import generate_compliance_pdf, generate_lot_csv
from components.kpi_ribbon import render_kpi_ribbon
from components.module_a_view import render_module_a_chart
from components.module_b_view import render_module_b_chart
from components.module_c_view import render_component_inspector

# =====================================================================
# 1. PAGE CONFIG & STYLESHEET
# =====================================================================
st.set_page_config(
    page_title="ASTRA-IC | Semiconductor Reliability Screening",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ingest Clean Light Theme CSS
css_file_path = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
if os.path.exists(css_file_path):
    with open(css_file_path, "r", encoding="utf-8") as f:
        custom_css = f.read()
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)


# =====================================================================
# 2. CACHED ENGINE INITIALIZATION
# =====================================================================
@st.cache_resource
def get_cached_gpr_engine() -> AstraGPREngine:
    engine = AstraGPREngine(random_state=42)
    engine.train_baseline_model()
    return engine


@st.cache_resource
def get_cached_xai_engine(_gpr_engine: AstraGPREngine) -> AstraXAIEngine:
    xai = AstraXAIEngine(_gpr_engine)
    xai.init_explainer(background_samples=25)
    return xai


gpr_engine = get_cached_gpr_engine()
xai_engine = get_cached_xai_engine(gpr_engine)


# =====================================================================
# 3. STRICT 2-STAGE STATE ROUTING
# =====================================================================
if "lot_df" not in st.session_state:
    st.session_state["lot_df"] = None

if "lot_name" not in st.session_state:
    st.session_state["lot_name"] = ""


# ---------------------------------------------------------------------
# STAGE 1: LANDING & FILE UPLOAD SCREEN (NO AUTO-LOADED DATASET)
# ---------------------------------------------------------------------
if st.session_state["lot_df"] is None:
    st.markdown("""
    <div style="height: 40px;"></div>
    """, unsafe_allow_html=True)

    landing_col_l, landing_col_center, landing_col_r = st.columns([1, 2.2, 1])

    with landing_col_center:
        st.markdown("""
        <div class="landing-container">
            <div style="font-size: 2.8rem; margin-bottom: 12px;">🛰️</div>
            <div class="landing-title">ASTRA-IC: Semiconductor Reliability Screening</div>
            <div class="landing-subtext">
                Upload your ATE Burn-In CSV telemetry lot (containing 0h and 24h readings) 
                to trigger the AI-driven 24-hour predictive reliability screening engine.
            </div>
            <div style="display: flex; justify-content: center; gap: 8px; margin-bottom: 24px;">
                <span class="standard-badge badge-blue">MIL-STD-883 Method 1015</span>
                <span class="standard-badge badge-green">AEC-Q001 Dynamic PAT</span>
                <span class="standard-badge badge-amber">GPR 3σ Dual-Engine</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop your ATE CSV Telemetry File Here",
            type=["csv"],
            key="landing_uploader",
            help="Accepts processed ASTRA CSV or raw ATE test CSV with chip_id, iddq_0h, and iddq_24h."
        )

        st.markdown("""
        <div style="text-align: center; margin: 16px 0 12px 0; color: #94A3B8; font-weight: 500; font-size: 0.88rem;">
            — OR FOR EVALUATION & DEMONSTRATION —
        </div>
        """, unsafe_allow_html=True)

        sample_btn = st.button("🚀 Load Sample Flight Batch (500 ICs)", use_container_width=True)

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                with st.spinner("Processing ATE Telemetry through Dynamic PAT & GPR Models..."):
                    proc_df, _ = process_lot_data(raw_df, gpr_engine=gpr_engine)
                    st.session_state["lot_df"] = proc_df
                    st.session_state["lot_name"] = uploaded_file.name
                    st.rerun()
            except Exception as e:
                st.error(f"Error ingesting CSV file: {e}")

        if sample_btn:
            sample_path = os.path.join(os.path.dirname(__file__), "astra_ic_processed_lot.csv")
            if os.path.exists(sample_path):
                raw_df = pd.read_csv(sample_path)
            else:
                raw_df = generate_ate_lot(n_chips=500, random_seed=42)
            
            with st.spinner("Analyzing Flight Lot #500 Telemetry..."):
                proc_df, _ = process_lot_data(raw_df, gpr_engine=gpr_engine)
                st.session_state["lot_df"] = proc_df
                st.session_state["lot_name"] = "Sample Flight Batch (500 ICs)"
                st.rerun()


# ---------------------------------------------------------------------
# STAGE 2: FULL RESULTS & SCREENING DASHBOARD
# ---------------------------------------------------------------------
else:
    processed_df = st.session_state["lot_df"]
    lot_name = st.session_state.get("lot_name", "Flight Batch")
    kpi_metrics = calculate_lot_kpis(processed_df)

    # Top Navigation Bar with Back Button
    nav_col1, nav_col2 = st.columns([3, 1])
    
    with nav_col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px;">
            <h2 style="margin: 0; font-size: 1.45rem; font-weight: 800; color: #0F172A;">
                🛰️ ASTRA-IC Reliability Screening Dashboard
            </h2>
            <span class="standard-badge badge-blue">{lot_name}</span>
            <span class="standard-badge badge-green">MIL-STD-883 Compliant</span>
        </div>
        """, unsafe_allow_html=True)

    with nav_col2:
        if st.button("⬅️ Upload Another Batch", use_container_width=True):
            st.session_state["lot_df"] = None
            st.session_state["lot_name"] = ""
            st.rerun()

    st.markdown("<hr style='border: 0; height: 1px; background: #E2E8F0; margin: 14px 0 20px 0;'/>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # SECTION 1: EXECUTIVE KPI SUMMARY (4 CLEAN METRIC CARDS)
    # -----------------------------------------------------------------
    st.markdown("### 📊 Section 1: Executive KPI Summary")
    render_kpi_ribbon(kpi_metrics)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # SECTION 2: INTERACTIVE LOT VISUALS (SIDE-BY-SIDE CLEAN LAYOUT)
    # -----------------------------------------------------------------
    st.markdown("### 🔬 Section 2: Interactive Lot Visuals")
    
    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        st.markdown("""
        <div class="clean-container" style="padding: 14px;">
            <div style="font-size: 0.85rem; color: #475569; margin-bottom: 8px;">
                <b>Module A (Hour 0):</b> Eliminates statistical maverick outliers using Median Absolute Deviation (MAD > 3.5), 
                catching parts that sneak past the standard 50 µA static limit.
            </div>
        </div>
        """, unsafe_allow_html=True)
        fig_a = render_module_a_chart(processed_df, height=380)
        st.plotly_chart(fig_a, use_container_width=True)

    with vis_col2:
        st.markdown("""
        <div class="clean-container" style="padding: 14px;">
            <div style="font-size: 0.85rem; color: #475569; margin-bottom: 8px;">
                <b>Module B (Hour 24):</b> Calibrated GPR predicts Day-7 degradation. 
                Nominal chips stay safe at ~11 µA while latent duds curve past the 30 µA ceiling.
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Trajectory chart for IC_0006 vs nominal
        fig_b = render_module_b_chart(processed_df, selected_chip_id="IC_0006", height=380)
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # SECTION 3: SINGLE COMPONENT INSPECTOR & PLAIN ENGLISH EXPLANATION
    # -----------------------------------------------------------------
    st.markdown("### 🔎 Section 3: Single Component Inspector & Human-Readable Explanation")

    all_chips = list(processed_df["chip_id"].values)
    default_ic_idx = all_chips.index("IC_0006") if "IC_0006" in all_chips else 0

    sel_col1, sel_col2 = st.columns([1.5, 2.5])
    with sel_col1:
        inspected_chip_id = st.selectbox(
            "Select Component ID to Audit:",
            options=all_chips,
            index=default_ic_idx,
            help="Select any chip to inspect its individual telemetry, GPR trajectory, and SHAP forces."
        )

    render_component_inspector(
        chip_id=inspected_chip_id,
        df=processed_df,
        xai_engine=xai_engine
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # SECTION 4: EXPORT DELIVERABLES
    # -----------------------------------------------------------------
    st.markdown("### 📄 Section 4: Export Deliverables")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("""
        <div class="clean-container">
            <div style="font-weight: 700; color: #0F172A; font-size: 1rem; margin-bottom: 4px;">
                📜 Mission-Ready PDF Compliance Audit Report
            </div>
            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 14px;">
                Conforming to MIL-STD-883 Method 1015 & AEC-Q001 standards, including QA digital signatures.
            </div>
        </div>
        """, unsafe_allow_html=True)

        pdf_bytes = generate_compliance_pdf(
            df=processed_df,
            kpis=kpi_metrics,
            lot_id=lot_name
        )

        st.download_button(
            label="📥 Download Official PDF Compliance Report",
            data=pdf_bytes,
            file_name="ASTRA_RELIABILITY_AUDIT_REPORT.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with exp_col2:
        st.markdown("""
        <div class="clean-container">
            <div style="font-weight: 700; color: #0F172A; font-size: 1rem; margin-bottom: 4px;">
                📊 Complete Screened Lot Telemetry Data (CSV)
            </div>
            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 14px;">
                Includes baseline measurements, Dynamic PAT modified Z-scores, GPR predictions, and abort flags.
            </div>
        </div>
        """, unsafe_allow_html=True)

        csv_bytes = generate_lot_csv(processed_df)

        st.download_button(
            label="📊 Download Processed CSV Data",
            data=csv_bytes,
            file_name="ASTRA_PROCESSED_LOT_TELEMETRY.csv",
            mime="text/csv",
            use_container_width=True
        )
