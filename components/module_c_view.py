"""
ASTRA-IC Clean Light Theme Module C: Component Inspector & Explainable AI
Plain-English triage justification and clean horizontal SHAP feature attribution bar chart.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any
from core.xai import AstraXAIEngine
from core.engine import CRITICAL_THRESHOLD


def render_component_inspector(
    chip_id: str,
    df: pd.DataFrame,
    xai_engine: AstraXAIEngine
) -> None:
    """
    Renders Section 3: Single Component Inspector & Plain-English Explanation.
    Includes status banner, human-readable rationale, and clean light-themed SHAP bar chart.
    """
    chip_row = df[df["chip_id"] == chip_id].iloc[0]
    
    # Calculate or retrieve SHAP explanation
    xai_result = xai_engine.explain_component(chip_row=chip_row, lot_df=df)
    contributions = xai_result["contributions"]
    pred_168h = xai_result["pred_168h"]
    upper_3sigma = xai_result["upper_3sigma"]
    early_abort = xai_result["early_abort"]
    top_driver = contributions[0]

    # 1. Status Banner
    if early_abort:
        st.markdown(f"""
        <div class="status-banner-alert">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 1.15rem; font-weight: 800; color: #DC2626;">
                    🚨 CRITICAL REJECT / EARLY ABORT AT 24h
                </div>
                <span class="standard-badge badge-red" style="font-size: 0.8rem; padding: 6px 12px;">
                    Upper 3σ: {upper_3sigma:.2f} µA (Exceeds {CRITICAL_THRESHOLD:.0f} µA Ceiling)
                </span>
            </div>
            <div class="explanation-box" style="border-left-color: #DC2626;">
                <b>Why rejected?</b> At 24 hours, its degradation velocity (<code>{top_driver['feature']}</code>) 
                contributed <b>{top_driver['shap_value']:+.2f} µA</b> toward the projected Day-7 failure of <b>{pred_168h:.2f} µA</b>. 
                Halting burn-in chamber testing at Hour 24 saves <b>144 testing chamber hours</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-banner-nominal">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 1.15rem; font-weight: 800; color: #16A34A;">
                    ✅ QUALIFIED FOR FLIGHT
                </div>
                <span class="standard-badge badge-green" style="font-size: 0.8rem; padding: 6px 12px;">
                    Upper 3σ: {upper_3sigma:.2f} µA (Safely below {CRITICAL_THRESHOLD:.0f} µA)
                </span>
            </div>
            <div class="explanation-box" style="border-left-color: #16A34A;">
                <b>Flight Ready:</b> Conforms to standard logarithmic Arrhenius drift. 
                Projected Day-7 leakage of <b>{pred_168h:.2f} µA</b> is well within the {CRITICAL_THRESHOLD:.0f} µA safety ceiling.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Clean Horizontal SHAP Bar Chart
    st.markdown("##### 🔬 Feature Attribution Impact Breakdown (SHAP Values)")

    feat_labels = [c["display_name"].split(" (")[0] for c in contributions][::-1]
    shap_vals = [c["shap_value"] for c in contributions][::-1]
    bar_colors = ["#DC2626" if v > 0 else "#16A34A" for v in shap_vals]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=feat_labels,
        x=shap_vals,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="#CBD5E1", width=0.5)
        ),
        text=[f"{v:+.2f} µA" for v in shap_vals],
        textposition="auto",
        textfont=dict(color="#FFFFFF", family="JetBrains Mono", size=11),
        hovertemplate="<b>%{y}</b><br>SHAP Force: <b>%{x:+.2f} µA</b><extra></extra>"
    ))

    fig.add_vline(x=0, line_width=1.5, line_color="#94A3B8")

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#475569", family="Inter"),
        xaxis=dict(
            title="<b>Impact on 168h Failure Projection (µA)</b>",
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            linecolor="#CBD5E1"
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1"
        ),
        height=260,
        margin=dict(l=20, r=20, t=10, b=35)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_module_c_view(chip_id: str, df: pd.DataFrame, xai_engine: AstraXAIEngine) -> None:
    """Wrapper for standalone Module C view."""
    render_component_inspector(chip_id=chip_id, df=df, xai_engine=xai_engine)
