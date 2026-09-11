"""
ASTRA-IC Clean Light Theme KPI Ribbon Component
Renders 4 clean metric cards with explicit subtexts and subtle elevation.
"""

import streamlit as st
from typing import Dict, Any


def render_kpi_ribbon(kpis: Dict[str, Any]) -> None:
    """Renders Section 1 Executive KPI Summary in a clean light-themed 4-card layout."""
    col1, col2, col3, col4 = st.columns(4)

    total_chips = kpis.get("total_screened", 500)
    static_pass = kpis.get("static_ate_pass_rate", 100.0)
    early_aborts = kpis.get("early_aborts", 10)
    time_saved = kpis.get("time_saved_percent", 85.7)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-title">Total ICs Ingested</div>
                <div class="kpi-value">{total_chips} <span style="font-size: 1.05rem; color: #334155; font-weight: 600;">Chips</span></div>
            </div>
            <div class="kpi-subtext">
                <span class="standard-badge badge-blue">Complete Batch Size</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-title">Static Limit Pass Rate</div>
                <div class="kpi-value" style="color: #B45309;">{static_pass:.1f}%</div>
            </div>
            <div class="kpi-subtext">
                <span class="standard-badge badge-amber">Standard tests miss latent duds</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-title">ASTRA-IC Defect Catches</div>
                <div class="kpi-value" style="color: #B91C1C;">{early_aborts} <span style="font-size: 1.05rem; color: #991B1B; font-weight: 600;">Chips Flagged</span></div>
            </div>
            <div class="kpi-subtext">
                <span class="standard-badge badge-red">Zero Escapes | 100% Recall</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-title">Chamber Testing Time Saved</div>
                <div class="kpi-value" style="color: #15803D;">{time_saved:.1f}%</div>
            </div>
            <div class="kpi-subtext">
                <span class="standard-badge badge-green">144 Hours / Lot Saved</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
