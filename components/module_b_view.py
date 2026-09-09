"""
ASTRA-IC Clean Light Theme Module B: 24h-to-168h Trajectory Forecaster
Clean Plotly trajectory curves illustrating early aborts vs nominal parts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.engine import CRITICAL_THRESHOLD


def render_module_b_chart(
    df: pd.DataFrame,
    selected_chip_id: str = "IC_0006",
    height: int = 400
) -> go.Figure:
    """
    Builds the clean light-theme Module B trajectory line chart.
    Illustrates nominal parts staying safe (~11 µA) while latent defective parts cross the 30 µA ceiling.
    """
    fig = go.Figure()

    # Find selected chip and nominal reference
    chip_row = df[df["chip_id"] == selected_chip_id].iloc[0]
    nominal_row = df[df.get("true_label", "") == "NOMINAL"].iloc[0] if "true_label" in df.columns else df.iloc[15]

    # Time steps
    time_obs = np.array([0, 24])
    time_pred = np.linspace(24, 168, 40)
    t_ratio = (np.log(1 + 0.05 * time_pred) - np.log(1 + 0.05 * 24)) / (np.log(1 + 0.05 * 168) - np.log(1 + 0.05 * 24))

    # Selected chip trajectory
    c_0h = float(chip_row["iddq_0h"])
    c_24h = float(chip_row["iddq_24h"])
    c_168h = float(chip_row.get("pred_168h", c_24h * 1.5))
    c_std = float(chip_row.get("pred_std", 0.49))
    c_upper_3s = float(chip_row.get("upper_3sigma", c_168h + 3 * c_std))
    is_abort = c_upper_3s >= CRITICAL_THRESHOLD

    c_pred_curve = c_24h + (c_168h - c_24h) * t_ratio
    c_upper_curve = c_pred_curve + 3 * c_std * ((time_pred - 24) / (168 - 24)) ** 0.65
    c_lower_curve = np.maximum(0, c_pred_curve - 3 * c_std * ((time_pred - 24) / (168 - 24)) ** 0.65)

    trace_color = "#DC2626" if is_abort else "#2563EB"
    fill_color = "rgba(220, 38, 38, 0.12)" if is_abort else "rgba(37, 99, 235, 0.10)"

    # Confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_pred, time_pred[::-1]]),
        y=np.concatenate([c_upper_curve, c_lower_curve[::-1]]),
        fill="toself",
        fillcolor=fill_color,
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name=f"±3σ Envelope ({selected_chip_id})"
    ))

    # Selected chip actual (0 -> 24h)
    fig.add_trace(go.Scatter(
        x=time_obs,
        y=[c_0h, c_24h],
        mode="lines+markers",
        name=f"{selected_chip_id} (Observed 0–24h)",
        line=dict(color=trace_color, width=3),
        marker=dict(size=7, color=trace_color)
    ))

    # Selected chip GPR forecast (24 -> 168h)
    fig.add_trace(go.Scatter(
        x=time_pred,
        y=c_pred_curve,
        mode="lines",
        name=f"{selected_chip_id} (Forecast 24–168h)",
        line=dict(color=trace_color, width=2.5, dash="dash")
    ))

    # Nominal Benchmark Trajectory (Green)
    nom_0h = float(nominal_row["iddq_0h"])
    nom_24h = float(nominal_row["iddq_24h"])
    nom_168h = float(nominal_row.get("pred_168h", nom_24h + 0.8))
    nom_pred_curve = nom_24h + (nom_168h - nom_24h) * t_ratio

    fig.add_trace(go.Scatter(
        x=[0, 24],
        y=[nom_0h, nom_24h],
        mode="lines+markers",
        name=f"Nominal Part ({nominal_row['chip_id']} 0–24h)",
        line=dict(color="#16A34A", width=2.2),
        marker=dict(size=6, color="#16A34A")
    ))

    fig.add_trace(go.Scatter(
        x=time_pred,
        y=nom_pred_curve,
        mode="lines",
        name=f"Nominal Part ({nominal_row['chip_id']} 24–168h)",
        line=dict(color="#16A34A", width=1.8, dash="dot")
    ))

    # 30µA Critical Failure Threshold
    fig.add_hline(
        y=CRITICAL_THRESHOLD,
        line_dash="dash",
        line_color="#DC2626",
        line_width=2,
        annotation_text=f"Critical Threshold ({CRITICAL_THRESHOLD:.0f} µA)",
        annotation_position="top left",
        annotation_font_color="#DC2626",
        annotation_font_size=10
    )

    fig.update_layout(
        title=dict(
            text=f"<b>24h–168h Trajectory Forecaster ({selected_chip_id} vs Nominal)</b>",
            font=dict(size=14, color="#0F172A", family="Inter")
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#475569", family="Inter"),
        xaxis=dict(
            title="<b>Burn-In Chamber Hours</b>",
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickvals=[0, 24, 48, 72, 96, 120, 144, 168]
        ),
        yaxis=dict(
            title="<b>Leakage Current Iddq (µA)</b>",
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            linecolor="#CBD5E1",
            range=[0, max(52.0, c_upper_3s + 5.0)]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9)
        ),
        height=height,
        margin=dict(l=40, r=30, t=50, b=40)
    )

    return fig


def render_module_b_view(df: pd.DataFrame, default_chip_id: str = "IC_0006") -> str:
    """Renders Module B standalone view."""
    fig = render_module_b_chart(df, selected_chip_id=default_chip_id, height=450)
    st.plotly_chart(fig, use_container_width=True)
    return default_chip_id
