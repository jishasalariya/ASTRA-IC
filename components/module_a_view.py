"""
ASTRA-IC Clean Light Theme Module A: Dynamic Part Average Testing (Hour 0)
Clean Plotly scatter chart complying with AEC-Q001 PAT standards.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.engine import STATIC_DATASHEET_LIMIT, AEC_Q001_Z_THRESHOLD


def render_module_a_chart(df: pd.DataFrame, height: int = 400) -> go.Figure:
    """Builds the clean light-theme Module A scatter chart."""
    lot_median = float(np.median(df["iddq_0h"]))
    flagged_df = df[df.get("pat_flagged", False)]
    nominal_df = df[~df.get("pat_flagged", False)]

    fig = go.Figure()

    # Nominal Parts (Blue circles)
    fig.add_trace(go.Scatter(
        x=nominal_df.index,
        y=nominal_df["iddq_0h"],
        mode="markers",
        name="Nominal Parts",
        marker=dict(
            color="#2563EB",
            size=6,
            opacity=0.75
        ),
        text=nominal_df["chip_id"],
        customdata=nominal_df.get("mod_z_score", np.zeros(len(nominal_df))),
        hovertemplate="<b>%{text}</b><br>0h Iddq: %{y:.2f} µA<br>Mod Z: %{customdata:.2f}<br>Status: NOMINAL<extra></extra>"
    ))

    # Outlier Parts (Crimson diamonds)
    if not flagged_df.empty:
        fig.add_trace(go.Scatter(
            x=flagged_df.index,
            y=flagged_df["iddq_0h"],
            mode="markers",
            name="PAT Outliers (M_i > 3.5)",
            marker=dict(
                color="#DC2626",
                size=10,
                symbol="diamond",
                line=dict(color="#991B1B", width=1)
            ),
            text=flagged_df["chip_id"],
            customdata=flagged_df.get("mod_z_score", np.zeros(len(flagged_df))),
            hovertemplate="<b>%{text}</b><br>0h Iddq: %{y:.2f} µA<br>Mod Z: %{customdata:.2f}<br>Status: 🚨 PAT OUTLIER<extra></extra>"
        ))

    # Static 50µA Datasheet Limit Line
    fig.add_hline(
        y=STATIC_DATASHEET_LIMIT,
        line_dash="dash",
        line_color="#DC2626",
        line_width=1.8,
        annotation_text="<b>Static Limit (50 µA)</b>",
        annotation_position="top right",
        annotation_font_color="#B91C1C",
        annotation_font_size=11
    )

    # Lot Median Line
    fig.add_hline(
        y=lot_median,
        line_dash="dot",
        line_color="#16A34A",
        line_width=1.8,
        annotation_text=f"<b>Lot Median ({lot_median:.1f} µA)</b>",
        annotation_position="bottom left",
        annotation_font_color="#15803D",
        annotation_font_size=11
    )

    fig.update_layout(
        title=dict(
            text="<b>Hour 0: Dynamic PAT Screening (AEC-Q001)</b>",
            font=dict(size=15, color="#0F172A", family="Inter")
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="Inter"),
        xaxis=dict(
            title=dict(text="<b>Chip Lot Index</b>", font=dict(color="#0F172A", size=12)),
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            linecolor="#94A3B8",
            tickfont=dict(color="#1E293B", size=11)
        ),
        yaxis=dict(
            title=dict(text="<b>0h Standby Current Iddq (µA)</b>", font=dict(color="#0F172A", size=12)),
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            linecolor="#94A3B8",
            tickfont=dict(color="#1E293B", size=11),
            range=[0, max(STATIC_DATASHEET_LIMIT + 5, df["iddq_0h"].max() + 5)]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#0F172A")
        ),
        height=height,
        margin=dict(l=40, r=30, t=50, b=40)
    )

    return fig


def render_module_a_view(df: pd.DataFrame, z_thresh: float = AEC_Q001_Z_THRESHOLD) -> None:
    """Renders Module A standalone view."""
    fig = render_module_a_chart(df, height=450)
    st.plotly_chart(fig, use_container_width=True)
