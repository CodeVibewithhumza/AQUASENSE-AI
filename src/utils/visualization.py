"""Visualization helper utilities and theme definitions for AquaSense AI."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
from src.utils.config import COLORS


def set_plot_style():
    """Applies the AquaSense dark navy scientific visual theme to Matplotlib and Seaborn."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg_primary'],
        'axes.facecolor': COLORS['bg_surface'],
        'axes.edgecolor': COLORS['border'],
        'axes.labelcolor': COLORS['text_primary'],
        'xtick.color': COLORS['text_muted'],
        'ytick.color': COLORS['text_muted'],
        'text.color': COLORS['text_primary'],
        'grid.color': COLORS['border'],
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.facecolor': COLORS['bg_surface'],
        'legend.edgecolor': COLORS['border'],
        'legend.fontsize': 10,
    })


def create_radar_chart(
    df: pd.DataFrame,
    categories: List[str] = None,
    models: List[str] = None
) -> go.Figure:
    """Creates a radar comparison chart using Plotly."""
    if categories is None:
        categories = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc']

    fig = go.Figure()
    palette = [COLORS['accent'], COLORS['safe'], '#FFAA00', '#FF4B4B', '#9D4EDD', '#00BBF9']

    if models is None:
        models = df.index.tolist()[:3]

    for idx, model_name in enumerate(models):
        if model_name in df.index:
            values = [df.loc[model_name, cat] for cat in categories if cat in df.columns]
            # Close the loop
            values.append(values[0])
            cats_closed = [c.upper() for c in categories] + [categories[0].upper()]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=cats_closed,
                fill='toself',
                name=model_name.replace('_', ' ').title(),
                line=dict(color=palette[idx % len(palette)], width=2),
                opacity=0.65
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                showticklabels=True,
                gridcolor=COLORS['border'],
                color=COLORS['text_muted'],
                tickfont=dict(size=9, color=COLORS['text_muted'])
            ),
            angularaxis=dict(
                gridcolor=COLORS['border'],
                color=COLORS['text_primary'],
                tickfont=dict(size=11, color=COLORS['text_primary'], family='sans-serif')
            ),
            bgcolor=COLORS['bg_surface']
        ),
        paper_bgcolor=COLORS['bg_primary'],
        font=dict(color=COLORS['text_primary']),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color=COLORS['text_primary'])
        ),
        margin=dict(l=40, r=40, t=30, b=40)
    )
    return fig


def plot_correlation_matrix_plotly(corr_matrix: pd.DataFrame) -> go.Figure:
    """Generates a styled Plotly heatmap for correlation matrices."""
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=[
            [0.0, COLORS['danger']],
            [0.5, COLORS['bg_surface']],
            [1.0, COLORS['accent']]
        ],
        zmin=-1.0,
        zmax=1.0
    )
    fig.update_layout(
        paper_bgcolor=COLORS['bg_primary'],
        plot_bgcolor=COLORS['bg_surface'],
        font=dict(color=COLORS['text_primary']),
        coloraxis_colorbar=dict(
            title=dict(text="Corr", font=dict(color=COLORS['text_primary'])),
            tickfont=dict(color=COLORS['text_muted'])
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig
