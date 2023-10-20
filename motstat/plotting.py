from motstat.data import load_tracks, Tracks, DataSpec, Track

import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

def plot_line(fig: go.Figure, x: List[float], y: List[float], col: str):
    trace = go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        name='lines+markers',
        marker=dict(color=col),
        line=dict(color=col, width=4),
        showlegend=False
        )
    fig.add_trace(trace)

def plot_track(track: Track) -> go.Figure:
    fig = go.Figure()

    x = [box.xyxy[0] for box in track.boxes]
    y = [box.xyxy[1] for box in track.boxes]
    plot_line(fig, x, y, col="blue")
    x = [box.xyxy[2] for box in track.boxes]
    y = [box.xyxy[3] for box in track.boxes]
    plot_line(fig, x, y, col="blue")
    
    return fig
