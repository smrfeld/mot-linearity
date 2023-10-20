from motstat.data import load_tracks, Tracks, DataSpec, Track, Box
from motstat.linear_analysis import LinSeg, LinSegs

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go


class Plotter:


    def __init__(self, fig: go.Figure):
        self.fig: go.Figure = fig
        self.default_layout()
    

    def default_layout(self):
        self.fig.update_layout(
            width=800,
            height=600,
            font=dict(size=24),
            xaxis=dict(
                title_text='x (Pixels)',
            ),
            yaxis=dict(
                title_text='y (Pixels)',
            ),
        )


    def add_line(self, x: List[float], y: List[float], col: str):
        trace = go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            name='lines+markers',
            marker=dict(color=col),
            line=dict(color=col, width=2),
            showlegend=False
            )
        self.fig.add_trace(trace)


    def add_box(self, box: Box, col: str):
        trace = go.Scatter(
            x=[box.xyxy[0], box.xyxy[2], box.xyxy[2], box.xyxy[0], box.xyxy[0]],
            y=[box.xyxy[1], box.xyxy[1], box.xyxy[3], box.xyxy[3], box.xyxy[1]],
            mode='lines',
            line=dict(color=col, width=1),
            showlegend=False
            )
        self.fig.add_trace(trace)


    def add_track(self, track: Track):
        self.add_box(track.boxes[0], col="gray")
        x = [box.xyxy[0] for box in track.boxes]
        y = [box.xyxy[1] for box in track.boxes]
        self.add_line(x, y, col="blue")
        x = [box.xyxy[2] for box in track.boxes]
        y = [box.xyxy[3] for box in track.boxes]
        self.add_line(x, y, col="blue")        
        self.add_box(track.boxes[-1], col="gray")


    def add_lin_segments(self, segments: LinSegs, track: Track):
        for seg in segments.segments:
            for i,j in [(0,1),(2,3)]:
                x = [track.boxes[x].xyxy[i] for x in range(seg.idx_start_incl, seg.idx_end_incl+1)]
                y = [track.boxes[x].xyxy[j] for x in range(seg.idx_start_incl, seg.idx_end_incl+1)]
                self.add_line(x, y, col="red")

                x = [track.boxes[seg.idx_start_incl].xyxy[i]]
                y = [track.boxes[seg.idx_start_incl].xyxy[j]]
                self.add_line(x, y, col="black")

                x = [track.boxes[seg.idx_end_incl].xyxy[i]]
                y = [track.boxes[seg.idx_end_incl].xyxy[j]]
                self.add_line(x, y, col="black")