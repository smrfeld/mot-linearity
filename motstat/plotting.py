from motstat.data import load_tracks, Tracks, DataSpec, Track
from motstat.linear_analysis import LinSeg

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go


class Plotter:


    def __init__(self, fig: go.Figure):
        self.fig: go.Figure = fig
    

    def add_line(self, x: List[float], y: List[float], col: str):
        trace = go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            name='lines+markers',
            marker=dict(color=col),
            line=dict(color=col, width=4),
            showlegend=False
            )
        self.fig.add_trace(trace)


    def add_track(self, track: Track):
        x = [box.xyxy[0] for box in track.boxes]
        y = [box.xyxy[1] for box in track.boxes]
        self.add_line(x, y, col="blue")
        x = [box.xyxy[2] for box in track.boxes]
        y = [box.xyxy[3] for box in track.boxes]
        self.add_line(x, y, col="blue")        


    def add_lin_segments(self, segments: List[LinSeg], track: Track):
        for seg in segments:
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