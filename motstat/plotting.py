from motstat.data import TrackXyxy, Box
from motstat.data_lin import LinSegs

from typing import List, Dict, Optional
import plotly.graph_objects as go


class PlotterTrajs:


    def __init__(self, fig: go.Figure):
        self.fig: go.Figure = fig
        self.default_layout()
    

    def default_layout(self):
        self.fig.update_layout(
            width=1200,
            height=600,
            font=dict(size=24),
            xaxis=dict(
                title_text='x (Pixels)',
            ),
            yaxis=dict(
                title_text='y (Pixels)',
            ),
        )


    def add_markers(self, x: List[float], y: List[float], color: str, row: Optional[int] = None, col: Optional[int] = None):
        trace = go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(color=color),
            showlegend=False
            )
        self.fig.add_trace(trace, row=row, col=col)


    def add_line(self, x: List[float], y: List[float], color: str, row: Optional[int] = None, col: Optional[int] = None):
        trace = go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
            )
        self.fig.add_trace(trace, row=row, col=col)


    def add_box(self, box: Box, color: str, row: Optional[int] = None, col: Optional[int] = None):
        trace = go.Scatter(
            x=[box.xyxy[0], box.xyxy[2], box.xyxy[2], box.xyxy[0], box.xyxy[0]],
            y=[box.xyxy[1], box.xyxy[1], box.xyxy[3], box.xyxy[3], box.xyxy[1]],
            mode='lines',
            line=dict(color=color, width=1),
            showlegend=False
            )
        self.fig.add_trace(trace, row=row, col=col)


    def add_track(self, track: TrackXyxy, excl_markers_for_idxs: List[int] = [], row: Optional[int] = None, col: Optional[int] = None):
        self.add_box(track.boxes[0], color="gray", row=row, col=col)

        x = [box.xyxy[0] for box in track.boxes]
        y = [box.xyxy[1] for box in track.boxes]
        self.add_line(x, y, color="blue", row=row, col=col)
        x_excl = [x[i] for i in range(len(x)) if i not in excl_markers_for_idxs]
        y_excl = [y[i] for i in range(len(y)) if i not in excl_markers_for_idxs]
        self.add_markers(x_excl, y_excl, color="blue", row=row, col=col)

        x = [box.xyxy[2] for box in track.boxes]
        y = [box.xyxy[3] for box in track.boxes]
        self.add_line(x, y, color="blue", row=row, col=col)
        x_excl = [x[i] for i in range(len(x)) if i not in excl_markers_for_idxs]
        y_excl = [y[i] for i in range(len(y)) if i not in excl_markers_for_idxs]
        self.add_markers(x_excl, y_excl, color="blue", row=row, col=col)

        self.add_box(track.boxes[-1], color="gray", row=row, col=col)


    def add_lin_segments(self, segments: LinSegs, track: TrackXyxy, row: Optional[int] = None, col: Optional[int] = None):
        for seg in segments.segments:
            for i,j in [(0,1),(2,3)]:
                x = [track.boxes[x].xyxy[i] for x in range(seg.idx_start_incl, seg.idx_end_incl+1)]
                y = [track.boxes[x].xyxy[j] for x in range(seg.idx_start_incl, seg.idx_end_incl+1)]
                self.add_line(x, y, color="red", row=row, col=col)

                x = [track.boxes[seg.idx_start_incl].xyxy[i]]
                y = [track.boxes[seg.idx_start_incl].xyxy[j]]
                self.add_markers(x, y, color="red", row=row, col=col)

                x = [track.boxes[seg.idx_end_incl].xyxy[i]]
                y = [track.boxes[seg.idx_end_incl].xyxy[j]]
                self.add_markers(x, y, color="red", row=row, col=col)



class PlotterHist:


    def __init__(self, fig: go.Figure):
        self.fig: go.Figure = fig
        self.default_layout()
    

    def default_layout(self):
        self.fig.update_layout(
            width=800,
            height=600,
            font=dict(size=24),
            xaxis=dict(
                title_text='Linear segments duration (number of frames)',
                range=[0,10],
            ),
            yaxis=dict(
                title_text='Percentage',
            ),
        )


    def add_hist(self, data: List[float], row: Optional[int] = None, col: Optional[int] = None):
        trace = go.Histogram(x=data, xbins=dict(size=1), histnorm='percent')
        self.fig.add_trace(trace, row=row, col=col)


class PlotterFrac:


    def __init__(self, fig: go.Figure):
        self.fig: go.Figure = fig
        self.default_layout()
    

    def default_layout(self):
        self.fig.update_layout(
            title="Fraction of points in linear segments<br>measured across all GT tracks",
            xaxis_title="Tolerance between neighboring slopes",
            yaxis_title="Fraction of points in linear segments",
            width=800,
            height=600,
            font=dict(size=24),
            yaxis=dict(
                range=[0,1]
            )
        )


    def add_tol_to_ave_frac(self, tol_to_ave_frac: Dict[float,float]):
        self.fig.add_trace(go.Scatter(
            x=list(tol_to_ave_frac.keys()),
            y=list(tol_to_ave_frac.values()),
            mode="lines+markers",
            marker=dict(size=10),
            line=dict(width=2),
            ))