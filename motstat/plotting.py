from motstat.data import TrackXyxy, Entry, Track, TrackXy
from motstat.data_lin import LinSegs

from typing import List, Dict, Optional, Union
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


    def add_markers(self, x: Union[List[float],List[int]], y: Union[List[float],List[int]], color: str, row: Optional[int] = None, col: Optional[int] = None):
        trace = go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(color=color),
            showlegend=False
            )
        self.fig.add_trace(trace, row=row, col=col)


    def add_line(self, x: Union[List[float],List[int]], y: Union[List[float],List[int]], color: str, row: Optional[int] = None, col: Optional[int] = None):
        trace = go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
            )
        self.fig.add_trace(trace, row=row, col=col)


    def add_box(self, box: Entry, color: str, row: Optional[int] = None, col: Optional[int] = None):
        trace = go.Scatter(
            x=[box.data[0], box.data[2], box.data[2], box.data[0], box.data[0]],
            y=[box.data[1], box.data[1], box.data[3], box.data[3], box.data[1]],
            mode='lines',
            line=dict(color=color, width=1),
            showlegend=False
            )
        self.fig.add_trace(trace, row=row, col=col)


    def add_track(self, track: Track, excl_markers_for_idxs: List[int] = [], row: Optional[int] = None, col: Optional[int] = None):
        if type(track) == TrackXyxy:
            self.add_track_xyxy(track, excl_markers_for_idxs, row, col)
        elif type(track) == TrackXy:
            self.add_track_xy(track, excl_markers_for_idxs, row, col)
        else:
            raise ValueError("Unknown track type: {}".format(type(track)))


    def add_track_xy(self, track: TrackXy, excl_markers_for_idxs: List[int] = [], row: Optional[int] = None, col: Optional[int] = None):

        x = [xy.data[0] for xy in track.entries]
        y = [xy.data[1] for xy in track.entries]
        self.add_line(x, y, color="blue", row=row, col=col)
        x_excl = [x[i] for i in range(len(x)) if i not in excl_markers_for_idxs]
        y_excl = [y[i] for i in range(len(y)) if i not in excl_markers_for_idxs]
        self.add_markers(x_excl, y_excl, color="blue", row=row, col=col)


    def add_track_xyxy(self, track: TrackXyxy, excl_markers_for_idxs: List[int] = [], row: Optional[int] = None, col: Optional[int] = None):
        self.add_box(track.entries[0], color="gray", row=row, col=col)

        x = [box.data[0] for box in track.entries]
        y = [box.data[1] for box in track.entries]
        self.add_line(x, y, color="blue", row=row, col=col)
        x_excl = [x[i] for i in range(len(x)) if i not in excl_markers_for_idxs]
        y_excl = [y[i] for i in range(len(y)) if i not in excl_markers_for_idxs]
        self.add_markers(x_excl, y_excl, color="blue", row=row, col=col)

        x = [box.data[2] for box in track.entries]
        y = [box.data[3] for box in track.entries]
        self.add_line(x, y, color="blue", row=row, col=col)
        x_excl = [x[i] for i in range(len(x)) if i not in excl_markers_for_idxs]
        y_excl = [y[i] for i in range(len(y)) if i not in excl_markers_for_idxs]
        self.add_markers(x_excl, y_excl, color="blue", row=row, col=col)

        self.add_box(track.entries[-1], color="gray", row=row, col=col)


    def add_lin_segments(self, segments: LinSegs, track: Track, row: Optional[int] = None, col: Optional[int] = None):
        for seg in segments.segments:

            if type(track) == TrackXyxy:
                ijs = [(0,1),(2,3)]
            elif type(track) == TrackXy:
                ijs = [(0,1)]
            else:
                raise ValueError("Unknown track type: {}".format(type(track)))
            
            for i,j in ijs:
                x = [track.entries[x].data[i] for x in range(seg.idx_start_incl, seg.idx_end_incl+1)]
                y = [track.entries[x].data[j] for x in range(seg.idx_start_incl, seg.idx_end_incl+1)]
                self.add_line(x, y, color="red", row=row, col=col)

                x = [track.entries[seg.idx_start_incl].data[i]]
                y = [track.entries[seg.idx_start_incl].data[j]]
                self.add_markers(x, y, color="red", row=row, col=col)

                x = [track.entries[seg.idx_end_incl].data[i]]
                y = [track.entries[seg.idx_end_incl].data[j]]
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
            width=1000,
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