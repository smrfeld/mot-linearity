import motstat as ms
from motstat.plotting import Plotter
import plotly.graph_objects as go

# Load the data
file_to_tracks = ms.load_tracks(ms.DataSpec(
    mot_dir="MOT17Labels",
    split=ms.DataSpec.Split.TRAIN,
    mode=ms.DataSpec.Mode.GT
    ))

# Plot lines for every track
# for tracks in file_to_tracks.values():
#     for track_id,track in tracks.tracks.items():
tracks = list(file_to_tracks.values())[3]
track_id,track = list(tracks.tracks.items())[4]

for tol in [0, 0.05, 0.1, 0.2]:
    segments = ms.find_linear_segments(track, tol=tol)

    stats = segments.stats()
    stats.report()

# Plot
fig = go.Figure()
pt = Plotter(fig)
pt.add_track(track)
pt.add_lin_segments(segments, track)
fig.show()