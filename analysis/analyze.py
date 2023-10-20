import motstat as ms
from motstat.plotting import PlotterTrajs, PlotterFrac
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mot-dir", type=str, help="MOT17Labels directory", required=True)
    parser.add_argument("--command", type=str, help="Command", required=True, choices=["plot-traj", "tol-analysis", "perturb-analysis"])
    parser.add_argument("--file", type=str, help="File to plot", required=False, default="MOT17-09-FRCNN")
    parser.add_argument("--track-id", type=int, help="Track index to plot", required=False, default=9)
    parser.add_argument("--tol", type=float, help="Tolerance", required=False, default=0.1)
    args = parser.parse_args()

    # Load the data
    file_to_tracks = ms.load_tracks(ms.DataSpec(
        mot_dir=args.mot_dir,
        split=ms.DataSpec.Split.TRAIN,
        mode=ms.DataSpec.Mode.GT
        ))

    if args.command == "plot-traj":

        tracks = file_to_tracks[args.file]
        print(list(file_to_tracks.keys())[3])
        track = tracks.tracks[args.track_id]

        checker = ms.LinTripletChecker(ms.LinTripletChecker.Options(mode=ms.LinTripletChecker.Options.Mode.TOL, tol=args.tol))
        segments = ms.find_linear_segments(track, checker)

        fig = go.Figure()
        pt = PlotterTrajs(fig)
        pt.add_track(track)
        fig.update_layout(
            title=f"Track {args.track_id} from {args.file}"
            )
        fig.show()

        fig.write_image(f"{args.file}_{args.track_id}.png")

        fig = go.Figure()
        pt = PlotterTrajs(fig)
        pt.add_track(track, excl_markers_for_idxs=segments.idxs_in_lin_segments)
        pt.add_lin_segments(segments, track)
        fig.update_layout(
            title=f"Track {args.track_id} from {args.file} including linear segments (red)"
            )
        fig.show()

        fig.write_image(f"{args.file}_{args.track_id}_incl_lin_segments.png")
    
    elif args.command == "tol-analysis":

        tol_to_ave_frac = ms.measure_tol_to_ave_frac(file_to_tracks)
        fig = go.Figure()
        pf = PlotterFrac(fig)
        pf.add_tol_to_ave_frac(tol_to_ave_frac)
        fig.show()

        fig.write_image(f"tol_analysis.png")

    elif args.command == "perturb-analysis":

        perturb_mag = 0.5
        ave_frac_perturb = ms.measure_ave_frac_perturb(file_to_tracks, perturb_mag)
        print(f"Ave fraction of linear points = {ave_frac_perturb:.2f} found by perturbing with magnitude {perturb_mag}")