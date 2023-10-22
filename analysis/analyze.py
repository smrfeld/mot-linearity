import motstat as ms
from motstat.plotting import PlotterTrajs, PlotterFrac, PlotterHist
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from typing import List
import json
import numpy as np
import os


def write_fig(fig: go.Figure, bname: str, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    fname = os.path.join(figures_dir, bname)
    fig.write_image(fname)
    print(f"Wrote to {fname}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mot-dir", type=str, help="MOT labels directory, e.g. MOT17Labels or MOT20Labels", required=True)
    parser.add_argument("--command", type=str, help="Command", required=True, choices=["plot-traj", "plot-traj-tog", "lin-analysis", "random-walk-sim", "random-walk-analysis"])
    parser.add_argument("--file", type=str, help="File to plot", required=False, default="MOT17-09-FRCNN")
    parser.add_argument("--track-ids", type=int, help="Track indexes to plot", required=False, nargs="+", default=[9,10,5])
    parser.add_argument("--tol", type=float, help="Tolerance", required=False, default=0.1)
    parser.add_argument("--show", action="store_true", help="Show plots")
    parser.add_argument("--disp-json", type=str, help="File name to write displacements to", required=False, default="displacements.json")
    parser.add_argument("--random-walk-json", type=str, help="File name to write random walk to", required=False, default="random_walk.json")
    parser.add_argument("--figures-dir", type=str, help="Directory to write figures to", required=False, default="figures")
    args = parser.parse_args()

    # Load the data
    file_to_tracks = ms.load_tracks(ms.DataSpec(
        mot=ms.DataSpec.Mot.MOT17,
        split=ms.DataSpec.Split.TRAIN,
        mode=ms.DataSpec.Mode.GT,
        mot17_method=ms.DataSpec.Mot17Method.FRCNN
        ))

    if args.command == "plot-traj-tog":

        assert args.file in file_to_tracks, f"File {args.file} not found in {args.mot_dir}"
        tracks = file_to_tracks[args.file]
        for track_id in args.track_ids:
            assert track_id in tracks.tracks, f"Track {track_id} not found in {args.file}"
            track = tracks.tracks[track_id]

            checker = ms.LinTripletChecker(ms.LinTripletChecker.Options(mode=ms.LinTripletChecker.Options.Mode.TOL, tol=args.tol))
            segments = ms.find_linear_segments(track, checker)

            fig = make_subplots(rows=1, cols=2)
            pt = PlotterTrajs(fig)
            pt.add_track(track, row=1, col=1)

            pt = PlotterTrajs(fig)
            pt.add_track(track, excl_markers_for_idxs=segments.idxs_in_lin_segments, row=1, col=2)
            pt.add_lin_segments(segments, track, row=1, col=2)

            fig.update_layout(
                title=f"Track {track_id} from {args.file} (slope difference tol={args.tol})",
                width=1600,
                )

            if args.show:
                fig.show()
            write_fig(fig, f"{args.file}_{track_id}_tog_tol_{args.tol:.2f}.png", args.figures_dir)

    elif args.command == "plot-traj":

        assert args.file in file_to_tracks, f"File {args.file} not found in {args.mot_dir}"
        tracks = file_to_tracks[args.file]
        for track_id in args.track_ids:
            assert track_id in tracks.tracks, f"Track {track_id} not found in {args.file}"
            track = tracks.tracks[track_id]

            checker = ms.LinTripletChecker(ms.LinTripletChecker.Options(mode=ms.LinTripletChecker.Options.Mode.TOL, tol=args.tol))
            segments = ms.find_linear_segments(track, checker)

            fig = go.Figure()
            pt = PlotterTrajs(fig)
            pt.add_track(track)
            fig.update_layout(
                title=f"Track {track_id} from {args.file}"
                )
            if args.show:
                fig.show()
            write_fig(fig, f"{args.file}_{track_id}.png", args.figures_dir)

            fig = go.Figure()
            pt = PlotterTrajs(fig)
            pt.add_track(track, excl_markers_for_idxs=segments.idxs_in_lin_segments)
            pt.add_lin_segments(segments, track)
            fig.update_layout(
                title=f"Track {track_id} from {args.file}<br>including linear segments (red, slope difference tol={args.tol})"
                )
            if args.show:
                fig.show()
            write_fig(fig, f"{args.file}_{track_id}_incl_lin_segments_tol_{args.tol:.2f}.png", args.figures_dir)

    elif args.command == "random-walk-sim":
        disps = ms.measure_bbox_coord_displacements(file_to_tracks)
        assert len(disps.xy_disps) > 0, "No displacements found"

        fig = make_subplots(rows=1, cols=2)
        ph = PlotterHist(fig)
        ph.add_hist([ xy[0] for xy in disps.xy_disps ], row=1, col=1)
        fig.update_xaxes(title_text="Displacements in x (pixels)", range=[-10,10], row=1, col=1)
        ph.add_hist([ xy[1] for xy in disps.xy_disps ], row=1, col=2)
        fig.update_xaxes(title_text="Displacements in y (pixels)", range=[-10,10], row=1, col=2)
        fig.update_layout(
            width=2000,
            title=f"Displacements of boxes between neighboring frames",
            )
        if args.show:
            fig.show()
        write_fig(fig, f"histogram_displacements.png", args.figures_dir)

        print(f"Mean displacement in x = {disps.xy_disp_mean[0]:.2f} +- {disps.xy_disp_std[0]:.2f} pixels")
        print(f"Mean displacement in y = {disps.xy_disp_mean[1]:.2f} +- {disps.xy_disp_std[1]:.2f} pixels")

        # Simulate random walk
        trajs = ms.sample_random_walk(no_trajs=100, no_pts_per_traj=100, disps_probs=disps.disps_probs)
        
        with open(args.random_walk_json, "w") as f:
            json.dump([t.to_dict() for t in trajs], f, indent=None)
            print(f"Wrote to {args.random_walk_json}")

    elif args.command == "random-walk-analysis":

        # Load displacements
        with open(args.random_walk_json, "r") as f:
            trajs = [ ms.RandomWalk.from_dict(d) for d in json.load(f) ]
            print(f"Loaded {len(trajs)} trajs from {args.random_walk_json}")

        res = ms.measure_lin_trajs(trajs, tol=args.tol)
        print(f"Ave fraction of linear points = {res.frac_of_pts_in_lin_segments_mean:.2f} +- {res.frac_of_pts_in_lin_segments_std:.2f} found by random walk simulation with slope difference tol={args.tol}")

        # Tolerance analysis
        print("---")
        tol_to_ave_frac = ms.measure_tol_to_ave_frac_all_files(file_to_tracks)
        print("Average fraction of points in linear segments by tolerance:")
        for tol,ave_frac in tol_to_ave_frac.items():
            print(f"\ttol={tol:.2f}, ave_frac={ave_frac:.2f}")

        fig = go.Figure()
        pf = PlotterFrac(fig)
        pf.add_tol_to_ave_frac(tol_to_ave_frac)
        if args.show:
            fig.show()
        write_fig(fig, f"tol_analysis.png", args.figures_dir)

    elif args.command == "lin-analysis":

        # Linear segments duration analysis
        lin_segments_duration_idxs = ms.measure_lin_segments_duration_idxs_all_files(file_to_tracks, tol=args.tol)
        mean = np.mean(lin_segments_duration_idxs, dtype=float)
        std = np.std(lin_segments_duration_idxs, dtype=float)
        print(f"Mean duration of linear segments = {mean:.2f} +- {std:.2f} frames")

        fig = go.Figure()
        ph = PlotterHist(fig)
        ph.add_hist(lin_segments_duration_idxs)
        fig.update_layout(
            title=f"Linear segments duration<br>(slope difference tol={args.tol})",
            )
        if args.show:
            fig.show()
        write_fig(fig, f"histogram_tol_{args.tol:.2f}.png", args.figures_dir)

        # Tolerance analysis
        print("---")
        tol_to_ave_frac = ms.measure_tol_to_ave_frac_all_files(file_to_tracks)
        print("Average fraction of points in linear segments by tolerance:")
        for tol,ave_frac in tol_to_ave_frac.items():
            print(f"\ttol={tol:.2f}, ave_frac={ave_frac:.2f}")

        fig = go.Figure()
        pf = PlotterFrac(fig)
        pf.add_tol_to_ave_frac(tol_to_ave_frac)
        if args.show:
            fig.show()
        write_fig(fig, f"tol_analysis.png", args.figures_dir)

        # Perturb analysis
        print("---")
        perturb_mag = 0.5
        frac_perturb_ave, frac_perturb_std = ms.measure_ave_frac_perturb_all_files(file_to_tracks, perturb_mag)
        print(f"Ave fraction of linear points = {frac_perturb_ave:.2f} +- {frac_perturb_std:.2f} found by perturbing with magnitude {perturb_mag}")

    else:
        raise NotImplementedError(f"Command {args.command} not implemented")