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


def linear_analysis(file_to_tracks: ms.FileToTracks, tol: float, show: bool, figures_dir: str, figures_tag: str):

    # Linear segments duration analysis
    lin_segments_duration_idxs = ms.measure_lin_segments_duration_idxs_all_files(file_to_tracks, tol=tol)
    mean = np.mean(lin_segments_duration_idxs, dtype=float)
    std = np.std(lin_segments_duration_idxs, dtype=float)
    print(f"Mean duration of linear segments = {mean:.2f} +- {std:.2f} frames")

    fig = go.Figure()
    ph = PlotterHist(fig)
    ph.add_hist(lin_segments_duration_idxs)
    fig.update_layout(
        title=f"Linear segments duration ({figures_tag})<br>(slope difference tol={tol})",
        )
    if show:
        fig.show()
    write_fig(fig, f"histogram_tol_{tol:.2f}_{figures_tag.replace(' ','_')}.png", figures_dir)

    # Tolerance analysis
    print("---")
    tol_to_frac_ave_std = ms.measure_tol_to_ave_frac_all_files(file_to_tracks).tol_to_frac_ave_std
    print("Average fraction of points in linear segments by tolerance:")
    for tol,(ave_frac,std_frac) in tol_to_frac_ave_std.items():
        print(f"\ttol={tol:.2f}, ave_frac={ave_frac:.2f} +- {std_frac:.2f}")

    fig = go.Figure()
    pf = PlotterFrac(fig)
    pf.add_tol_to_ave_frac({ tol: ave_frac for tol,(ave_frac,std_frac) in tol_to_frac_ave_std.items() })
    if show:
        fig.show()
    fig.update_layout(
        title=f"Fraction of points in linear segments ({figures_tag})<br>measured across all GT tracks",
        )
    write_fig(fig, f"tol_analysis_{figures_tag.replace(' ','_')}.png", figures_dir)

    # Perturb analysis
    print("---")
    perturb_mag = 0.5
    perturb = ms.measure_ave_frac_perturb_all_files(file_to_tracks, perturb_mag)
    print(f"Ave fraction of linear points = {perturb.mean:.2f} +- {perturb.std:.2f} found by perturbing with magnitude {perturb_mag}")


def plot_traj_tog(track_ids: List[int], tracks: ms.Tracks, tol: float, src_str: str, show: bool, figures_dir: str):
    for track_id in track_ids:
        assert track_id in tracks.tracks, f"Track {track_id} not found"
        track = tracks.tracks[track_id]

        checker = ms.LinTripletChecker(ms.LinTripletChecker.Options(mode=ms.LinTripletChecker.Options.Mode.TOL, tol=tol))
        segments = ms.find_linear_segments(track, checker)

        fig = make_subplots(rows=1, cols=2)
        pt = PlotterTrajs(fig)
        pt.add_track(track, row=1, col=1)

        pt = PlotterTrajs(fig)
        pt.add_track(track, excl_markers_for_idxs=segments.idxs_in_lin_segments, row=1, col=2)
        pt.add_lin_segments(segments, track, row=1, col=2)

        fig.update_layout(
            title=f"Track {track_id} from {src_str} (slope difference tol={tol})",
            width=1600,
            )

        if show:
            fig.show()
        write_fig(fig, f"{src_str.replace(' ','_')}_{track_id}_tog_tol_{tol:.2f}.png", figures_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mot", type=str, help="MOT", required=True, choices=[ms.DataSpec.Mot.MOT17.value, ms.DataSpec.Mot.MOT20.value])
    parser.add_argument("--command", type=str, help="Command", required=True, choices=["plot-traj", "plot-traj-tog", "lin-analysis", "random-walk-sim", "random-walk-analysis", "plot-traj-tog-random-walk"])
    parser.add_argument("--file", type=str, help="File to plot", required=False, default="MOT17-09-FRCNN")
    parser.add_argument("--track-ids", type=int, help="Track indexes to plot", required=False, nargs="+", default=[9,10,5])
    parser.add_argument("--tol", type=float, help="Tolerance", required=False, default=0.1)
    parser.add_argument("--show", action="store_true", help="Show plots")
    parser.add_argument("--disp-json", type=str, help="File name to write displacements to", required=False, default="displacements.json")
    parser.add_argument("--random-walk-json", type=str, help="File name to write random walk to", required=False, default="random_walk.json")
    parser.add_argument("--figures-dir", type=str, help="Directory to write figures to", required=False, default="figures")
    args = parser.parse_args()

    # Load the data
    mot_file_to_tracks = ms.load_tracks(ms.DataSpec(
        mot=ms.DataSpec.Mot(args.mot),
        split=ms.DataSpec.Split.TRAIN,
        mode=ms.DataSpec.Mode.GT,
        mot17_method=ms.DataSpec.Mot17Method.FRCNN
        ))

    if args.command == "plot-traj-tog":

        assert args.file in mot_file_to_tracks, f"File {args.file} not found in {args.mot_dir}"
        tracks = mot_file_to_tracks[args.file]
        plot_traj_tog(track_ids=args.track_ids, tracks=tracks, tol=args.tol, src_str=args.file, show=args.show, figures_dir=args.figures_dir)

    elif args.command == "plot-traj-tog-random-walk":

        assert os.path.exists(args.random_walk_json), f"File {args.random_walk_json} not found - run random-walk-sim first"
        with open(args.random_walk_json, "r") as f:
            tracks = ms.TracksXy.from_dict(json.load(f))
            print(f"Loaded {len(tracks.tracks)} trajs from {args.random_walk_json}")

        plot_traj_tog(track_ids=[0,1,2,3], tracks=tracks, tol=args.tol, src_str="random walk", show=args.show, figures_dir=args.figures_dir)

    elif args.command == "plot-traj":

        assert args.file in mot_file_to_tracks, f"File {args.file} not found in {args.mot_dir}"
        tracks = mot_file_to_tracks[args.file]
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
        disps = ms.measure_bbox_coord_displacements(mot_file_to_tracks)
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
        tracks = ms.sample_random_walk(no_trajs=100, no_pts_per_traj=100, disps_probs=disps.disps_probs)
        
        with open(args.random_walk_json, "w") as f:
            json.dump(tracks.to_dict(), f, indent=None)
            print(f"Wrote to {args.random_walk_json}")

    elif args.command == "random-walk-analysis":

        # Load displacements
        assert os.path.exists(args.random_walk_json), f"File {args.random_walk_json} not found - run random-walk-sim first"
        with open(args.random_walk_json, "r") as f:
            tracks = ms.TracksXy.from_dict(json.load(f))
            print(f"Loaded {len(tracks.tracks)} trajs from {args.random_walk_json}")

        linear_analysis({ "random_walk": tracks }, tol=args.tol, show=args.show, figures_dir=args.figures_dir, figures_tag="Random Walk")

    elif args.command == "lin-analysis":
        linear_analysis(mot_file_to_tracks, tol=args.tol, show=args.show, figures_dir=args.figures_dir, figures_tag=args.mot)

    else:
        raise NotImplementedError(f"Command {args.command} not implemented")