import motstat as ms
from motstat.plotting import PlotterTrajs, PlotterFrac, PlotterLinSegmentsHist
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mot-dir", type=str, help="MOT labels directory, e.g. MOT17Labels or MOT20Labels", required=True)
    parser.add_argument("--command", type=str, help="Command", required=True, choices=["plot-traj", "plot-traj-tog", "hist-analysis", "tol-analysis", "perturb-analysis"])
    parser.add_argument("--file", type=str, help="File to plot", required=False, default="MOT17-09-FRCNN")
    parser.add_argument("--track-ids", type=int, help="Track indexes to plot", required=False, nargs="+", default=[9,10,5])
    parser.add_argument("--tol", type=float, help="Tolerance", required=False, default=0.1)
    parser.add_argument("--show", action="store_true", help="Show plots")
    args = parser.parse_args()

    # Load the data
    file_to_tracks = ms.load_tracks(ms.DataSpec(
        mot_dir=args.mot_dir,
        split=ms.DataSpec.Split.TRAIN,
        mode=ms.DataSpec.Mode.GT
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
            fname = f"{args.file}_{track_id}_tog_tol_{args.tol:.2f}.png"
            fig.write_image(fname)
            print(f"Wrote to {fname}")

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
            fname = f"{args.file}_{track_id}.png"
            fig.write_image(fname)
            print(f"Wrote to {fname}")

            fig = go.Figure()
            pt = PlotterTrajs(fig)
            pt.add_track(track, excl_markers_for_idxs=segments.idxs_in_lin_segments)
            pt.add_lin_segments(segments, track)
            fig.update_layout(
                title=f"Track {track_id} from {args.file}<br>including linear segments (red, slope difference tol={args.tol})"
                )
            if args.show:
                fig.show()
            fname = f"{args.file}_{track_id}_incl_lin_segments_tol_{args.tol:.2f}.png"
            fig.write_image(fname)
            print(f"Wrote to {fname}")

    elif args.command == "hist-analysis":

        lin_segments_duration_idxs = []
        for fname,tracks in tqdm(file_to_tracks.items(), desc="Measuring linear stats for each file"):
            for track_id,track in tracks.tracks.items():

                checker = ms.LinTripletChecker(ms.LinTripletChecker.Options(mode=ms.LinTripletChecker.Options.Mode.TOL, tol=args.tol))
                segments = ms.find_linear_segments(track, checker)
                stats = segments.stats()
                lin_segments_duration_idxs += stats.lin_segments_duration_idxs

        fig = go.Figure()
        ph = PlotterLinSegmentsHist(fig)
        ph.add_lin_segments_hist(lin_segments_duration_idxs)
        fig.update_layout(
            title=f"Linear segments duration<br>(slope difference tol={args.tol})",
            )
        if args.show:
            fig.show()
        fname = f"histogram_tol_{args.tol:.2f}.png"
        fig.write_image(fname)
        print(f"Wrote to {fname}")

    elif args.command == "tol-analysis":

        tol_to_ave_frac = ms.measure_tol_to_ave_frac(file_to_tracks)
        for tol,ave_frac in tol_to_ave_frac.items():
            print(f"tol={tol:.2f}, ave_frac={ave_frac:.2f}")
        fig = go.Figure()
        pf = PlotterFrac(fig)
        pf.add_tol_to_ave_frac(tol_to_ave_frac)
        if args.show:
            fig.show()

        fig.write_image(f"tol_analysis.png")

    elif args.command == "perturb-analysis":

        perturb_mag = 0.5
        ave_frac_perturb = ms.measure_ave_frac_perturb(file_to_tracks, perturb_mag)
        print(f"Ave fraction of linear points = {ave_frac_perturb:.2f} found by perturbing with magnitude {perturb_mag}")

    else:
        raise NotImplementedError(f"Command {args.command} not implemented")