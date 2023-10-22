from motstat.data import Tracks, Track
from motstat.linear_triplet import LinTripletChecker
from motstat.linear_analysis import find_linear_segments
from motstat.random_walk import DispProb, RandomWalk


from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass


def measure_ave_frac_perturb(file_to_tracks: Dict[str,Tracks], perturb_mag: float) -> Tuple[float,float]:
    checker = LinTripletChecker(LinTripletChecker.Options(mode=LinTripletChecker.Options.Mode.TOL, perturb_mag=perturb_mag))

    frac_list = []
    for fname,tracks in tqdm(file_to_tracks.items(), desc="Measuring linear stats for each file"):
        for track_id,track in tracks.tracks.items():
            segments = find_linear_segments(track, checker)
            stats = segments.stats()
            frac_list.append(stats.frac_of_points_in_linear_segments)

    return np.mean(frac_list, dtype=float), np.std(frac_list, dtype=float)


def measure_tol_to_ave_frac(file_to_tracks: Dict[str,Tracks]) -> Dict[float,float]:
    tol_to_frac_list = {}
    for fname,tracks in tqdm(file_to_tracks.items(), desc="Measuring linear stats for each file"):
        for track_id,track in tracks.tracks.items():
            tol_to_frac = measure_tol_to_frac_for_track(track)
            for tol,frac in tol_to_frac.items():
                tol_to_frac_list.setdefault(tol, []).append(frac)

    tol_to_ave_frac = {}
    for tol,frac_list in tol_to_frac_list.items():
        tol_to_ave_frac[tol] = np.mean(frac_list, dtype=float) if len(frac_list) > 0 else 0.0
    return tol_to_ave_frac


def measure_tol_to_frac_for_track(track: Track) -> Dict[float,float]:
    checker = LinTripletChecker(LinTripletChecker.Options(mode=LinTripletChecker.Options.Mode.TOL))

    tol_to_frac = {}
    for tol in [0, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0]:
        checker.options.tol = tol
        segments = find_linear_segments(track, checker)
        stats = segments.stats()
        tol_to_frac[tol] = stats.frac_of_points_in_linear_segments
    return tol_to_frac


@dataclass
class BoxDisps:
    xy_disps: List[Tuple[float,float]]
    xy_disp_mean: Tuple[float,float]
    xy_disp_std: Tuple[float,float]
    disps_probs: List[DispProb]


def measure_bbox_coord_displacements(file_to_tracks: Dict[str,Tracks]) -> BoxDisps:
    assert len(file_to_tracks) > 0, "No files found"

    disps = BoxDisps([], (0,0), (0,0), [])
    for fname,tracks in tqdm(file_to_tracks.items(), desc="Measuring linear stats for each file"):
        for track_id,track in tracks.tracks.items():
            
            disps.xy_disps += [ (track.boxes[i+1].xyxy[0] - track.boxes[i].xyxy[0], track.boxes[i+1].xyxy[1] - track.boxes[i].xyxy[1]) for i in range(len(track.boxes)-1) ]
            disps.xy_disps += [ (track.boxes[i+1].xyxy[2] - track.boxes[i].xyxy[2], track.boxes[i+1].xyxy[3] - track.boxes[i].xyxy[3]) for i in range(len(track.boxes)-1) ]

    disps.xy_disp_mean = (np.mean([ xy_disp[0] for xy_disp in disps.xy_disps ], dtype=float), np.mean([ xy_disp[1] for xy_disp in disps.xy_disps ], dtype=float))
    disps.xy_disp_std = (np.std([ xy_disp[0] for xy_disp in disps.xy_disps ], dtype=float), np.std([ xy_disp[1] for xy_disp in disps.xy_disps ], dtype=float))

    # Measure dist
    for disp_x in range(-10,10):
        for disp_y in range(-10,10):
            disp_xmin = disp_x - 0.5
            disp_xmax = disp_x + 0.5
            disp_ymin = disp_y - 0.5
            disp_ymax = disp_y + 0.5
            disps.disps_probs.append(DispProb(
                disp_x=disp_x,
                disp_y=disp_y,
                prob=len([ xy for xy in disps.xy_disps if disp_xmin <= xy[0] < disp_xmax and disp_ymin <= xy[1] < disp_ymax ])
                ))
    tot = sum([ dp.prob for dp in disps.disps_probs ])
    for dp in disps.disps_probs:
        dp.prob /= tot

    return disps


def measure_lin_segments_duration_idxs(file_to_tracks: Dict[str,Tracks], tol: float) -> List[float]:
    lin_segments_duration_idxs = []
    for fname,tracks in tqdm(file_to_tracks.items(), desc="Measuring linear stats for each file"):
        for track_id,track in tracks.tracks.items():

            checker = LinTripletChecker(LinTripletChecker.Options(mode=LinTripletChecker.Options.Mode.TOL, tol=tol))
            segments = find_linear_segments(track, checker)
            stats = segments.stats()
            lin_segments_duration_idxs += stats.lin_segments_duration_idxs
    return lin_segments_duration_idxs


@dataclass
class LinTrajRes:
    frac_of_pts_in_lin_segments_mean: float
    frac_of_pts_in_lin_segments_std: float
    lin_seg_durations_idxs: List[float]


def measure_lin_trajs(trajs: List[RandomWalk], tol: float) -> LinTrajRes:
    checker = LinTripletChecker(LinTripletChecker.Options(mode=LinTripletChecker.Options.Mode.TOL, tol=tol))

    fracs_of_pts_in_lin_segments = []
    lin_seg_durations_idxs = []
    for pts in trajs:
        pts = [ [float(pt[0]), float(pt[1])] for pt in pts.traj ]
        linear_idxs = checker.find_linear_triplets(pts)
        linear_idxs_unique = sorted(list(set(linear_idxs)))
        frac_of_points_in_linear_segments = len(linear_idxs_unique) / len(pts)
        fracs_of_pts_in_lin_segments.append(frac_of_points_in_linear_segments)

        # Segments durations
        segments = checker.lin_idxs_to_segments(linear_idxs)
        lin_seg_durations_idxs += [seg.idx_end_incl - seg.idx_start_incl + 1 for seg in segments]
    
    frac_mean = np.mean(fracs_of_pts_in_lin_segments, dtype=float)
    frac_std = np.std(fracs_of_pts_in_lin_segments, dtype=float)

    return LinTrajRes(frac_mean, frac_std, lin_seg_durations_idxs)