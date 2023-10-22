from motstat.data import Track, Tracks
from motstat.lin_detection import find_linear_segments, LinTripletChecker


from typing import List, Dict, Tuple, Union
import numpy as np
from tqdm import tqdm


def measure_ave_frac_perturb_all_files(file_to_tracks: Dict[str,Tracks], perturb_mag: float) -> Tuple[float,float,List[float]]:
    frac_list = []
    for fname,tracks in tqdm(file_to_tracks.items(), desc="Measuring linear stats for each file"):
        _,_,frac_list_for_file = measure_ave_frac_perturb(tracks, perturb_mag)
        frac_list += frac_list_for_file

    return np.mean(frac_list, dtype=float), np.std(frac_list, dtype=float), frac_list


def measure_ave_frac_perturb(tracks: Tracks, perturb_mag: float) -> Tuple[float,float, List[float]]:
    checker = LinTripletChecker(LinTripletChecker.Options(mode=LinTripletChecker.Options.Mode.TOL, perturb_mag=perturb_mag))

    frac_list = []
    for track_id,track in tracks.tracks.items():
        segments = find_linear_segments(track, checker)
        stats = segments.stats()
        frac_list.append(stats.frac_of_points_in_linear_segments)

    return np.mean(frac_list, dtype=float), np.std(frac_list, dtype=float), frac_list


def measure_tol_to_ave_frac_all_files(file_to_tracks: Dict[str,Tracks]) -> Tuple[Dict[float,float],Dict[float,List[float]]]:
    tol_to_frac_list: Dict[float,List[float]] = {}
    for fname,tracks in tqdm(file_to_tracks.items(), desc="Measuring linear stats for each file"):
        _, tol_to_frac = measure_tol_to_ave_frac(tracks)
        for tol,fracs in tol_to_frac.items():
            tol_to_frac_list.setdefault(tol, []).extend(fracs)

    tol_to_ave_frac = {}
    for tol,frac_list in tol_to_frac_list.items():
        tol_to_ave_frac[tol] = np.mean(frac_list, dtype=float) if len(frac_list) > 0 else 0.0
    return tol_to_ave_frac, tol_to_frac_list


def measure_tol_to_ave_frac(tracks: Tracks) -> Tuple[Dict[float,float],Dict[float,List[float]]]:
    tol_to_frac_list: Dict[float,List[float]] = {}
    for track_id,track in tracks.tracks.items():
        tol_to_frac = measure_tol_to_frac_for_track(track)
        for tol,frac in tol_to_frac.items():
            tol_to_frac_list.setdefault(tol, []).append(frac)

    tol_to_ave_frac = {}
    for tol,frac_list in tol_to_frac_list.items():
        tol_to_ave_frac[tol] = np.mean(frac_list, dtype=float) if len(frac_list) > 0 else 0.0
    return tol_to_ave_frac, tol_to_frac_list


def measure_tol_to_frac_for_track(track: Track) -> Dict[float,float]:
    checker = LinTripletChecker(LinTripletChecker.Options(mode=LinTripletChecker.Options.Mode.TOL))

    tol_to_frac = {}
    for tol in [0, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0]:
        checker.options.tol = tol
        segments = find_linear_segments(track, checker)
        stats = segments.stats()
        tol_to_frac[tol] = stats.frac_of_points_in_linear_segments
    return tol_to_frac


def measure_lin_segments_duration_idxs_all_files(file_to_tracks: Dict[str,Tracks], tol: float) -> List[float]:
    lin_segments_duration_idxs = []
    for fname,tracks in tqdm(file_to_tracks.items(), desc="Measuring linear stats for each file"):
        lin_segments_duration_idxs += measure_lin_segments_duration_idxs(tracks, tol)
    return lin_segments_duration_idxs


def measure_lin_segments_duration_idxs(tracks: Tracks, tol: float) -> List[float]:
    lin_segments_duration_idxs = []
    for track_id,track in tracks.tracks.items():

        checker = LinTripletChecker(LinTripletChecker.Options(mode=LinTripletChecker.Options.Mode.TOL, tol=tol))
        segments = find_linear_segments(track, checker)
        stats = segments.stats()
        lin_segments_duration_idxs += stats.lin_segments_duration_idxs
    return lin_segments_duration_idxs