from motstat.data import Tracks, Track
from motstat.linear_triplet import LinTripletChecker
from motstat.linear_analysis import find_linear_segments


from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

def measure_ave_frac_perturb(file_to_tracks: Dict[str,Tracks], perturb_mag: float) -> float:
    checker = LinTripletChecker(LinTripletChecker.Options(mode=LinTripletChecker.Options.Mode.TOL, perturb_mag=perturb_mag))

    frac_list = []
    for fname,tracks in tqdm(file_to_tracks.items(), desc="Measuring linear stats for each file"):
        for track_id,track in tracks.tracks.items():
            segments = find_linear_segments(track, checker)
            stats = segments.stats()
            frac_list.append(stats.frac_of_points_in_linear_segments)

    return np.mean(frac_list, dtype=float)

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