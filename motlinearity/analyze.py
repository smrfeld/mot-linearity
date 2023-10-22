from motlinearity.data import Track, Tracks, FileToTracks
from motlinearity.lin_detection import find_linear_segments, LinTripletChecker


from typing import List, Dict, Tuple, Union
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class AveFracPerturb:
    mean: float
    std: float
    frac_list: List[float]

    @classmethod
    def from_list(cls, frac_list: List[float]):
        return cls(np.mean(frac_list, dtype=float), np.std(frac_list, dtype=float), frac_list)


def measure_ave_frac_perturb_all_files(file_to_tracks: FileToTracks, perturb_mag: float) -> AveFracPerturb:
    frac_list = []
    for fname,tracks in tqdm(file_to_tracks.items(), desc="Measuring linear stats for each file"):
        r = measure_ave_frac_perturb(tracks, perturb_mag)
        frac_list += r.frac_list
    return AveFracPerturb.from_list(frac_list)


def measure_ave_frac_perturb(tracks: Tracks, perturb_mag: float) -> AveFracPerturb:
    checker = LinTripletChecker(LinTripletChecker.Options(mode=LinTripletChecker.Options.Mode.TOL, perturb_mag=perturb_mag))

    frac_list = []
    for track_id,track in tracks.tracks.items():
        segments = find_linear_segments(track, checker)
        stats = segments.stats()
        frac_list.append(stats.frac_of_points_in_linear_segments)

    return AveFracPerturb.from_list(frac_list)


@dataclass
class TolToFrac:
    tol_to_frac_ave_std: Dict[float,Tuple[float,float]]
    tol_to_frac_list: Dict[float,List[float]]

    @classmethod
    def from_dict(cls, tol_to_frac_list: Dict[float,List[float]]):
        tol_to_frac_ave_std = {}
        for tol,frac_list in tol_to_frac_list.items():
            if len(frac_list) == 0:
                tol_to_frac_ave_std[tol] = (0,0)
            else:
                tol_to_frac_ave_std[tol] = (np.mean(frac_list, dtype=float), np.std(frac_list, dtype=float))
        return cls(tol_to_frac_ave_std, tol_to_frac_list)


def measure_tol_to_ave_frac_all_files(file_to_tracks: FileToTracks) -> TolToFrac:
    tol_to_frac_list: Dict[float,List[float]] = {}
    for fname,tracks in tqdm(file_to_tracks.items(), desc="Measuring linear stats for each file"):
        tf = measure_tol_to_ave_frac(tracks)
        for tol,fracs in tf.tol_to_frac_list.items():
            tol_to_frac_list.setdefault(tol, []).extend(fracs)

    return TolToFrac.from_dict(tol_to_frac_list)


def measure_tol_to_ave_frac(tracks: Tracks) -> TolToFrac:
    tol_to_frac_list: Dict[float,List[float]] = {}
    for track_id,track in tracks.tracks.items():
        tol_to_frac = measure_tol_to_frac_for_track(track)
        for tol,frac in tol_to_frac.items():
            tol_to_frac_list.setdefault(tol, []).append(frac)

    return TolToFrac.from_dict(tol_to_frac_list)


def measure_tol_to_frac_for_track(track: Track) -> Dict[float,float]:
    checker = LinTripletChecker(LinTripletChecker.Options(mode=LinTripletChecker.Options.Mode.TOL))

    tol_to_frac = {}
    for tol in [0, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0]:
        checker.options.tol = tol
        segments = find_linear_segments(track, checker)
        stats = segments.stats()
        tol_to_frac[tol] = stats.frac_of_points_in_linear_segments
    return tol_to_frac


def measure_lin_segments_duration_idxs_all_files(file_to_tracks: FileToTracks, tol: float) -> List[float]:
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