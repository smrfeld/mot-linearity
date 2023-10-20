from motstat.data import load_tracks, Tracks, DataSpec, Track, length_boxes_center
from motstat.linear_triplet import LinTripletChecker, LinTriplet


import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import numpy as np


def find_linear_triplets(track: Track, checker: LinTripletChecker) -> List[int]:
    
    idx_linear = []
    for i in range(1,len(track.boxes)-1):
        xyxy1 = track.boxes[i-1].xyxy
        xyxy2 = track.boxes[i].xyxy
        xyxy3 = track.boxes[i+1].xyxy

        # Check if in a line
        xy_bl_1 = [xyxy1[0], xyxy1[1]]
        xy_bl_2 = [xyxy2[0], xyxy2[1]]
        xy_bl_3 = [xyxy3[0], xyxy3[1]]
        t_bl = checker.check_if_triplet_in_line(xy_bl_1, xy_bl_2, xy_bl_3)

        xy_tr_1 = [xyxy1[2], xyxy1[3]]
        xy_tr_2 = [xyxy2[2], xyxy2[3]]
        xy_tr_3 = [xyxy3[2], xyxy3[3]]
        t_tr = checker.check_if_triplet_in_line(xy_tr_1, xy_tr_2, xy_tr_3)

        # logger.debug(f"Slopes: {t_bl.m12} {t_bl.m23} {t_bl.is_linear} {t_tr.m12} {t_tr.m23} {t_tr.is_linear}")

        if not t_tr.is_linear or not t_bl.is_linear:
            continue
        
        # In a line
        idx_linear.append(i)

    return idx_linear


@dataclass
class LinSeg:
    idx_start_incl: int
    idx_end_incl: int


@dataclass
class LinStats:
    track_id: int
    no_lin_segments: int

    no_points_in_lin_segments: int
    no_points_in_track: int
    frac_of_points_in_linear_segments: float
    
    track_duration_idxs: int
    track_length_pixels: float

    lin_segments_mean_duration_idxs: float
    lin_segments_std_duration_idxs: float
    
    lin_segments_mean_length_pixels: float
    lin_segments_std_length_pixels: float

    def report(self):
        logger.info(f"Track {self.track_id} has {self.no_lin_segments} linear segments:")

        logger.info(f"  No points in linear segments: {self.no_points_in_lin_segments}")
        logger.info(f"  No points in track: {self.no_points_in_track}")
        logger.info(f"  Frac of points in linear segments: {self.frac_of_points_in_linear_segments:.2f}")

        logger.info(f"  Ave duration of linear segments: {self.lin_segments_mean_duration_idxs:.2f} +- {self.lin_segments_std_duration_idxs:.2f} (idxs) - track duration: {self.track_duration_idxs} (idxs)")
        logger.info(f"  Ave length of linear segments: {self.lin_segments_mean_length_pixels:.2f} +- {self.lin_segments_std_length_pixels:.2f} (pixels) - track length: {self.track_length_pixels:.2f} (pixels)")

@dataclass
class LinSegs:
    segments: List[LinSeg]
    track: Track

    @property
    def idxs_in_lin_segments(self) -> List[int]:
        return sorted(list(set([ idx for seg in self.segments for idx in range(seg.idx_start_incl, seg.idx_end_incl+1) ])))

    def stats(self) -> LinStats:        
        no_points_in_linear_segments = len(self.idxs_in_lin_segments)
        no_points_in_track = len(self.track.boxes)
        frac_of_points_in_linear_segments = no_points_in_linear_segments / no_points_in_track if no_points_in_track > 0 else 0

        track_duration_idxs = self.track.boxes[-1].frame_id - self.track.boxes[0].frame_id + 1
        track_length_pixels = self.track.length_pixels_center()

        lin_segments_duration_idxs = [seg.idx_end_incl - seg.idx_start_incl + 1 for seg in self.segments]
        lin_segments_mean_duration_idxs = np.mean(lin_segments_duration_idxs,dtype=float) if len(lin_segments_duration_idxs) > 0 else 0
        lin_segments_std_duration_idxs = np.std(lin_segments_duration_idxs,dtype=float) if len(lin_segments_duration_idxs) > 0 else 0

        lin_segments_length_pixels = [ length_boxes_center(self.track.boxes[seg.idx_start_incl:seg.idx_end_incl+1]) for seg in self.segments]
        lin_segments_mean_length_pixels = np.mean(lin_segments_length_pixels,dtype=float) if len(lin_segments_length_pixels) > 0 else 0
        lin_segments_std_length_pixels = np.std(lin_segments_length_pixels,dtype=float) if len(lin_segments_length_pixels) > 0 else 0

        return LinStats(
            track_id=self.track.track_id,
            no_lin_segments=len(self.segments),
            no_points_in_lin_segments=no_points_in_linear_segments,
            no_points_in_track=no_points_in_track,
            frac_of_points_in_linear_segments=frac_of_points_in_linear_segments,
            track_duration_idxs=track_duration_idxs,
            track_length_pixels=track_length_pixels,
            lin_segments_mean_duration_idxs=lin_segments_mean_duration_idxs,
            lin_segments_std_duration_idxs=lin_segments_std_duration_idxs,
            lin_segments_mean_length_pixels=lin_segments_mean_length_pixels,
            lin_segments_std_length_pixels=lin_segments_std_length_pixels
            )


def find_linear_segments(track: Track, checker: LinTripletChecker) -> LinSegs:
    idxs = find_linear_triplets(track, checker)

    segments = []
    no_boxes = len(track.boxes)
    for i in range(no_boxes):
        
        if i in idxs:
            # Not any segments yet
            if len(segments) == 0:
                segments.append(LinSeg(i,i))
                continue

            # Check if continues => extend
            if segments[-1].idx_end_incl == i-1:
                # Continues
                segments[-1].idx_end_incl = i
            else:
                # New segment
                segments.append(LinSeg(i,i))

    # All segments go "one further" because they are the center pts of linear triplets
    # Add 1 to the start and end idxs
    segments = [LinSeg(seg.idx_start_incl-1, seg.idx_end_incl+1) for seg in segments]
    return LinSegs(segments, track)