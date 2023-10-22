from motstat.data import Track, length_boxes_center
from motstat.linear_triplet import LinTripletChecker, LinSeg


from typing import List
from dataclasses import dataclass
from loguru import logger
import numpy as np


def find_linear_triplets(track: Track, checker: LinTripletChecker) -> List[int]:
    xyxys = [box.xyxy for box in track.boxes]
    return checker.find_linear_triplets_xyxy(xyxys)


@dataclass
class LinStats:
    track_id: int
    no_lin_segments: int

    no_points_in_lin_segments: int
    no_points_in_track: int
    frac_of_points_in_linear_segments: float
    
    track_duration_idxs: int
    track_length_pixels: float
    track_displacements_pixels: List[float]

    lin_segments_duration_idxs: List[int]
    lin_segments_mean_duration_idxs: float
    lin_segments_std_duration_idxs: float
    
    lin_segments_length_pixels: List[float]
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

        track_displacements_pixels = [ length_boxes_center([self.track.boxes[i], self.track.boxes[i+1]]) for i in range(len(self.track.boxes)-1) ]
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
            track_displacements_pixels=track_displacements_pixels,
            track_duration_idxs=track_duration_idxs,
            track_length_pixels=track_length_pixels,
            lin_segments_duration_idxs=lin_segments_duration_idxs,
            lin_segments_mean_duration_idxs=lin_segments_mean_duration_idxs,
            lin_segments_std_duration_idxs=lin_segments_std_duration_idxs,
            lin_segments_length_pixels=lin_segments_length_pixels,
            lin_segments_mean_length_pixels=lin_segments_mean_length_pixels,
            lin_segments_std_length_pixels=lin_segments_std_length_pixels
            )


def find_linear_segments(track: Track, checker: LinTripletChecker) -> LinSegs:
    idxs = find_linear_triplets(track, checker)
    segments = checker.lin_idxs_to_segments(idxs)
    return LinSegs(segments, track)