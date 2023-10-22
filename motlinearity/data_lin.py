from motlinearity.lin_detection_triplets import LinSeg


from typing import List
from dataclasses import dataclass
from loguru import logger
import numpy as np


@dataclass
class LinStats:
    track_id: int
    no_lin_segments: int

    no_points_in_lin_segments: int
    no_points_in_track: int
    frac_of_points_in_linear_segments: float
    
    lin_segments_duration_idxs: List[int]
    lin_segments_mean_duration_idxs: float
    lin_segments_std_duration_idxs: float
    
    def report(self):
        logger.info(f"Track {self.track_id} has {self.no_lin_segments} linear segments:")

        logger.info(f"  No points in linear segments: {self.no_points_in_lin_segments}")
        logger.info(f"  No points in track: {self.no_points_in_track}")
        logger.info(f"  Frac of points in linear segments: {self.frac_of_points_in_linear_segments:.2f}")

        logger.info(f"  Ave duration of linear segments: {self.lin_segments_mean_duration_idxs:.2f} +- {self.lin_segments_std_duration_idxs:.2f} (idxs)")


@dataclass
class LinSegs:
    segments: List[LinSeg]
    no_points_in_track: int
    track_id: int

    @property
    def idxs_in_lin_segments(self) -> List[int]:
        return sorted(list(set([ idx for seg in self.segments for idx in range(seg.idx_start_incl, seg.idx_end_incl+1) ])))

    def stats(self) -> LinStats:        
        no_points_in_linear_segments = len(self.idxs_in_lin_segments)
        frac_of_points_in_linear_segments = no_points_in_linear_segments / self.no_points_in_track if self.no_points_in_track > 0 else 0

        lin_segments_duration_idxs = [seg.idx_end_incl - seg.idx_start_incl + 1 for seg in self.segments]
        lin_segments_mean_duration_idxs = np.mean(lin_segments_duration_idxs,dtype=float) if len(lin_segments_duration_idxs) > 0 else 0
        lin_segments_std_duration_idxs = np.std(lin_segments_duration_idxs,dtype=float) if len(lin_segments_duration_idxs) > 0 else 0

        return LinStats(
            track_id=self.track_id,
            no_lin_segments=len(self.segments),
            no_points_in_lin_segments=no_points_in_linear_segments,
            no_points_in_track=self.no_points_in_track,
            frac_of_points_in_linear_segments=frac_of_points_in_linear_segments,
            lin_segments_duration_idxs=lin_segments_duration_idxs,
            lin_segments_mean_duration_idxs=lin_segments_mean_duration_idxs,
            lin_segments_std_duration_idxs=lin_segments_std_duration_idxs,
            )