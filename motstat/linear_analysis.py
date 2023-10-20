from motstat.data import load_tracks, Tracks, DataSpec, Track

import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class LinTriplet:
    is_linear: bool
    m12: Optional[float] = None
    m23: Optional[float] = None


def check_if_triplet_in_line(xy1: List[float], xy2: List[float], xy3: List[float], tol: float) -> LinTriplet:
    delta_x12 = xy2[0] - xy1[0]
    delta_y12 = xy2[1] - xy1[1]
    delta_x23 = xy3[0] - xy2[0]
    delta_y23 = xy3[1] - xy2[1]

    # Handle 0
    if delta_x12 == 0:
        return LinTriplet(delta_x23 == 0)
    if delta_x23 == 0:
        return LinTriplet(delta_x12 == 0)
    
    m12 = delta_y12 / delta_x12
    m23 = delta_y23 / delta_x23
    is_linear = abs(m12 - m23) <= tol
    return LinTriplet(is_linear, m12, m23)


def find_linear_triplets(track: Track, tol: float) -> List[int]:
    
    idx_linear = []
    for i in range(1,len(track.boxes)-1):
        xyxy1 = track.boxes[i-1].xyxy
        xyxy2 = track.boxes[i].xyxy
        xyxy3 = track.boxes[i+1].xyxy

        # Check if in a line
        xy_bl_1 = [xyxy1[0], xyxy1[1]]
        xy_bl_2 = [xyxy2[0], xyxy2[1]]
        xy_bl_3 = [xyxy3[0], xyxy3[1]]
        t_bl = check_if_triplet_in_line(xy_bl_1, xy_bl_2, xy_bl_3, tol)

        xy_tr_1 = [xyxy1[2], xyxy1[3]]
        xy_tr_2 = [xyxy2[2], xyxy2[3]]
        xy_tr_3 = [xyxy3[2], xyxy3[3]]
        t_tr = check_if_triplet_in_line(xy_tr_1, xy_tr_2, xy_tr_3, tol)

        logger.debug("Slopes:", t_bl.m12, t_bl.m23, t_bl.is_linear, t_tr.m12, t_tr.m23, t_tr.is_linear)

        if not t_tr.is_linear or not t_bl.is_linear:
            continue
        
        # In a line
        idx_linear.append(i)

    return idx_linear


@dataclass
class LinSeg:
    idx_start_incl: int
    idx_end_incl: int


def find_linear_segments(track: Track, tol: float) -> List[LinSeg]:
    idxs = find_linear_triplets(track, tol)

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
    return segments