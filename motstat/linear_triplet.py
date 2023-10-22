import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import numpy as np
from mashumaro import DataClassDictMixin
from enum import Enum


@dataclass
class LinTriplet:
    is_linear: bool
    m12: Optional[float] = None
    m23: Optional[float] = None

    m12_min: Optional[float] = None
    m12_max: Optional[float] = None
    m23_min: Optional[float] = None
    m23_max: Optional[float] = None


@dataclass
class LinSeg:
    idx_start_incl: int
    idx_end_incl: int


class LinTripletChecker:


    @dataclass
    class Options(DataClassDictMixin):
        

        class Mode(Enum):
            PERTURB = "perturb"
            TOL = "tol"


        mode: Mode = Mode.PERTURB
        perturb_mag: float = 1.0
        tol: float = 0.1


    def __init__(self, options: Options):
        self.options = options


    def check_if_triplet_in_line(self, xy1: List[float], xy2: List[float], xy3: List[float]) -> LinTriplet:
        if self.options.mode == self.Options.Mode.PERTURB:
            return self._check_if_triplet_in_line_perturb(xy1, xy2, xy3)
        elif self.options.mode == self.Options.Mode.TOL:
            return self._check_if_triplet_in_line_tol(xy1, xy2, xy3)
        else:
            raise NotADirectoryError(f"Unknown mode {self.options.mode}")


    def _check_if_triplet_in_line_perturb(self, xy1: List[float], xy2: List[float], xy3: List[float]) -> LinTriplet:
        delta_x12 = xy2[0] - xy1[0]
        delta_y12 = xy2[1] - xy1[1]
        delta_x23 = xy3[0] - xy2[0]
        delta_y23 = xy3[1] - xy2[1]

        # Handle 0
        if delta_x12 == 0:
            return LinTriplet(delta_x23 == 0)
        if delta_x23 == 0:
            return LinTriplet(delta_x12 == 0)
        
        # Slopes:
        # m12 = delta_y12 / delta_x12
        # m23 = delta_y23 / delta_x23

        # Allow perturbation each point in x,y be perturb
        # This means max perturbation in each delta is 2*perturb and min is 0
        # This means biggest slope is (delta_y12 + 2*perturb) / (delta_x12 - 2*perturb) and smallest is (delta_y12 - 2*perturb_px) / (delta_x12 + 2*perturb) and similarly for m23

        p = self.options.perturb_mag
        m12_min = (delta_y12 - 2*p) / (delta_x12 + 2*p) if (delta_x12 + 2*p) != 0 else 0
        m12_max = (delta_y12 + 2*p) / (delta_x12 - 2*p) if (delta_x12 - 2*p) != 0 else 0
        m23_min = (delta_y23 - 2*p) / (delta_x23 + 2*p) if (delta_x23 + 2*p) != 0 else 0
        m23_max = (delta_y23 + 2*p) / (delta_x23 - 2*p) if (delta_x23 - 2*p) != 0 else 0

        # Check if there is overlap in the ranges => linear
        is_linear = (m12_min <= m23_max and m12_max >= m23_min) or (m23_min <= m12_max and m23_max >= m12_min)
        return LinTriplet(is_linear, m12_min=m12_min, m12_max=m12_max, m23_min=m23_min, m23_max=m23_max)


    def _check_if_triplet_in_line_tol(self, xy1: List[float], xy2: List[float], xy3: List[float]) -> LinTriplet:
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
        is_linear = abs(m12 - m23) <= self.options.tol
        return LinTriplet(is_linear, m12, m23)


    def find_linear_triplets(self, pts: List[List[float]]) -> List[int]:
        
        idx_linear = []
        for i in range(1,len(pts)-1):
            # Check if in a line
            assert len(pts[i-1]) == 2
            assert len(pts[i]) == 2
            assert len(pts[i+1]) == 2
            t = self.check_if_triplet_in_line(pts[i-1], pts[i], pts[i+1])

            if not t.is_linear:
                continue
            
            # In a line
            idx_linear.append(i)

        return idx_linear


    def find_linear_triplets_xyxy(self, xyxys: List[List[float]]) -> List[int]:

        idx_linear = []
        for i in range(1,len(xyxys)-1):
            xyxy1 = xyxys[i-1]
            xyxy2 = xyxys[i]
            xyxy3 = xyxys[i+1]

            # Check if in a line
            xy_bl_1 = [xyxy1[0], xyxy1[1]]
            xy_bl_2 = [xyxy2[0], xyxy2[1]]
            xy_bl_3 = [xyxy3[0], xyxy3[1]]
            t_bl = self.check_if_triplet_in_line(xy_bl_1, xy_bl_2, xy_bl_3)

            xy_tr_1 = [xyxy1[2], xyxy1[3]]
            xy_tr_2 = [xyxy2[2], xyxy2[3]]
            xy_tr_3 = [xyxy3[2], xyxy3[3]]
            t_tr = self.check_if_triplet_in_line(xy_tr_1, xy_tr_2, xy_tr_3)

            # logger.debug(f"Slopes: {t_bl.m12} {t_bl.m23} {t_bl.is_linear} {t_tr.m12} {t_tr.m23} {t_tr.is_linear}")

            if not t_tr.is_linear or not t_bl.is_linear:
                continue
            
            # In a line
            idx_linear.append(i)

        return idx_linear

    def lin_idxs_to_segments(self, idxs: List[int]) -> List[LinSeg]:
        if len(idxs) == 0:
            return []
        
        max_idx = max(idxs)
        segments = []
        for idx in range(max_idx+1):
            
            if idx in idxs:
                # Not any segments yet
                if len(segments) == 0:
                    segments.append(LinSeg(idx,idx))
                    continue

                # Check if continues => extend
                if segments[-1].idx_end_incl == idx-1:
                    # Continues
                    segments[-1].idx_end_incl = idx
                else:
                    # New segment
                    segments.append(LinSeg(idx,idx))

        # All segments go "one further" because they are the center pts of linear triplets
        # Add 1 to the start and end idxs
        segments = [LinSeg(seg.idx_start_incl-1, seg.idx_end_incl+1) for seg in segments]

        # Check min length = 3
        for seg in segments:
            assert seg.idx_end_incl - seg.idx_start_incl + 1 >= 3, f"Segment {seg} is too short, should be at least 3 pts"

        return segments