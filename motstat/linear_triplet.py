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


