import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from mashumaro import DataClassDictMixin


@dataclass
class RandomWalk(DataClassDictMixin):

    # List of (x,y) points
    traj: List[List[int]]


@dataclass
class DispProb(DataClassDictMixin):
    disp_x: int
    disp_y: int
    prob: float


def sample_random_walk(no_trajs: int, no_pts_per_traj: int, disps_probs: List[DispProb]) -> List[RandomWalk]:
    trajs = []
    for _ in range(0,no_trajs):

        pts = [[0,0]]
        trajs.append(RandomWalk(traj=pts))
        for j in range(0,no_pts_per_traj):
            # Sample displacement
            disp_x = np.random.choice([ dp.disp_x for dp in disps_probs ], p=[ dp.prob for dp in disps_probs ])
            disp_y = np.random.choice([ dp.disp_y for dp in disps_probs ], p=[ dp.prob for dp in disps_probs ])
            
            # Add to points
            pts.append([pts[-1][0]+int(disp_x), pts[-1][1]+int(disp_y)])
    return trajs