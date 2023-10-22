import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from mashumaro import DataClassDictMixin


@dataclass
class RandomWalk(DataClassDictMixin):

    # List of (x,y) points
    traj: List[List[int]]


def sample_random_walk(no_trajs: int, no_pts_per_traj: int, displacement_to_prob: Dict[int,float]) -> List[RandomWalk]:
    trajs = []
    for _ in range(0,no_trajs):

        pts = [[0,0]]
        trajs.append(RandomWalk(traj=pts))
        for j in range(0,no_pts_per_traj):
            # Sample displacement
            disp_x = np.random.choice(list(displacement_to_prob.keys()), p=list(displacement_to_prob.values()))
            disp_y = np.random.choice(list(displacement_to_prob.keys()), p=list(displacement_to_prob.values()))
            
            # Add to points
            pts.append([pts[-1][0]+int(disp_x), pts[-1][1]+int(disp_y)])
    return trajs