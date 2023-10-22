from motstat.data import DispProb, TracksXy, TrackXy


import numpy as np
from typing import List


def sample_random_walk(no_trajs: int, no_pts_per_traj: int, disps_probs: List[DispProb]) -> TracksXy:
    tracks = TracksXy({})
    for idx in range(0,no_trajs):

        pts = [[0,0]]
        tracks.tracks[idx] = TrackXy(track_id=idx, pts=pts)
        for j in range(0,no_pts_per_traj):
            # Sample displacement
            disp_x = np.random.choice([ dp.disp_x for dp in disps_probs ], p=[ dp.prob for dp in disps_probs ])
            disp_y = np.random.choice([ dp.disp_y for dp in disps_probs ], p=[ dp.prob for dp in disps_probs ])
            
            # Add to points
            pts.append([pts[-1][0]+int(disp_x), pts[-1][1]+int(disp_y)])
    return tracks
