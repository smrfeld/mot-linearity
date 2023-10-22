from motlinearity.data import DispProb, TracksXy, TrackXy, Entry


import numpy as np
from typing import List


def sample_random_walk(no_trajs: int, no_pts_per_traj: int, disps_probs: List[DispProb]) -> TracksXy:
    tracks = TracksXy({})
    for track_id in range(0,no_trajs):

        entries: List[Entry] = [ Entry(data=[0,0], frame_id=0, track_id=track_id) ]
        tracks.tracks[track_id] = TrackXy(track_id=track_id, entries=entries)
        for j in range(1,no_pts_per_traj):
            # Sample displacement
            track_id = np.random.choice(list(range(len(disps_probs))), p=[ dp.prob for dp in disps_probs ])
            disp_x = disps_probs[track_id].disp_x
            disp_y = disps_probs[track_id].disp_y
            
            # Add to points
            entries.append(Entry(frame_id=j, track_id=track_id, data=[entries[-1].data[0]+int(disp_x), entries[-1].data[1]+int(disp_y)]))
    return tracks
