from motstat.data import TrackXyxy
from motstat.lin_detection_triplets import LinTripletChecker
from motstat.data_lin import LinSegs


from typing import List


def find_linear_triplets(track: TrackXyxy, checker: LinTripletChecker) -> List[int]:
    xyxys = [box.xyxy for box in track.boxes]
    return checker.find_linear_triplets_xyxy(xyxys)


def find_linear_segments(track: TrackXyxy, checker: LinTripletChecker) -> LinSegs:
    idxs = find_linear_triplets(track, checker)
    segments = checker.lin_idxs_to_segments(idxs)
    return LinSegs(segments, no_points_in_track=len(track.boxes), track_id=track.track_id)