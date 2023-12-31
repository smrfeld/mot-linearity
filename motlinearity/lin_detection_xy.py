from motlinearity.data import TrackXy
from motlinearity.lin_detection_triplets import LinTripletChecker
from motlinearity.data_lin import LinSegs


from typing import List


def find_linear_triplets(track: TrackXy, checker: LinTripletChecker) -> List[int]:
    xys = [ [ float(z) for z in entry.data ] for entry in track.entries]
    return checker.find_linear_triplets(xys)


def find_linear_segments(track: TrackXy, checker: LinTripletChecker) -> LinSegs:
    idxs = find_linear_triplets(track, checker)
    segments = checker.lin_idxs_to_segments(idxs)
    return LinSegs(segments, no_points_in_track=len(track.entries), track_id=track.track_id)