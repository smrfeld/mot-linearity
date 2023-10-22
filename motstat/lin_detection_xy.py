from motstat.data import TrackXy
from motstat.lin_detection_triplets import LinTripletChecker
from motstat.data_lin import LinSegs


from typing import List


def find_linear_triplets(track: TrackXy, checker: LinTripletChecker) -> List[int]:
    xys = [ [ float(z) for z in pt ] for pt in track.pts]
    return checker.find_linear_triplets(xys)


def find_linear_segments(track: TrackXy, checker: LinTripletChecker) -> LinSegs:
    idxs = find_linear_triplets(track, checker)
    segments = checker.lin_idxs_to_segments(idxs)
    return LinSegs(segments, no_points_in_track=len(track.pts), track_id=track.track_id)