from motstat.data import TrackXy, TrackXyxy
from motstat.lin_detection_triplets import LinTripletChecker
from motstat.data_lin import LinSegs


from typing import Union


def find_linear_segments(track: Union[TrackXyxy,TrackXy], checker: LinTripletChecker) -> LinSegs:
    if type(track) == TrackXyxy:
        from motstat.lin_detection_xyxy import find_linear_segments
        return find_linear_segments(track, checker)
    elif type(track) == TrackXy:
        from motstat.lin_detection_xy import find_linear_segments
        return find_linear_segments(track, checker)
    else:
        raise ValueError("Unknown track type: {}".format(type(track)))