from dataclasses import dataclass
from mashumaro import DataClassDictMixin
import glob
import os
import json
from typing import List, Optional, Dict

@dataclass
class Box(DataClassDictMixin):
    frame_id: int
    track_id: int
    xyxy: List[float]
    is_gt: bool
    conf: Optional[float] = None
    consider: Optional[bool] = None


@dataclass
class Track(DataClassDictMixin):
    track_id: int
    boxes: List[Box]
    is_gt: bool


@dataclass
class Tracks(DataClassDictMixin):
    tracks: Dict[int,Track]


def xywh_to_xyxy(xywh: List[float]) -> List[float]:
    x, y, w, h = xywh
    return [x, y, x + w, y + h]


def parse_line(s: str, is_gt: bool) -> Box:
    # Line format:
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <consider_entry or conf>, <x>, <y>, <z>
    items = s.split(",")
    assert len(items) >= 7, f"Line must have at least 7 items, got {len(items)}: {s}"
    frame_id, track_id, bb_left, bb_top, bb_width, bb_height, consider_entry_or_conf = items[:7]
    xywh = [float(bb_left), float(bb_top), float(bb_width), float(bb_height)]
    xyxy = xywh_to_xyxy(xywh)
    return Box(
        frame_id=int(frame_id),
        track_id=int(track_id),
        xyxy=xyxy,
        is_gt=is_gt,
        conf=float(consider_entry_or_conf) if not is_gt else None,
        consider=bool(consider_entry_or_conf) if is_gt else None
        )


def read_file(fname: str, is_gt: bool) -> Tracks:
    with open(fname, "r") as f:
        lines = f.readlines()
    boxes = [parse_line(line, is_gt) for line in lines]

    tracks = Tracks({})
    for box in boxes:
        tracks.tracks.setdefault(box.track_id, Track(track_id=box.track_id, boxes=[], is_gt=is_gt))
        tracks.tracks[box.track_id].boxes.append(box)

    # Sort
    for track in tracks.tracks.values():
        track.boxes.sort(key=lambda box: box.frame_id)

    return tracks


@dataclass
class DataSpec(DataClassDictMixin):
    dir_name: str = "/Users/oernst/Downloads/MOT17Labels"
    split: str = "train"
    mode: str = "gt"


def load_tracks(spec: DataSpec) -> Dict[str, Tracks]:
    fnames = glob.glob(os.path.join(spec.dir_name, spec.split, "*", spec.mode, "%s.txt" % spec.mode))
    tracks = {}
    for fname in fnames:
        tracks[os.path.basename(os.path.dirname(os.path.dirname(fname)))] = read_file(fname, spec.mode == "gt")
    return tracks