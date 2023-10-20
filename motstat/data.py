from dataclasses import dataclass
from mashumaro import DataClassDictMixin
import glob
import os
import json
from typing import List, Optional, Dict
from enum import Enum


@dataclass
class Box(DataClassDictMixin):
    frame_id: int
    track_id: int
    xyxy: List[float]
    is_gt: bool
    conf: Optional[float] = None
    consider: Optional[bool] = None


def length_boxes_center(boxes: List[Box]) -> float:
    dist_center = 0
    for i in range(len(boxes)-1):
        xyxy1 = boxes[i].xyxy
        xyxy2 = boxes[i+1].xyxy

        center1 = [(xyxy1[0] + xyxy1[2]) / 2, (xyxy1[1] + xyxy1[3]) / 2]
        center2 = [(xyxy2[0] + xyxy2[2]) / 2, (xyxy2[1] + xyxy2[3]) / 2]
        dist_center += ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

    return dist_center


@dataclass
class Track(DataClassDictMixin):
    track_id: int
    boxes: List[Box]
    is_gt: bool

    def length_pixels_center(self) -> float:
        return length_boxes_center(self.boxes)


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

    class Split(Enum):
        TRAIN = "train"
        TEST = "test"

    class Mode(Enum):
        GT = "gt"
        DET = "det"

    mot_dir: str
    split: Split = Split.TRAIN
    mode: Mode = Mode.GT


def load_tracks(spec: DataSpec) -> Dict[str, Tracks]:
    fnames = glob.glob(os.path.join(spec.mot_dir, spec.split.value, "*", spec.mode.value, "%s.txt" % spec.mode.value))
    tracks = {}
    for fname in fnames:
        tracks[os.path.basename(os.path.dirname(os.path.dirname(fname)))] = read_file(fname, spec.mode == "gt")
    return tracks