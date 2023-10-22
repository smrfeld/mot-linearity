from dataclasses import dataclass
from mashumaro import DataClassDictMixin
import glob
import os
from typing import List, Optional, Dict, Tuple, Union
from enum import Enum
from tqdm import tqdm
import numpy as np


@dataclass
class Entry(DataClassDictMixin):
    frame_id: int
    track_id: int
    data: List[float]
    is_gt: bool = True
    conf: Optional[float] = None
    consider: Optional[bool] = None


def length_boxes_center(boxes: List[Entry]) -> float:
    dist_center = 0
    for i in range(len(boxes)-1):
        xyxy1 = boxes[i].data
        xyxy2 = boxes[i+1].data

        center1 = [(xyxy1[0] + xyxy1[2]) / 2, (xyxy1[1] + xyxy1[3]) / 2]
        center2 = [(xyxy2[0] + xyxy2[2]) / 2, (xyxy2[1] + xyxy2[3]) / 2]
        dist_center += ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

    return dist_center


@dataclass
class TrackXyxy(DataClassDictMixin):
    track_id: int
    entries: List[Entry]
    is_gt: bool

    def length_pixels_center(self) -> float:
        return length_boxes_center(self.entries)


@dataclass
class TrackXy(DataClassDictMixin):
    track_id: int
    entries: List[Entry]


@dataclass
class TracksXyxy(DataClassDictMixin):
    tracks: Dict[int,TrackXyxy]


@dataclass
class TracksXy(DataClassDictMixin):
    tracks: Dict[int,TrackXy]


Track = Union[TrackXyxy, TrackXy]
Tracks = Union[TracksXyxy, TracksXy]
FileToTracks = Union[Dict[str,TracksXyxy], Dict[str,TracksXy]]

def xywh_to_xyxy(xywh: List[float]) -> List[float]:
    x, y, w, h = xywh
    return [x, y, x + w, y + h]


def parse_line(s: str, is_gt: bool) -> Entry:
    # Line format:
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <consider_entry or conf>, <x>, <y>, <z>
    items = s.split(",")
    assert len(items) >= 7, f"Line must have at least 7 items, got {len(items)}: {s}"
    frame_id, track_id, bb_left, bb_top, bb_width, bb_height, consider_entry_or_conf = items[:7]
    xywh = [float(bb_left), float(bb_top), float(bb_width), float(bb_height)]
    xyxy = xywh_to_xyxy(xywh)
    return Entry(
        frame_id=int(frame_id),
        track_id=int(track_id),
        data=xyxy,
        is_gt=is_gt,
        conf=float(consider_entry_or_conf) if not is_gt else None,
        consider=bool(consider_entry_or_conf) if is_gt else None
        )


def read_file(fname: str, is_gt: bool) -> TracksXyxy:
    with open(fname, "r") as f:
        lines = f.readlines()
    boxes = [parse_line(line, is_gt) for line in lines]

    tracks = TracksXyxy({})
    for box in boxes:
        tracks.tracks.setdefault(box.track_id, TrackXyxy(track_id=box.track_id, entries=[], is_gt=is_gt))
        tracks.tracks[box.track_id].entries.append(box)

    # Sort
    for track in tracks.tracks.values():
        track.entries.sort(key=lambda box: box.frame_id)

    return tracks


@dataclass
class DataSpec(DataClassDictMixin):

    class Mot(Enum):
        MOT17 = "MOT17"
        MOT20 = "MOT20"

    class Split(Enum):
        TRAIN = "train"
        TEST = "test"

    class Mode(Enum):
        GT = "gt"
        DET = "det"

    class Mot17Method(Enum):
        DPM = "DPM"
        FRCNN = "FRCNN"
        SDP = "SDP"

    mot: Mot
    split: Split = Split.TRAIN
    mode: Mode = Mode.GT
    mot17_method: Mot17Method = Mot17Method.DPM

    @property
    def mot_dir(self):
        if self.mot == self.Mot.MOT17:
            return "MOT17Labels"
        elif self.mot == self.Mot.MOT20:
            return "MOT20Labels"
        else:
            raise NotImplementedError(f"Unknown MOT dataset {self.mot}")


def load_tracks(spec: DataSpec) -> Dict[str, TracksXyxy]:
    if spec.mot == DataSpec.Mot.MOT17:
        fnames_glob = os.path.join(spec.mot_dir, spec.split.value, "*-%s" % spec.mot17_method.value, spec.mode.value, "%s.txt" % spec.mode.value)
    elif spec.mot == DataSpec.Mot.MOT20:
        fnames_glob = os.path.join(spec.mot_dir, spec.split.value, "*", spec.mode.value, "*", "%s.txt" % spec.mode.value)
    else:
        raise NotImplementedError(f"Unknown MOT dataset {spec.mot}")
    fnames = glob.glob(fnames_glob)
    assert len(fnames) > 0, f"No files found in {fnames_glob}"

    tracks = {}
    for fname in fnames:
        tracks[os.path.basename(os.path.dirname(os.path.dirname(fname)))] = read_file(fname, spec.mode == "gt")
    return tracks



@dataclass
class DispProb(DataClassDictMixin):
    disp_x: int
    disp_y: int
    prob: float


@dataclass
class BoxDisps:
    xy_disps: List[Tuple[float,float]]
    xy_disp_mean: Tuple[float,float]
    xy_disp_std: Tuple[float,float]
    disps_probs: List[DispProb]


def measure_bbox_coord_displacements(file_to_tracks: Dict[str,TracksXyxy]) -> BoxDisps:
    assert len(file_to_tracks) > 0, "No files found"

    disps = BoxDisps([], (0,0), (0,0), [])
    for fname,tracks in tqdm(file_to_tracks.items(), desc="Measuring linear stats for each file"):
        for track_id,track in tracks.tracks.items():
            
            disps.xy_disps += [ (track.entries[i+1].data[0] - track.entries[i].data[0], track.entries[i+1].data[1] - track.entries[i].data[1]) for i in range(len(track.entries)-1) ]
            disps.xy_disps += [ (track.entries[i+1].data[2] - track.entries[i].data[2], track.entries[i+1].data[3] - track.entries[i].data[3]) for i in range(len(track.entries)-1) ]

    disps.xy_disp_mean = (np.mean([ xy_disp[0] for xy_disp in disps.xy_disps ], dtype=float), np.mean([ xy_disp[1] for xy_disp in disps.xy_disps ], dtype=float))
    disps.xy_disp_std = (np.std([ xy_disp[0] for xy_disp in disps.xy_disps ], dtype=float), np.std([ xy_disp[1] for xy_disp in disps.xy_disps ], dtype=float))

    # Measure dist
    for disp_x in range(-10,10):
        for disp_y in range(-10,10):
            disp_xmin = disp_x - 0.5
            disp_xmax = disp_x + 0.5
            disp_ymin = disp_y - 0.5
            disp_ymax = disp_y + 0.5
            disps.disps_probs.append(DispProb(
                disp_x=disp_x,
                disp_y=disp_y,
                prob=len([ xy for xy in disps.xy_disps if disp_xmin <= xy[0] < disp_xmax and disp_ymin <= xy[1] < disp_ymax ])
                ))
    tot = sum([ dp.prob for dp in disps.disps_probs ])
    for dp in disps.disps_probs:
        dp.prob /= tot

    return disps
