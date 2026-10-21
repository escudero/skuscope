"""Video: follows each product across frames, so a moving camera counts it once.

Matching boxes of consecutive frames by overlap (IoU) is not enough on a
handheld or cart-mounted camera: products move tens of pixels per frame and
vanish for a few frames in motion blur, then come back as new products and get
counted again. So the tracker first estimates the camera motion between frames
(a homography from ORB features + RANSAC), moves every track by it, seen or
lost, and only then matches by IoU. A product lost for up to max_misses frames
is found again where the camera motion put it, keeping its identity.
"""

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from skuscope.catalog import Catalog
from skuscope.pipeline import UNKNOWN, decide

Rect = tuple[float, float, float, float]  # x1, y1, x2, y2


def iou_matrix(a: list[Rect], b: list[Rect]) -> np.ndarray:
    """Intersection over union of every box in a with every box in b."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def warp_box(box: Rect, h: np.ndarray) -> Rect:
    """The axis-aligned box around the four corners of box moved by the homography h."""
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], float) @ h.T
    corners = corners[:, :2] / corners[:, 2:]
    return (*corners.min(axis=0), *corners.max(axis=0))


class CameraMotion:
    """Homography from the previous frame to the current one, in full-resolution pixels.

    Features are found on a frame scaled to `width` pixels, which is enough for
    the camera motion and several times faster. When too few features match
    (a blank wall, heavy blur), the camera is assumed still.

    One homography describes one plane. Rolling down an aisle, the left shelf,
    the right shelf and the floor move differently, and a fit over all of them
    follows none well. `region` (left, right, as fractions of the width) keeps
    only the features of the shelf you care about.
    """

    def __init__(
        self,
        width: int = 960,
        features: int = 3000,
        min_matches: int = 12,
        region: tuple[float, float] = (0.0, 1.0),
    ):
        self.width = width
        self.min_matches = min_matches
        self.region = region
        self.orb = cv2.ORB_create(features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.previous = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """frame: an RGB or grayscale array. Returns identity for the first frame."""
        scale = min(1.0, self.width / frame.shape[1])
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        gray = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
        mask = np.zeros(gray.shape, np.uint8)
        mask[:, round(self.region[0] * gray.shape[1]) : round(self.region[1] * gray.shape[1])] = 255
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
        h = np.eye(3)
        if self.previous is not None and descriptors is not None and self.previous[1] is not None:
            matches = self.matcher.match(self.previous[1], descriptors)
            if len(matches) >= self.min_matches:
                src = np.float32([self.previous[0][m.queryIdx].pt for m in matches])
                dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
                found, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                if found is not None:
                    to_small = np.diag([scale, scale, 1.0])
                    h = np.linalg.inv(to_small) @ found @ to_small
        self.previous = (keypoints, descriptors)
        return h


@dataclass
class Track:
    id: int
    box: Rect
    scores: np.ndarray  # running average of the per-SKU scores
    label: str = UNKNOWN
    age: int = 1  # frames seen
    misses: int = 0  # frames lost in a row
    named_frames: int = 0  # frames seen with a name
    counted_as: str | None = None

    @property
    def score(self) -> float:
        return float(self.scores.max())


@dataclass
class Tracker:
    """Gives each product one identity across frames and counts it once.

    Each track keeps a running average of its per-SKU scores, so one blurry
    frame does not flip its name. A product is counted after being named in
    count_after frames, once, with the name it had at that moment.
    """

    catalog: Catalog
    threshold: float
    margin: float
    smoothing: float = 0.35  # weight of the new frame in the running average
    max_misses: int = 30  # 1 s at 30 fps; chained homographies drift ~10 px over that
    min_iou: float = 0.3
    count_after: int = 3
    tracks: list[Track] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)  # products counted, per label
    seen: int = 0  # products seen in at least count_after frames, named or not
    _next_id: int = 0

    def update(self, boxes: list[Rect], scores: np.ndarray, motion: np.ndarray, size: tuple[int, int]) -> list[Track]:
        """Feeds one frame: its boxes, their catalog.sku_scores, the camera motion since the
        previous frame and the frame (width, height). Returns the tracks seen in this frame."""
        for track in self.tracks:
            track.box = warp_box(track.box, motion)

        overlap = iou_matrix([t.box for t in self.tracks], boxes)
        pairs = sorted(zip(*np.nonzero(overlap >= self.min_iou)), key=lambda p: -overlap[p])
        matched_tracks, matched_boxes = set(), set()
        for i, j in pairs:  # best overlaps first
            if i in matched_tracks or j in matched_boxes:
                continue
            matched_tracks.add(i)
            matched_boxes.add(j)
            track = self.tracks[i]
            # This frame's box: smoothing it would lag behind the moving camera.
            track.box = tuple(boxes[j])
            track.scores = track.scores * (1 - self.smoothing) + scores[j] * self.smoothing
            track.age += 1
            track.misses = 0
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.misses += 1
        for j, box in enumerate(boxes):
            if j not in matched_boxes:
                self.tracks.append(Track(self._next_id, tuple(box), np.array(scores[j], dtype=np.float32)))
                self._next_id += 1

        width, height = size
        self.tracks = [
            t for t in self.tracks
            if t.misses <= self.max_misses and t.box[2] > 0 and t.box[0] < width and t.box[3] > 0 and t.box[1] < height
        ]
        live = [t for t in self.tracks if t.misses == 0]
        for track in live:
            track.label = decide(track.scores, self.catalog, self.threshold, self.margin)
            if track.age == self.count_after:
                self.seen += 1
            if track.label != UNKNOWN:
                track.named_frames += 1
                if track.named_frames == self.count_after:
                    track.counted_as = track.label
                    self.counts[track.label] += 1
        return live
