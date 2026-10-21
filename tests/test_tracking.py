import cv2
import numpy as np
import pytest

from skuscope.catalog import Catalog
from skuscope.pipeline import UNKNOWN
from skuscope.tracking import CameraMotion, Tracker, iou_matrix, warp_box

SIZE = (640, 480)
STILL = np.eye(3)
COLA = np.array([0.95, 0.50], dtype=np.float32)  # per-SKU scores: cola, guarana
UNSURE = np.array([0.60, 0.55], dtype=np.float32)


def shift(dx: float, dy: float = 0.0) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)


@pytest.fixture
def tracker() -> Tracker:
    catalog = Catalog(np.eye(2, dtype=np.float32), ["cola", "guarana"], ["a.jpg", "b.jpg"], "dinov2-small")
    return Tracker(catalog, threshold=0.8, margin=0.03)


def test_iou_matrix():
    m = iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10), (5, 0, 15, 10), (20, 20, 30, 30)])

    assert m[0] == pytest.approx([1.0, 1 / 3, 0.0], abs=1e-6)
    assert iou_matrix([], [(0, 0, 1, 1)]).shape == (0, 1)


def test_warp_box_by_a_translation():
    assert warp_box((10, 20, 30, 40), shift(5, -3)) == pytest.approx((15, 17, 35, 37))


def test_still_product_keeps_its_identity_and_is_counted_once(tracker):
    for _ in range(10):
        live = tracker.update([(100, 100, 150, 200)], COLA[None], STILL, SIZE)

    assert [t.id for t in live] == [0]
    assert live[0].label == "cola"
    assert tracker.counts == {"cola": 1}
    assert tracker.seen == 1


def test_product_is_counted_only_after_count_after_named_frames(tracker):
    tracker.update([(100, 100, 150, 200)], COLA[None], STILL, SIZE)
    tracker.update([(100, 100, 150, 200)], COLA[None], STILL, SIZE)

    assert tracker.counts == {}


def test_moving_camera_keeps_identity_through_a_gap(tracker):
    """The camera pans 40 px per frame, wider than the overlap of plain IoU, and loses the product for 5 frames."""
    x = 100
    for frame in range(12):
        x += 40
        boxes = [] if 4 <= frame < 9 else [(x, 100, x + 50, 200)]
        live = tracker.update(boxes, np.array([COLA] * len(boxes)).reshape(-1, 2), shift(40), (2000, 480))

    assert [t.id for t in live] == [0]
    assert tracker.counts == {"cola": 1}


def test_lost_too_long_is_a_new_product(tracker):
    tracker.max_misses = 2
    for _ in range(3):
        tracker.update([(100, 100, 150, 200)], COLA[None], STILL, SIZE)
    for _ in range(3):
        tracker.update([], np.empty((0, 2)), STILL, SIZE)
    for _ in range(3):
        live = tracker.update([(100, 100, 150, 200)], COLA[None], STILL, SIZE)

    assert [t.id for t in live] == [1]
    assert tracker.counts == {"cola": 2}


def test_one_bad_frame_does_not_flip_the_name(tracker):
    for _ in range(5):
        tracker.update([(100, 100, 150, 200)], COLA[None], STILL, SIZE)
    live = tracker.update([(100, 100, 150, 200)], UNSURE[None], STILL, SIZE)

    assert live[0].label == "cola"


def test_unknown_products_are_seen_but_not_counted(tracker):
    for _ in range(5):
        live = tracker.update([(100, 100, 150, 200)], UNSURE[None], STILL, SIZE)

    assert live[0].label == UNKNOWN
    assert tracker.seen == 1
    assert tracker.counts == {}


def test_products_leaving_the_frame_are_dropped(tracker):
    tracker.update([(100, 100, 150, 200)], COLA[None], STILL, SIZE)
    tracker.update([], np.empty((0, 2)), shift(-200), SIZE)

    assert tracker.tracks == []


def test_camera_motion_finds_a_pan():
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (600, 1600), dtype=np.uint8), (0, 0), 2)
    first, second = texture[:, 100:1300], texture[:, 112:1312]  # the camera moves 12 px right

    motion = CameraMotion(width=1200)
    assert np.allclose(motion(first), np.eye(3))
    h = motion(second)

    assert warp_box((500, 200, 560, 300), h) == pytest.approx((488, 200, 548, 300), abs=1.0)


def test_camera_motion_region_ignores_the_rest_of_the_frame():
    rng = np.random.default_rng(1)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (600, 1600), dtype=np.uint8), (0, 0), 2)
    first, second = texture[:, 100:1300].copy(), texture[:, 112:1312].copy()
    first[:, 600:] = second[:, 600:] = 128  # only the right half would be left, and it is blank

    motion = CameraMotion(width=1200, region=(0.5, 1.0))
    motion(first)

    assert np.allclose(motion(second), np.eye(3))
