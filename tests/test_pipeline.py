from PIL import Image

from skuscope.detector import Box
from skuscope.pipeline import largest_product


class FakeDetector:
    def __init__(self, boxes: list[Box]):
        self.boxes = boxes

    def detect(self, image: Image.Image) -> list[Box]:
        return self.boxes


def test_crops_to_the_largest_product():
    photo = Image.new("RGB", (100, 200))
    detector = FakeDetector([Box(0, 0, 10, 10, 0.9), Box(10, 20, 90, 180, 0.8)])

    assert largest_product(photo, detector).size == (80, 160)


def test_keeps_whole_photo_when_only_a_fragment_is_found():
    photo = Image.new("RGB", (100, 200))
    label_fragment = Box(20, 80, 60, 120, 0.3)  # 8% of the photo

    assert largest_product(photo, FakeDetector([label_fragment])).size == (100, 200)


def test_keeps_whole_photo_when_nothing_is_found():
    photo = Image.new("RGB", (100, 200))

    assert largest_product(photo, FakeDetector([])).size == (100, 200)
