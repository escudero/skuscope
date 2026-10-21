"""Product detection with a YOLO11 model trained on SKU-110K."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

from skuscope.config import DETECTOR_REPO, DETECTOR_REVISION, DETECTOR_WEIGHTS


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop((round(self.x1), round(self.y1), round(self.x2), round(self.y2)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_weights(size: str = "s") -> Path:
    filename, expected = DETECTOR_WEIGHTS[size]
    path = Path(hf_hub_download(DETECTOR_REPO, filename, revision=DETECTOR_REVISION))
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {filename}: expected {expected}, got {actual}")
    return path


class ProductDetector:
    """Finds every product on a shelf photo, without telling them apart."""

    def __init__(self, size: str = "s", conf: float = 0.25):
        from ultralytics import YOLO  # slow import, only pay for it when detecting

        self.model = YOLO(str(download_weights(size)), task="detect")
        self.conf = conf

    def detect(self, image: Image.Image) -> list[Box]:
        result = self.model.predict(image, imgsz=640, conf=self.conf, max_det=1000, verbose=False)[0]
        return [
            Box(*xyxy, score=score)
            for xyxy, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
        ]
