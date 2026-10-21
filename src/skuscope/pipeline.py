"""Detect → crop → embed → match."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from skuscope.catalog import Catalog, Match, scan_photos
from skuscope.detector import Box

if TYPE_CHECKING:
    from skuscope.detector import ProductDetector
    from skuscope.embedder import Embedder

register_heif_opener()  # iPhone photos are HEIC by default

UNKNOWN = "unknown"

# The detector was trained on dense shelves. On a close-up of a single product it
# often finds nothing, or just a piece of the label, which would become the SKU's
# fingerprint. Below this share of the photo, keep the whole photo instead.
MIN_PRODUCT_AREA = 0.2


@dataclass(frozen=True)
class Detection:
    box: Box
    sku: str
    score: float


def label(match: Match, threshold: float) -> str:
    return match.sku if match.score >= threshold else UNKNOWN


def load_photo(path: Path) -> Image.Image:
    # Phone cameras store rotation in EXIF instead of rotating the pixels.
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def largest_product(image: Image.Image, detector: "ProductDetector") -> Image.Image:
    """Crops the reference photo to its product, so background does not leak into the vector."""
    boxes = detector.detect(image)
    if not boxes:
        return image
    largest = max(boxes, key=lambda b: b.area)
    if largest.area < MIN_PRODUCT_AREA * image.width * image.height:
        return image
    return largest.crop(image)


def build_catalog(photos_dir: Path, embedder: "Embedder", detector: "ProductDetector | None") -> Catalog:
    pairs = scan_photos(photos_dir)
    if not pairs:
        raise ValueError(f"No photos found. Expected {photos_dir}/<sku>/<photo>.jpg")
    images = [load_photo(path) for _, path in pairs]
    if detector is not None:
        images = [largest_product(image, detector) for image in images]
    vectors = embedder.embed(images)
    return Catalog(vectors, [sku for sku, _ in pairs], [str(p) for _, p in pairs], embedder.name)


def recognize(
    image: Image.Image,
    detector: "ProductDetector",
    embedder: "Embedder",
    catalog: Catalog,
    threshold: float,
) -> list[Detection]:
    boxes = detector.detect(image)
    matches = catalog.match(embedder.embed([box.crop(image) for box in boxes]))
    return [Detection(box, label(m, threshold), m.score) for box, m in zip(boxes, matches)]


def count_skus(detections: list[Detection]) -> Counter:
    return Counter(d.sku for d in detections)
