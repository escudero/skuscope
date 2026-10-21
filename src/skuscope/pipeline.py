"""Detect → crop → embed → match."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from skuscope.catalog import Catalog, read_families, scan_photos
from skuscope.detector import Box

if TYPE_CHECKING:
    from skuscope.detector import ProductDetector
    from skuscope.embedder import Embedder

register_heif_opener()  # iPhone photos are HEIC by default

UNKNOWN = "unknown"
# DINOv2 scores even different products at 0.6-0.8, so the bar sits high.
DEFAULT_THRESHOLD = 0.8
# A lookalike scores almost as high as the right SKU. Asking the winner to lead
# the runner-up by this much turns most of those into "unknown" instead of a wrong name.
DEFAULT_MARGIN = 0.03

# The detector was trained on dense shelves. On a close-up of a single product it
# often finds nothing, or just a piece of the label, which would become the SKU's
# fingerprint. Below this share of the photo, keep the whole photo instead.
MIN_PRODUCT_AREA = 0.2


@dataclass(frozen=True)
class Detection:
    box: Box
    sku: str
    score: float


def decide(scores: np.ndarray, catalog: Catalog, threshold: float, margin: float) -> str:
    """Labels one product from its per-SKU scores (catalog.sku_scores).

    The SKU, when it clears the threshold and beats every other SKU by the margin.
    Else its family, when the doubt is only among siblings of one product line
    (flavors, sizes) and the family beats every other SKU by the margin. Else unknown.
    """
    order = np.argsort(-scores)
    best = catalog.sku_names[order[0]]
    top = scores[order[0]]
    if top < threshold:
        return UNKNOWN
    if len(order) == 1 or top - scores[order[1]] >= margin:
        return best
    family = catalog.family(best)
    rivals = [scores[i] for i in order[1:] if catalog.family(catalog.sku_names[i]) != family]
    if family != best and (not rivals or top - max(rivals) >= margin):
        return family
    return UNKNOWN


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
    skus = [sku for sku, _ in pairs]
    return Catalog(vectors, skus, [str(p) for _, p in pairs], embedder.name, read_families(photos_dir))


def recognize(
    image: Image.Image,
    detector: "ProductDetector",
    embedder: "Embedder",
    catalog: Catalog,
    threshold: float = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
) -> list[Detection]:
    boxes = detector.detect(image)
    scores = catalog.sku_scores(embedder.embed([box.crop(image) for box in boxes]))
    return [Detection(box, decide(s, catalog, threshold, margin), float(s.max())) for box, s in zip(boxes, scores)]


def count_skus(detections: list[Detection]) -> Counter:
    return Counter(d.sku for d in detections)
