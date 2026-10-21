"""Recognize the products in a video and count each one once."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from skuscope.catalog import Catalog
from skuscope.detector import Box
from skuscope.draw import annotate
from skuscope.pipeline import Detection
from skuscope.tracking import CameraMotion, Tracker

if TYPE_CHECKING:
    from collections import Counter

    from skuscope.detector import ProductDetector
    from skuscope.embedder import Embedder


@dataclass(frozen=True)
class VideoResult:
    frames: int
    seen: int  # different products seen, named or not
    counts: "Counter"  # different products counted, per label


def _draw_totals(image: Image.Image, seen: int, counted: int) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(14, image.width // 50))
    text = f"seen {seen}   counted {counted}"
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad = font.size // 2
    draw.rectangle((0, 0, right - left + 2 * pad, bottom - top + 2 * pad), fill="black")
    draw.text((pad - left, pad - top), text, fill="white", font=font)


def recognize_video(
    path: Path,
    detector: "ProductDetector",
    embedder: "Embedder",
    catalog: Catalog,
    threshold: float,
    margin: float,
    out: Path | None = None,
    colors: dict[str, str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    motion_region: tuple[float, float] = (0.0, 1.0),
) -> VideoResult:
    """Runs every frame through detect → embed → match, tracks the products and,
    when out is given, writes the annotated video there. motion_region: see CameraMotion."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    motion, tracker, writer, frames = CameraMotion(region=motion_region), Tracker(catalog, threshold, margin), None, 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            boxes = detector.detect(image)
            scores = catalog.sku_scores(embedder.embed([box.crop(image) for box in boxes]))
            live = tracker.update([(b.x1, b.y1, b.x2, b.y2) for b in boxes], scores, motion(rgb), image.size)
            frames += 1
            if out is not None:
                detections = [Detection(Box(*t.box, score=1.0), t.label, t.score) for t in live]
                annotated = annotate(image, detections, colors or {})
                _draw_totals(annotated, tracker.seen, sum(tracker.counts.values()))
                if writer is None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, image.size)
                writer.write(cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))
            if progress is not None:
                progress(frames, total)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return VideoResult(frames, tracker.seen, tracker.counts)
