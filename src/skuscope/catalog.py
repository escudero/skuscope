"""The catalog: reference vectors of every known SKU."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def scan_photos(root: Path) -> list[tuple[str, Path]]:
    """Lists (sku, photo) pairs from a folder laid out as root/<sku>/<photo>."""
    pairs = []
    for sku_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for photo in sorted(sku_dir.iterdir()):
            if photo.suffix.lower() in IMAGE_SUFFIXES:
                pairs.append((sku_dir.name, photo))
    return pairs


@dataclass(frozen=True)
class Match:
    sku: str
    score: float


class Catalog:
    """Exact nearest-neighbor search with one matrix product.

    Same result as a FAISS IndexFlatIP, without its bundled libomp, which
    clashes with PyTorch's and crashes the process on macOS. Brute force is
    instant for catalogs of thousands of photos.
    """

    def __init__(self, vectors: np.ndarray, skus: list[str], photos: list[str], embedder: str):
        if len(vectors) != len(skus):
            raise ValueError(f"Got {len(vectors)} vectors but {len(skus)} labels")
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.skus = skus
        self.photos = photos
        self.embedder = embedder

    def match(self, queries: np.ndarray) -> list[Match]:
        """Nearest reference photo for each query vector."""
        if len(queries) == 0:
            return []
        # Vectors are L2-normalized, so the dot product is the cosine similarity.
        scores = np.asarray(queries, dtype=np.float32) @ self.vectors.T
        best = scores.argmax(axis=1)
        return [Match(self.skus[i], float(scores[row, i])) for row, i in enumerate(best)]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        meta = {"embedder": self.embedder, "skus": self.skus, "photos": self.photos}
        (folder / "catalog.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, folder: Path) -> "Catalog":
        meta = json.loads((folder / "catalog.json").read_text())
        return cls(np.load(folder / "vectors.npy"), meta["skus"], meta["photos"], meta["embedder"])
