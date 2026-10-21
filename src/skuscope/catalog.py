"""The catalog: reference vectors of every known SKU."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
FAMILIES_FILE = "families.json"


def scan_photos(root: Path) -> list[tuple[str, Path]]:
    """Lists (sku, photo) pairs from a folder laid out as root/<sku>/<photo>."""
    pairs = []
    for sku_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for photo in sorted(sku_dir.iterdir()):
            if photo.suffix.lower() in IMAGE_SUFFIXES:
                pairs.append((sku_dir.name, photo))
    return pairs


def read_families(root: Path) -> dict[str, list[str]]:
    """Optional root/families.json: {"family": ["sku", ...]}, variants of one product line that look alike."""
    path = root / FAMILIES_FILE
    return json.loads(path.read_text()) if path.exists() else {}


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

    def __init__(
        self,
        vectors: np.ndarray,
        skus: list[str],
        photos: list[str],
        embedder: str,
        families: dict[str, list[str]] | None = None,
    ):
        if len(vectors) != len(skus):
            raise ValueError(f"Got {len(vectors)} vectors but {len(skus)} labels")
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.skus = skus
        self.photos = photos
        self.embedder = embedder
        self.families = families or {}
        # One column per SKU in sku_scores, in this order.
        self.sku_names = sorted(set(skus))
        self._columns = [np.array([s == name for s in skus]) for name in self.sku_names]
        self._family_of = {}
        for name, members in self.families.items():
            for sku in members:
                if sku not in self.sku_names:
                    raise ValueError(f"Family {name!r} lists {sku!r}, which has no photos in the catalog")
                self._family_of[sku] = name

    def family(self, sku: str) -> str:
        """The product line of a SKU, or the SKU itself when it belongs to no family."""
        return self._family_of.get(sku, sku)

    @property
    def labels(self) -> list[str]:
        """Every label a product can get: SKUs and family names."""
        return sorted(set(self.sku_names) | set(self.families))

    def match(self, queries: np.ndarray) -> list[Match]:
        """Nearest reference photo for each query vector."""
        if len(queries) == 0:
            return []
        # Vectors are L2-normalized, so the dot product is the cosine similarity.
        scores = np.asarray(queries, dtype=np.float32) @ self.vectors.T
        best = scores.argmax(axis=1)
        return [Match(self.skus[i], float(scores[row, i])) for row, i in enumerate(best)]

    def sku_scores(self, queries: np.ndarray) -> np.ndarray:
        """Similarity of each query to each SKU (columns in sku_names order): the SKU's best photo."""
        scores = np.asarray(queries, dtype=np.float32).reshape(-1, self.vectors.shape[1]) @ self.vectors.T
        return np.stack([scores[:, column].max(axis=1) for column in self._columns], axis=1)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        meta = {"embedder": self.embedder, "skus": self.skus, "photos": self.photos, "families": self.families}
        (folder / "catalog.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, folder: Path) -> "Catalog":
        meta = json.loads((folder / "catalog.json").read_text())
        return cls(
            np.load(folder / "vectors.npy"), meta["skus"], meta["photos"], meta["embedder"], meta.get("families")
        )
