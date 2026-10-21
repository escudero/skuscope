import numpy as np
import pytest

from skuscope.catalog import Catalog, Match, scan_photos
from skuscope.pipeline import UNKNOWN, label


def unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def catalog() -> Catalog:
    vectors = np.stack([unit(1, 0, 0), unit(0.9, 0.1, 0), unit(0, 1, 0)])
    return Catalog(vectors, ["cola", "cola", "guarana"], ["a.jpg", "b.jpg", "c.jpg"], "dinov2-small")


def test_match_returns_nearest_sku(catalog):
    matches = catalog.match(np.stack([unit(1, 0.05, 0), unit(0.1, 1, 0)]))
    assert [m.sku for m in matches] == ["cola", "guarana"]
    assert matches[0].score == pytest.approx(float(unit(1, 0.05, 0) @ unit(1, 0, 0)), abs=1e-5)


def test_match_with_no_queries(catalog):
    assert catalog.match(np.empty((0, 3), dtype=np.float32)) == []


def test_save_and_load_roundtrip(catalog, tmp_path):
    catalog.save(tmp_path / "index")
    loaded = Catalog.load(tmp_path / "index")
    assert loaded.skus == catalog.skus
    assert loaded.embedder == "dinov2-small"
    assert loaded.match(unit(0, 1, 0)[None])[0].sku == "guarana"


def test_mismatched_labels_are_rejected():
    with pytest.raises(ValueError):
        Catalog(np.stack([unit(1, 0)]), ["a", "b"], ["x", "y"], "dinov2-small")


def test_label_below_threshold_is_unknown():
    assert label(Match("cola", 0.80), threshold=0.5) == "cola"
    assert label(Match("cola", 0.30), threshold=0.5) == UNKNOWN


def test_scan_photos_reads_one_folder_per_sku(tmp_path):
    for sku, names in {"cola": ["1.jpg", "2.HEIC"], "guarana": ["1.png", "notes.txt"]}.items():
        (tmp_path / sku).mkdir()
        for name in names:
            (tmp_path / sku / name).touch()
    (tmp_path / ".hidden").mkdir()

    pairs = scan_photos(tmp_path)

    assert [(sku, p.name) for sku, p in pairs] == [("cola", "1.jpg"), ("cola", "2.HEIC"), ("guarana", "1.png")]
