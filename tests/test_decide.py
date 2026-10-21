import numpy as np
import pytest

from skuscope.catalog import Catalog
from skuscope.pipeline import UNKNOWN, decide

SKUS = ["cola", "cola-zero", "guarana"]


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(np.eye(3, dtype=np.float32), SKUS, ["a.jpg", "b.jpg", "c.jpg"], "dinov2-small", {"cola-line": ["cola", "cola-zero"]})


def scores(cola: float, cola_zero: float, guarana: float) -> np.ndarray:
    return np.array([cola, cola_zero, guarana], dtype=np.float32)


def test_clear_winner_gets_its_sku(catalog):
    assert decide(scores(0.90, 0.70, 0.60), catalog, threshold=0.8, margin=0.03) == "cola"


def test_below_threshold_is_unknown(catalog):
    assert decide(scores(0.75, 0.50, 0.40), catalog, threshold=0.8, margin=0.03) == UNKNOWN


def test_close_call_between_siblings_gets_the_family(catalog):
    assert decide(scores(0.90, 0.89, 0.60), catalog, threshold=0.8, margin=0.03) == "cola-line"


def test_close_call_across_families_is_unknown(catalog):
    assert decide(scores(0.90, 0.60, 0.89), catalog, threshold=0.8, margin=0.03) == UNKNOWN


def test_family_needs_the_margin_over_other_products(catalog):
    assert decide(scores(0.90, 0.89, 0.88), catalog, threshold=0.8, margin=0.03) == UNKNOWN


def test_no_margin_behaves_like_threshold_only(catalog):
    assert decide(scores(0.90, 0.60, 0.899), catalog, threshold=0.8, margin=0.0) == "cola"


def test_single_sku_catalog():
    single = Catalog(np.eye(1, dtype=np.float32), ["cola"], ["a.jpg"], "dinov2-small")

    assert decide(np.array([0.9], dtype=np.float32), single, threshold=0.8, margin=0.03) == "cola"
