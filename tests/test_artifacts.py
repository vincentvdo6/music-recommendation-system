"""Slow tests against the real model artifacts in models/ (marked slow)."""

from pathlib import Path

import pytest

from services.recommendation.features import FEATURE_NAMES

pytestmark = pytest.mark.slow

MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(scope="module")
def real_engine():
    from services.recommendation.factory import create_engine

    engine = create_engine()
    if engine is None:
        pytest.skip("model artifacts not present")
    return engine


def test_engine_loads_with_real_artifacts(real_engine):
    assert real_engine.embeddings.item2vec.wv is not None


def test_ranker_feature_contract(real_engine):
    """If the LightGBM v2 model is present it must match FEATURE_NAMES exactly."""
    ranker_path = MODELS / "ranker" / "lightgbm_ranker_v2.txt"
    if not ranker_path.exists():
        assert real_engine.ranker.name == "linear-fallback"
        return
    assert real_engine.ranker.name == "lightgbm-v2"
    assert real_engine.ranker.model.feature_name() == FEATURE_NAMES


def test_real_recommendation_smoke(real_engine):
    wv = real_engine.embeddings.item2vec.wv
    playlist = wv.index_to_key[:20]
    seed = wv.index_to_key[50]
    recs = real_engine.recommend(playlist, input_track_id=seed, limit=10)
    assert recs
    assert all(r["id"] not in set(playlist) for r in recs)
    ids_a = [r["id"] for r in recs]

    other_seed = wv.index_to_key[5000]
    ids_b = [r["id"] for r in real_engine.recommend(playlist, input_track_id=other_seed, limit=10)]
    assert ids_a != ids_b, "different seeds must produce different recommendations"
