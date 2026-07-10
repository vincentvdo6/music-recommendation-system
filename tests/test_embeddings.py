"""Embedding fallback behavior when the optional ANN artifact is unavailable."""

import numpy as np
from gensim.models import KeyedVectors

from services.recommendation.embeddings import EmbeddingService


def test_corrupt_optional_ann_index_falls_back(monkeypatch, tmp_path):
    ann_dir = tmp_path / "ann"
    ann_dir.mkdir()

    def fail_to_load(self, path):
        raise ValueError("corrupt index")

    monkeypatch.setattr(EmbeddingService, "load_ann_index", fail_to_load)
    service = EmbeddingService(ann_index_path=str(ann_dir))
    assert service.ann_index is None


def test_exact_fallback_uses_vectorized_keyed_vectors():
    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(
        ["near", "far"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    service = EmbeddingService()
    service.item2vec.wv = vectors
    service.item2vec.dim = 2

    assert service._brute_force_neighbors(np.array([1.0, 0.0]), k=1) == ["near"]
    assert service._brute_force_neighbors(np.zeros(2), k=1) == []
