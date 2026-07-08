"""Feature contract tests: column order and missing-subsystem behavior."""

import numpy as np

from services.recommendation import features as F


def _minimal_matrix(n=3, **overrides):
    ids = [f"c{i}" for i in range(n)]
    args = {
        "ids": ids,
        "vectors": np.eye(n, 8, dtype=np.float32),
        "artists": [""] * n,
        "log_pop": np.zeros(n, dtype=np.float32),
        "durations_ms": np.zeros(n, dtype=np.float32),
        "moods": None,
        "ncf_scores": None,
        "ncf_mask": None,
        "ctx": F.RankingContext(),
    }
    args.update(overrides)
    return F.build_matrix(**args)


def test_columns_match_contract_in_order():
    X = _minimal_matrix()
    assert list(X.columns) == F.FEATURE_NAMES


def test_training_feature_spec_is_in_sync():
    """training/features_spec.py must be a byte-identical mirror of features.py."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    serving = (root / "services" / "recommendation" / "features.py").read_text(encoding="utf-8")
    training = (root / "training" / "features_spec.py").read_text(encoding="utf-8")
    assert serving == training, (
        "training/features_spec.py drifted from services/recommendation/features.py — "
        "re-copy it and re-upload the Kaggle base-models dataset"
    )


def test_missing_subsystems_produce_constants():
    X = _minimal_matrix()
    # No seed, no playlist, no NCF, no meta -> zero features
    for col in ["seed_i2v_cos", "playlist_i2v_cos", "playlist_i2v_max", "i2v_seed_rr",
                "i2v_playlist_rr", "ncf_score", "has_ncf", "log_pop",
                "same_artist_as_seed", "artist_in_playlist", "artist_playlist_share"]:
        assert (X[col] == 0.0).all(), col
    # Missing mood -> neutral constants (must not reorder candidates)
    for col in ["valence_diff", "energy_diff", "acousticness_diff", "danceability_diff", "mood_sim"]:
        assert (X[col] == 0.5).all(), col
    assert (X["duration_diff"] == 1.0).all()


def test_seed_cosine_and_rank_features():
    seed = np.zeros(8, dtype=np.float32)
    seed[0] = 1.0
    vectors = np.zeros((3, 8), dtype=np.float32)
    vectors[0, 0] = 1.0   # identical to seed
    vectors[1, 1] = 1.0   # orthogonal
    vectors[2, 0] = -1.0  # opposite
    ctx = F.RankingContext(seed_vec=seed, seed_rank={"c0": 0, "c2": 4})

    X = _minimal_matrix(vectors=vectors, ctx=ctx)
    assert np.isclose(X["seed_i2v_cos"][0], 1.0)
    assert np.isclose(X["seed_i2v_cos"][1], 0.0)
    assert np.isclose(X["seed_i2v_cos"][2], -1.0)
    assert np.isclose(X["i2v_seed_rr"][0], 1.0)      # rank 0 -> 1
    assert np.isclose(X["i2v_seed_rr"][1], 0.0)      # absent
    assert np.isclose(X["i2v_seed_rr"][2], 1.0 / 5)  # rank 4


def test_mood_and_artist_features():
    moods = np.array([[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    ctx = F.RankingContext(
        seed_artist="drake",
        playlist_artist_share={"drake": 0.4},
        target_mood=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    )
    X = _minimal_matrix(n=2, vectors=np.eye(2, 8, dtype=np.float32),
                        artists=["drake", "someone"], moods=moods, ctx=ctx)

    assert np.isclose(X["mood_sim"][0], 1.0)
    assert np.isclose(X["mood_sim"][1], 0.5)
    assert X["same_artist_as_seed"].tolist() == [1.0, 0.0]
    assert X["artist_in_playlist"].tolist() == [1.0, 0.0]
    assert np.isclose(X["artist_playlist_share"][0], 0.4)


def test_duration_diff_capped():
    ctx = F.RankingContext(playlist_mean_duration_ms=200000.0)
    durations = np.array([200000.0, 200000.0 + 10 * 60000, 0.0], dtype=np.float32)
    X = _minimal_matrix(durations_ms=durations, ctx=ctx)
    assert np.isclose(X["duration_diff"][0], 0.0)
    assert np.isclose(X["duration_diff"][1], 5.0)   # capped
    assert np.isclose(X["duration_diff"][2], 1.0)   # unknown duration -> neutral
