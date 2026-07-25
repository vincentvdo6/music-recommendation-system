"""Privacy-minimal local impression, feedback, and identity persistence."""

from services.storage import LocalStore


def _recommendation(track_id="r1"):
    return {
        "id": track_id,
        "rank_score": 0.9,
        "similarity_score": 0.8,
        "discovery": False,
        "recommendation": {"components": {"audio_similarity": 0.4}},
    }


def test_impression_feedback_roundtrip(tmp_path):
    store = LocalStore(tmp_path / "one_rec.sqlite3")
    store.record_impression(
        impression_id="imp-12345678",
        request_id="req1",
        seed_id="seed",
        mode="seed_only",
        playlist_ids=[],
        source="ml-seed",
        recommendations=[_recommendation()],
        candidates=[_recommendation(), _recommendation("hidden")],
        session_id="session-12345678",
        profile="explorer",
        provenance={"ranker": "test"},
    )

    assert store.record_feedback(
        impression_id="imp-12345678", track_id="r1", event="like", position=0
    )
    assert not store.record_feedback(
        impression_id="imp-12345678", track_id="not-shown", event="like", position=1
    )
    signals = dict(store.recent_session_signals("session-12345678"))
    assert signals["r1"] > 0
    assert store.recent_session_exclusions("session-12345678") == ["r1"]
    candidate_count = store._db.execute("SELECT COUNT(*) FROM candidate_items").fetchone()[0]
    assert candidate_count == 2
    impression = store._db.execute(
        "SELECT profile, provenance_json FROM impressions"
    ).fetchone()
    assert impression["profile"] == "explorer"
    assert '"ranker":"test"' in impression["provenance_json"]
    store.close()


def test_session_exclusions_use_catalog_identity_and_are_session_scoped(tmp_path):
    store = LocalStore(tmp_path / "one_rec.sqlite3")
    served = _recommendation("spotify-1")
    served["catalog_id"] = "dz:1"
    store.record_impression(
        impression_id="imp-12345678",
        request_id="req1",
        seed_id="seed",
        mode="seed_only",
        playlist_ids=[],
        source="ml-seed",
        recommendations=[served],
        session_id="session-12345678",
    )

    assert store.recent_session_exclusions("session-12345678") == ["dz:1"]
    assert store.recent_session_exclusions("another-session") == []
    store.close()


def test_extension_identity_positive_and_negative_results_persist(tmp_path):
    path = tmp_path / "one_rec.sqlite3"
    store = LocalStore(path)
    store.put_extension_identity("dz:1", {"id": "sp1", "name": "Track"})
    store.put_extension_identity("dz:2", None)
    store.close()

    reopened = LocalStore(path)
    assert reopened.get_extension_identity("dz:1") == (
        True,
        {"id": "sp1", "name": "Track"},
    )
    assert reopened.get_extension_identity("dz:2") == (True, None)
    assert reopened.get_extension_identity("dz:3") == (False, None)
    reopened.close()


def test_feedback_store_has_session_query_indexes_and_busy_timeout(tmp_path):
    store = LocalStore(tmp_path / "one_rec.sqlite3")
    index_names = {
        row["name"]
        for row in store._db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
    }
    busy_timeout = store._db.execute("PRAGMA busy_timeout").fetchone()[0]

    assert "idx_impressions_session_created" in index_names
    assert "idx_impression_items_catalog" in index_names
    assert busy_timeout == 5000
    store.close()
