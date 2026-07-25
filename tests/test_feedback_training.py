"""Feedback export contracts."""

import pandas as pd

from scripts.export_feedback import main as export_main
from services.storage import LocalStore


def _rec(track_id, audio_similarity):
    return {
        "id": track_id,
        "rank_score": 0.5,
        "similarity_score": 0.4,
        "recommendation": {
            "components": {
                "audio_similarity": audio_similarity,
                "audio_retrieval_rr": 0.1,
            }
        },
    }


def test_export_does_not_train_on_view_only_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "feedback.sqlite3"
    output_path = tmp_path / "feedback.parquet"
    store = LocalStore(db_path)
    store.record_impression(
        impression_id="impression-1",
        request_id="request-1",
        seed_id="seed",
        mode="seed_only",
        playlist_ids=[],
        source="ml-seed",
        recommendations=[
            _rec("viewed", 0.2),
            _rec("liked", 0.8),
            _rec("dismissed", 0.3),
            _rec("too-familiar", 0.7),
            _rec("neutral", 0.5),
        ],
        candidates=[
            _rec("viewed", 0.2),
            _rec("liked", 0.8),
            _rec("dismissed", 0.3),
            _rec("too-familiar", 0.7),
            _rec("unserved", 0.6),
            _rec("neutral", 0.5),
        ],
        session_id="session-12345678",
        profile="explorer",
    )
    store.record_feedback(impression_id="impression-1", track_id="viewed", event="view")
    store.record_feedback(impression_id="impression-1", track_id="liked", event="like")
    store.record_feedback(impression_id="impression-1", track_id="dismissed", event="dismiss")
    store.record_feedback(
        impression_id="impression-1",
        track_id="too-familiar",
        event="not_similar_enough",
    )
    store.record_feedback(impression_id="impression-1", track_id="neutral", event="neutral")
    store.close()

    monkeypatch.setattr(
        "sys.argv", ["export_feedback.py", str(db_path), str(output_path)]
    )
    assert export_main() == 0

    exported = pd.read_parquet(output_path).set_index("track_id")
    assert exported.loc["viewed", "observed"] == 1
    assert exported.loc["viewed", "training_eligible"] == 0
    assert exported.loc["liked", "training_eligible"] == 1
    assert exported.loc["liked", "label"] == 3
    assert exported.loc["dismissed", "training_eligible"] == 0
    assert exported.loc["too-familiar", "training_eligible"] == 0
    assert exported.loc["neutral", "training_eligible"] == 0
    assert exported.loc["unserved", "observed"] == 0
    assert exported.loc["unserved", "training_eligible"] == 0
    assert pd.isna(exported.loc["unserved", "served_position"])
    assert exported.loc["liked", "ips_weight"] >= 1.0
