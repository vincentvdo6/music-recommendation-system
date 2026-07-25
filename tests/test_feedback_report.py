import pandas as pd

from scripts.report_feedback import build_profile_report


def test_profile_report_counts_feedback_discovery_and_repeats():
    impressions = pd.DataFrame([
        {"impression_id": "i1", "session_id": "s1", "profile": "balanced"},
        {"impression_id": "i2", "session_id": "s1", "profile": "balanced"},
        {"impression_id": "i3", "session_id": "s2", "profile": "explorer"},
    ])
    exposures = pd.DataFrame([
        {"impression_id": "i1", "session_id": "s1", "profile": "balanced",
         "track_id": "a", "catalog_track_id": "a", "discovery": 0},
        {"impression_id": "i2", "session_id": "s1", "profile": "balanced",
         "track_id": "a", "catalog_track_id": "a", "discovery": 0},
        {"impression_id": "i3", "session_id": "s2", "profile": "explorer",
         "track_id": "b", "catalog_track_id": "dz:b", "discovery": 1},
    ])
    feedback = pd.DataFrame([
        {"impression_id": "i1", "profile": "balanced", "track_id": "a", "event": "view"},
        {"impression_id": "i1", "profile": "balanced", "track_id": "a", "event": "like"},
        {"impression_id": "i3", "profile": "explorer", "track_id": "b", "event": "view"},
        {"impression_id": "i3", "profile": "explorer",
         "track_id": "b", "event": "not_similar_enough"},
    ])

    report = build_profile_report(impressions, exposures, feedback)

    assert report.loc["balanced", "strong_positive_rate"] == 0.5
    assert report.loc["balanced", "repeat_exposure_rate"] == 0.5
    assert report.loc["explorer", "discovery_share"] == 1.0
    assert report.loc["explorer", "not_similar_enough_per_impression"] == 1.0
