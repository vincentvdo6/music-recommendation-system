"""Extension catalog: audio-ANN retrieval channel + catalog stores.

The extension artifacts (harvested tracks beyond the MPD vocabulary) are
optional — without them every path here must leave the pipeline untouched.
"""

import numpy as np
import pandas as pd


def _audio_row(track_id, axis, dim=16, noise=0.0, rng=None):
    vec = np.zeros(dim, dtype=np.float32)
    vec[axis] = 1.0
    if noise and rng is not None:
        vec = vec + rng.normal(0, noise, dim)
    return {"track_id": track_id, "embedding": vec.astype(np.float16).tobytes()}


# --------------------------------------------------------------- AudioStore


def test_nearest_returns_cosine_order_and_excludes(audio_store):
    q = np.zeros(16, dtype=np.float32)
    q[0] = 1.0  # the t-cluster audio axis
    got = audio_store.nearest(q, k=5)
    assert len(got) == 5
    assert all(t.startswith("t") for t in got)  # u-cluster is orthogonal

    excluded = set(got[:2])
    got2 = audio_store.nearest(q, k=5, exclude=excluded)
    assert not excluded & set(got2)


def test_audio_extension_appends_without_clobbering(audio_store, tmp_path):
    base_size = audio_store.size
    t00_before = audio_store.vector("t00").copy()

    ext = pd.DataFrame([
        _audio_row("dz:111", axis=0),                       # new, near t-cluster
        {"track_id": "t00", "embedding": (np.ones(16, dtype=np.float16) * 9).tobytes()},
    ])
    path = tmp_path / "extension_audio_emb.parquet"
    ext.to_parquet(path)
    audio_store.load_extension(str(path))

    assert audio_store.size == base_size + 1  # t00 conflict ignored
    assert np.allclose(audio_store.vector("t00"), t00_before)

    q = np.zeros(16, dtype=np.float32)
    q[0] = 1.0
    assert "dz:111" in audio_store.nearest(q, k=3)  # exact axis vector ranks top


# ------------------------------------------------------------ TrackMetaStore


def test_meta_extension_maps_harvest_schema(track_meta, tmp_path):
    ext = pd.DataFrame([
        {"id": "dz:111", "deezer_id": 111, "title": "New Song",
         "artist_deezer": "Fresh Artist", "artist_mpd": "", "duration_ms": 201000,
         "pos": 0, "stage": "related", "kind": "new", "matched_track_id": "",
         "embedded": True},
        {"id": "extmpd1", "deezer_id": 222, "title": "Deep Cut",
         "artist_deezer": "Artist 0 ", "artist_mpd": "Artist 0", "duration_ms": 185000,
         "pos": 7, "stage": "artist", "kind": "mpd_ext", "matched_track_id": "extmpd1",
         "embedded": True},
        # No audio vector -> unreachable by any channel -> must be filtered out.
        {"id": "dz:999", "deezer_id": 999, "title": "Never Embedded",
         "artist_deezer": "Nobody", "artist_mpd": "", "duration_ms": 100000,
         "pos": 90, "stage": "artist", "kind": "new", "matched_track_id": "",
         "embedded": False},
    ])
    path = tmp_path / "extension_tracks.parquet"
    ext.to_parquet(path)
    base_len = len(track_meta._df)
    track_meta.load_extension(str(path))

    assert len(track_meta._df) == base_len + 2
    row = track_meta.lookup(["dz:111"]).iloc[0]
    assert row["name"] == "New Song"
    assert row["artist"] == "Fresh Artist"
    assert track_meta.artists(["dz:111", "extmpd1"]) == ["fresh artist", "artist 0"]
    # extension tracks carry no popularity prior
    assert track_meta.log_pop(["dz:111"])[0] == 0.0
    # same-artist proxy lookup now sees the extension row too
    assert "extmpd1" in track_meta.tracks_by_artist("Artist 0", limit=30)


def test_meta_extension_never_overrides_base(track_meta, tmp_path):
    original = track_meta.lookup(["t00"]).iloc[0]["name"]
    ext = pd.DataFrame([
        {"id": "t00", "deezer_id": 1, "title": "WRONG NAME", "artist_deezer": "X",
         "artist_mpd": "", "duration_ms": 1000, "pos": 0, "stage": "chart",
         "kind": "backfill", "matched_track_id": "t00", "embedded": True},
    ])
    path = tmp_path / "ext.parquet"
    ext.to_parquet(path)
    track_meta.load_extension(str(path))
    assert track_meta.lookup(["t00"]).iloc[0]["name"] == original


# ------------------------------------------------------------------- Engine


def _engine_with_audio(embeddings, track_meta, audio_store):
    from services.recommendation.engine import RecommendationEngine
    from services.recommendation.ranker import LinearFallbackRanker

    return RecommendationEngine(
        embeddings=embeddings,
        ranker=LinearFallbackRanker(),
        track_meta=track_meta,
        audio=audio_store,
    )


def test_audio_channel_retrieves_out_of_vocab_tracks(embeddings, track_meta, audio_store, tmp_path):
    # dz:900 exists ONLY in the extension: audio near the t-cluster, no i2v vector.
    ext_audio = pd.DataFrame([_audio_row("dz:900", axis=0)])
    ext_audio.to_parquet(tmp_path / "ea.parquet")
    audio_store.load_extension(str(tmp_path / "ea.parquet"))
    ext_meta = pd.DataFrame([
        {"id": "dz:900", "deezer_id": 900, "title": "Unreachable Gem",
         "artist_deezer": "Ghost Artist", "artist_mpd": "", "duration_ms": 200000,
         "pos": 0, "stage": "related", "kind": "new", "matched_track_id": "",
         "embedded": True},
    ])
    ext_meta.to_parquet(tmp_path / "et.parquet")
    track_meta.load_extension(str(tmp_path / "et.parquet"))

    engine = _engine_with_audio(embeddings, track_meta, audio_store)
    seed_audio = engine._seed_audio("t10", None)
    assert seed_audio is not None
    ids, ctx = engine._retrieve([], "t10", "t10", seed_audio)
    assert "dz:900" in ids  # unreachable by co-occurrence, reached by sound
    assert ctx.seed_audio_vec is not None


def test_audio_channel_inert_without_extension(embeddings, track_meta, audio_store):
    # Audio channel only adds in-catalog tracks here — everything it returns
    # already has an i2v vector, and the pipeline shape is unchanged.
    engine = _engine_with_audio(embeddings, track_meta, audio_store)
    recs = engine.recommend(["t00", "t01", "t02"], input_track_id="t10", limit=5)
    assert recs
    assert all(not r["id"].startswith("dz:") for r in recs)


def test_extension_track_can_reach_results_when_floor_allows(
    embeddings, track_meta, audio_store, tmp_path
):
    from services.recommendation.policy import ScoringPolicy

    ext_audio = pd.DataFrame([_audio_row("dz:901", axis=0)])
    ext_audio.to_parquet(tmp_path / "ea2.parquet")
    audio_store.load_extension(str(tmp_path / "ea2.parquet"))
    ext_meta = pd.DataFrame([
        {"id": "dz:901", "deezer_id": 901, "title": "Sound Twin",
         "artist_deezer": "Ghost Artist", "artist_mpd": "", "duration_ms": 200000,
         "pos": 0, "stage": "related", "kind": "new", "matched_track_id": "",
         "embedded": True},
    ])
    ext_meta.to_parquet(tmp_path / "et2.parquet")
    track_meta.load_extension(str(tmp_path / "et2.parquet"))

    engine = _engine_with_audio(embeddings, track_meta, audio_store)
    # The current seed-cos floor is i2v-based, which walls off no-i2v tracks by
    # design until the per-channel floor lands with the next retrain; with the
    # floor off, the extension track must be able to reach the final list.
    engine.policy = ScoringPolicy(seed_affinity=0.0, discovery=0.0, floor=0.0)
    recs = engine.recommend([], input_track_id="t10", limit=60)
    ranks = {r["id"]: i for i, r in enumerate(recs)}
    assert "dz:901" in ranks
    # Sound-only evidence beats no-evidence: the extension track outranks nearly
    # all u-cluster tracks (orthogonal in both i2v and audio space) — a couple
    # may still edge ahead on the fallback ranker's popularity weight.
    u_ranks = [ranks[u] for u in ranks if u.startswith("u")]
    assert sum(ranks["dz:901"] < ur for ur in u_ranks) >= len(u_ranks) - 3
    twin = next(r for r in recs if r["id"] == "dz:901")
    assert twin["name"] == "Sound Twin"


def test_seed_audio_prefers_the_searched_tracks_own_vector(
    embeddings, track_meta, audio_store, tmp_path
):
    # The searched track is OUT of the i2v vocab (resolves via artist proxy),
    # but its own audio vector exists — that vector must win over the proxy's.
    ext_audio = pd.DataFrame([_audio_row("spotify_new_song", axis=1)])  # u-axis sound
    ext_audio.to_parquet(tmp_path / "ea3.parquet")
    audio_store.load_extension(str(tmp_path / "ea3.parquet"))

    engine = _engine_with_audio(embeddings, track_meta, audio_store)
    vec = engine._seed_audio("t10", "Artist 5", "spotify_new_song")
    assert vec is not None
    assert vec[1] > 0.9  # the track's own (u-axis) sound, not t10's t-axis audio


# ----------------------------------------------------------- contract v4


def test_channel_indicator_features(embeddings, track_meta, audio_store, tmp_path):
    import services.recommendation.features as F

    ext_audio = pd.DataFrame([_audio_row("dz:902", axis=0)])
    ext_audio.to_parquet(tmp_path / "ea4.parquet")
    audio_store.load_extension(str(tmp_path / "ea4.parquet"))
    ext_meta = pd.DataFrame([
        {"id": "dz:902", "deezer_id": 902, "title": "Channel Test",
         "artist_deezer": "Ghost", "artist_mpd": "", "duration_ms": 200000,
         "pos": 0, "stage": "related", "kind": "new", "matched_track_id": "",
         "embedded": True},
    ])
    ext_meta.to_parquet(tmp_path / "et4.parquet")
    track_meta.load_extension(str(tmp_path / "et4.parquet"))

    engine = _engine_with_audio(embeddings, track_meta, audio_store)
    seed_audio = engine._seed_audio("t10", None)
    ids, ctx = engine._retrieve([], "t10", "t10", seed_audio)
    assert "dz:902" in ids and ctx.audio_rank  # audio channel populated

    vectors = engine._vectors(ids)
    X = F.build_matrix(ids, vectors, track_meta.artists(ids),
                       track_meta.log_pop(ids), track_meta.durations_ms(ids),
                       None, None, None, ctx,
                       audio_vecs=audio_store.matrix_for(ids)[0])
    row = X.iloc[ids.index("dz:902")]
    assert row["has_i2v"] == 0.0        # no co-occurrence vector by design
    assert row["audio_seed_rr"] > 0.0   # retrieved by the audio channel
    assert row["has_audio"] == 1.0
    in_vocab = X.iloc[ids.index([i for i in ids if i.startswith("t")][0])]
    assert in_vocab["has_i2v"] == 1.0

    # effective seed cos: acoustic for the extension row, co-listen otherwise
    eff = F.effective_seed_cos(X)
    i = ids.index("dz:902")
    assert eff[i] == row["audio_cos_seed"]
    j = ids.index([x for x in ids if x.startswith("t")][0])
    assert eff[j] == X.iloc[j]["seed_i2v_cos"]


def test_extension_track_passes_default_floor(embeddings, track_meta, audio_store, tmp_path):
    # v4 policy floors extension candidates on ACOUSTIC closeness — a sound-
    # twin extension track is eligible under the default policy, no override.
    ext_audio = pd.DataFrame([_audio_row("dz:903", axis=0)])
    ext_audio.to_parquet(tmp_path / "ea5.parquet")
    audio_store.load_extension(str(tmp_path / "ea5.parquet"))
    ext_meta = pd.DataFrame([
        {"id": "dz:903", "deezer_id": 903, "title": "Twin", "artist_deezer": "Ghost",
         "artist_mpd": "", "duration_ms": 200000, "pos": 0, "stage": "related",
         "kind": "new", "matched_track_id": "", "embedded": True},
    ])
    ext_meta.to_parquet(tmp_path / "et5.parquet")
    track_meta.load_extension(str(tmp_path / "et5.parquet"))

    engine = _engine_with_audio(embeddings, track_meta, audio_store)
    recs = engine.recommend([], input_track_id="t10", limit=60)
    ranks = {r["id"]: i for i, r in enumerate(recs)}
    assert "dz:903" in ranks
    # audio cos ~1.0 to the seed: it must NOT be relegated to ineligible filler
    # (the old i2v floor pushed every extension track behind all 29 eligibles)
    assert ranks["dz:903"] < 29
