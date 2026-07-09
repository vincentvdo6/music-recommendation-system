"""
Engine factory: loads model artifacts with per-artifact graceful degradation.

item2vec embeddings are the only hard requirement — without them there is no
retrieval and create_engine returns None (the service then reports the engine
as unavailable). Every other artifact degrades independently:

    ranker      -> LinearFallbackRanker over the same v2 features
    item-NCF    -> ncf_score/has_ncf features are zero
    track meta  -> artist/popularity/duration features are zero, no proxy seeds
    mood        -> mood features are neutral constants
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from services.recommendation.audio_store import AudioStore
from services.recommendation.embeddings import EmbeddingService
from services.recommendation.engine import DEFAULT_DISCOVERY, DEFAULT_SEED_AFFINITY, RecommendationEngine
from services.recommendation.mood import MoodPredictor
from services.recommendation.ncf import load_item_ncf
from services.recommendation.ranker import LightGBMRanker, LinearFallbackRanker
from services.recommendation.track_meta import TrackMetaStore

logger = logging.getLogger(__name__)

ITEM2VEC_PATH = "models/item2vec/item2vec.wordvectors"
ANN_INDEX_PATH = "models/item2vec/ann_index"
RANKER_PATH = "models/ranker/lightgbm_ranker_v2.txt"
NCF_PATH = "models/ncf/ncf_item_v2.pt"
TRACK_META_PATH = "models/meta/track_meta.parquet"
MOOD_PATH = "models/mood_predictor/mood_predictor.pkl"
AUDIO_EMB_PATH = "models/audio/audio_emb.parquet"


def create_engine(
    item2vec_path: str = ITEM2VEC_PATH,
    ann_index_path: str = ANN_INDEX_PATH,
    ranker_path: str = RANKER_PATH,
    ncf_path: str = NCF_PATH,
    track_meta_path: str = TRACK_META_PATH,
    mood_path: str = MOOD_PATH,
    audio_path: str = AUDIO_EMB_PATH,
) -> Optional[RecommendationEngine]:
    """Build the engine from whatever artifacts exist on disk."""
    if not Path(item2vec_path).exists():
        logger.error("item2vec embeddings not found at %s — engine unavailable", item2vec_path)
        return None

    embeddings = EmbeddingService(
        item2vec_path=item2vec_path,
        ann_index_path=ann_index_path if Path(ann_index_path).exists() else None,
    )
    if embeddings.item2vec.wv is None:
        logger.error("Failed to load item2vec embeddings — engine unavailable")
        return None

    try:
        ranker = LightGBMRanker(ranker_path)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.info("LightGBM ranker unavailable (%s) — using linear fallback", exc)
        ranker = LinearFallbackRanker()

    ncf = load_item_ncf(ncf_path)

    track_meta = TrackMetaStore()
    if Path(track_meta_path).exists():
        try:
            track_meta.load(track_meta_path)
        except Exception as exc:
            logger.warning("Failed to load track metadata: %s", exc)

    mood = MoodPredictor()
    if Path(mood_path).exists():
        try:
            mood.load(mood_path)
        except Exception as exc:
            logger.warning("Failed to load mood predictor: %s", exc)

    audio = AudioStore()
    if Path(audio_path).exists():
        try:
            audio.load(audio_path)
        except Exception as exc:
            logger.warning("Failed to load audio embeddings: %s", exc)

    def env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except ValueError:
            logger.warning("Invalid %s env value — using default %.1f", name, default)
            return default

    seed_affinity = env_float("SEED_AFFINITY", DEFAULT_SEED_AFFINITY)
    discovery = env_float("DISCOVERY", DEFAULT_DISCOVERY)

    engine = RecommendationEngine(
        embeddings=embeddings,
        ranker=ranker,
        mood=mood,
        ncf=ncf,
        track_meta=track_meta,
        audio=audio,
        seed_affinity=seed_affinity,
        discovery=discovery,
    )

    logger.info(
        "Engine capabilities | embeddings: %d tracks | ann: %s | ranker: %s | ncf: %s | track_meta: %s | mood: %s | audio: %s | seed_affinity: %.1f | discovery: %.1f",
        len(embeddings.item2vec.wv),
        "yes" if embeddings.ann_index else "no (gensim fallback)",
        ranker.name,
        f"{ncf.vocab_size} tracks" if ncf else "no",
        "yes" if track_meta.available else "no",
        "yes" if mood.available else "no",
        f"{len(audio._row_of)} tracks" if audio.available else "no",
        seed_affinity,
        discovery,
    )
    return engine


_engine_instance: Optional[RecommendationEngine] = None
_engine_loaded = False


def get_engine() -> Optional[RecommendationEngine]:
    """Process-wide singleton (models are memory-mapped once)."""
    global _engine_instance, _engine_loaded
    if not _engine_loaded:
        _engine_instance = create_engine()
        _engine_loaded = True
    return _engine_instance
