"""Context-aware recommendation utilities built on a curated music catalogue."""

from .catalogue import TrackCatalogue  # noqa: F401
from .contextual_engine import ContextualRecommendationEngine  # noqa: F401
from .audio_similarity import AudioSimilarityEngine  # noqa: F401

__all__ = [
    "TrackCatalogue",
    "ContextualRecommendationEngine",
    "AudioSimilarityEngine",
]
