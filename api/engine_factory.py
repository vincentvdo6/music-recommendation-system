"""
Factory for initializing recommendation engines.

Handles loading models and graceful fallbacks.
"""

import logging
from pathlib import Path
from typing import Optional

from services.recommendation.hybrid_engine import HybridRecommendationEngine
from services.recommendation.contextual_engine import ContextualRecommendationEngine
from services.recommendation.embeddings import EmbeddingService
from services.recommendation.ranker import LearnedRanker
from services.recommendation.catalogue import TrackCatalogue

logger = logging.getLogger(__name__)


class EngineFactory:
    """Factory for creating recommendation engines with model loading."""

    @staticmethod
    def create_engine(
        use_hybrid: bool = True,
        item2vec_path: str = "models/item2vec/item2vec.wordvectors",
        ann_index_path: str = "models/item2vec/ann_index",
        ranker_path: str = "models/ranker/lightgbm_ranker.txt",
        use_mmr: bool = True,
    ):
        """
        Create recommendation engine with optional model loading.

        Args:
            use_hybrid: Try to use hybrid engine (falls back to baseline if models missing)
            item2vec_path: Path to item2vec embeddings
            ann_index_path: Path to ANN index
            ranker_path: Path to learned ranker
            use_mmr: Use MMR diversity reranking

        Returns:
            HybridRecommendationEngine or ContextualRecommendationEngine
        """
        # Load catalogue (always required)
        catalogue = TrackCatalogue()

        if not use_hybrid:
            logger.info("Using baseline ContextualRecommendationEngine")
            return ContextualRecommendationEngine(catalogue=catalogue)

        # Try to load item2vec
        embedding_service = None
        if Path(item2vec_path).exists():
            try:
                logger.info(f"Loading item2vec from {item2vec_path}")
                embedding_service = EmbeddingService(
                    item2vec_path=item2vec_path,
                    ann_index_path=ann_index_path if Path(ann_index_path).exists() else None,
                )
                logger.info("✓ Item2vec loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load item2vec: {e}")
                embedding_service = None
        else:
            logger.info(f"Item2vec not found at {item2vec_path}, using baseline")

        # Try to load ranker
        ranker = None
        if Path(ranker_path).exists():
            try:
                logger.info(f"Loading ranker from {ranker_path}")
                ranker = LearnedRanker(model_path=ranker_path)
                logger.info("✓ Ranker loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load ranker: {e}")
                ranker = None
        else:
            logger.info(f"Ranker not found at {ranker_path}, using SimpleRanker")

        # Determine if we should use MMR
        use_mmr_final = use_mmr and (embedding_service is not None)

        # Create hybrid engine
        if embedding_service or ranker:
            logger.info("Creating HybridRecommendationEngine")
            logger.info(f"  - Embeddings: {'✓' if embedding_service else '✗'}")
            logger.info(f"  - Learned ranker: {'✓' if ranker else '✗'}")
            logger.info(f"  - MMR diversity: {'✓' if use_mmr_final else '✗'}")

            return HybridRecommendationEngine(
                catalogue=catalogue,
                embedding_service=embedding_service,
                ranker=ranker,
                use_mmr=use_mmr_final,
            )
        else:
            logger.info("No models found, using baseline ContextualRecommendationEngine")
            return ContextualRecommendationEngine(catalogue=catalogue)


# Singleton instance (initialized on first import)
_engine_instance: Optional[HybridRecommendationEngine] = None


def get_engine():
    """Get singleton recommendation engine instance."""
    global _engine_instance

    if _engine_instance is None:
        _engine_instance = EngineFactory.create_engine(use_hybrid=True)

    return _engine_instance
