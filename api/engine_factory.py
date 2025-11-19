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
from services.recommendation.ncf_loader import load_ncf_model

logger = logging.getLogger(__name__)


class EngineFactory:
    """Factory for creating recommendation engines with model loading."""

    @staticmethod
    def create_engine(
        use_hybrid: bool = True,
        item2vec_path: str = "models/item2vec/item2vec.wordvectors",
        ann_index_path: str = "models/item2vec/ann_index",
        ranker_path: str = "models/ranker/lightgbm_ranker.txt",
        ncf_path: str = "models/ncf/neumf_bpr_best.pt",
        mood_predictor_path: str = "models/mood_predictor/mood_predictor.pkl",
        use_mmr: bool = True,
    ):
        """
        Create recommendation engine with optional model loading.

        Args:
            use_hybrid: Try to use hybrid engine (falls back to baseline if models missing)
            item2vec_path: Path to item2vec embeddings
            ann_index_path: Path to ANN index
            ranker_path: Path to learned ranker
            ncf_path: Path to NCF model checkpoint
            mood_predictor_path: Path to mood predictor model
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

        # Try to load NCF model (Phase 2)
        ncf_recommender = None
        if Path(ncf_path).exists():
            try:
                ncf_recommender = load_ncf_model(Path(ncf_path))
                if ncf_recommender:
                    logger.info("✓ NCF model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load NCF model: {e}")
                ncf_recommender = None
        else:
            logger.info(f"NCF model not found at {ncf_path}")

        # Determine if we should use MMR
        use_mmr_final = use_mmr and (embedding_service is not None)

        # Check mood predictor availability
        mood_predictor_available = Path(mood_predictor_path).exists()

        # Create hybrid engine
        if embedding_service or ranker or ncf_recommender:
            logger.info("Creating HybridRecommendationEngine")
            logger.info(f"  - Embeddings (item2vec): {'✓' if embedding_service else '✗'}")
            logger.info(f"  - Learned ranker: {'✓' if ranker else '✗'}")
            logger.info(f"  - NCF (deep learning): {'✓' if ncf_recommender else '✗'}")
            logger.info(f"  - Mood predictor: {'✓' if mood_predictor_available else '✗'}")
            logger.info(f"  - MMR diversity: {'✓' if use_mmr_final else '✗'}")

            return HybridRecommendationEngine(
                catalogue=catalogue,
                embedding_service=embedding_service,
                ranker=ranker,
                ncf_recommender=ncf_recommender,
                use_mmr=use_mmr_final,
                mood_predictor_path=mood_predictor_path if mood_predictor_available else None,
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
