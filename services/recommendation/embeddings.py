"""
Embedding service for track representations.

Provides item2vec embeddings and fast nearest-neighbor search using Annoy.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


class Item2VecModel:
    """
    Item2vec embedding model trained on playlist co-occurrence.

    Uses Word2Vec skip-gram on playlist sequences to learn
    "goes-with" relationships between tracks.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.wv: Optional[KeyedVectors] = None
        self.dim: int = 0

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def load(self, model_path: str):
        """Load trained word vectors."""
        logger.info(f"Loading item2vec model from {model_path}")
        self.wv = KeyedVectors.load(model_path, mmap='r')
        self.dim = self.wv.vector_size
        logger.info(f"Loaded {len(self.wv):,} track embeddings (dim={self.dim})")

    def has_vector(self, track_id: str) -> bool:
        """Check if track has an embedding."""
        if self.wv is None:
            return False
        return track_id in self.wv

    def vector(self, track_id: str) -> Optional[np.ndarray]:
        """Get embedding for a single track."""
        if not self.has_vector(track_id):
            return None
        return self.wv[track_id]

    def mean_vector(self, track_ids: List[str]) -> Optional[np.ndarray]:
        """
        Compute mean embedding for a list of tracks.

        Returns None if no tracks have embeddings.
        """
        if self.wv is None:
            return None

        vectors = []
        for track_id in track_ids:
            if self.has_vector(track_id):
                vectors.append(self.wv[track_id])

        if not vectors:
            return None

        return np.mean(vectors, axis=0)

    def neighbors(self, track_id: str, k: int = 200) -> List[Tuple[str, float]]:
        """
        Get k nearest neighbors by cosine similarity.

        Returns list of (track_id, similarity_score) tuples.
        """
        if not self.has_vector(track_id):
            return []

        try:
            return self.wv.most_similar(track_id, topn=k)
        except KeyError:
            return []

    def similarity(self, track_id1: str, track_id2: str) -> float:
        """Compute cosine similarity between two tracks."""
        if not (self.has_vector(track_id1) and self.has_vector(track_id2)):
            return 0.0

        try:
            return float(self.wv.similarity(track_id1, track_id2))
        except KeyError:
            return 0.0


class ANNIndex:
    """
    Approximate Nearest Neighbor index using Annoy.

    Enables fast retrieval from embeddings for real-time recommendations.
    """

    def __init__(self, dim: int, metric: str = "angular"):
        """
        Initialize ANN index.

        Args:
            dim: Embedding dimension
            metric: 'angular' (cosine), 'euclidean', or 'manhattan'
        """
        try:
            from annoy import AnnoyIndex
            self.index = AnnoyIndex(dim, metric)
        except ImportError:
            logger.error("Annoy not installed. Run: pip install annoy")
            raise

        self.dim = dim
        self.metric = metric
        self.id_map: List[str] = []
        self.id_to_pos: Dict[str, int] = {}

    def build(self, id_to_vec: Dict[str, np.ndarray], n_trees: int = 50):
        """
        Build index from track embeddings.

        Args:
            id_to_vec: Mapping of track_id -> embedding vector
            n_trees: More trees = better accuracy, slower build
        """
        logger.info(f"Building ANN index with {len(id_to_vec):,} vectors")

        self.id_map = list(id_to_vec.keys())
        self.id_to_pos = {tid: i for i, tid in enumerate(self.id_map)}

        for tid, vec in id_to_vec.items():
            pos = self.id_to_pos[tid]
            vec_array = vec.tolist() if isinstance(vec, np.ndarray) else vec
            self.index.add_item(pos, vec_array)

        logger.info(f"Building index with {n_trees} trees...")
        self.index.build(n_trees)
        logger.info("Index built!")

    def query(self, vec: np.ndarray, k: int = 200) -> List[str]:
        """
        Find k nearest neighbors to query vector.

        Returns list of track IDs.
        """
        vec_array = vec.tolist() if isinstance(vec, np.ndarray) else vec
        positions = self.index.get_nns_by_vector(vec_array, k, include_distances=False)
        return [self.id_map[p] for p in positions]

    def query_by_id(self, track_id: str, k: int = 200) -> List[str]:
        """Find k nearest neighbors to a track by ID."""
        if track_id not in self.id_to_pos:
            return []

        pos = self.id_to_pos[track_id]
        positions = self.index.get_nns_by_item(pos, k + 1, include_distances=False)

        # Exclude the query track itself
        return [self.id_map[p] for p in positions if self.id_map[p] != track_id]

    def save(self, path: str):
        """Save index to disk."""
        import json
        import os

        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save Annoy index - use absolute path for Windows compatibility
        index_path = output_path / "annoy.index"
        abs_index_path = os.path.abspath(str(index_path))
        self.index.save(abs_index_path)
        logger.info(f"Saved Annoy index to {index_path}")

        # Save ID mapping
        id_map_path = output_path / "id_map.json"
        with open(id_map_path, 'w') as f:
            json.dump(self.id_map, f)
        logger.info(f"Saved ID map to {id_map_path}")

    def load(self, path: str):
        """Load index from disk."""
        import json

        input_path = Path(path)

        # Load Annoy index
        index_path = input_path / "annoy.index"
        self.index.load(str(index_path))
        logger.info(f"Loaded Annoy index from {index_path}")

        # Load ID mapping
        id_map_path = input_path / "id_map.json"
        with open(id_map_path, 'r') as f:
            self.id_map = json.load(f)

        item_count = self.index.get_n_items()
        if len(self.id_map) != item_count:
            raise ValueError(
                f"ANN ID map has {len(self.id_map)} entries, but the index has {item_count} items"
            )

        self.id_to_pos = {tid: i for i, tid in enumerate(self.id_map)}
        logger.info(f"Loaded ID map with {len(self.id_map):,} tracks")


class EmbeddingService:
    """
    Unified item2vec embedding service.

    Provides fast nearest neighbor search and similarity computation.
    """

    def __init__(
        self,
        item2vec_path: Optional[str] = None,
        ann_index_path: Optional[str] = None,
    ):
        self.item2vec = Item2VecModel(item2vec_path)
        self.ann_index: Optional[ANNIndex] = None

        if ann_index_path and Path(ann_index_path).exists():
            try:
                self.load_ann_index(ann_index_path)
            except Exception as exc:
                logger.warning("ANN index unavailable (%s) — using gensim fallback", exc)
                self.ann_index = None

    def load_ann_index(self, path: str):
        """Load pre-built ANN index."""
        if self.item2vec.wv is None:
            logger.warning("Item2vec not loaded, cannot load ANN index")
            return

        ann_index = ANNIndex(self.item2vec.dim, metric="angular")
        ann_index.load(path)
        self.ann_index = ann_index

    def build_ann_index(self, n_trees: int = 50) -> Optional[ANNIndex]:
        """Build ANN index from item2vec embeddings."""
        if self.item2vec.wv is None:
            raise ValueError("Item2vec model not loaded")

        try:
            logger.info("Building ANN index from item2vec embeddings")

            # Extract all embeddings
            id_to_vec = {
                track_id: self.item2vec.wv[track_id]
                for track_id in self.item2vec.wv.index_to_key
            }

            # Build index
            self.ann_index = ANNIndex(self.item2vec.dim, metric="angular")
            self.ann_index.build(id_to_vec, n_trees=n_trees)

            return self.ann_index
        except ImportError:
            logger.warning("Annoy not installed. Skipping ANN index (will use gensim fallback)")
            logger.info("For small datasets (<10k tracks), gensim search is fast enough")
            self.ann_index = None
            return None

    def get_neighbors(
        self,
        track_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        k: int = 200,
    ) -> List[str]:
        """
        Get nearest neighbors using ANN index.

        Provide either track_id or embedding vector.
        """
        if self.ann_index is None:
            logger.debug("ANN index not built, using gensim search")
            if track_id:
                neighbors = self.item2vec.neighbors(track_id, k=k)
                return [tid for tid, _ in neighbors]
            elif embedding is not None:
                # Fallback: compute cosine similarity to all tracks
                return self._brute_force_neighbors(embedding, k=k)
            return []

        if track_id:
            return self.ann_index.query_by_id(track_id, k=k)
        elif embedding is not None:
            return self.ann_index.query(embedding, k=k)
        else:
            return []

    def _brute_force_neighbors(self, query_vec: np.ndarray, k: int = 200) -> List[str]:
        """Exact nearest neighbors via gensim's vectorized cosine search."""
        if self.item2vec.wv is None:
            return []
        if np.linalg.norm(query_vec) == 0:
            return []
        neighbors = self.item2vec.wv.similar_by_vector(query_vec, topn=k)
        return [track_id for track_id, _ in neighbors]

    def get_playlist_neighbors(
        self,
        track_ids: List[str],
        k: int = 500,
    ) -> List[str]:
        """
        Get neighbors for a playlist by mean-pooling embeddings.

        Returns candidate track IDs for recommendation.
        """
        # Compute mean embedding
        mean_vec = self.item2vec.mean_vector(track_ids)

        if mean_vec is None:
            logger.warning("No embeddings found for playlist tracks")
            return []

        # Query ANN index
        return self.get_neighbors(embedding=mean_vec, k=k)
