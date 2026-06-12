"""
recommendation_engine.py
========================
Core recommendation engine for the Semantic Movie Recommendation system.

Loads movie metadata, builds or loads a pre-computed embedding index, and
exposes a recommend() method that returns ranked movie suggestions for a
free-text query using cosine similarity over sentence embeddings.

Author: Umesh Pandey
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from embedding_service import EmbeddingService
from text_preprocessor import TextPreprocessor

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# Expected CSV columns
_REQUIRED_COLUMNS = {"movie_id", "title", "genre", "year", "plot", "rating"}


class RecommendationEngine:
    """
    End-to-end recommendation engine that combines:

    * Metadata loading (CSV → DataFrame)
    * Document construction via :class:`~text_preprocessor.TextPreprocessor`
    * Embedding generation / loading via :class:`~embedding_service.EmbeddingService`
    * Cosine-similarity search and result enrichment

    Example usage::

        from embedding_service import EmbeddingService

        engine = RecommendationEngine(
            metadata_path="embeddings/movies_metadata.csv",
            embeddings_path="embeddings/movie_embeddings.npy",
        )
        svc = EmbeddingService()
        svc.load_model()
        engine.load_metadata()
        engine.build_index(svc)

        results = engine.recommend("space exploration adventure", k=5)
        for r in results:
            print(r["title"], r["score"])
    """

    def __init__(
        self,
        metadata_path: str,
        embeddings_path: Optional[str] = None,
    ) -> None:
        """
        Initialise the RecommendationEngine.

        Args:
            metadata_path:   Path to ``movies_metadata.csv``.
            embeddings_path: Optional path to a pre-computed
                             ``movie_embeddings.npy`` file.  If ``None`` or
                             the file is absent, embeddings are generated at
                             :meth:`build_index` time.
        """
        self.metadata_path = metadata_path
        self.embeddings_path = embeddings_path

        self._df: Optional[pd.DataFrame] = None
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_service: Optional[EmbeddingService] = None
        self._preprocessor = TextPreprocessor()

        logger.info(
            "RecommendationEngine created (metadata=%s, embeddings=%s).",
            metadata_path,
            embeddings_path,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_metadata(self) -> pd.DataFrame:
        """
        Load movie metadata from the CSV file into a DataFrame.

        Returns:
            Pandas DataFrame with at least the columns ``movie_id``, ``title``,
            ``genre``, ``year``, ``plot``, and ``rating``.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError:        If required columns are missing.
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: '{self.metadata_path}'."
            )

        logger.info("Loading metadata from '%s'…", self.metadata_path)
        df = pd.read_csv(self.metadata_path)

        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Metadata CSV is missing required columns: {missing}."
            )

        # Sanitise: fill NaN plots / genres with empty strings
        df["plot"] = df["plot"].fillna("")
        df["genre"] = df["genre"].fillna("Unknown")
        df["year"] = df["year"].fillna(0).astype(int)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)

        self._df = df.reset_index(drop=True)
        logger.info(
            "Loaded %d movies from metadata CSV.", len(self._df)
        )
        return self._df

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, embedding_service: EmbeddingService) -> None:
        """
        Build (or load) the embedding index for all movies in the metadata.

        If ``embeddings_path`` points to an existing ``.npy`` file it is
        loaded directly; otherwise, each movie's document is constructed via
        :meth:`~text_preprocessor.TextPreprocessor.create_movie_document`
        and embedded in batch.

        Args:
            embedding_service: A loaded :class:`~embedding_service.EmbeddingService`
                                instance.

        Raises:
            RuntimeError: If metadata has not been loaded first.
        """
        if self._df is None:
            raise RuntimeError(
                "Metadata not loaded. Call load_metadata() before build_index()."
            )

        self._embedding_service = embedding_service

        # Attempt to load pre-computed embeddings
        if (
            self.embeddings_path
            and os.path.exists(self.embeddings_path)
        ):
            logger.info(
                "Loading pre-computed embeddings from '%s'…",
                self.embeddings_path,
            )
            self._embeddings = embedding_service.load_precomputed_embeddings(
                self.embeddings_path
            )
            if self._embeddings.shape[0] != len(self._df):
                logger.warning(
                    "Embedding count (%d) ≠ metadata count (%d). "
                    "Re-generating embeddings.",
                    self._embeddings.shape[0],
                    len(self._df),
                )
                self._embeddings = None

        # Generate embeddings if not loaded
        if self._embeddings is None:
            logger.info("Generating embeddings for %d movies…", len(self._df))
            documents = [
                self._preprocessor.create_movie_document(
                    title=row["title"],
                    genre=row["genre"],
                    plot=row["plot"],
                    year=int(row["year"]) if row["year"] else None,
                )
                for _, row in self._df.iterrows()
            ]
            self._embeddings = embedding_service.embed_batch(documents)
            logger.info(
                "Embeddings generated. Shape: %s.", str(self._embeddings.shape)
            )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k movie recommendations for a free-text query.

        Args:
            query:     Natural-language query string (e.g. ``'space adventure'``).
            k:         Number of recommendations to return.
            min_score: Minimum cosine similarity threshold (``0.0`` keeps all).

        Returns:
            List of result dictionaries each containing ``movie_id``, ``title``,
            ``genre``, ``year``, ``plot``, ``rating``, and ``score``, sorted by
            descending similarity.

        Raises:
            RuntimeError: If the index has not been built yet.
            ValueError:   If ``query`` is empty.
        """
        if self._embeddings is None or self._embedding_service is None:
            raise RuntimeError(
                "Index not built. Call build_index() first."
            )
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        logger.info("Recommend query='%s', k=%d, min_score=%.3f.", query, k, min_score)

        # Preprocess and embed the query
        normalised_query = self._preprocessor.normalize(query)
        query_vec = self._embedding_service.embed_text(normalised_query)

        # Retrieve top-k candidates (retrieve extra to allow score filtering)
        candidates = self._embedding_service.top_k_similar(
            query_vec, self._embeddings, k=min(k * 3, len(self._df))
        )

        # Enrich results and apply min_score filter
        results: List[Dict[str, Any]] = []
        for idx, score in candidates:
            if score < min_score:
                continue
            results.append(self._enrich_result(idx, score))
            if len(results) >= k:
                break

        logger.info("Returning %d results for query='%s'.", len(results), query)
        return results

    # ------------------------------------------------------------------
    # Individual movie access
    # ------------------------------------------------------------------

    def get_movie_by_id(self, movie_id: int) -> Dict[str, Any]:
        """
        Retrieve a single movie record by its integer ``movie_id``.

        Args:
            movie_id: The movie's unique integer identifier.

        Returns:
            Dictionary with all movie fields.

        Raises:
            RuntimeError:  If metadata has not been loaded.
            KeyError:      If no movie with ``movie_id`` exists.
        """
        if self._df is None:
            raise RuntimeError("Metadata not loaded. Call load_metadata() first.")

        matches = self._df[self._df["movie_id"] == movie_id]
        if matches.empty:
            raise KeyError(f"Movie with id={movie_id} not found.")

        row = matches.iloc[0]
        return row.to_dict()

    def get_all_movies(self) -> List[Dict[str, Any]]:
        """
        Return all movies as a list of dictionaries.

        Returns:
            List of movie record dictionaries.

        Raises:
            RuntimeError: If metadata has not been loaded.
        """
        if self._df is None:
            raise RuntimeError("Metadata not loaded. Call load_metadata() first.")

        return self._df.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def filter_by_genre(
        self, results: List[Dict[str, Any]], genre: str
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of result dictionaries to only include movies matching
        the given genre string (case-insensitive substring match).

        Args:
            results: List of recommendation result dicts (from :meth:`recommend`).
            genre:   Genre string to filter by (e.g. ``'sci-fi'``).

        Returns:
            Filtered list preserving original order.
        """
        genre_lower = genre.lower().strip()
        filtered = [
            r for r in results
            if genre_lower in str(r.get("genre", "")).lower()
        ]
        logger.debug(
            "filter_by_genre('%s'): %d/%d results kept.",
            genre,
            len(filtered),
            len(results),
        )
        return filtered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enrich_result(self, idx: int, score: float) -> Dict[str, Any]:
        """
        Build an enriched result dictionary from a DataFrame row index.

        Args:
            idx:   Integer row index into ``self._df``.
            score: Cosine similarity score.

        Returns:
            Dictionary with movie fields plus ``'score'`` key.
        """
        row = self._df.iloc[idx]
        return {
            "movie_id": int(row["movie_id"]),
            "title": str(row["title"]),
            "genre": str(row["genre"]),
            "year": int(row["year"]),
            "plot": str(row["plot"]),
            "rating": float(row["rating"]),
            "score": round(float(score), 6),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def movies_count(self) -> int:
        """Number of movies in the loaded metadata."""
        return len(self._df) if self._df is not None else 0

    @property
    def model_name(self) -> str:
        """Name of the underlying embedding model."""
        if self._embedding_service is not None:
            return self._embedding_service.model_name
        return "not_loaded"
