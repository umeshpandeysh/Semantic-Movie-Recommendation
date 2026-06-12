"""
test_recommendation_engine.py
==============================
Unit tests for EmbeddingService and RecommendationEngine.

Exercises:
    - EmbeddingService cosine similarity, top-k retrieval, type checking
    - RecommendationEngine with mocked pandas and EmbeddingService dependencies

Run with::

    pytest tests/test_recommendation_engine.py -v

Author: Umesh Pandey
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List

import numpy as np
import pandas as pd

# Allow importing from the backend package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from embedding_service import EmbeddingService  # noqa: E402
from recommendation_engine import RecommendationEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_matrix(n: int = 20, dim: int = 384, seed: int = 42) -> np.ndarray:
    """Create a reproducible random embedding matrix."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def _make_random_vector(dim: int = 384, seed: int = 99) -> np.ndarray:
    """Create a reproducible random unit vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _make_dummy_df(n: int = 30) -> pd.DataFrame:
    """Create a minimal movies DataFrame for testing."""
    return pd.DataFrame(
        {
            "movie_id": list(range(1, n + 1)),
            "title": [f"Movie {i}" for i in range(1, n + 1)],
            "genre": ["Drama" if i % 2 == 0 else "Sci-Fi" for i in range(1, n + 1)],
            "year": [2000 + i for i in range(1, n + 1)],
            "plot": [f"A plot about event {i} and its consequences." for i in range(1, n + 1)],
            "rating": [round(6.0 + (i % 30) * 0.1, 1) for i in range(1, n + 1)],
        }
    )


# ---------------------------------------------------------------------------
# EmbeddingService tests
# ---------------------------------------------------------------------------

class TestEmbeddingServiceCosineSimilarity(unittest.TestCase):
    """Tests for EmbeddingService.cosine_similarity_matrix()."""

    def setUp(self) -> None:
        self.svc = EmbeddingService()

    def test_output_shape_matches_corpus_size(self) -> None:
        """The similarity array should have the same length as the corpus."""
        query = _make_random_vector()
        corpus = _make_random_matrix(n=20)
        sims = self.svc.cosine_similarity_matrix(query, corpus)
        self.assertEqual(sims.shape, (20,))

    def test_self_similarity_is_one(self) -> None:
        """A vector's cosine similarity with itself should be 1.0."""
        vec = _make_random_vector()
        corpus = vec.reshape(1, -1)
        sim = self.svc.cosine_similarity_matrix(vec, corpus)
        self.assertAlmostEqual(float(sim[0]), 1.0, places=5)

    def test_scores_are_in_valid_range(self) -> None:
        """All cosine similarity scores should be in [-1, 1]."""
        query = _make_random_vector()
        corpus = _make_random_matrix(n=50)
        sims = self.svc.cosine_similarity_matrix(query, corpus)
        self.assertTrue(np.all(sims >= -1.0 - 1e-6))
        self.assertTrue(np.all(sims <= 1.0 + 1e-6))

    def test_returns_ndarray(self) -> None:
        """Return type should be np.ndarray."""
        query = _make_random_vector()
        corpus = _make_random_matrix(n=5)
        result = self.svc.cosine_similarity_matrix(query, corpus)
        self.assertIsInstance(result, np.ndarray)

    def test_dimension_mismatch_raises_value_error(self) -> None:
        """Mismatched dimensions should raise ValueError."""
        query = _make_random_vector(dim=384)
        corpus = _make_random_matrix(n=5, dim=256)
        with self.assertRaises(ValueError):
            self.svc.cosine_similarity_matrix(query, corpus)

    def test_two_d_query_raises_value_error(self) -> None:
        """A 2-D query vector should raise ValueError (must be 1-D)."""
        query = _make_random_vector().reshape(1, -1)  # 2-D
        corpus = _make_random_matrix(n=5)
        with self.assertRaises(ValueError):
            self.svc.cosine_similarity_matrix(query, corpus)


class TestEmbeddingServiceTopK(unittest.TestCase):
    """Tests for EmbeddingService.top_k_similar()."""

    def setUp(self) -> None:
        self.svc = EmbeddingService()
        self.corpus = _make_random_matrix(n=50)

    def test_returns_exactly_k_results(self) -> None:
        """top_k_similar should return exactly k results when corpus ≥ k."""
        query = _make_random_vector()
        results = self.svc.top_k_similar(query, self.corpus, k=10)
        self.assertEqual(len(results), 10)

    def test_k_larger_than_corpus_returns_all(self) -> None:
        """When k > corpus size, all corpus items should be returned."""
        small_corpus = _make_random_matrix(n=5)
        query = _make_random_vector()
        results = self.svc.top_k_similar(query, small_corpus, k=100)
        self.assertEqual(len(results), 5)

    def test_results_are_sorted_descending(self) -> None:
        """Results should be ordered from highest to lowest similarity."""
        query = _make_random_vector()
        results = self.svc.top_k_similar(query, self.corpus, k=10)
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_returns_list_of_tuples(self) -> None:
        """Each result should be a (int, float) tuple."""
        query = _make_random_vector()
        results = self.svc.top_k_similar(query, self.corpus, k=5)
        for item in results:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            idx, score = item
            self.assertIsInstance(idx, int)
            self.assertIsInstance(score, float)

    def test_index_within_corpus_bounds(self) -> None:
        """Returned indices should all be valid corpus row indices."""
        query = _make_random_vector()
        results = self.svc.top_k_similar(query, self.corpus, k=10)
        corpus_size = self.corpus.shape[0]
        for idx, _ in results:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, corpus_size)


class TestEmbeddingServiceEmbedText(unittest.TestCase):
    """Tests for EmbeddingService.embed_text() using the fallback embedder."""

    def setUp(self) -> None:
        self.svc = EmbeddingService()
        # Force the fallback hash-based embedder so no model download needed
        from embedding_service import _HashEmbedder
        self.svc._model = _HashEmbedder()

    def test_returns_ndarray(self) -> None:
        """embed_text should return a numpy ndarray."""
        result = self.svc.embed_text("A robot uprising in a dystopian future.")
        self.assertIsInstance(result, np.ndarray)

    def test_returns_one_dimensional_vector(self) -> None:
        """The returned embedding should be 1-D."""
        result = self.svc.embed_text("space exploration adventure")
        self.assertEqual(result.ndim, 1)

    def test_vector_is_unit_normalised(self) -> None:
        """The fallback hash embedder should produce unit-normalised vectors."""
        result = self.svc.embed_text("any text here")
        norm = float(np.linalg.norm(result))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_empty_text_raises_value_error(self) -> None:
        """An empty string should raise ValueError."""
        with self.assertRaises(ValueError):
            self.svc.embed_text("")

    def test_deterministic_for_same_input(self) -> None:
        """The hash embedder should return the same vector for the same text."""
        text = "machine learning project"
        v1 = self.svc.embed_text(text)
        v2 = self.svc.embed_text(text)
        np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# RecommendationEngine tests (with mocking)
# ---------------------------------------------------------------------------

class TestRecommendationEngine(unittest.TestCase):
    """Tests for RecommendationEngine using mocked dependencies."""

    N_MOVIES = 30

    def _build_engine(self) -> RecommendationEngine:
        """
        Build a RecommendationEngine pre-loaded with a dummy DataFrame and
        a hash-based EmbeddingService (no real model downloads).
        """
        from embedding_service import _HashEmbedder

        engine = RecommendationEngine(
            metadata_path="fake/path/movies_metadata.csv",
            embeddings_path=None,
        )
        # Inject dummy metadata directly (bypass file I/O)
        engine._df = _make_dummy_df(self.N_MOVIES)

        # Use a real EmbeddingService backed by the hash embedder
        svc = EmbeddingService()
        svc._model = _HashEmbedder()
        engine._embedding_service = svc

        # Generate embeddings directly from dummy documents
        from text_preprocessor import TextPreprocessor
        tp = TextPreprocessor()
        docs = [
            tp.create_movie_document(
                title=row["title"],
                genre=row["genre"],
                plot=row["plot"],
                year=row["year"],
            )
            for _, row in engine._df.iterrows()
        ]
        engine._embeddings = svc.embed_batch(docs)
        return engine

    # ------------------------------------------------------------------
    # recommend() tests
    # ------------------------------------------------------------------

    def test_recommend_returns_list(self) -> None:
        """recommend() should return a list."""
        engine = self._build_engine()
        result = engine.recommend("space adventure robot")
        self.assertIsInstance(result, list)

    def test_recommend_contains_title_key(self) -> None:
        """Each result dict should contain a 'title' key."""
        engine = self._build_engine()
        results = engine.recommend("space adventure robot", k=5)
        for r in results:
            self.assertIn("title", r)

    def test_recommend_contains_score_key(self) -> None:
        """Each result dict should contain a 'score' key."""
        engine = self._build_engine()
        results = engine.recommend("space adventure robot", k=5)
        for r in results:
            self.assertIn("score", r)

    def test_k_parameter_respected(self) -> None:
        """The number of results should not exceed k."""
        engine = self._build_engine()
        for k in (1, 5, 10, 20):
            results = engine.recommend("drama crime thriller", k=k)
            self.assertLessEqual(len(results), k)

    def test_min_score_filters_results(self) -> None:
        """Results with score below min_score should be excluded."""
        engine = self._build_engine()
        # Using a very high threshold should filter most results
        results = engine.recommend("test", k=10, min_score=0.99)
        for r in results:
            self.assertGreaterEqual(r["score"], 0.99)

    def test_recommend_raises_on_empty_query(self) -> None:
        """An empty query string should raise ValueError."""
        engine = self._build_engine()
        with self.assertRaises(ValueError):
            engine.recommend("")

    def test_recommend_raises_if_index_not_built(self) -> None:
        """Calling recommend without building the index should raise RuntimeError."""
        engine = RecommendationEngine(metadata_path="fake.csv")
        with self.assertRaises(RuntimeError):
            engine.recommend("query")

    # ------------------------------------------------------------------
    # filter_by_genre() tests
    # ------------------------------------------------------------------

    def test_filter_by_genre_returns_matching_results(self) -> None:
        """filter_by_genre should only return movies whose genre matches."""
        engine = self._build_engine()
        results = engine.recommend("adventure drama", k=20)
        sci_fi_results = engine.filter_by_genre(results, "sci-fi")
        for r in sci_fi_results:
            self.assertIn("sci-fi", r["genre"].lower())

    def test_filter_by_genre_no_match_returns_empty(self) -> None:
        """A genre with no matches should return an empty list."""
        engine = self._build_engine()
        results = engine.recommend("drama film", k=10)
        filtered = engine.filter_by_genre(results, "xyznonexistent")
        self.assertEqual(filtered, [])

    def test_filter_by_genre_case_insensitive(self) -> None:
        """Genre filtering should be case-insensitive."""
        engine = self._build_engine()
        all_results = engine.get_all_movies()
        lower_filtered = engine.filter_by_genre(all_results, "drama")
        upper_filtered = engine.filter_by_genre(all_results, "DRAMA")
        self.assertEqual(len(lower_filtered), len(upper_filtered))

    # ------------------------------------------------------------------
    # get_movie_by_id() tests
    # ------------------------------------------------------------------

    def test_get_movie_by_id_returns_dict(self) -> None:
        """get_movie_by_id should return a dictionary."""
        engine = self._build_engine()
        result = engine.get_movie_by_id(1)
        self.assertIsInstance(result, dict)

    def test_get_movie_by_id_correct_movie(self) -> None:
        """get_movie_by_id should return the movie with the matching ID."""
        engine = self._build_engine()
        result = engine.get_movie_by_id(5)
        self.assertEqual(int(result["movie_id"]), 5)

    def test_get_movie_by_id_raises_key_error_for_missing(self) -> None:
        """Requesting a non-existent movie_id should raise KeyError."""
        engine = self._build_engine()
        with self.assertRaises(KeyError):
            engine.get_movie_by_id(99999)

    # ------------------------------------------------------------------
    # get_all_movies() tests
    # ------------------------------------------------------------------

    def test_get_all_movies_returns_list(self) -> None:
        """get_all_movies should return a list."""
        engine = self._build_engine()
        result = engine.get_all_movies()
        self.assertIsInstance(result, list)

    def test_get_all_movies_length_matches_metadata(self) -> None:
        """The returned list should have the same length as the DataFrame."""
        engine = self._build_engine()
        result = engine.get_all_movies()
        self.assertEqual(len(result), self.N_MOVIES)

    # ------------------------------------------------------------------
    # movies_count property
    # ------------------------------------------------------------------

    def test_movies_count_property(self) -> None:
        """movies_count should return the number of loaded movies."""
        engine = self._build_engine()
        self.assertEqual(engine.movies_count, self.N_MOVIES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
