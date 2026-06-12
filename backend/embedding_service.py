"""
embedding_service.py
====================
Provides the EmbeddingService class for generating and managing sentence
embeddings used by the Semantic Movie Recommendation system.

The service uses ``sentence-transformers`` as its primary backend. If the
library is not installed, it gracefully falls back to a deterministic
hash-based random-embedding generator so that the rest of the system can
still be tested without the heavy ML dependency.

Author: Umesh Pandey
"""

import hashlib
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Try importing sentence-transformers; fall back gracefully
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("sentence-transformers library detected.")
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers is not installed. "
        "Using deterministic hash-based fallback embeddings (dim=384). "
        "Install with: pip install sentence-transformers"
    )


# ---------------------------------------------------------------------------
# Fallback: deterministic random embedder
# ---------------------------------------------------------------------------

class _HashEmbedder:
    """
    Deterministic pseudo-random embedder used when sentence-transformers is
    unavailable.  Maps any string to a unit-normalised 384-dimensional vector
    derived from its SHA-256 digest.  Results are reproducible across runs.
    """

    EMBED_DIM: int = 384

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of sentences to fake embeddings.

        Args:
            sentences:         List of text strings.
            batch_size:        Ignored (kept for API compatibility).
            show_progress_bar: Ignored.

        Returns:
            NumPy array of shape ``(len(sentences), EMBED_DIM)`` with each
            row unit-normalised.
        """
        vectors = []
        for text in sentences:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # Repeat digest bytes to fill embed_dim floats
            seed = int.from_bytes(digest, "big") % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.EMBED_DIM).astype(np.float32)
            # Unit-normalise
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.vstack(vectors)


# ---------------------------------------------------------------------------
# EmbeddingService
# ---------------------------------------------------------------------------

class EmbeddingService:
    """
    High-level service for generating, saving, and comparing sentence
    embeddings.

    Uses the ``sentence-transformers`` library when available, otherwise
    transparently falls back to :class:`_HashEmbedder` for CI / offline use.

    Example usage::

        svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
        svc.load_model()

        vec = svc.embed_text("A robot uprising in a dystopian future.")
        matrix = svc.embed_batch(["Movie A plot.", "Movie B plot."])
        scores = svc.cosine_similarity_matrix(vec, matrix)
        top = svc.top_k_similar(vec, matrix, k=5)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embeddings_dir: str = "embeddings",
    ) -> None:
        """
        Initialise the EmbeddingService.

        Args:
            model_name:     HuggingFace model identifier for
                            ``sentence-transformers``.
            embeddings_dir: Directory used by :meth:`save_embeddings` and
                            :meth:`load_precomputed_embeddings` by default.
        """
        self.model_name = model_name
        self.embeddings_dir = embeddings_dir
        self._model: Optional[object] = None
        self._using_fallback: bool = False
        logger.info(
            "EmbeddingService created (model=%s, dir=%s).",
            model_name,
            embeddings_dir,
        )

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the sentence-transformer model into memory.

        If ``sentence-transformers`` is not installed, loads the
        :class:`_HashEmbedder` fallback instead.  This method is idempotent:
        calling it multiple times does not reload the model.
        """
        if self._model is not None:
            logger.debug("Model already loaded; skipping reload.")
            return

        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info("Loading SentenceTransformer model '%s'…", self.model_name)
            try:
                self._model = SentenceTransformer(self.model_name)
                self._using_fallback = False
                logger.info("Model '%s' loaded successfully.", self.model_name)
            except Exception as exc:
                logger.error(
                    "Failed to load SentenceTransformer model '%s': %s. "
                    "Falling back to hash embedder.",
                    self.model_name,
                    exc,
                )
                self._model = _HashEmbedder()
                self._using_fallback = True
        else:
            logger.warning("Using _HashEmbedder fallback (no sentence-transformers).")
            self._model = _HashEmbedder()
            self._using_fallback = True

    def _ensure_model_loaded(self) -> None:
        """Auto-load model if not already done."""
        if self._model is None:
            logger.info("Model not yet loaded; calling load_model() automatically.")
            self.load_model()

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Input string to embed.

        Returns:
            1-D NumPy array of shape ``(embed_dim,)``.

        Raises:
            ValueError: If ``text`` is empty.
        """
        if not text or not text.strip():
            raise ValueError("embed_text received an empty string.")

        self._ensure_model_loaded()
        logger.debug("Embedding single text of length %d.", len(text))
        vectors = self._model.encode([text], show_progress_bar=False)  # type: ignore[union-attr]
        return vectors[0]

    def embed_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """
        Embed a list of text strings in batches.

        Args:
            texts:      List of input strings.
            batch_size: Number of texts to encode per forward pass.

        Returns:
            2-D NumPy array of shape ``(len(texts), embed_dim)``.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("embed_batch received an empty list.")

        self._ensure_model_loaded()
        logger.info(
            "Embedding batch of %d texts (batch_size=%d).",
            len(texts),
            batch_size,
        )
        vectors: np.ndarray = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        logger.info("Batch embedding complete. Shape: %s.", str(vectors.shape))
        return vectors

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_embeddings(
        self, embeddings: np.ndarray, filepath: str
    ) -> None:
        """
        Save a NumPy embedding array to disk in ``.npy`` format.

        Args:
            embeddings: Array to save.
            filepath:   Destination file path (should end with ``.npy``).
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.save(filepath, embeddings)
        logger.info(
            "Saved embeddings of shape %s to '%s'.",
            str(embeddings.shape),
            filepath,
        )

    def load_precomputed_embeddings(self, filepath: str) -> np.ndarray:
        """
        Load a pre-computed embedding array from a ``.npy`` file.

        Args:
            filepath: Path to the ``.npy`` file.

        Returns:
            NumPy array of pre-computed embeddings.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Embeddings file not found: '{filepath}'. "
                "Run generate_embeddings.py first."
            )
        embeddings = np.load(filepath)
        logger.info(
            "Loaded precomputed embeddings of shape %s from '%s'.",
            str(embeddings.shape),
            filepath,
        )
        return embeddings

    # ------------------------------------------------------------------
    # Similarity utilities
    # ------------------------------------------------------------------

    def cosine_similarity_matrix(
        self,
        query_vec: np.ndarray,
        corpus_vecs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between a query vector and a corpus matrix.

        Args:
            query_vec:   1-D array of shape ``(embed_dim,)``.
            corpus_vecs: 2-D array of shape ``(n_docs, embed_dim)``.

        Returns:
            1-D NumPy array of shape ``(n_docs,)`` containing cosine
            similarity scores in ``[-1, 1]``.

        Raises:
            ValueError: If shapes are incompatible.
        """
        if query_vec.ndim != 1:
            raise ValueError(
                f"query_vec must be 1-D, got shape {query_vec.shape}."
            )
        if corpus_vecs.ndim != 2:
            raise ValueError(
                f"corpus_vecs must be 2-D, got shape {corpus_vecs.shape}."
            )
        if query_vec.shape[0] != corpus_vecs.shape[1]:
            raise ValueError(
                f"Dimension mismatch: query_vec has dim {query_vec.shape[0]}, "
                f"corpus_vecs has dim {corpus_vecs.shape[1]}."
            )

        # Unit-normalise query
        q_norm = np.linalg.norm(query_vec)
        q = query_vec / q_norm if q_norm > 0 else query_vec

        # Unit-normalise corpus rows
        c_norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
        c_norms = np.where(c_norms == 0, 1.0, c_norms)
        c = corpus_vecs / c_norms

        similarities = c @ q
        logger.debug(
            "cosine_similarity_matrix: query dim=%d, corpus shape=%s.",
            q.shape[0],
            str(corpus_vecs.shape),
        )
        return similarities

    def top_k_similar(
        self,
        query_vec: np.ndarray,
        corpus_vecs: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Return the top-k most similar corpus indices and their scores.

        Args:
            query_vec:   1-D embedding vector for the query.
            corpus_vecs: 2-D embedding matrix for the corpus.
            k:           Number of top results to return.

        Returns:
            List of ``(index, score)`` tuples sorted by descending similarity.
            The list has at most ``min(k, len(corpus_vecs))`` entries.
        """
        k = min(k, corpus_vecs.shape[0])
        similarities = self.cosine_similarity_matrix(query_vec, corpus_vecs)

        # Partial sort for efficiency (O(n) vs O(n log n))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results: List[Tuple[int, float]] = [
            (int(idx), float(similarities[idx])) for idx in top_indices
        ]
        logger.debug(
            "top_k_similar: returning %d results (best score=%.4f).",
            len(results),
            results[0][1] if results else float("nan"),
        )
        return results


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    svc = EmbeddingService()
    svc.load_model()

    texts = [
        "A young wizard discovers his magical heritage.",
        "A hacker discovers the world is a simulation.",
        "A crime family struggles to maintain power.",
    ]
    matrix = svc.embed_batch(texts)
    query = svc.embed_text("magic spell and wizardry school adventure")
    top = svc.top_k_similar(query, matrix, k=2)
    print("Top-2 similar movies:")
    for idx, score in top:
        print(f"  [{idx}] '{texts[idx]}' — score={score:.4f}")
