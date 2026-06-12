"""
text_preprocessor.py
====================
Text preprocessing utilities for the Semantic Movie Recommendation system.

This module provides the TextPreprocessor class, which handles cleaning,
stopword removal, and document construction for movie metadata before
embedding generation.

Author: Umesh Pandey
"""

import logging
import re
import string
from typing import List, Optional

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Built-in English stopword list (no NLTK dependency)
# ---------------------------------------------------------------------------
STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "if", "as", "it", "its", "this", "that",
    "these", "those", "not", "no", "nor", "so", "yet", "both", "either",
    "neither", "just", "also", "then", "than", "when", "where", "which",
    "who", "whom", "what", "how", "all", "each", "every", "any", "few",
    "more", "most", "other", "some", "such", "very", "too", "while",
    "about", "above", "after", "before", "between", "into", "through",
    "during", "up", "down", "out", "off", "over", "under", "again",
    "further", "once", "here", "there", "own", "same", "him", "his",
    "her", "she", "he", "they", "them", "their", "we", "our", "us",
    "you", "your", "my", "me", "i", "am",
})


class TextPreprocessor:
    """
    Provides text preprocessing utilities for the movie recommendation system.

    Methods cover cleaning raw text, removing common English stopwords,
    normalising documents, and constructing structured movie documents
    suitable for sentence embedding.

    Example usage::

        preprocessor = TextPreprocessor()
        doc = preprocessor.create_movie_document(
            title="Inception",
            genre="Sci-Fi/Thriller",
            plot="A thief who steals corporate secrets through dream-sharing.",
            year=2010,
        )
        clean = preprocessor.normalize(doc)
    """

    def __init__(self, extra_stopwords: Optional[frozenset] = None) -> None:
        """
        Initialise the TextPreprocessor.

        Args:
            extra_stopwords: Optional additional stopwords to merge with the
                             built-in STOPWORDS set.
        """
        self._stopwords = STOPWORDS
        if extra_stopwords:
            self._stopwords = STOPWORDS | frozenset(
                w.lower() for w in extra_stopwords
            )
        logger.debug(
            "TextPreprocessor initialised with %d stopwords.",
            len(self._stopwords),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """
        Lowercase, remove punctuation, and normalise whitespace.

        Args:
            text: Raw input string.

        Returns:
            Cleaned string with all characters lowercased, punctuation
            removed, and consecutive whitespace collapsed to a single space.

        Example::

            >>> preprocessor.clean_text("Hello, World!  How's it going?")
            'hello world  hows it going'
        """
        if not isinstance(text, str):
            logger.warning(
                "clean_text received non-string input (%s); converting.",
                type(text).__name__,
            )
            text = str(text)

        # Lowercase
        text = text.lower()

        # Remove punctuation (keep hyphens as word separators → space)
        text = text.replace("-", " ")
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        logger.debug("clean_text: %d chars → %d chars", len(text), len(text))
        return text

    def remove_stopwords(
        self, text: str, language: str = "english"
    ) -> str:
        """
        Remove stopwords from the given text.

        Only English is supported natively. Passing a different ``language``
        logs a warning and still applies the built-in English stopword list.

        Args:
            text:     Input string (expected to already be lowercased).
            language: Language for stopword removal (default ``'english'``).

        Returns:
            String with stopwords removed, tokens re-joined with a space.
        """
        if language != "english":
            logger.warning(
                "Language '%s' is not supported; falling back to English "
                "stopwords.",
                language,
            )

        tokens = text.split()
        filtered = [t for t in tokens if t not in self._stopwords]
        logger.debug(
            "remove_stopwords: %d tokens → %d tokens (removed %d).",
            len(tokens),
            len(filtered),
            len(tokens) - len(filtered),
        )
        return " ".join(filtered)

    def normalize(self, text: str) -> str:
        """
        Full normalisation pipeline: clean then remove stopwords.

        Args:
            text: Raw input string.

        Returns:
            Normalised string suitable for downstream embedding.
        """
        cleaned = self.clean_text(text)
        return self.remove_stopwords(cleaned)

    def batch_normalize(self, texts: List[str]) -> List[str]:
        """
        Apply :meth:`normalize` to a list of strings.

        Args:
            texts: List of raw input strings.

        Returns:
            List of normalised strings, preserving input order and length.

        Raises:
            TypeError: If ``texts`` is not a list.
        """
        if not isinstance(texts, list):
            raise TypeError(
                f"batch_normalize expects a list, got {type(texts).__name__}."
            )

        logger.info("batch_normalize: processing %d documents.", len(texts))
        result = [self.normalize(t) for t in texts]
        logger.info("batch_normalize: completed.")
        return result

    def create_movie_document(
        self,
        title: str,
        genre: str,
        plot: str,
        year: Optional[int] = None,
    ) -> str:
        """
        Construct a single text document from movie metadata fields.

        The document combines title, genre, plot (and optionally year) with
        ``|`` separators so that each field contributes distinctly to the
        resulting embedding.

        Args:
            title: Movie title string.
            genre: Genre string (e.g. ``'Sci-Fi/Thriller'``).
            plot:  Plot summary string.
            year:  Optional release year (integer).

        Returns:
            A combined document string ready for normalisation and embedding.

        Example::

            >>> preprocessor.create_movie_document(
            ...     "Inception", "Sci-Fi", "A thief enters dreams.", 2010
            ... )
            'title: inception | genre: sci-fi | year: 2010 | plot: a thief enters dreams'
        """
        title_clean = self.clean_text(str(title))
        genre_clean = self.clean_text(str(genre))
        plot_clean = self.clean_text(str(plot))

        parts = [
            f"title: {title_clean}",
            f"genre: {genre_clean}",
        ]
        if year is not None:
            parts.append(f"year: {year}")
        parts.append(f"plot: {plot_clean}")

        document = " | ".join(parts)
        logger.debug(
            "create_movie_document: '%s' → %d chars.", title, len(document)
        )
        return document


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------
_default_preprocessor: Optional[TextPreprocessor] = None


def get_default_preprocessor() -> TextPreprocessor:
    """
    Return (or lazily create) a module-level default TextPreprocessor instance.

    Returns:
        Singleton :class:`TextPreprocessor` instance.
    """
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = TextPreprocessor()
    return _default_preprocessor


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tp = TextPreprocessor()

    sample = (
        "A young boy discovers he has magical powers and is enrolled in "
        "a school for wizards, where he makes friends and enemies while "
        "uncovering the truth about his parents."
    )
    doc = tp.create_movie_document(
        title="Harry Potter and the Philosopher's Stone",
        genre="Fantasy/Adventure",
        plot=sample,
        year=2001,
    )
    print("Raw document:")
    print(doc)
    print("\nNormalised:")
    print(tp.normalize(doc))
