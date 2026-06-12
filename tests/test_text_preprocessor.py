"""
test_text_preprocessor.py
=========================
Unit tests for the TextPreprocessor class.

Run with::

    pytest tests/test_text_preprocessor.py -v

Author: Umesh Pandey
"""

import sys
import os
import unittest

# Allow importing from the backend package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from text_preprocessor import TextPreprocessor, STOPWORDS  # noqa: E402


class TestCleanText(unittest.TestCase):
    """Tests for TextPreprocessor.clean_text()."""

    def setUp(self) -> None:
        """Instantiate the preprocessor before each test."""
        self.tp = TextPreprocessor()

    def test_uppercase_converted_to_lowercase(self) -> None:
        """All uppercase characters should be lowercased."""
        result = self.tp.clean_text("HELLO WORLD")
        self.assertEqual(result, "hello world")

    def test_mixed_case_converted_to_lowercase(self) -> None:
        """Mixed-case input should be fully lowercased."""
        result = self.tp.clean_text("The Dark Knight Rises")
        self.assertEqual(result, "the dark knight rises")

    def test_punctuation_removed(self) -> None:
        """Punctuation characters should be stripped from the output."""
        result = self.tp.clean_text("Hello, World! How's it going?")
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)
        self.assertNotIn("?", result)

    def test_hyphens_become_spaces(self) -> None:
        """Hyphens should be replaced by spaces, not simply deleted."""
        result = self.tp.clean_text("science-fiction thriller")
        # Both words should still be present
        self.assertIn("science", result)
        self.assertIn("fiction", result)

    def test_multiple_spaces_collapsed(self) -> None:
        """Consecutive whitespace should be normalised to a single space."""
        result = self.tp.clean_text("hello   world")
        self.assertEqual(result, "hello world")

    def test_leading_trailing_whitespace_stripped(self) -> None:
        """Leading and trailing whitespace should be removed."""
        result = self.tp.clean_text("  hello  ")
        self.assertEqual(result, "hello")

    def test_empty_string_returns_empty(self) -> None:
        """An empty string should return an empty string."""
        result = self.tp.clean_text("")
        self.assertEqual(result, "")

    def test_numeric_characters_preserved(self) -> None:
        """Digits should be preserved in the cleaned output."""
        result = self.tp.clean_text("Alien 3")
        self.assertIn("3", result)

    def test_non_string_input_does_not_raise(self) -> None:
        """Non-string input should be coerced without raising an exception."""
        result = self.tp.clean_text(12345)  # type: ignore[arg-type]
        self.assertIsInstance(result, str)


class TestRemoveStopwords(unittest.TestCase):
    """Tests for TextPreprocessor.remove_stopwords()."""

    def setUp(self) -> None:
        self.tp = TextPreprocessor()

    def test_common_stopwords_removed(self) -> None:
        """Typical stopwords ('the', 'is', 'a') should be absent from output."""
        result = self.tp.remove_stopwords("the hero is a young man")
        self.assertNotIn("the", result.split())
        self.assertNotIn("is", result.split())
        self.assertNotIn("a", result.split())

    def test_content_words_preserved(self) -> None:
        """Non-stopword tokens should remain in the output."""
        result = self.tp.remove_stopwords("hero journey adventure redemption")
        self.assertIn("hero", result)
        self.assertIn("journey", result)
        self.assertIn("adventure", result)
        self.assertIn("redemption", result)

    def test_empty_string_returns_empty(self) -> None:
        """An empty input should return an empty string."""
        result = self.tp.remove_stopwords("")
        self.assertEqual(result, "")

    def test_all_stopwords_returns_empty(self) -> None:
        """A string made entirely of stopwords should return an empty string."""
        result = self.tp.remove_stopwords("the a an is are was were")
        self.assertEqual(result.strip(), "")

    def test_unsupported_language_logs_warning_and_uses_english(self) -> None:
        """
        Passing an unsupported language should not raise but should still
        remove English stopwords (since English is the built-in fallback).
        """
        result = self.tp.remove_stopwords("the quick brown fox", language="spanish")
        # 'the' is in STOPWORDS, so it should still be removed
        self.assertNotIn("the", result.split())

    def test_case_sensitive_matching(self) -> None:
        """
        remove_stopwords expects already-lowercased text; uppercase tokens
        should NOT be matched against the lowercase stopword list.
        """
        # 'The' with capital T should not be removed (input not yet lowercased)
        result = self.tp.remove_stopwords("The hero")
        # 'The' ≠ 'the' so it remains
        self.assertIn("The", result.split())


class TestNormalize(unittest.TestCase):
    """Tests for TextPreprocessor.normalize()."""

    def setUp(self) -> None:
        self.tp = TextPreprocessor()

    def test_output_is_lowercase(self) -> None:
        """Normalised output should be entirely lowercase."""
        result = self.tp.normalize("The DARK Knight")
        self.assertEqual(result, result.lower())

    def test_no_punctuation_in_output(self) -> None:
        """Normalised output should contain no punctuation."""
        import string
        result = self.tp.normalize("Hello, World! It's amazing.")
        for ch in string.punctuation:
            self.assertNotIn(ch, result)

    def test_stopwords_removed_in_pipeline(self) -> None:
        """Stopwords should be absent after the full pipeline."""
        result = self.tp.normalize("The film is about a young hero")
        tokens = result.split()
        for stopword in ("the", "is", "about", "a"):
            self.assertNotIn(stopword, tokens)

    def test_content_words_survive_pipeline(self) -> None:
        """Domain words should survive the full normalisation pipeline."""
        result = self.tp.normalize("A robot uprising in a dystopian future")
        self.assertIn("robot", result)
        self.assertIn("uprising", result)
        self.assertIn("dystopian", result)
        self.assertIn("future", result)

    def test_empty_string_returns_empty(self) -> None:
        result = self.tp.normalize("")
        self.assertEqual(result, "")


class TestCreateMovieDocument(unittest.TestCase):
    """Tests for TextPreprocessor.create_movie_document()."""

    def setUp(self) -> None:
        self.tp = TextPreprocessor()

    def test_title_included_in_document(self) -> None:
        """The movie title should appear in the document."""
        doc = self.tp.create_movie_document(
            title="Inception",
            genre="Sci-Fi",
            plot="A thief enters dreams.",
        )
        self.assertIn("inception", doc)

    def test_genre_included_in_document(self) -> None:
        """The genre should appear in the document."""
        doc = self.tp.create_movie_document(
            title="Inception",
            genre="Sci-Fi/Thriller",
            plot="A thief enters dreams.",
        )
        self.assertIn("sci", doc)

    def test_plot_included_in_document(self) -> None:
        """The plot text should appear in the document."""
        doc = self.tp.create_movie_document(
            title="Inception",
            genre="Sci-Fi",
            plot="A thief enters dreams.",
        )
        self.assertIn("thief", doc)

    def test_year_included_when_provided(self) -> None:
        """When year is given, it should appear in the document."""
        doc = self.tp.create_movie_document(
            title="Inception",
            genre="Sci-Fi",
            plot="A thief enters dreams.",
            year=2010,
        )
        self.assertIn("2010", doc)

    def test_year_absent_when_not_provided(self) -> None:
        """When year is None, the output should not contain a year field."""
        doc = self.tp.create_movie_document(
            title="Inception",
            genre="Sci-Fi",
            plot="A thief enters dreams.",
            year=None,
        )
        self.assertNotIn("year:", doc)

    def test_separator_present_in_document(self) -> None:
        """Fields should be separated by the '|' character."""
        doc = self.tp.create_movie_document(
            title="Inception",
            genre="Sci-Fi",
            plot="A thief enters dreams.",
        )
        self.assertIn("|", doc)

    def test_return_type_is_string(self) -> None:
        """The method should return a string."""
        doc = self.tp.create_movie_document(
            title="Inception",
            genre="Sci-Fi",
            plot="A thief enters dreams.",
        )
        self.assertIsInstance(doc, str)

    def test_document_is_non_empty(self) -> None:
        """The returned document should be non-empty."""
        doc = self.tp.create_movie_document(
            title="X",
            genre="Y",
            plot="Z",
        )
        self.assertGreater(len(doc), 0)


class TestBatchNormalize(unittest.TestCase):
    """Tests for TextPreprocessor.batch_normalize()."""

    def setUp(self) -> None:
        self.tp = TextPreprocessor()

    def test_output_length_matches_input(self) -> None:
        """Output list should have the same length as the input list."""
        texts = [
            "The Shawshank Redemption is a drama film.",
            "Inception is a sci-fi thriller about dreams.",
            "The Dark Knight features Batman fighting the Joker.",
        ]
        result = self.tp.batch_normalize(texts)
        self.assertEqual(len(result), len(texts))

    def test_each_element_is_string(self) -> None:
        """Every element in the output should be a string."""
        texts = ["Hello World", "Another sentence."]
        result = self.tp.batch_normalize(texts)
        for item in result:
            self.assertIsInstance(item, str)

    def test_empty_list_returns_empty_list(self) -> None:
        """An empty list input should produce an empty list output."""
        result = self.tp.batch_normalize([])
        self.assertEqual(result, [])

    def test_single_element_list(self) -> None:
        """A single-element list should produce a single-element output."""
        result = self.tp.batch_normalize(["The quick brown fox."])
        self.assertEqual(len(result), 1)

    def test_non_list_input_raises_type_error(self) -> None:
        """Passing a non-list should raise a TypeError."""
        with self.assertRaises(TypeError):
            self.tp.batch_normalize("not a list")  # type: ignore[arg-type]

    def test_normalisation_applied_to_each_element(self) -> None:
        """Each element should be individually normalised."""
        texts = ["The HERO is brave.", "A VILLAIN lurks."]
        result = self.tp.batch_normalize(texts)
        # Output should be lowercase and free of stopwords
        self.assertNotIn("the", result[0].split())
        self.assertNotIn("is", result[0].split())
        self.assertNotIn("a", result[1].split())


class TestStopwordsSet(unittest.TestCase):
    """Tests for the module-level STOPWORDS constant."""

    def test_stopwords_is_frozenset(self) -> None:
        """STOPWORDS should be a frozenset."""
        self.assertIsInstance(STOPWORDS, frozenset)

    def test_common_words_in_stopwords(self) -> None:
        """Common English stopwords should be members of STOPWORDS."""
        for word in ("the", "a", "an", "is", "are", "was", "were"):
            self.assertIn(word, STOPWORDS)

    def test_content_words_not_in_stopwords(self) -> None:
        """Domain-relevant content words should not be in STOPWORDS."""
        for word in ("hero", "villain", "space", "dragon", "robot"):
            self.assertNotIn(word, STOPWORDS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
