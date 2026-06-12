"""pytest shared fixtures for Semantic-Movie-Recommendation tests."""
from __future__ import annotations

import sys
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(str(pathlib.Path(__file__).resolve().parent.parent), 'backend'))


@pytest.fixture()
def random_embeddings_matrix():
    """20 normalised 384-dim random vectors."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((20, 384)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


@pytest.fixture()
def random_query_vector():
    """One normalised 384-dim query vector."""
    rng = np.random.default_rng(99)
    vec = rng.standard_normal(384).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm


@pytest.fixture()
def dummy_movies_df():
    """Minimal DataFrame matching movies_metadata.csv schema."""
    return pd.DataFrame({
        'movie_id': list(range(1, 21)),
        'title':    [f'Movie {i}' for i in range(1, 21)],
        'genre':    ['Drama' if i % 2 == 0 else 'Sci-Fi' for i in range(1, 21)],
        'year':     [2000 + i for i in range(1, 21)],
        'plot':     [f'A story about event {i} and its outcome.' for i in range(1, 21)],
        'rating':   [round(6.0 + (i % 10) * 0.3, 1) for i in range(1, 21)],
    })
