"""
embedding_service.py
Semantic Movie Recommendation — Python Embedding Service

Generates sentence embeddings for user queries and computes cosine similarity
against pre-computed movie embeddings. Called by the Node.js API server.

Usage:
    python embedding_service.py --query "space adventure with a robot" --top_k 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_embeddings(embeddings_dir: str = "embeddings"):
    """
    Load pre-computed movie embeddings and metadata.

    Args:
        embeddings_dir: Directory containing embeddings.npy and movies.csv

    Returns:
        embeddings (np.ndarray), metadata (pd.DataFrame)
    """
    emb_path = Path(embeddings_dir) / "movie_embeddings.npy"
    meta_path = Path(embeddings_dir) / "movies.csv"

    if not emb_path.exists() or not meta_path.exists():
        # Return empty results if no embeddings exist yet
        return None, None

    embeddings = np.load(str(emb_path))
    metadata = pd.read_csv(str(meta_path))
    return embeddings, metadata


def embed_query(query: str):
    """
    Generate a sentence embedding for the user query.

    Args:
        query: User query string.

    Returns:
        Query embedding as numpy array.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode([query], convert_to_numpy=True)
        return embedding[0]
    except ImportError:
        # Fallback: random vector (for testing without sentence-transformers)
        return np.random.rand(384)


def cosine_similarity(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query vector and all corpus vectors.

    Args:
        query_vec: 1D query embedding.
        corpus_matrix: 2D matrix of corpus embeddings.

    Returns:
        1D array of similarity scores.
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    corpus_norms = corpus_matrix / (np.linalg.norm(corpus_matrix, axis=1, keepdims=True) + 1e-10)
    return corpus_norms @ query_norm


def recommend(query: str, top_k: int = 10) -> list:
    """
    Return top-K movie recommendations for the given query.

    Args:
        query: User query string.
        top_k: Number of recommendations to return.

    Returns:
        List of recommendation dictionaries.
    """
    embeddings, metadata = load_embeddings()

    if embeddings is None:
        # Return placeholder results when no data is loaded
        return [
            {
                "rank": 1,
                "title": "Placeholder — Add movie embeddings to embeddings/ folder",
                "similarity_score": 0.0,
                "genre": "N/A",
                "overview": "Run generate_embeddings.py with your movie dataset to enable real recommendations."
            }
        ]

    query_vec = embed_query(query)
    scores = cosine_similarity(query_vec, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        row = metadata.iloc[idx]
        results.append({
            "rank": rank,
            "title": row.get("title", "Unknown"),
            "similarity_score": float(scores[idx]),
            "genre": row.get("genres", "N/A"),
            "overview": row.get("overview", "")
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Movie Recommendation Service")
    parser.add_argument("--query", type=str, required=True, help="User query text")
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations")
    args = parser.parse_args()

    results = recommend(args.query, args.top_k)
    print(json.dumps(results))
