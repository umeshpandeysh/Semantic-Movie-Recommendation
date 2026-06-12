"""
evaluation.py
=============
Evaluation utilities for the Semantic Movie Recommendation system.

Implements standard information-retrieval metrics:
    - Precision@K
    - Recall@K
    - NDCG@K (Normalised Discounted Cumulative Gain)

Also provides an ``evaluate_recommender`` function that runs a batch
evaluation against a set of labelled test queries, and a ``run_evaluation``
convenience wrapper that uses a built-in demo query set.

Author: Umesh Pandey
"""

import logging
import math
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Demo test queries
# ---------------------------------------------------------------------------
# Each entry contains:
#   query         — natural-language search string
#   relevant_ids  — list of movie_ids that are considered relevant answers
#
# The relevant_ids correspond to movie_ids in embeddings/movies_metadata.csv
# and are chosen based on semantic relevance to the query.

TEST_QUERIES: List[Dict[str, Any]] = [
    {
        "query": "space exploration science fiction",
        "relevant_ids": [4, 8, 15, 101, 110, 125, 140, 155],  # Interstellar, Matrix, …
        "description": "Sci-fi movies about space or technology",
    },
    {
        "query": "prison escape redemption friendship",
        "relevant_ids": [1, 5, 7, 50, 65],  # Shawshank, Godfather, Forrest Gump, …
        "description": "Drama about friendship and resilience",
    },
    {
        "query": "superhero crime fighting dark city",
        "relevant_ids": [2, 22, 30, 45, 60],  # Dark Knight, …
        "description": "Action/thriller superhero films",
    },
    {
        "query": "animated fantasy magical journey children",
        "relevant_ids": [10, 80, 90, 120, 130],  # Spirited Away, …
        "description": "Animated fantasy adventures",
    },
    {
        "query": "psychological thriller mind bending twist",
        "relevant_ids": [3, 11, 12, 40, 55],  # Inception, Parasite, Get Out, …
        "description": "Mind-bending thriller films",
    },
    {
        "query": "romantic love story music dancing",
        "relevant_ids": [13, 70, 75, 85, 95],  # La La Land, …
        "description": "Romance and musical films",
    },
    {
        "query": "sports underdog triumph competition",
        "relevant_ids": [17, 19, 35, 48, 62],  # Whiplash, Dangal, …
        "description": "Sports drama and underdog stories",
    },
    {
        "query": "Indian comedy social commentary satire",
        "relevant_ids": [18, 20, 100, 112],  # 3 Idiots, PK, …
        "description": "Indian comedy-drama films",
    },
    {
        "query": "post apocalyptic survival action",
        "relevant_ids": [14, 25, 38, 56, 78],  # Mad Max, …
        "description": "Post-apocalyptic action films",
    },
    {
        "query": "true story biography genius mathematics",
        "relevant_ids": [9, 16, 43, 67, 88],  # Good Will Hunting, Social Network, …
        "description": "Biographical dramas about real-life brilliance",
    },
]


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def precision_at_k(
    recommended_ids: List[int],
    relevant_ids: List[int],
    k: int,
) -> float:
    """
    Compute Precision@K.

    Precision@K is the fraction of the top-k recommended items that are
    relevant.

    Args:
        recommended_ids: Ordered list of recommended movie IDs (most relevant
                         first).
        relevant_ids:    Ground-truth list of relevant movie IDs.
        k:               Cutoff rank.

    Returns:
        Precision@K in ``[0.0, 1.0]``.

    Example::

        >>> precision_at_k([1, 2, 3, 4, 5], relevant_ids=[1, 3, 7], k=5)
        0.4
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not recommended_ids:
        return 0.0

    top_k = set(recommended_ids[:k])
    hits = top_k & set(relevant_ids)
    return len(hits) / k


def recall_at_k(
    recommended_ids: List[int],
    relevant_ids: List[int],
    k: int,
) -> float:
    """
    Compute Recall@K.

    Recall@K is the fraction of all relevant items that appear in the top-k
    recommendations.

    Args:
        recommended_ids: Ordered list of recommended movie IDs.
        relevant_ids:    Ground-truth list of relevant movie IDs.
        k:               Cutoff rank.

    Returns:
        Recall@K in ``[0.0, 1.0]``.

    Example::

        >>> recall_at_k([1, 2, 3, 4, 5], relevant_ids=[1, 3, 7], k=5)
        0.6667
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not relevant_ids:
        return 0.0
    if not recommended_ids:
        return 0.0

    top_k = set(recommended_ids[:k])
    hits = top_k & set(relevant_ids)
    return len(hits) / len(relevant_ids)


def ndcg_at_k(
    recommended_ids: List[int],
    relevant_ids: List[int],
    k: int,
) -> float:
    """
    Compute NDCG@K (Normalised Discounted Cumulative Gain).

    Uses binary relevance: an item is relevant (gain=1) if it appears in
    ``relevant_ids``, otherwise irrelevant (gain=0).

    Args:
        recommended_ids: Ordered list of recommended movie IDs.
        relevant_ids:    Ground-truth list of relevant movie IDs.
        k:               Cutoff rank.

    Returns:
        NDCG@K in ``[0.0, 1.0]``.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not recommended_ids or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = recommended_ids[:k]

    # DCG
    dcg = 0.0
    for rank, movie_id in enumerate(top_k, start=1):
        if movie_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: place all relevant items at the top
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_recommender(
    engine: Any,
    test_queries: List[Dict[str, Any]],
    k: int = 10,
) -> Dict[str, Any]:
    """
    Run batch evaluation of a :class:`~recommendation_engine.RecommendationEngine`
    over a set of labelled test queries.

    Args:
        engine:        Loaded recommendation engine with a ``recommend()`` method.
        test_queries:  List of query dicts, each with keys ``'query'`` and
                       ``'relevant_ids'``.
        k:             Evaluation cutoff rank.

    Returns:
        Dictionary with:
            - ``k``: cutoff used
            - ``num_queries``: number of queries evaluated
            - ``per_query``: list of per-query metric dicts
            - ``avg_precision_at_k``: mean P@K
            - ``avg_recall_at_k``: mean R@K
            - ``avg_ndcg_at_k``: mean NDCG@K
    """
    logger.info(
        "Evaluating recommender on %d queries at k=%d.", len(test_queries), k
    )

    per_query_results = []
    precisions, recalls, ndcgs = [], [], []

    for entry in test_queries:
        query = entry["query"]
        relevant_ids = entry["relevant_ids"]

        try:
            recommendations = engine.recommend(query, k=k)
            recommended_ids = [r["movie_id"] for r in recommendations]
        except Exception as exc:
            logger.warning("Failed to get recommendations for '%s': %s", query, exc)
            recommended_ids = []

        p_k = precision_at_k(recommended_ids, relevant_ids, k)
        r_k = recall_at_k(recommended_ids, relevant_ids, k)
        n_k = ndcg_at_k(recommended_ids, relevant_ids, k)

        precisions.append(p_k)
        recalls.append(r_k)
        ndcgs.append(n_k)

        per_query_results.append(
            {
                "query": query,
                "description": entry.get("description", ""),
                "relevant_ids": relevant_ids,
                "recommended_ids": recommended_ids,
                f"precision@{k}": round(p_k, 4),
                f"recall@{k}": round(r_k, 4),
                f"ndcg@{k}": round(n_k, 4),
            }
        )
        logger.info(
            "  Query: '%-45s'  P@%d=%.3f  R@%d=%.3f  NDCG@%d=%.3f",
            query[:45],
            k, p_k,
            k, r_k,
            k, n_k,
        )

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0
    avg_n = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

    summary = {
        "k": k,
        "num_queries": len(test_queries),
        "per_query": per_query_results,
        f"avg_precision@{k}": round(avg_p, 4),
        f"avg_recall@{k}": round(avg_r, 4),
        f"avg_ndcg@{k}": round(avg_n, 4),
    }

    logger.info(
        "=== Evaluation Summary (k=%d) ===  P@%d=%.4f  R@%d=%.4f  NDCG@%d=%.4f",
        k, k, avg_p, k, avg_r, k, avg_n,
    )
    return summary


def run_evaluation(engine: Any, k: int = 10) -> Dict[str, Any]:
    """
    Run evaluation using the built-in :data:`TEST_QUERIES` demo set.

    Args:
        engine: Loaded recommendation engine.
        k:      Evaluation cutoff rank.

    Returns:
        Evaluation summary dictionary from :func:`evaluate_recommender`.
    """
    logger.info("Running built-in demo evaluation with %d test queries.", len(TEST_QUERIES))
    return evaluate_recommender(engine, TEST_QUERIES, k=k)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Demonstrate metric functions without a live engine
    print("=== Metric Demo ===")

    recs = [1, 4, 2, 8, 15, 3, 6, 11, 20, 5]
    rels = [1, 4, 8, 11, 25, 30]
    k_val = 10

    print(f"Recommended : {recs}")
    print(f"Relevant    : {rels}")
    print(f"Precision@{k_val} : {precision_at_k(recs, rels, k_val):.4f}")
    print(f"Recall@{k_val}    : {recall_at_k(recs, rels, k_val):.4f}")
    print(f"NDCG@{k_val}      : {ndcg_at_k(recs, rels, k_val):.4f}")
