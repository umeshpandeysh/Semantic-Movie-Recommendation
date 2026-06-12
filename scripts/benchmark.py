#!/usr/bin/env python3
"""Semantic Movie Recommendation — Recommendation Quality Benchmark.

Usage
-----
    python scripts/benchmark.py --metadata embeddings/movies_metadata.csv \\
                                --embeddings embeddings/movie_embeddings.npy

Measures:
    - precision@k and recall@k on a hand-curated set of test queries
    - Latency per query (wall-clock time for embed + search)
    - Top-k retrieval consistency across k values

All evaluation is done without real user feedback.
Ground truth is manually curated genre+keyword relevance.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

import numpy as np

# ---------------------------------------------------------------------------
# Hand-curated test queries with expected genre tags.
# A result is "relevant" if it shares the expected genre.
# ---------------------------------------------------------------------------
TEST_QUERIES: List[Dict] = [
    {'query': 'space exploration adventure with a lone astronaut',
     'expected_genre': 'sci-fi', 'k': 10},
    {'query': 'crime thriller with a corrupt detective',
     'expected_genre': 'crime', 'k': 10},
    {'query': 'redemption story of a wrongly imprisoned man',
     'expected_genre': 'drama', 'k': 10},
    {'query': 'animated film for children with talking animals',
     'expected_genre': 'animation', 'k': 10},
    {'query': 'romantic drama set in 1940s wartime Paris',
     'expected_genre': 'romance', 'k': 10},
    {'query': 'psychological horror with supernatural elements',
     'expected_genre': 'horror', 'k': 10},
    {'query': 'biographical film about a famous musician',
     'expected_genre': 'biography', 'k': 10},
    {'query': 'heist movie with an elaborate plan',
     'expected_genre': 'thriller', 'k': 10},
]


def precision_at_k(results: List[Dict], expected_genre: str, k: int) -> float:
    """Fraction of top-k results whose genre contains expected_genre."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = sum(
        1 for r in top_k
        if expected_genre.lower() in r.get('genre', '').lower()
    )
    return hits / len(top_k)


def mean_score_at_k(results: List[Dict], k: int) -> float:
    """Average similarity score of top-k results."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    return float(np.mean([r.get('score', 0.0) for r in top_k]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Recommendation quality benchmark')
    p.add_argument('--metadata',   default='embeddings/movies_metadata.csv')
    p.add_argument('--embeddings', default='embeddings/movie_embeddings.npy')
    p.add_argument('--model',      default='all-MiniLM-L6-v2')
    p.add_argument('--k',          type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from embedding_service import EmbeddingService
    from recommendation_engine import RecommendationEngine

    print('Loading recommendation engine...')
    svc = EmbeddingService(model_name=args.model)
    svc.load_model()

    embeddings_path = args.embeddings if os.path.exists(args.embeddings) else None
    engine = RecommendationEngine(
        metadata_path=args.metadata,
        embeddings_path=embeddings_path,
    )
    engine.load_metadata()
    engine.build_index(svc)
    print(f'Engine ready — {engine.movies_count} movies indexed.\n')

    results_summary = []
    total_latency = 0.0

    for q in TEST_QUERIES:
        t0 = time.perf_counter()
        results = engine.recommend(query=q['query'], k=args.k)
        latency_ms = (time.perf_counter() - t0) * 1000
        total_latency += latency_ms

        p_at_k = precision_at_k(results, q['expected_genre'], args.k)
        mean_sim = mean_score_at_k(results, args.k)

        results_summary.append({
            'query': q['query'][:50],
            'p@k': round(p_at_k, 3),
            'mean_sim': round(mean_sim, 3),
            'latency_ms': round(latency_ms, 1),
        })

    # Print results table
    print(f'{"Query":<52} {"P@" + str(args.k):<8} {"AvgSim":<8} {"ms":<8}')
    print('-' * 78)
    for r in results_summary:
        print(f"{r['query']:<52} {r['p@k']:<8.3f} {r['mean_sim']:<8.3f} {r['latency_ms']:<8.1f}")
    print('-' * 78)
    avg_p = float(np.mean([r['p@k'] for r in results_summary]))
    avg_sim = float(np.mean([r['mean_sim'] for r in results_summary]))
    avg_lat = total_latency / len(TEST_QUERIES)
    print(f'Mean P@{args.k}: {avg_p:.3f}   Mean Similarity: {avg_sim:.3f}   Avg Latency: {avg_lat:.1f}ms')
    print()
    print('NOTE: Precision is computed against genre tags, not true user relevance.')
    print('These metrics are indicative only. Actual relevance requires user feedback.')


if __name__ == '__main__':
    main()
