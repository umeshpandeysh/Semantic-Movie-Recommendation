"""
app.py
======
FastAPI application for the Semantic Movie Recommendation system.

Exposes a RESTful API to:
    GET  /api/health              — Service health check
    POST /api/recommend           — Get movie recommendations for a query
    GET  /api/movies              — List all movies
    GET  /api/movies/{movie_id}   — Retrieve a single movie by ID

Run locally::

    uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

Or from within the backend directory::

    uvicorn app:app --reload

Author: Umesh Pandey
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ensure backend package is importable when run from project root
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND_DIR))

from embedding_service import EmbeddingService  # noqa: E402
from recommendation_engine import RecommendationEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Environment-driven configuration
# ---------------------------------------------------------------------------
_METADATA_PATH = os.environ.get(
    "METADATA_PATH",
    str(Path(_BACKEND_DIR).parent / "embeddings" / "movies_metadata.csv"),
)
_EMBEDDINGS_PATH = os.environ.get(
    "EMBEDDINGS_PATH",
    str(Path(_BACKEND_DIR).parent / "embeddings" / "movie_embeddings.npy"),
)
_MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Global engine (populated during startup)
# ---------------------------------------------------------------------------
_engine: Optional[RecommendationEngine] = None
_embedding_service: Optional[EmbeddingService] = None


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.

    Loads the recommendation engine and embedding model on startup so they
    are ready for the first request without cold-start latency.
    """
    global _engine, _embedding_service

    logger.info("=== Semantic Movie Recommender API — starting up ===")
    logger.info("Metadata  : %s", _METADATA_PATH)
    logger.info("Embeddings: %s", _EMBEDDINGS_PATH)
    logger.info("Model     : %s", _MODEL_NAME)

    try:
        _embedding_service = EmbeddingService(
            model_name=_MODEL_NAME,
            embeddings_dir=str(Path(_EMBEDDINGS_PATH).parent),
        )
        _embedding_service.load_model()

        embeddings_path = (
            _EMBEDDINGS_PATH if os.path.exists(_EMBEDDINGS_PATH) else None
        )
        _engine = RecommendationEngine(
            metadata_path=_METADATA_PATH,
            embeddings_path=embeddings_path,
        )
        _engine.load_metadata()
        _engine.build_index(_embedding_service)

        logger.info(
            "Engine ready — %d movies indexed.", _engine.movies_count
        )
    except Exception as exc:
        logger.error("Failed to initialise recommendation engine: %s", exc)
        # Allow the app to start in degraded mode; endpoints will return 503
        _engine = None

    yield  # ← application runs here

    logger.info("=== Shutting down ===")
    _engine = None
    _embedding_service = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Semantic Movie Recommendation API",
    description=(
        "NLP-powered movie recommendation service using sentence-transformers "
        "and cosine similarity. Built as a 2nd-year B.Tech portfolio project."
    ),
    version="1.0.0",
    contact={"name": "Umesh Pandey"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """Request body for the /api/recommend endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Free-text query for movie recommendations.",
        examples=["space exploration adventure with a hero"],
    )
    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of recommendations to return.",
    )
    genre_filter: Optional[str] = Field(
        default=None,
        description="Optional genre substring to filter results (e.g. 'sci-fi').",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold.",
    )


class MovieResult(BaseModel):
    """Single movie recommendation result."""

    movie_id: int
    title: str
    genre: str
    year: int
    plot: str
    rating: float
    score: float


class RecommendResponse(BaseModel):
    """Response body for the /api/recommend endpoint."""

    query: str
    k: int
    genre_filter: Optional[str]
    count: int
    results: List[MovieResult]


class MovieRecord(BaseModel):
    """Full movie record (no score)."""

    movie_id: int
    title: str
    genre: str
    year: int
    plot: str
    rating: float


class MoviesListResponse(BaseModel):
    """Response for the movie listing endpoint."""

    count: int
    movies: List[MovieRecord]


class HealthResponse(BaseModel):
    """Response for the health-check endpoint."""

    status: str
    model: str
    movies_count: int
    embeddings_loaded: bool


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_engine() -> RecommendationEngine:
    """
    Return the global engine, raising HTTP 503 if not available.

    Returns:
        Loaded :class:`~recommendation_engine.RecommendationEngine` instance.

    Raises:
        HTTPException(503): If the engine failed to initialise.
    """
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Recommendation engine is not available. "
                "Check server logs for initialisation errors."
            ),
        )
    return _engine


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health() -> Dict[str, Any]:
    """
    Return current service status and basic statistics.

    Returns ``status: ok`` when the engine is ready; ``status: degraded``
    when initialisation failed.
    """
    if _engine is None:
        return {
            "status": "degraded",
            "model": _MODEL_NAME,
            "movies_count": 0,
            "embeddings_loaded": False,
        }
    return {
        "status": "ok",
        "model": _engine.model_name,
        "movies_count": _engine.movies_count,
        "embeddings_loaded": _engine._embeddings is not None,
    }


@app.post(
    "/api/recommend",
    response_model=RecommendResponse,
    summary="Get movie recommendations",
    tags=["Recommendations"],
)
async def recommend(body: RecommendRequest) -> Dict[str, Any]:
    """
    Return the top-k movie recommendations for a free-text query.

    The query is normalised, embedded, and compared against the corpus
    using cosine similarity.  An optional ``genre_filter`` applies a
    post-retrieval genre substring filter.

    Args:
        body: Request body containing ``query``, ``k``, ``genre_filter``,
              and ``min_score``.

    Returns:
        JSON response with ``query``, ``k``, ``genre_filter``, ``count``,
        and a ``results`` list.
    """
    engine = _require_engine()

    try:
        results = engine.recommend(
            query=body.query,
            k=body.k,
            min_score=body.min_score,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Recommendation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

    if body.genre_filter:
        results = engine.filter_by_genre(results, body.genre_filter)

    return {
        "query": body.query,
        "k": body.k,
        "genre_filter": body.genre_filter,
        "count": len(results),
        "results": results,
    }


@app.get(
    "/api/movies",
    response_model=MoviesListResponse,
    summary="List all movies",
    tags=["Movies"],
)
async def list_movies(
    limit: int = Query(default=100, ge=1, le=500, description="Max movies to return."),
    offset: int = Query(default=0, ge=0, description="Number of movies to skip."),
) -> Dict[str, Any]:
    """
    Return a paginated list of all movies in the database.

    Args:
        limit:  Maximum number of movies to return (default 100, max 500).
        offset: Number of movies to skip (for pagination).

    Returns:
        JSON with ``count`` (total) and ``movies`` (paginated slice).
    """
    engine = _require_engine()
    all_movies = engine.get_all_movies()
    paginated = all_movies[offset: offset + limit]
    return {"count": len(all_movies), "movies": paginated}


@app.get(
    "/api/movies/{movie_id}",
    response_model=MovieRecord,
    summary="Get a movie by ID",
    tags=["Movies"],
)
async def get_movie(movie_id: int) -> Dict[str, Any]:
    """
    Retrieve a single movie record by its integer ID.

    Args:
        movie_id: The unique movie identifier.

    Returns:
        Full movie record dictionary.

    Raises:
        HTTPException(404): If no movie with the given ID exists.
    """
    engine = _require_engine()
    try:
        movie = engine.get_movie_by_id(movie_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Movie with id={movie_id} not found.",
        ) from exc

    return movie


# ---------------------------------------------------------------------------
# Optional: run directly with uvicorn
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
