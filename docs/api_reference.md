# API Reference — Semantic Movie Recommendation

> **Version:** 1.0.0  
> **Base URL (local):** `http://localhost:8000`  
> **Author:** Umesh Pandey  
> **License:** MIT

---

## Section 1: REST API Endpoints

All endpoints return JSON. Errors follow the standard FastAPI error format:

```json
{
  "detail": "Human-readable error message"
}
```

---

### `GET /api/health`

**Description:** Returns the current health status of the service, including
the loaded model name and the number of indexed movies.

#### Response Body

| Field | Type | Description |
|---|---|---|
| `status` | `string` | `"ok"` when healthy, `"degraded"` when engine failed to load |
| `model` | `string` | Name of the loaded sentence-transformers model |
| `movies_count` | `integer` | Number of movies in the embedding index |
| `embeddings_loaded` | `boolean` | Whether embeddings are in memory |

#### Example Response

```json
{
  "status": "ok",
  "model": "all-MiniLM-L6-v2",
  "movies_count": 250,
  "embeddings_loaded": true
}
```

#### Example curl

```bash
curl -X GET http://localhost:8000/api/health
```

---

### `POST /api/recommend`

**Description:** Returns a ranked list of movie recommendations for a
free-text natural language query using cosine similarity over sentence
embeddings.

#### Request Body

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `string` | ✅ | — | Natural language query (1–500 chars) |
| `k` | `integer` | ❌ | `10` | Number of results (1–50) |
| `genre_filter` | `string` | ❌ | `null` | Genre substring filter (e.g. `"sci-fi"`) |
| `min_score` | `float` | ❌ | `0.0` | Minimum cosine similarity threshold (0.0–1.0) |

#### Response Body

| Field | Type | Description |
|---|---|---|
| `query` | `string` | Original query string |
| `k` | `integer` | Requested number of results |
| `genre_filter` | `string\|null` | Applied genre filter |
| `count` | `integer` | Number of results returned |
| `results` | `array` | List of movie result objects |

Each element of `results` contains:

| Field | Type | Description |
|---|---|---|
| `movie_id` | `integer` | Unique movie ID |
| `title` | `string` | Movie title |
| `genre` | `string` | Genre string (e.g. `"Sci-Fi/Thriller"`) |
| `year` | `integer` | Release year |
| `plot` | `string` | Plot summary |
| `rating` | `float` | IMDb-style rating |
| `score` | `float` | Cosine similarity score (higher = more relevant) |

#### Example Request Body

```json
{
  "query": "space exploration adventure with a crew",
  "k": 5,
  "genre_filter": "sci-fi",
  "min_score": 0.2
}
```

#### Example Response

```json
{
  "query": "space exploration adventure with a crew",
  "k": 5,
  "genre_filter": "sci-fi",
  "count": 3,
  "results": [
    {
      "movie_id": 4,
      "title": "Interstellar",
      "genre": "Sci-Fi",
      "year": 2014,
      "plot": "A team of explorers travel through a wormhole...",
      "rating": 8.6,
      "score": 0.812345
    }
  ]
}
```

#### Example curl

```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "space exploration adventure", "k": 5, "genre_filter": "sci-fi"}'
```

#### Error Responses

| Status | Condition |
|---|---|
| `422 Unprocessable Entity` | Query is empty or validation fails |
| `503 Service Unavailable` | Recommendation engine failed to initialise |

---

### `GET /api/movies`

**Description:** Returns a paginated list of all movies in the database.

#### Query Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | `integer` | `100` | Max movies per page (1–500) |
| `offset` | `integer` | `0` | Number of movies to skip |

#### Response Body

| Field | Type | Description |
|---|---|---|
| `count` | `integer` | Total number of movies in the database |
| `movies` | `array` | Paginated list of movie records |

#### Example Response

```json
{
  "count": 250,
  "movies": [
    {
      "movie_id": 1,
      "title": "The Shawshank Redemption",
      "genre": "Drama",
      "year": 1994,
      "plot": "Two imprisoned men bond over a number of years...",
      "rating": 9.3
    }
  ]
}
```

#### Example curl

```bash
curl "http://localhost:8000/api/movies?limit=20&offset=0"
```

---

### `GET /api/movies/{movie_id}`

**Description:** Retrieve a single movie record by its integer ID.

#### Path Parameters

| Parameter | Type | Description |
|---|---|---|
| `movie_id` | `integer` | Unique movie identifier |

#### Response Body

Full movie record (same schema as individual items in `/api/movies`).

#### Example curl

```bash
curl http://localhost:8000/api/movies/4
```

#### Example Response

```json
{
  "movie_id": 4,
  "title": "Interstellar",
  "genre": "Sci-Fi",
  "year": 2014,
  "plot": "A team of explorers travel through a wormhole in space...",
  "rating": 8.6
}
```

#### Error Responses

| Status | Condition |
|---|---|
| `404 Not Found` | No movie exists with the given `movie_id` |

---

## Section 2: Python Module Reference

### `TextPreprocessor`

**Module:** `backend.text_preprocessor`

**Description:**
Provides text cleaning, stopword removal, normalisation, and movie document
construction for the recommendation pipeline. Designed to produce consistent
input strings for the embedding model, improving retrieval quality.

#### Constructor

```python
TextPreprocessor(extra_stopwords: Optional[frozenset] = None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `extra_stopwords` | `frozenset \| None` | `None` | Additional stopwords merged with the built-in set |

#### Key Methods

---

##### `clean_text(text: str) -> str`

Lowercase, remove punctuation, replace hyphens with spaces, normalise
consecutive whitespace.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Raw input string |

**Returns:** `str` — Cleaned string.

---

##### `remove_stopwords(text: str, language: str = 'english') -> str`

Remove stopwords from a (preferably already-lowercased) string.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Input string |
| `language` | `str` | Stopword language (`'english'` only natively supported) |

**Returns:** `str` — String with stopwords removed.

---

##### `normalize(text: str) -> str`

Full pipeline: `clean_text` → `remove_stopwords`.

**Returns:** `str` — Normalised string suitable for embedding.

---

##### `batch_normalize(texts: List[str]) -> List[str]`

Apply `normalize` to every element of a list.

| Parameter | Type | Description |
|---|---|---|
| `texts` | `List[str]` | List of raw strings |

**Returns:** `List[str]` — Normalised strings, same length as input.  
**Raises:** `TypeError` if `texts` is not a list.

---

##### `create_movie_document(title, genre, plot, year=None) -> str`

Combine movie metadata fields into a single document string for embedding.
Fields are joined with ` | ` separators and individually cleaned.

| Parameter | Type | Description |
|---|---|---|
| `title` | `str` | Movie title |
| `genre` | `str` | Genre string |
| `plot` | `str` | Plot summary |
| `year` | `int \| None` | Optional release year |

**Returns:** `str` — Formatted document string.

---

### `EmbeddingService`

**Module:** `backend.embedding_service`

**Description:**
High-level service for generating sentence embeddings using `sentence-transformers`.
Falls back to a deterministic hash-based embedder if the library is not installed,
allowing the rest of the pipeline to function in CI/CD environments without
heavy ML dependencies.

#### Constructor

```python
EmbeddingService(model_name: str = 'all-MiniLM-L6-v2', embeddings_dir: str = 'embeddings')
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `'all-MiniLM-L6-v2'` | HuggingFace sentence-transformers model |
| `embeddings_dir` | `str` | `'embeddings'` | Default directory for saving/loading `.npy` files |

#### Key Methods

---

##### `load_model() -> None`

Load the sentence-transformer model (or fallback hash embedder) into memory.
Idempotent: safe to call multiple times.

---

##### `embed_text(text: str) -> np.ndarray`

Embed a single text string.

**Returns:** `np.ndarray` of shape `(embed_dim,)`.  
**Raises:** `ValueError` if text is empty.

---

##### `embed_batch(texts: List[str], batch_size: int = 32) -> np.ndarray`

Embed a list of texts in batches.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `texts` | `List[str]` | — | Input strings |
| `batch_size` | `int` | `32` | Documents per forward pass |

**Returns:** `np.ndarray` of shape `(len(texts), embed_dim)`.

---

##### `cosine_similarity_matrix(query_vec, corpus_vecs) -> np.ndarray`

Compute cosine similarity between a 1-D query and a 2-D corpus matrix.

| Parameter | Type | Description |
|---|---|---|
| `query_vec` | `np.ndarray` | 1-D array of shape `(embed_dim,)` |
| `corpus_vecs` | `np.ndarray` | 2-D array of shape `(n_docs, embed_dim)` |

**Returns:** `np.ndarray` of shape `(n_docs,)` with scores in `[-1, 1]`.

---

##### `top_k_similar(query_vec, corpus_vecs, k=10) -> List[Tuple[int, float]]`

Return the k most similar documents as `(index, score)` tuples, sorted by
descending similarity.

---

##### `save_embeddings(embeddings, filepath) -> None`

Save a NumPy array to `.npy` format.

---

##### `load_precomputed_embeddings(filepath) -> np.ndarray`

Load a pre-saved `.npy` embedding file.  
**Raises:** `FileNotFoundError` if the file does not exist.

---

### `RecommendationEngine`

**Module:** `backend.recommendation_engine`

**Description:**
End-to-end recommendation engine that chains metadata loading, embedding
index construction, and cosine-similarity retrieval. Exposes a simple
`recommend(query)` interface.

#### Constructor

```python
RecommendationEngine(metadata_path: str, embeddings_path: Optional[str] = None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metadata_path` | `str` | — | Path to `movies_metadata.csv` |
| `embeddings_path` | `str \| None` | `None` | Path to pre-computed `movie_embeddings.npy` |

#### Key Methods

---

##### `load_metadata() -> pd.DataFrame`

Load and validate the movies CSV.  
**Raises:** `FileNotFoundError`, `ValueError`.

---

##### `build_index(embedding_service: EmbeddingService) -> None`

Load or generate the embedding index for all movies.  
Must be called after `load_metadata()`.

---

##### `recommend(query: str, k: int = 10, min_score: float = 0.0) -> List[Dict]`

Return top-k recommendations for a free-text query.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | — | Natural language query |
| `k` | `int` | `10` | Number of results |
| `min_score` | `float` | `0.0` | Minimum similarity threshold |

**Returns:** List of dicts with keys `movie_id`, `title`, `genre`, `year`, `plot`, `rating`, `score`.

---

##### `filter_by_genre(results, genre: str) -> List[Dict]`

Post-filter a results list by genre substring (case-insensitive).

---

##### `get_movie_by_id(movie_id: int) -> Dict`

Retrieve a single movie record.  
**Raises:** `KeyError` if not found.

---

##### `get_all_movies() -> List[Dict]`

Return all movies as a list of dictionaries.

---

##### Properties

| Property | Type | Description |
|---|---|---|
| `movies_count` | `int` | Number of loaded movies |
| `model_name` | `str` | Name of the underlying embedding model |
