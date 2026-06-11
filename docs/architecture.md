# Semantic Movie Recommendation — System Architecture

## Component Overview

| Component | Technology | Role |
|-----------|-----------|------|
| Frontend | React 18 | User interface for query input and result display |
| API Server | Node.js + Express | REST API gateway |
| Embedding Service | Python + sentence-transformers | NLP processing and similarity computation |
| Data Store | NumPy (.npy) + CSV | Pre-computed embeddings and movie metadata |

## Key Technical Decisions

- **Sentence embeddings over TF-IDF**: Captures semantic meaning, not just keyword frequency
- **Cosine similarity over Euclidean**: Magnitude-invariant, better for text embeddings
- **Pre-computation**: Movie embeddings computed once and stored, reducing query latency
- **Python-Node bridge**: Keeps NLP logic in Python (richer ecosystem) while API stays in Node.js
