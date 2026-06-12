"""
generate_embeddings.py
======================
Standalone script to pre-compute sentence embeddings for all movies in the
metadata CSV and persist them to disk.

Run::

    python backend/generate_embeddings.py \\
        --metadata embeddings/movies_metadata.csv \\
        --output_dir embeddings \\
        --model_name all-MiniLM-L6-v2 \\
        --batch_size 64

Outputs:
    embeddings/movie_embeddings.npy   — NumPy float32 array (n_movies, embed_dim)
    embeddings/movie_documents.txt    — One normalised document per line

Author: Umesh Pandey
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from the backend package when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from embedding_service import EmbeddingService  # noqa: E402
from text_preprocessor import TextPreprocessor  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("generate_embeddings")

# ---------------------------------------------------------------------------
# Required CSV columns
# ---------------------------------------------------------------------------
_REQUIRED_COLUMNS = {"movie_id", "title", "genre", "year", "plot"}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_movies(metadata_path: str) -> pd.DataFrame:
    """
    Load and validate the movies metadata CSV.

    Args:
        metadata_path: Path to ``movies_metadata.csv``.

    Returns:
        Validated Pandas DataFrame.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError:        If required columns are absent.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: '{metadata_path}'. "
            "Make sure the path is correct."
        )

    logger.info("Loading metadata from '%s'…", metadata_path)
    df = pd.read_csv(metadata_path)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata CSV is missing required columns: {missing}."
        )

    # Sanitise
    df["plot"] = df["plot"].fillna("")
    df["genre"] = df["genre"].fillna("Unknown")
    df["year"] = df["year"].fillna(0).astype(int)

    logger.info("Loaded %d movies.", len(df))
    return df


def build_documents(df: pd.DataFrame, preprocessor: TextPreprocessor) -> list:
    """
    Construct one normalised document string per movie.

    Args:
        df:           Movies DataFrame.
        preprocessor: :class:`~text_preprocessor.TextPreprocessor` instance.

    Returns:
        List of document strings in the same order as ``df``.
    """
    logger.info("Building documents for %d movies…", len(df))
    documents = []
    for i, row in df.iterrows():
        year = int(row["year"]) if row["year"] else None
        doc = preprocessor.create_movie_document(
            title=row["title"],
            genre=row["genre"],
            plot=row["plot"],
            year=year,
        )
        documents.append(doc)
        if (i + 1) % 50 == 0:
            logger.info("  Built %d / %d documents.", i + 1, len(df))

    logger.info("Document construction complete.")
    return documents


def save_documents(documents: list, output_dir: str) -> str:
    """
    Save document strings to a plain-text file (one per line).

    Args:
        documents:  List of document strings.
        output_dir: Directory to write ``movie_documents.txt`` into.

    Returns:
        Absolute path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "movie_documents.txt")
    with open(out_path, "w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(doc.replace("\n", " ") + "\n")
    logger.info("Saved %d documents to '%s'.", len(documents), out_path)
    return out_path


def generate_and_save_embeddings(
    documents: list,
    output_dir: str,
    model_name: str,
    batch_size: int,
) -> str:
    """
    Embed all documents and save to ``movie_embeddings.npy``.

    Args:
        documents:  List of document strings.
        output_dir: Directory to write the ``.npy`` file into.
        model_name: HuggingFace sentence-transformers model name.
        batch_size: Embedding batch size.

    Returns:
        Absolute path of the saved ``.npy`` file.
    """
    svc = EmbeddingService(model_name=model_name, embeddings_dir=output_dir)
    svc.load_model()

    logger.info(
        "Starting embedding of %d documents (batch_size=%d)…",
        len(documents),
        batch_size,
    )
    t0 = time.time()
    embeddings: np.ndarray = svc.embed_batch(documents, batch_size=batch_size)
    elapsed = time.time() - t0
    logger.info(
        "Embedding complete in %.1f s. Shape: %s.",
        elapsed,
        str(embeddings.shape),
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "movie_embeddings.npy")
    svc.save_embeddings(embeddings, out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-compute sentence embeddings for all movies in the "
                    "metadata CSV and save them to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=os.path.join("embeddings", "movies_metadata.csv"),
        help="Path to movies_metadata.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="embeddings",
        help="Directory to save embeddings and documents",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="HuggingFace sentence-transformers model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of documents to embed per batch",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the embedding generation script."""
    args = parse_args()

    logger.info("=== Semantic Movie Recommendation — Embedding Generator ===")
    logger.info("Metadata  : %s", args.metadata)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Model     : %s", args.model_name)
    logger.info("Batch size: %d", args.batch_size)

    # Step 1: Load metadata
    df = load_movies(args.metadata)

    # Step 2: Build documents
    preprocessor = TextPreprocessor()
    documents = build_documents(df, preprocessor)

    # Step 3: Save raw documents
    doc_path = save_documents(documents, args.output_dir)
    logger.info("Documents saved → %s", doc_path)

    # Step 4: Generate and save embeddings
    emb_path = generate_and_save_embeddings(
        documents=documents,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    logger.info("Embeddings saved → %s", emb_path)

    logger.info("=== Done! ===")


if __name__ == "__main__":
    main()
