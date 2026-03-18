"""
RAG retrieval for workflow extraction.

Loads extracted_train.json, embeds procedure texts with all-MiniLM-L6-v2,
and retrieves the most similar examples at query time via cosine similarity.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def build_example_pool(train_path: Path) -> tuple[list, np.ndarray, SentenceTransformer]:
    """Load training records and pre-compute embeddings for all procedure texts.

    Returns:
        pool      : list of records from extracted_train.json
        embeddings: normalised (N, D) float32 array — one row per record
        embedder  : loaded SentenceTransformer model (reused at query time)
    """
    with open(train_path, encoding="utf-8") as f:
        pool = json.load(f)

    # all-MiniLM-L6-v2 is small (80MB), fast on CPU, and good enough for
    # semantic similarity between short procedural texts
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [r["procedure_text"] for r in pool]

    # normalize_embeddings=True means retrieval is just a dot product —
    # no need to compute full cosine similarity at query time
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    return pool, embeddings, embedder


def retrieve_similar_workflows(
    query: str,
    pool: list,
    embeddings: np.ndarray,
    embedder: SentenceTransformer,
    k: int = 2,
) -> list[tuple[dict, float]]:
    """Return the k most similar training records to the query string."""

    # embed the query the same way as the pool (normalized)
    query_emb = embedder.encode([query], normalize_embeddings=True)

    # dot product against the full pool matrix — fast because embeddings are normalized
    scores = (embeddings @ query_emb.T).squeeze()

    # argsort ascending, reverse to get highest scores first, take top k
    top_k_indices = np.argsort(scores)[::-1][:k]
    return [(pool[i], float(scores[i])) for i in top_k_indices]


def format_retrieval_results(results: list[tuple[dict, float]]) -> str:
    """Format retrieved examples as procedure text + workflow JSON.
    
    The similarity score is included so the model can judge how relevant
    each example actually is — a score below ~0.5 means weak match.
    """
    parts = []
    for i, (record, score) in enumerate(results, 1):
        parts.append(f"--- Retrieved Example {i} (similarity: {score:.2f}) ---")
        parts.append(f"PROCEDURE:\n{record['procedure_text']}")
        # no CoT here — the model generates its own reasoning,
        # these examples just provide structural reference
        parts.append(f"WORKFLOW:\n{json.dumps(record['workflow'], indent=2, ensure_ascii=False)}")
    return "\n\n".join(parts)