
#RAG retrieval for workflow extraction.
#we load extracted_train.json and embed procedure texts using OpenAI text-embedding-3-small
#supports up to 8191 tokens per text — no truncation issues with long procedures
#retrieves the most similar example at query time via cosine similarity (dot product)


import json
import numpy as np
from pathlib import Path
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims, 8191 token limit, $0.02/1M tokens


def _embed(texts: list[str], client: OpenAI) -> np.ndarray:
    #OpenAI API accepts up to 2048 inputs per request so we batch in chunks
    #returns a (N, 1536) float32 array, already normalized by the API
    all_embeddings = []
    batch_size = 512  # safe batch size well under the 2048 limit

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        # response.data is a list of Embedding objects sorted by index
        batch_vecs = np.array([e.embedding for e in response.data], dtype=np.float32)
        all_embeddings.append(batch_vecs)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} procedures...")

    return np.vstack(all_embeddings)


def build_example_pool(train_path: Path, client: OpenAI) -> tuple[list, np.ndarray]:
    #load training records and pre-compute embeddings for all procedure texts
    #we pass the OpenAI client so no local model is needed — same API key as extraction

    with open(train_path, encoding="utf-8") as f:
        pool = json.load(f)

    texts = [r["procedure_text"] for r in pool]
    embeddings = _embed(texts, client)
    return pool, embeddings


def retrieve_similar_workflows(
    query: str,
    pool: list,
    embeddings: np.ndarray,
    client: OpenAI,
    k: int = 1,  # retrieve only the single most similar procedure
) -> list[tuple[dict, float]]:
    #embed the query using the same model as the pool — single API call
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_emb = np.array(response.data[0].embedding, dtype=np.float32)

    #dot product works as cosine similarity because OpenAI embeddings are already normalized
    scores = (embeddings @ query_emb).squeeze()

    #return top k with their similarity scores
    top_k_indices = np.argsort(scores)[::-1][:k]
    return [(pool[i], float(scores[i])) for i in top_k_indices]


def format_retrieval_results(results: list[tuple[dict, float]]) -> str:
    #format retrieved examples as procedure text + workflow JSON as extra context
    #similarity score included so model can judge relevance (below ~0.5 = weak match)

    parts = []
    for i, (record, score) in enumerate(results, 1):
        parts.append(f"--- Retrieved Example {i} (similarity: {score:.2f}) ---")
        parts.append(f"PROCEDURE:\n{record['procedure_text']}")
        #no CoT — just the procedure and workflow JSON as structural reference
        parts.append(f"WORKFLOW:\n{json.dumps(record['workflow'], indent=2, ensure_ascii=False)}")
    return "\n\n".join(parts)
