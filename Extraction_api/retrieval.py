
#RAG retrieval for workflow extraction.
#we load extracted_train.json embed procedure texts with MiniLM-L6-v2 (because it s very small and fast to downlaod)
#and we retrieves the most 2 similar examples at query time with cos sim


import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

#methid for emmbeding our proeceudres texts
def build_example_pool(train_path: Path) -> tuple[list, np.ndarray, SentenceTransformer]:
 
    with open(train_path, encoding="utf-8") as f:
        pool = json.load(f)

    #all-MiniLM-L6-v2 is small (80MB) and good enough for
    #semantic similarity between short procedural texts
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [r["procedure_text"] for r in pool]

    #normalize_embeddings=True means retrieval is just a dot product 
    #tahts why as usual we do matrix dot product since both vectors have lenght 1 
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
    k: int = 2, #for now lets keep it 2 so we keep the costs low, 2 similar procedures should be enough
) -> list[tuple[dict, float]]:


    #we embed the query this comes crom _RETREIVAL_TOOL check that and only when model is uncertian
    query_emb = embedder.encode([query], normalize_embeddings=True)

    #the dot prod between the embebded prcodures and the query
    scores = (embeddings @ query_emb.T).squeeze()

    #we take top k from reverse but for now k is 2
    top_k_indices = np.argsort(scores)[::-1][:k]
    return [(pool[i], float(scores[i])) for i in top_k_indices]


def format_retrieval_results(results: list[tuple[dict, float]]) -> str:
    #we now format retrieved examples as procedure text + workflow JSON as extra context
    #the similarity score is included so the model can judge how relevant
    #each example actually is 

    parts = []
    for i, (record, score) in enumerate(results, 1):
        parts.append(f"--- Retrieved Example {i} (similarity: {score:.2f}) ---")
        parts.append(f"PROCEDURE:\n{record['procedure_text']}")

        #these examples just provide structural reference, no CoT or anything
        parts.append(f"WORKFLOW:\n{json.dumps(record['workflow'], indent=2, ensure_ascii=False)}")
    return "\n\n".join(parts)