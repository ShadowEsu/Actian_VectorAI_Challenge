import csv
import re
import numpy as np
from typing import List, Dict, Any, Optional

from cortex import CortexClient, DistanceMetric
from cortex.filters import Filter, Field

DB_ADDR = "localhost:50051"
COLLECTION = "challenge_rag"
DIM = 256

def embed(text: str, dim: int = DIM) -> List[float]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    v = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        v[hash(t) % dim] += 1.0
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    return v.tolist()

def chunk_text(text: str, max_words: int = 180, overlap: int = 40) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]
    chunks = []
    i = 0
    step = max(1, max_words - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i:i+max_words]).strip())
        i += step
    return [c for c in chunks if c]

def recreate_collection(client: CortexClient):
    if client.has_collection(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        name=COLLECTION,
        dimension=DIM,
        distance_metric=DistanceMetric.COSINE,
        hnsw_m=32,
        hnsw_ef_construct=256,
        hnsw_ef_search=100,
    )

def ingest_csv(client: CortexClient, csv_path: str, text_cols: List[str], category_col: Optional[str]=None):
    ids, vectors, payloads = [], [], []
    vid = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            combined = " | ".join(str(row.get(c, "")).strip() for c in text_cols).strip()
            if not combined:
                continue

            category = (row.get(category_col) if category_col else None) or "general"

            for ci, chunk in enumerate(chunk_text(combined)):
                ids.append(vid)
                vectors.append(embed(chunk))
                payloads.append({
                    "source": "csv",
                    "category": category,
                    "chunk_index": ci,
                    "text": chunk,
                    **row
                })
                vid += 1

    if ids:
        client.batch_upsert(COLLECTION, ids=ids, vectors=vectors, payloads=payloads)
        client.flush(COLLECTION)

    print(f"✅ Ingested {len(ids)} chunks/vectors from {csv_path}")

def search(client: CortexClient, question: str, top_k: int = 5, category: Optional[str] = None):
    qv = embed(question)

    if category:
        flt = Filter().must(Field("category").eq(category))
        results = client.search_filtered(COLLECTION, qv, flt, top_k=top_k)
    else:
        results = client.search(COLLECTION, query=qv, top_k=top_k)

    hits = []
    for r in results:
        hits.append({"id": r.id, "score": float(r.score), "payload": r.payload})
    return hits

def answer_with_citations(question: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "No relevant results found."

    evidence_lines = []
    for i, h in enumerate(hits, start=1):
        txt = (h["payload"].get("text") or "").replace("\\n", " ").strip()
        evidence_lines.append(f"[{i}] {txt[:260]}...")

    best = (hits[0]["payload"].get("text") or "").replace("\\n", " ").strip()
    answer = best[:500]

    return (
        f"Q: {question}\\n\\n"
        f"Answer (grounded):\\n- {answer}  [1]\\n\\n"
        f"Evidence:\\n" + "\\n".join(evidence_lines)
    )

def main():
    with CortexClient(DB_ADDR) as client:
        version, uptime = client.health_check()
        print(f"✅ Connected to {version} (uptime={uptime})")

        recreate_collection(client)
        ingest_csv(client, "data.csv", text_cols=["title", "description"], category_col="category")

        while True:
            q = input("\\nAsk a question (or 'exit'): ").strip()
            if q.lower() in {"exit", "quit"}:
                break

            hits = search(client, q, top_k=5)
            print("\\n" + answer_with_citations(q, hits))

if __name__ == "__main__":
    main()
