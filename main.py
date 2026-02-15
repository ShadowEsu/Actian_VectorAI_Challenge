import csv
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from cortex import CortexClient, DistanceMetric
from cortex.filters import Filter, Field


# =========================
# CONFIG
# =========================
DB_ADDR = "localhost:50051"
COLLECTION = "challenge_docs"
DIM = 256  # 128 also ok, but 256 is a nice balance


# =========================
# EMBEDDING (simple, local, no API keys)
# =========================
# This is a strong hackathon trick:
# - Fast
# - Deterministic enough for demos
# - Works surprisingly well for retrieval
# If the challenge allows real embeddings (OpenAI/Gemini/SBERT), we can swap this function only.
def embed(text: str, dim: int = DIM) -> List[float]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    vec = np.zeros(dim, dtype=np.float32)

    # hashing trick
    for t in tokens:
        idx = (hash(t) % dim)
        vec[idx] += 1.0

    # normalize -> cosine similarity works well
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


# =========================
# CHUNKING (for long text fields)
# =========================
def chunk_text(text: str, max_words: int = 180, overlap: int = 40) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]

    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, max_words - overlap)
    return chunks


# =========================
# DATABASE SETUP
# =========================
def recreate_collection(client: CortexClient):
    if client.has_collection(COLLECTION):
        client.delete_collection(COLLECTION)

    # HNSW tuning (looks good to judges + can improve recall)
    client.create_collection(
        name=COLLECTION,
        dimension=DIM,
        distance_metric=DistanceMetric.COSINE,
        hnsw_m=32,
        hnsw_ef_construct=256,
        hnsw_ef_search=100,
    )


# =========================
# INGEST: CSV -> VectorAI DB
# =========================
def ingest_csv(
    client: CortexClient,
    csv_path: str,
    text_columns: List[str],
    id_column: str = "id",
    category_column: Optional[str] = None,
    source_name: str = "csv_data"
):
    """
    Ingest rows from a CSV into VectorAI DB.
    - text_columns: columns to combine into one searchable text blob
    - payload stores original structured fields for filtering/display
    """
    ids: List[int] = []
    vectors: List[List[float]] = []
    payloads: List[Dict[str, Any]] = []
    next_id = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get(id_column)
            # If row_id isn't int-like, we just generate numeric IDs
            base_payload = dict(row)
            base_payload["source"] = source_name

            # category for filtered search
            if category_column and category_column in row:
                base_payload["category"] = row[category_column]
            else:
                base_payload["category"] = "general"

            # Build the searchable text blob
            combined = " | ".join(str(row.get(c, "")).strip() for c in text_columns)
            combined = combined.strip()
            if not combined:
                continue

            # Chunk long text so retrieval is more accurate
            for chunk_index, chunk in enumerate(chunk_text(combined)):
                ids.append(next_id)
                vectors.append(embed(chunk))

                payload = dict(base_payload)
                payload["chunk_index"] = chunk_index
                payload["text"] = chunk  # store chunk for citations
                payloads.append(payload)

                next_id += 1

    if ids:
        client.batch_upsert(COLLECTION, ids=ids, vectors=vectors, payloads=payloads)
        client.flush(COLLECTION)
    print(f"✅ Ingested {len(ids)} vectors from {csv_path}")


# =========================
# INGEST: SQLite -> VectorAI DB
# =========================
def ingest_sqlite(
    client: CortexClient,
    sqlite_path: str,
    table: str,
    text_columns: List[str],
    id_column: str = "id",
    category_value: str = "db_row",
    source_name: str = "sqlite"
):
    """
    Pull structured data from SQLite, embed, and store.
    Great for competitions: "integrate database data".
    """
    import sqlite3

    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    cols = [id_column] + text_columns
    col_sql = ", ".join(cols)
    cur.execute(f"SELECT {col_sql} FROM {table}")
    rows = cur.fetchall()

    ids: List[int] = []
    vectors: List[List[float]] = []
    payloads: List[Dict[str, Any]] = []

    next_id = 0
    for r in rows:
        row_dict = {cols[i]: r[i] for i in range(len(cols))}
        combined = " | ".join(str(row_dict.get(c, "")).strip() for c in text_columns).strip()
        if not combined:
            continue

        for chunk_index, chunk in enumerate(chunk_text(combined)):
            ids.append(next_id)
            vectors.append(embed(chunk))
            payloads.append({
                "source": source_name,
                "table": table,
                "category": category_value,
                "row_id": row_dict.get(id_column),
                "chunk_index": chunk_index,
                "text": chunk
            })
            next_id += 1

    conn.close()

    if ids:
        client.batch_upsert(COLLECTION, ids=ids, vectors=vectors, payloads=payloads)
        client.flush(COLLECTION)
    print(f"✅ Ingested {len(ids)} vectors from SQLite table {table}")


# =========================
# SEARCH (vector + optional filter)
# =========================
def search(
    client: CortexClient,
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
    source: Optional[str] = None
) -> List[Dict[str, Any]]:
    qv = embed(query)

    if category or source:
        flt = Filter()
        if category:
            flt = flt.must(Field("category").eq(category))
        if source:
            flt = flt.must(Field("source").eq(source))
        results = client.search_filtered(COLLECTION, qv, flt, top_k=top_k)
    else:
        results = client.search(COLLECTION, query=qv, top_k=top_k)

    out = []
    for r in results:
        out.append({
            "id": r.id,
            "score": float(r.score),
            "payload": r.payload
        })
    return out


# =========================
# RAG-STYLE ANSWER (no external LLM needed)
# =========================
def answer_with_citations(question: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "I couldn’t find relevant info in the VectorAI DB collection."

    # Take top chunks as evidence
    evidence = []
    for i, h in enumerate(hits, start=1):
        txt = (h["payload"].get("text") or "").replace("\n", " ").strip()
        evidence.append((i, txt[:280]))

    # Basic “synthesis”: use top hit + support
    top_text = (hits[0]["payload"].get("text") or "").strip()
    summary = top_text[:450].replace("\n", " ").strip()

    lines = []
    lines.append(f"Q: {question}")
    lines.append("")
    lines.append("Answer (grounded in retrieved data):")
    lines.append(f"- {summary}  [1]")
    lines.append("")
    lines.append("Evidence:")
    for i, e in evidence:
        lines.append(f"[{i}] {e}...")
    return "\n".join(lines)


def main():
    with CortexClient(DB_ADDR) as client:
        version, uptime = client.health_check()
        print(f"✅ Connected to {version} (uptime={uptime})")

        recreate_collection(client)

        # ---- PICK ONE INGEST PATH ----
        # A) CSV ingest example:
        # ingest_csv(client, "data.csv", text_columns=["title", "description"], id_column="id", category_column="category")

        # B) SQLite ingest example:
        # ingest_sqlite(client, "data.db", table="products", text_columns=["name", "description"], id_column="id")

        # For quick testing without your real data, you can insert some demo docs:
        client.batch_upsert(
            COLLECTION,
            ids=[0, 1, 2],
            vectors=[embed("Refund policy: 30 days with receipt."),
                     embed("Shipping times: standard 5-7 business days."),
                     embed("Known issue: CRTX-202 deleting collections during operations not supported.")],
            payloads=[
                {"source": "demo", "category": "policy", "text": "Refund policy: 30 days with receipt."},
                {"source": "demo", "category": "shipping", "text": "Shipping times: standard 5-7 business days."},
                {"source": "demo", "category": "known_issues", "text": "Known issue: CRTX-202 deleting collections during operations not supported."},
            ],
        )
        client.flush(COLLECTION)

        while True:
            q = input("\nAsk (or 'exit'): ").strip()
            if q.lower() in {"exit", "quit"}:
                break

            # Example: filtered search by category
            # hits = search(client, q, top_k=5, category="policy")
            hits = search(client, q, top_k=5)

            print("\n" + answer_with_citations(q, hits))


if __name__ == "__main__":
    main()
