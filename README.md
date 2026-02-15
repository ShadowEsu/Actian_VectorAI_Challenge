# Actian_VectorAI_Challenge
SF Hacks Actian VectorAI Use of Database Challenge Submission - Preston Susanto

Actian VectorAI DB + Python RAG Demo (Challenge Submission)
This project demonstrates an end-to-end Vector Search + RAG-style answering workflow using Actian VectorAI DB (gRPC on localhost:50051) and the Actian Cortex Python client (actiancortex).
It supports:
✅ Create / recreate vector collections (HNSW indexing)
✅ Ingest structured data (CSV) into vectors
✅ KNN search (semantic retrieval)
✅ Filtered search with Type-safe Filter DSL
✅ RAG-style answers with citations (evidence chunks)
What this demo does
Takes structured data (example: data.csv)
Embeds each row into a vector (simple local embedding for no-key setup)
Stores vectors in Actian VectorAI DB
On query:
embeds the question
runs vector search (optionally filtered)
returns a grounded answer + top evidence chunks
This demonstrates how VectorAI DB can power semantic retrieval on real “database-like” datasets.


Challenges Encountered & Lessons Learned

1. Environment & Setup Friction
Challenge:
Initial setup required aligning multiple components (Docker Desktop, Python ≥3.10, virtual environments, and the Actian Cortex client). Version mismatches (e.g., older Python versions) caused installation errors during early attempts.
Fix / Lesson:
Standardizing on Python 3.11 and isolating dependencies in a project-specific virtual environment resolved these issues and made the setup reproducible across machines.

2.  Docker & Platform Compatibility
Challenge:
Running the VectorAI DB Docker image on Apple Silicon (ARM) introduced minor compatibility concerns since the image targets linux/amd64.
Fix / Lesson:
Docker Desktop’s built-in x86 emulation handled this seamlessly once enabled. No runtime performance issues were observed during local testing, demonstrating Docker’s robustness for cross-architecture development.

3. Data Chunking & Retrieval Quality
Challenge:
Using large text fields without chunking initially reduced retrieval accuracy, as long documents diluted semantic similarity scores.
Fix / Lesson:
Introducing text chunking significantly improved search relevance. This reinforced the importance of structuring data properly before vector ingestion.



✅ Fast & Reliable Vector Search
Actian VectorAI DB delivered consistently low-latency similarity search, even after batch ingestion of multiple vectors. Search results were stable and deterministic across runs.

✅ Clean & Intuitive Python Client
The Cortex Python client provided a clear, well-structured API with strong typing and async support. Collection management, batch upserts, and filtered searches were easy to implement.

✅ Powerful Filtering with Minimal Complexity
The type-safe Filter DSL allowed combining structured filtering (e.g., category, source) with vector similarity search, enabling hybrid search patterns without additional infrastructure.

✅ Production-Oriented Design
Persistent storage, HNSW indexing, and gRPC communication made the system feel production-ready rather than a purely academic demo. Restarting the database preserved collections without re-ingestion.
