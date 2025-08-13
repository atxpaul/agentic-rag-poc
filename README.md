# Pan GPT (RAG PoC)

A lightweight Retrieval-Augmented Generation (RAG) proof of concept that:

-   Ingests local Markdown/PDF files into a vector DB (Qdrant)
-   Optionally mirrors structure to a Neo4j graph (Doc/Chunk + NEXT relations)
-   Serves a FastAPI endpoint with a simple ChatGPT‑style frontend (Vue + Tailwind CDN)
-   Logs retrieval traces as JSONL for observability

## Features

-   Chunking via `RecursiveCharacterTextSplitter` (size 800, overlap 120)
-   Embeddings and chat via an OpenAI‑compatible LM Studio endpoint
-   Vector store: Qdrant (local via Docker)
-   Optional graph enrichments in Neo4j (NEXT neighbors)
-   Incremental/idempotent ingest using deterministic `chunk_id`
-   JSONL logging with latency and selection metadata

## Project layout

-   `ingest.py`: loads docs from `data/`, chunks, embeds, upserts into Qdrant, and optionally stores graph nodes/edges in Neo4j
-   `rag.py`: builds the retrieval + LLM chain; includes structured logging
-   `server.py`: FastAPI server exposing `/health` and `/query`, and serving the web UI
-   `static/index.html`: Dark UI similar to ChatGPT (Vue 3 + Tailwind CDN)

## Requirements

-   Python 3.10+
-   LM Studio (or any OpenAI‑compatible server) listening at `LMSTUDIO_BASE`
-   Docker (for Qdrant and optionally Neo4j)

Recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Install Python deps:

```bash
python -m pip install -U \
  langchain-core langchain-text-splitters langchain-community \
  langchain-openai langchain-qdrant qdrant-client \
  requests python-dotenv pypdf neo4j \
  fastapi "uvicorn[standard]"
```

## Environment variables (.env)

```env
# LM Studio / OpenAI-compatible endpoints
LMSTUDIO_BASE=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
LMSTUDIO_CHAT_MODEL=gpt-oss-20b

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=docs
# QDRANT_API_KEY=   # leave empty for local HTTP

# Graph (optional)
GRAPH_ENABLED=false
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password

# Ingest behavior
CHROMA_REPLACE_BY_SOURCE=false  # if true, re-write per document source
RAG_INDEX_VERSION=v1
RAG_LOG_PATH=logs/rag.log
```

## Docker (Qdrant and Neo4j)

Create `docker-compose.yml` (if you don’t have one):

```yaml
version: '3.8'
services:
    qdrant:
        image: qdrant/qdrant:latest
        container_name: qdrant
        restart: unless-stopped
        ports:
            - '6333:6333' # REST
            - '6334:6334' # gRPC
        volumes:
            - ./qdrant/storage:/qdrant/storage

    neo4j:
        image: neo4j:5
        container_name: neo4j
        restart: unless-stopped
        ports:
            - '7474:7474' # Browser
            - '7687:7687' # Bolt
        environment:
            - NEO4J_AUTH=neo4j/neo4j_password
            - NEO4J_PLUGINS=["apoc"]
            - NEO4J_dbms_security_procedures_unrestricted=apoc.*
        volumes:
            - ./neo4j/data:/data
            - ./neo4j/logs:/logs
            - ./neo4j/plugins:/plugins
```

Start services:

```bash
docker compose up -d
```

## Ingest documents

Place `.md`, `.markdown`, `.txt`, and `.pdf` under `data/`. Then run:

```bash
python ingest.py
```

What happens:

-   Documents are chunked
-   Each chunk gets a deterministic `chunk_id` (UUID5 of source+content)
-   If `CHROMA_REPLACE_BY_SOURCE=true`, existing chunks for a source are removed before upsert; otherwise, upsert by `ids`
-   Qdrant collection is created on first run using the detected embedding dimension
-   If `GRAPH_ENABLED=true`, Doc/Chunk nodes and CONTAINS/NEXT relations are upserted in Neo4j (idempotent via MERGE)

## Run the server

```bash
uvicorn server:app --reload --port 8000
```

-   Web UI: `http://localhost:8000/` (dark, ChatGPT‑style)
-   Health: `http://localhost:8000/health`
-   API:

```bash
curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I add an external USB drive?"}'
```

## Graph‑augmented retrieval (optional)

Set `GRAPH_ENABLED=true` and ensure Neo4j is running. The retriever will:

-   Retrieve top‑k chunks by vector similarity
-   Expand context with immediate `NEXT` neighbors from the graph for continuity

## Logging and tracing

-   JSONL written to `logs/rag.log` (configurable via `RAG_LOG_PATH`)
-   Example retrieval event schema:

```json
{
    "ts": "2025-08-10T12:34:56.789Z",
    "event": "retrieval",
    "trace_id": "...",
    "question_hash": "...",
    "question_meta": { "lang": "es", "len": 42, "intent": "task" },
    "index_version": "v1",
    "model_version": {
        "embedder": "nomic-ai/nomic-embed-text-v1.5",
        "llm": "gpt-oss-20b"
    },
    "latency_ms": { "embed": 8, "retrieve": 22, "rerank": 15 },
    "selected": [
        {
            "doc_id": "…",
            "source": "proxmox/...md",
            "chunk_index": 0,
            "score": 0.73,
            "rank": 1
        }
    ]
}
```

Read logs:

```bash
tail -f logs/rag.log
# or pretty-print
jq . logs/rag.log | less -R
```

## Configuration tips

-   Local dev: omit `QDRANT_API_KEY` to avoid insecure-key warning on HTTP
-   LM Studio must expose `/v1/embeddings` and `/v1/chat/completions`
-   Adjust chunking in `ingest.py` (`chunk_size`, `chunk_overlap`) for your corpus
-   Tune retriever `k` in `rag.py`

## Incremental ingest

-   Deterministic `chunk_id` enables safe re‑runs without duplicates
-   To fully refresh a modified file, set `CHROMA_REPLACE_BY_SOURCE=true` or just re‑run ingest (it deletes by `ids` first)

## Switching vector DBs

-   The code currently uses Qdrant. If you prefer Chroma, revert to `langchain_chroma` and `Chroma` (the ingest logic already supports idempotent upserts by IDs)

## Troubleshooting

-   Error: `Not found: Collection 'docs' doesn't exist!`
    -   The first run of `ingest.py` will create it automatically. Ensure Qdrant is up.
-   Error: `Api key is used with an insecure connection.`
    -   For local HTTP, leave `QDRANT_API_KEY` unset.
-   Empty results
    -   Check `data/` actually contains supported files; review `logs/rag.log` for selected chunks.
-   Neo4j errors
    -   Verify `GRAPH_ENABLED=true`, credentials, and container is healthy. Open `http://localhost:7474`.

## License

PoC for personal use. Replace with your preferred license if publishing.
