import os
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

DATA_DIR = Path("data")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


class LMStudioEmbeddings(Embeddings):
    """
    Minimal OpenAI-compatible embeddings client for LM Studio.
    Sends ONE string per request to avoid batch payload issues.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "lm-studio"
        self.model = model
        self.timeout = timeout

    def _embed_one(self, text: str) -> List[float]:
        payload = {"model": self.model, "input": str(text)}
        r = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


def load_docs(path: Path):
    docs = []
    exts = {".txt", ".md", ".markdown", ".pdf"}
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            if p.suffix.lower() in {".txt", ".md", ".markdown"}:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())

    # Normalize source paths to be relative to DATA_DIR (handy for citations)
    for d in docs:
        src = d.metadata.get("source")
        try:
            d.metadata["source"] = str(
                Path(src).resolve().relative_to(path.resolve()))
        except Exception:
            # if not under DATA_DIR, keep whatever the loader set
            d.metadata["source"] = src
    return docs


# -------------------- Optional Graph (Neo4j) utils --------------------

def _graph_enabled() -> bool:
    return os.getenv("GRAPH_ENABLED", "false").lower() in ("1", "true", "yes")


def _get_graph_driver():
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "GRAPH_ENABLED is true but neo4j driver is not installed. Run: pip install neo4j"
        ) from e

    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j_password")
    return GraphDatabase.driver(uri, auth=(user, password))


def _build_graph_for_chunks(chunks: List):
    """Create Doc/Chunk nodes and CONTAINS/NEXT relationships for navigation."""
    driver = _get_graph_driver()

    # Group chunks by source and ensure chunk_index exists
    by_source = {}
    for ch in chunks:
        src = ch.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(ch)

    with driver.session() as session:
        for source, docs in by_source.items():
            # Create Doc node
            session.run("""
                MERGE (d:Doc {id: $doc_id})
                ON CREATE SET d.source = $source
                ON MATCH SET d.source = $source
            """, doc_id=source, source=source)

            prev_chunk_id = None
            for d in docs:
                chunk_index = int(d.metadata.get("chunk_index", 0))
                chunk_id = d.metadata.get(
                    "chunk_id") or f"{source}:::{chunk_index}"
                content = d.page_content

                session.run("""
                    MERGE (c:Chunk {id: $chunk_id})
                    ON CREATE SET c.source = $source, c.index = $index, c.content = $content
                    ON MATCH SET c.source = $source, c.index = $index, c.content = $content
                """, chunk_id=chunk_id, source=source, index=chunk_index, content=content)

                session.run("""
                    MATCH (d:Doc {id: $doc_id}), (c:Chunk {id: $chunk_id})
                    MERGE (d)-[:CONTAINS]->(c)
                """, doc_id=source, chunk_id=chunk_id)

                if prev_chunk_id is not None:
                    session.run("""
                        MATCH (p:Chunk {id: $prev_id}), (c:Chunk {id: $curr_id})
                        MERGE (p)-[:NEXT]->(c)
                    """, prev_id=prev_chunk_id, curr_id=chunk_id)
                prev_chunk_id = chunk_id


def _stable_chunk_id(source: str, content: str) -> str:
    import uuid

    name = f"{source}\n{content}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def main():
    docs = load_docs(DATA_DIR)
    if not docs:
        raise SystemExit("No documents found in ./data")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    # Assign a stable chunk_index per source and compute chunk_id
    by_source = {}
    for ch in chunks:
        src = ch.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(ch)
    for source, ds in by_source.items():
        for idx, d in enumerate(ds):
            d.metadata["chunk_index"] = idx
            d.metadata["chunk_id"] = _stable_chunk_id(source, d.page_content)

    embeddings = LMStudioEmbeddings(
        base_url=os.getenv("LMSTUDIO_BASE", "http://localhost:1234/v1"),
        api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        model=os.getenv("LMSTUDIO_EMBED_MODEL",
                        "nomic-ai/nomic-embed-text-v1.5"),
    )

    # Open or create Qdrant vectorstore and perform idempotent add via delete+add
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    # Ensure collection exists with proper vector size
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION)
    except Exception:
        probe_dim = len(embeddings.embed_query("dimension probe"))
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=probe_dim, distance=Distance.COSINE),
        )

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )

    # Build deterministic ids list (UUID strings)
    ids = [d.metadata["chunk_id"] for d in chunks]

    # Optional replacement strategy per source
    replace_by_source = os.getenv(
        "CHROMA_REPLACE_BY_SOURCE", "false").lower() in ("1", "true", "yes")
    if replace_by_source:
        for source in by_source.keys():
            try:
                vectorstore.delete(where={"source": source})
            except Exception:
                pass
    else:
        # Ensure idempotency on re-runs: delete by ids first (no-op if not present)
        try:
            vectorstore.delete(ids=ids)
        except Exception:
            pass

    vectorstore.add_documents(documents=chunks, ids=ids)

    # Optionally mirror structure into a graph database for structural navigation
    if _graph_enabled():
        _build_graph_for_chunks(chunks)

    print(
        f"Ingested {len(chunks)} chunks into Qdrant collection '{QDRANT_COLLECTION}'")


if __name__ == "__main__":
    main()
