import os
from typing import List

import requests
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
import logging
import json
from datetime import datetime
from pathlib import Path
import time
import uuid
import hashlib

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
RAG_LOG_PATH = os.getenv("RAG_LOG_PATH", "logs/rag.log")
RAG_INDEX_VERSION = os.getenv("RAG_INDEX_VERSION", "v1")


# -------- Logging setup --------
def _setup_logger():
    logger = logging.getLogger("rag")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_path = Path(RAG_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


_LOGGER = _setup_logger()


def _log_event(event: str, data: dict):
    payload = {"ts": datetime.utcnow().isoformat() + "Z", "event": event}
    payload.update(data)
    try:
        _LOGGER.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        # Evitar que logging rompa el flujo de RAG
        pass


def _hash_question(q: str) -> str:
    return hashlib.sha1(q.encode("utf-8")).hexdigest()


def _doc_id_from_source(source: str) -> str:
    return hashlib.sha1((source or "").encode("utf-8")).hexdigest()


def _detect_lang_basic(text: str) -> str:
    lowered = text.lower()
    spanish_markers = ["¿", "¡", "cómo", "qué", "cuál",
                       "dónde", "por qué", "porque", "para", "con"]
    if any(tok in lowered for tok in spanish_markers):
        return "es"
    return "unknown"


def _classify_intent_basic(text: str) -> str:
    lowered = text.lower().strip()
    chit = ["hola", "qué tal", "como estás", "gracias", "ok", "vale"]
    if any(lowered.startswith(x) or x in lowered for x in chit):
        return "chitchat"
    return "task"


class LMStudioEmbeddings(Embeddings):
    """OpenAI-compatible embeddings for LM Studio. Envía 1 texto por request."""

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


def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] {d.page_content}\nSOURCE: {d.metadata.get('source')}"
        for i, d in enumerate(docs)
    )


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


def _expand_with_graph_context(initial_docs: List):
    if not _graph_enabled() or not initial_docs:
        return []

    driver = _get_graph_driver()

    # Gather chunk_ids and fetch immediate NEXT neighbors
    ids = []
    for d in initial_docs:
        cid = d.metadata.get("chunk_id")
        if cid:
            ids.append(cid)

    if not ids:
        return []

    # Query neighbors and return candidate chunk_ids
    neighbors: List[str] = []
    with driver.session() as session:
        for cid in ids:
            result = session.run(
                """
                MATCH (c:Chunk {id: $cid})
                OPTIONAL MATCH (c)-[:NEXT]->(n)
                OPTIONAL MATCH (p)-[:NEXT]->(c)
                RETURN collect(distinct n.id) + collect(distinct p.id) AS ids
                """,
                cid=cid,
            )
            row = result.single()
            if not row:
                continue
            for nid in (row["ids"] or []):
                if nid:
                    neighbors.append(nid)

    return neighbors


def build_chain():
    embeddings = LMStudioEmbeddings(
        base_url=os.getenv("LMSTUDIO_BASE", "http://localhost:1234/v1"),
        api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        model=os.getenv("LMSTUDIO_EMBED_MODEL",
                        "nomic-ai/nomic-embed-text-v1.5"),
    )

    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        base_url=os.getenv("LMSTUDIO_BASE", "http://localhost:1234/v1"),
        api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        model=os.getenv("LMSTUDIO_CHAT_MODEL", "gpt-oss-20b"),
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise assistant. Answer the user's question using ONLY the context. "
         "If the answer isn't in the context, say you don't know."),
        ("human", "Question:\n{question}\n\nContext:\n{context}")
    ])

    def _common_meta(question: str):
        return {
            "trace_id": str(uuid.uuid4()),
            "question_hash": _hash_question(question),
            "question_meta": {
                "lang": _detect_lang_basic(question),
                "len": len(question),
                "intent": _classify_intent_basic(question),
            },
            "index_version": RAG_INDEX_VERSION,
            "model_version": {
                "embedder": os.getenv("LMSTUDIO_EMBED_MODEL", "unknown"),
                "llm": os.getenv("LMSTUDIO_CHAT_MODEL", "unknown"),
            },
        }

    def retrieve_with_graph(question: str):
        meta = _common_meta(question)

        # Time embedding (probe only)
        t0 = time.perf_counter()
        try:
            _ = embeddings.embed_query(question)
        finally:
            t1 = time.perf_counter()
        embed_ms = int((t1 - t0) * 1000)

        # Retrieve base with scores
        t2 = time.perf_counter()
        base_with_scores = vectorstore.similarity_search_with_score(
            question, k=4)
        t3 = time.perf_counter()
        retrieve_ms = int((t3 - t2) * 1000)

        base_docs = [d for d, _ in base_with_scores]
        neighbor_ids = _expand_with_graph_context(base_docs)

        # Neighbor fetch (no scores)
        t4 = time.perf_counter()
        extra_docs = []
        for cid in neighbor_ids:
            try:
                extra_docs.extend(vectorstore.similarity_search(
                    "", k=1, filter={"chunk_id": cid}))
            except Exception:
                pass
        t5 = time.perf_counter()
        rerank_ms = int((t5 - t4) * 1000)

        # Merge and dedup
        seen = set()
        merged = []
        ranked = []
        # First, base with their scores
        for rank, (doc, score) in enumerate(base_with_scores, start=1):
            key = doc.metadata.get("chunk_id") or (
                doc.metadata.get("source"), doc.metadata.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            ranked.append((doc, score, rank))
        # Then neighbors without score
        for doc in extra_docs:
            key = doc.metadata.get("chunk_id") or (
                doc.metadata.get("source"), doc.metadata.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            ranked.append((doc, None, len(ranked) + 1))

        _log_event("retrieval", {
            **meta,
            "stage": "retrieval",
            "latency_ms": {"embed": embed_ms, "retrieve": retrieve_ms, "rerank": rerank_ms},
            "selected": [
                {
                    "doc_id": _doc_id_from_source(d.metadata.get("source")),
                    "source": d.metadata.get("source"),
                    "chunk_index": d.metadata.get("chunk_index"),
                    "score": score,
                    "rank": rnk,
                }
                for (d, score, rnk) in ranked
            ],
        })
        return merged[:8]

    def retrieve_with_graph_formatted(question: str) -> str:
        return format_docs(retrieve_with_graph(question))

    def retrieve_standard_formatted(question: str) -> str:
        meta = _common_meta(question)

        t0 = time.perf_counter()
        try:
            _ = embeddings.embed_query(question)
        finally:
            t1 = time.perf_counter()
        embed_ms = int((t1 - t0) * 1000)

        t2 = time.perf_counter()
        with_scores = vectorstore.similarity_search_with_score(question, k=4)
        t3 = time.perf_counter()
        retrieve_ms = int((t3 - t2) * 1000)

        docs = [d for d, _ in with_scores]

        _log_event("retrieval", {
            **meta,
            "stage": "retrieval",
            "latency_ms": {"embed": embed_ms, "retrieve": retrieve_ms, "rerank": 0},
            "selected": [
                {
                    "doc_id": _doc_id_from_source(d.metadata.get("source")),
                    "source": d.metadata.get("source"),
                    "chunk_index": d.metadata.get("chunk_index"),
                    "score": score,
                    "rank": i + 1,
                }
                for i, (d, score) in enumerate(with_scores)
            ],
        })
        return format_docs(docs)

    rag = (
        {
            "context": (retrieve_with_graph_formatted if _graph_enabled() else retrieve_standard_formatted),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag


if __name__ == "__main__":
    chain = build_chain()
    print(chain.invoke("Cómo puedo añadir una unidad externa USB?"))
