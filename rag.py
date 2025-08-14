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

# Optional reranker
try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
RAG_LOG_PATH = os.getenv("RAG_LOG_PATH", "logs/rag.log")
RAG_INDEX_VERSION = os.getenv("RAG_INDEX_VERSION", "v1")
# e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "")

# Prompt/config (env-overridable)
PROMPT_SYSTEM_GENERIC = os.getenv(
    "PROMPT_SYSTEM_GENERIC",
    "You are a precise assistant. Answer using ONLY the provided context. If the answer isn't in the context, say you don't know.",
)
# Domain → system-prompt JSON mapping (optional). If not set, uses only default prompt.
PROMPT_SYSTEM_BY_DOMAIN_RAW = os.getenv("PROMPT_SYSTEM_BY_DOMAIN", "")

VERIFY_PROMPT_SYSTEM = os.getenv(
    "VERIFY_PROMPT_SYSTEM",
    "You are a strict verifier. Extract claims. If any claim is not explicitly supported by the context, grounded=false with reason. Return JSON only.",
)
VERIFY_PROMPT_HUMAN = os.getenv(
    "VERIFY_PROMPT_HUMAN",
    "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}\n\nReturn JSON with fields: grounded (true/false), reason (string).",
)

# Router thresholds and behavior (env-overridable)
TOPSCORE_THRESHOLD = float(os.getenv("ROUTER_TOPSCORE_THRESHOLD", "0.35"))
MARGIN_THRESHOLD = float(os.getenv("ROUTER_MARGIN_THRESHOLD", "0.10"))
CONFIDENCE_HIGH = float(os.getenv("ROUTER_CONF_HIGH", "0.70"))
CONFIDENCE_MED = float(os.getenv("ROUTER_CONF_MED", "0.40"))
K_HIGH = int(os.getenv("RETRIEVAL_K_HIGH", "6"))
K_MED = int(os.getenv("RETRIEVAL_K_MED", "12"))
K_LOW = int(os.getenv("RETRIEVAL_K_LOW", "20"))
# Continuity keywords (empty by default; fully configurable)
CONTINUITY_KEYWORDS = [
    k.strip().lower()
    for k in os.getenv("CONTINUITY_KEYWORDS", "").split(",")
    if k.strip()
]

# Domain detection keywords (env JSON mapping). Empty by default.
DOMAIN_KEYWORDS_RAW = os.getenv("DOMAIN_KEYWORDS", "{}")

# Recovery synonyms (comma-separated). Empty by default.
RECOVERY_SYNONYMS = [s.strip() for s in os.getenv(
    "RECOVERY_SYNONYMS", "").split(",") if s.strip()]


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
        pass


def _hash_question(q: str) -> str:
    return hashlib.sha1(q.encode("utf-8")).hexdigest()


def _doc_id_from_source(source: str) -> str:
    return hashlib.sha1((source or "").encode("utf-8")).hexdigest()


def _parse_json_env(raw: str, fallback):
    if not raw:
        return fallback
    try:
        val = json.loads(raw)
        return val if isinstance(val, (dict, list)) else fallback
    except Exception:
        return fallback


def _load_prompts_by_domain():
    mapping = _parse_json_env(PROMPT_SYSTEM_BY_DOMAIN_RAW, {
                              "default": PROMPT_SYSTEM_GENERIC})
    if "default" not in mapping:
        mapping["default"] = PROMPT_SYSTEM_GENERIC
    return mapping


PROMPTS_BY_DOMAIN = _load_prompts_by_domain()
DOMAIN_KEYWORDS = _parse_json_env(DOMAIN_KEYWORDS_RAW, {})


def _detect_lang_basic(text: str) -> str:
    # Try langdetect if available
    try:
        from langdetect import detect  # type: ignore
        return detect(text)
    except Exception:
        lowered = text.lower()
        if any(tok in lowered for tok in ["¿", "¡"]):
            return "es"
        return "unknown"


def _classify_intent_basic(text: str) -> str:
    lowered = text.lower().strip()
    chit = ["hola", "hi", "hello", "gracias", "thanks", "ok", "vale"]
    if any(lowered.startswith(x) or x in lowered for x in chit):
        return "chitchat"
    return "task"


def _detect_domain(text: str) -> str:
    t = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in (keywords or []):
            if kw and kw.lower() in t:
                return domain
    return "default"


class LMStudioEmbeddings(Embeddings):
    """OpenAI-compatible embeddings for LM Studio. Sends 1 text per request."""

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

    ids = []
    for d in initial_docs:
        cid = d.metadata.get("chunk_id")
        if cid:
            ids.append(cid)

    if not ids:
        return []

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


# -------- Baseline chain (kept for compatibility) --------

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
        ("system", PROMPT_SYSTEM_GENERIC),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ])

    def retrieve_with_graph_formatted(question: str) -> str:
        base_docs = retriever.get_relevant_documents(question)
        neighbor_ids = _expand_with_graph_context(base_docs)
        extra_docs = []
        for cid in neighbor_ids:
            try:
                extra_docs.extend(vectorstore.similarity_search(
                    "", k=1, filter={"chunk_id": cid}))
            except Exception:
                pass
        seen = set()
        merged = []
        for d in base_docs + extra_docs:
            key = d.metadata.get("chunk_id") or (
                d.metadata.get("source"), d.metadata.get("chunk_index"))
            if key not in seen:
                seen.add(key)
                merged.append(d)
        return format_docs(merged[:8])

    rag = (
        {
            "context": (retrieve_with_graph_formatted if _graph_enabled() else (retriever | format_docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag


# -------- Multi-agent pipeline --------

class AgenticRAG:
    def __init__(self, vectorstore: QdrantVectorStore, embeddings: Embeddings, llm: ChatOpenAI):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.llm = llm
        self.graph_on = _graph_enabled()
        self.reranker = None
        if RERANKER_MODEL and CrossEncoder is not None:
            try:
                self.reranker = CrossEncoder(RERANKER_MODEL)
            except Exception:
                self.reranker = None

    def _common_meta(self, question: str) -> dict:
        return {
            "trace_id": str(uuid.uuid4()),
            "question_hash": _hash_question(question),
            "question_meta": {
                "lang": _detect_lang_basic(question),
                "len": len(question),
                "intent": _classify_intent_basic(question),
                "domain": _detect_domain(question),
            },
            "index_version": RAG_INDEX_VERSION,
            "model_version": {
                "embedder": os.getenv("LMSTUDIO_EMBED_MODEL", "unknown"),
                "llm": os.getenv("LMSTUDIO_CHAT_MODEL", "unknown"),
                "reranker": RERANKER_MODEL or "",
            },
        }

    # Agent 1: Router
    def router(self, question: str, meta: dict) -> dict:
        decision = {"need": True, "use_graph": False,
                    "k": K_HIGH, "reason": "default"}
        ql = question.lower()
        if _classify_intent_basic(ql) == "chitchat":
            decision.update({"need": False, "reason": "chitchat"})
        try:
            with_scores = self.vectorstore.similarity_search_with_score(
                question, k=2)
            if with_scores:
                top = with_scores[0][1]
                second = with_scores[1][1] if len(with_scores) > 1 else 0.0
                margin = top - second
                confidence = max(
                    0.0, min(1.0, 0.5 * top + 0.5 * (margin / max(1e-6, MARGIN_THRESHOLD * 2))))
                bucket = "high" if confidence >= CONFIDENCE_HIGH else (
                    "medium" if confidence >= CONFIDENCE_MED else "low")
                k = K_HIGH if bucket == "high" else (
                    K_MED if bucket == "medium" else K_LOW)
                need = (top < TOPSCORE_THRESHOLD) or (
                    margin < MARGIN_THRESHOLD) or bucket != "high"
                decision.update({
                    "need": need,
                    "reason": f"scores(top={top:.3f},margin={margin:.3f})",
                    "k": k,
                    "retrieval_confidence": confidence,
                    "retrieval_confidence_bucket": bucket,
                })
        except Exception:
            decision.update({"need": True, "reason": "retrieval_error"})
        if any(x in ql for x in CONTINUITY_KEYWORDS):
            decision["use_graph"] = True
        _log_event("route", {**meta, "stage": "route", **decision})
        return decision

    # Agent 2: Retriever
    def retrieve(self, question: str, decision: dict, meta: dict) -> List:
        t0 = time.perf_counter()
        try:
            _ = self.embeddings.embed_query(question)
        finally:
            t1 = time.perf_counter()
        embed_ms = int((t1 - t0) * 1000)

        t2 = time.perf_counter()
        with_scores = self.vectorstore.similarity_search_with_score(
            question, k=decision.get("k", K_HIGH))
        t3 = time.perf_counter()
        retrieve_ms = int((t3 - t2) * 1000)

        base_docs = [d for d, _ in with_scores]
        extra_docs = []
        if decision.get("use_graph") and self.graph_on:
            neighbor_ids = _expand_with_graph_context(base_docs)
            t4 = time.perf_counter()
            for cid in neighbor_ids:
                try:
                    extra_docs.extend(self.vectorstore.similarity_search(
                        "", k=1, filter={"chunk_id": cid}))
                except Exception:
                    pass
            t5 = time.perf_counter()
            rerank_ms = int((t5 - t4) * 1000)
        else:
            rerank_ms = 0

        # Rerank (optional cross-encoder)
        ranked_pairs = list(with_scores)
        if self.reranker is not None:
            try:
                pairs = [(question, d.page_content) for d, _ in with_scores]
                scores = self.reranker.predict(pairs)
                ranked_pairs = sorted(
                    zip([d for d, _ in with_scores], scores), key=lambda x: x[1], reverse=True)
                ranked_pairs = [(d, float(s)) for d, s in ranked_pairs]
            except Exception:
                pass

        seen = set()
        merged = []
        ranked = []
        for rank, (doc, score) in enumerate(ranked_pairs, start=1):
            key = doc.metadata.get("chunk_id") or (
                doc.metadata.get("source"), doc.metadata.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            ranked.append((doc, score, rank))
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

    # Agent 3a: Answerer
    def answer(self, question: str, docs: List, meta: dict) -> str:
        domain = meta.get("question_meta", {}).get("domain", "default")
        system = PROMPTS_BY_DOMAIN.get(domain) or PROMPTS_BY_DOMAIN.get(
            "default", PROMPT_SYSTEM_GENERIC)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Question:\n{question}\n\nContext:\n{context}"),
        ])
        context = format_docs(docs)
        t0 = time.perf_counter()
        out = (prompt | self.llm | StrOutputParser()).invoke(
            {"question": question, "context": context})
        t1 = time.perf_counter()
        _log_event("answer", {**meta, "stage": "answer",
                   "latency_ms": int((t1 - t0) * 1000)})
        return out

    # Agent 3b: Verifier
    def verify(self, question: str, answer: str, docs: List, meta: dict) -> dict:
        verify_prompt = ChatPromptTemplate.from_messages([
            ("system", VERIFY_PROMPT_SYSTEM),
            ("human", VERIFY_PROMPT_HUMAN),
        ])
        context = format_docs(docs)
        t0 = time.perf_counter()
        resp = (verify_prompt | self.llm | StrOutputParser()).invoke(
            {"question": question, "answer": answer, "context": context})
        t1 = time.perf_counter()
        latency = int((t1 - t0) * 1000)
        result = {"grounded": None, "reason": ""}
        try:
            parsed = json.loads(resp)
            result["grounded"] = bool(parsed.get("grounded"))
            result["reason"] = str(parsed.get("reason", ""))
        except Exception:
            result = {"grounded": None, "reason": resp[:500]}
        _log_event("verify", {**meta, "stage": "verify",
                   "latency_ms": latency, **result})
        return result

    def invoke(self, question: str) -> str:
        meta = self._common_meta(question)
        decision = self.router(question, meta)
        if not decision.get("need"):
            ans = self.answer(question, [], meta)
            ver = self.verify(question, ans, [], meta)
            if ver.get("grounded") is False:
                safe = (
                    "I don't have sufficient grounded context to provide exact commands. "
                    "Would you like me to expand the search or outline high-level steps?"
                )
                _log_event("citation_gate", {
                           **meta, "required": True, "passed": False})
                return safe
            _log_event("citation_gate", {
                       **meta, "required": True, "passed": True})
            return ans

        docs = self.retrieve(question, decision, meta)
        ans = self.answer(question, docs, meta)
        ver = self.verify(question, ans, docs, meta)
        if ver.get("grounded") is False:
            recovery = {"attempts": 0, "action": "", "outcome": ""}
            try:
                recovery["attempts"] = 1
                if RECOVERY_SYNONYMS:
                    q2 = question + " | " + " | ".join(RECOVERY_SYNONYMS)
                else:
                    q2 = question
                more_pairs = self.vectorstore.similarity_search_with_score(
                    q2, k=max(K_LOW, decision.get("k", K_HIGH) * 2))
                more_docs = [d for d, _ in more_pairs]
                docs2 = self._dedupe_docs(docs + more_docs)
                ans2 = self.answer(question, docs2, meta)
                ver2 = self.verify(question, ans2, docs2, meta)
                if ver2.get("grounded"):
                    recovery["action"] = "expand_k+rerank+query_rewrite"
                    recovery["outcome"] = "fixed"
                    _log_event("recovery", {**meta, **recovery})
                    _log_event("citation_gate", {
                               **meta, "required": True, "passed": True})
                    return ans2
                else:
                    recovery["action"] = "expand_k+rerank+query_rewrite"
                    recovery["outcome"] = "degraded"
                    _log_event("recovery", {**meta, **recovery})
            except Exception:
                recovery["outcome"] = "aborted"
                _log_event("recovery", {**meta, **recovery})
            _log_event("citation_gate", {
                       **meta, "required": True, "passed": False})
            return (
                "I couldn't find grounded evidence for exact commands in the current context. "
                "Here are high-level steps you can follow. If you want, I can expand the search to related notes."
            )
        _log_event("citation_gate", {**meta, "required": True, "passed": True})
        return ans

    def _dedupe_docs(self, docs: List) -> List:
        seen = set()
        out = []
        for d in docs:
            key = d.metadata.get("chunk_id") or (
                d.metadata.get("source"), d.metadata.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out


def build_agentic():
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
    llm = ChatOpenAI(
        base_url=os.getenv("LMSTUDIO_BASE", "http://localhost:1234/v1"),
        api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        model=os.getenv("LMSTUDIO_CHAT_MODEL", "gpt-oss-20b"),
        temperature=0.1,
    )
    return AgenticRAG(vectorstore, embeddings, llm)


if __name__ == "__main__":
    # Demo run with agentic pipeline
    chain = build_agentic()
    print(chain.invoke("Hello"))
