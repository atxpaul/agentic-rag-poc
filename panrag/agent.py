import hashlib
import json
import time
import uuid
from typing import List

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI

from . import config
from .embeddings import LMStudioEmbeddings
from .logging_utils import log_event
from .agents import RouterAgent, RetrieverAgent, AnswerAgent, VerifierAgent

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None


def _hash_question(q: str) -> str:
    return hashlib.sha1(q.encode("utf-8")).hexdigest()


def _detect_lang_basic(text: str) -> str:
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
    if any(x in lowered for x in ["hola", "hi", "hello", "gracias", "thanks", "ok", "vale"]):
        return "chitchat"
    return "task"


class AgenticRAG:
    def __init__(self, vectorstore: QdrantVectorStore, embeddings: LMStudioEmbeddings, llm: ChatOpenAI):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.llm = llm
        self.reranker = None
        if config.RERANKER_MODEL and CrossEncoder is not None:
            try:
                self.reranker = CrossEncoder(config.RERANKER_MODEL)
            except Exception:
                self.reranker = None

        self.router = RouterAgent()
        self.retriever = RetrieverAgent(vectorstore, reranker=self.reranker)
        self.answerer = AnswerAgent(llm)
        self.verifier = VerifierAgent(llm)

    def _common_meta(self, question: str) -> dict:
        return {
            "trace_id": str(uuid.uuid4()),
            "question_hash": _hash_question(question),
            "question_meta": {
                "lang": _detect_lang_basic(question),
                "len": len(question),
                "intent": _classify_intent_basic(question),
            },
            "index_version": config.RAG_INDEX_VERSION,
            "model_version": {
                "embedder": config.LMSTUDIO_EMBED_MODEL,
                "llm": config.LMSTUDIO_CHAT_MODEL,
                "reranker": config.RERANKER_MODEL or "",
            },
        }

    def invoke(self, question: str) -> str:
        meta = self._common_meta(question)
        decision = self.router.decide(question, self.vectorstore, meta)
        if not decision.get("need"):
            ans = self.answerer.run(question, [], meta)
            ver = self.verifier.run(question, ans, [], meta)
            if ver.get("grounded") is False:
                safe = (
                    "I don't have sufficient grounded context to provide exact commands. "
                    "Would you like me to expand the search or outline high-level steps?"
                )
                log_event("citation_gate", {
                          **meta, "required": True, "passed": False})
                return safe
            log_event("citation_gate", {
                      **meta, "required": True, "passed": True})
            return ans

        docs = self.retriever.run(question, decision, meta)
        ans = self.answerer.run(question, docs, meta)
        ver = self.verifier.run(question, ans, docs, meta)
        if not ver.get("passes_policy"):
            recovery = {"attempts": 0, "action": "", "outcome": ""}
            try:
                recovery["attempts"] = 1
                q2 = question if not config.RECOVERY_SYNONYMS else question + \
                    " | " + " | ".join(config.RECOVERY_SYNONYMS)
                more_pairs = self.vectorstore.similarity_search_with_score(q2, k=max(
                    config.RETRIEVAL_K_LOW, decision.get("k", config.RETRIEVAL_K_HIGH) * 2))
                more_docs = [d for d, _ in more_pairs]
                docs2 = self._dedupe_docs(docs + more_docs)
                ans2 = self.answerer.run(question, docs2, meta)
                ver2 = self.verifier.run(question, ans2, docs2, meta)
                if ver2.get("passes_policy"):
                    log_event("policy", {**meta, "citation_gate": {"required": True,
                              "min_coverage": config.POLICY_CITATION_MIN_COVERAGE}, "passed": True})
                    return ans2
                else:
                    log_event("recovery", {
                              **meta, "attempts": recovery["attempts"], "action": "expand_k+rerank+query_rewrite", "outcome": "degraded"})
            except Exception:
                log_event(
                    "recovery", {**meta, "attempts": recovery["attempts"], "outcome": "aborted"})
            log_event("policy", {**meta, "citation_gate": {"required": True,
                      "min_coverage": config.POLICY_CITATION_MIN_COVERAGE}, "passed": False})
            return (
                "I couldn't find grounded evidence for exact commands in the current context. "
                "Here are high-level steps you can follow. If you want, I can expand the search to related notes."
            )
        log_event("policy", {**meta, "citation_gate": {"required": True,
                  "min_coverage": config.POLICY_CITATION_MIN_COVERAGE}, "passed": True})
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
