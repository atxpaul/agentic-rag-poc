import time
from typing import List, Tuple, Any

from .. import config
from ..logging_utils import log_event
from ..graph_utils import graph_enabled, expand_neighbors


class RetrieverAgent:
    def __init__(self, vectorstore, reranker=None):
        self.vectorstore = vectorstore
        self.reranker = reranker

    def run(self, question: str, decision: dict, meta: dict) -> List:
        t2 = time.perf_counter()
        with_scores: List[Tuple[Any, float]] = self.vectorstore.similarity_search_with_score(
            question, k=decision.get("k", config.RETRIEVAL_K_HIGH)
        )
        t3 = time.perf_counter()
        retrieve_ms = int((t3 - t2) * 1000)

        base_docs = [d for d, _ in with_scores]
        extra_docs = []
        if decision.get("use_graph") and graph_enabled():
            t4 = time.perf_counter()
            neighbor_ids = expand_neighbors(base_docs)
            for cid in neighbor_ids:
                try:
                    extra_docs.extend(self.vectorstore.similarity_search(
                        "", k=1, filter={"chunk_id": cid}))
                except Exception:
                    pass
            t5 = time.perf_counter()
            graph_ms = int((t5 - t4) * 1000)
        else:
            graph_ms = 0

        # Conditional reranker
        rerank_ms = 0
        ranked_pairs = list(with_scores)
        if config.RERANKER_ENABLE_BY_CONF and self.reranker is not None and with_scores:
            top = with_scores[0][1]
            second = with_scores[1][1] if len(with_scores) > 1 else 0.0
            margin = top - second
            confidence = max(0.0, min(
                1.0, 0.5 * top + 0.5 * (margin / max(1e-6, config.ROUTER_MARGIN_THRESHOLD * 2))))
            should = (confidence < config.RERANKER_CONF_THRESHOLD) or (
                top < config.RERANKER_TOP_THRESHOLD) or (margin < config.RERANKER_MARGIN_THRESHOLD)
            if should:
                t6 = time.perf_counter()
                try:
                    pairs = [(question, d.page_content)
                             for d, _ in with_scores]
                    scores = self.reranker.predict(pairs)
                    ranked_pairs = sorted(
                        zip([d for d, _ in with_scores], scores), key=lambda x: x[1], reverse=True)
                    ranked_pairs = [(d, float(s)) for d, s in ranked_pairs]
                finally:
                    t7 = time.perf_counter()
                    rerank_ms = int((t7 - t6) * 1000)

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

        log_event("retrieval", {
            **meta,
            "stage": "retrieval",
            "latency_ms": {"retrieve": retrieve_ms, "graph": graph_ms, "rerank": rerank_ms},
            "reranker": config.RERANKER_MODEL if rerank_ms > 0 else "",
            "selected": [
                {
                    "doc_id": (d.metadata.get("source") or ""),
                    "source": d.metadata.get("source"),
                    "chunk_index": d.metadata.get("chunk_index"),
                    "score": score,
                    "rank": rnk,
                }
                for (d, score, rnk) in ranked
            ],
        })
        return merged[:8]
