import time
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..prompts import verifier_prompts
from ..logging_utils import log_event
from .. import config


class VerifierAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, question: str, answer: str, docs: list, meta: dict) -> dict:
        sys_prompt, human_prompt = verifier_prompts()
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", human_prompt),
        ])
        context = "\n\n".join(
            f"[{i+1}] {d.page_content}\nSOURCE: {d.metadata.get('source')}" for i, d in enumerate(docs)
        )
        t0 = time.perf_counter()
        resp = (prompt | self.llm | StrOutputParser()).invoke(
            {"question": question, "answer": answer, "context": context})
        t1 = time.perf_counter()
        latency = int((t1 - t0) * 1000)
        result = {"grounded": None, "reason": "", "claims_total": 0,
                  "claims_supported": 0, "citations": []}
        try:
            parsed = json.loads(resp)
            result["grounded"] = bool(parsed.get("grounded"))
            result["reason"] = str(parsed.get("reason", ""))
            result["claims_total"] = int(parsed.get("claims_total", 0))
            result["claims_supported"] = int(parsed.get("claims_supported", 0))
            result["citations"] = parsed.get("citations", []) or []
        except Exception:
            result["reason"] = resp[:500]
        # KPIs
        coverage = 0.0
        if result["claims_total"] > 0:
            coverage = result["claims_supported"] / \
                max(1, result["claims_total"])
        diversity = len({c.get("source")
                        for c in result["citations"] if c.get("source")})
        result["attribution_coverage"] = coverage
        result["evidence_diversity"] = diversity
        log_event("verify", {**meta, "stage": "verify",
                  "latency_ms": latency, **result})
        # Policy decision can be consumed by caller
        result["passes_policy"] = bool(result.get(
            "grounded") and coverage >= config.POLICY_CITATION_MIN_COVERAGE)
        return result
