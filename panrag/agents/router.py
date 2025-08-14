import time
from typing import Dict

from .. import config
from ..logging_utils import log_event

try:
    from langdetect import detect  # type: ignore
except Exception:
    detect = None


class RouterAgent:
    def decide(self, question: str, vectorstore, meta: Dict) -> Dict:
        decision = {"need": True, "use_graph": False,
                    "k": config.RETRIEVAL_K_HIGH, "reason": "default"}
        ql = question.lower()
        if any(x in ql for x in ["hola", "hi", "hello", "gracias", "thanks", "ok", "vale"]):
            decision.update({"need": False, "reason": "chitchat"})
        try:
            with_scores = vectorstore.similarity_search_with_score(
                question, k=2)
            if with_scores:
                top = with_scores[0][1]
                second = with_scores[1][1] if len(with_scores) > 1 else 0.0
                margin = top - second
                confidence = max(0.0, min(
                    1.0, 0.5 * top + 0.5 * (margin / max(1e-6, config.ROUTER_MARGIN_THRESHOLD * 2))))
                bucket = "high" if confidence >= config.ROUTER_CONF_HIGH else (
                    "medium" if confidence >= config.ROUTER_CONF_MED else "low")
                k = config.RETRIEVAL_K_HIGH if bucket == "high" else (
                    config.RETRIEVAL_K_MED if bucket == "medium" else config.RETRIEVAL_K_LOW)
                need = (top < config.ROUTER_TOPSCORE_THRESHOLD) or (
                    margin < config.ROUTER_MARGIN_THRESHOLD) or bucket != "high"
                decision.update({
                    "need": need,
                    "reason": f"scores(top={top:.3f},margin={margin:.3f})",
                    "k": k,
                    "retrieval_confidence": confidence,
                    "retrieval_confidence_bucket": bucket,
                })
        except Exception:
            decision.update({"need": True, "reason": "retrieval_error"})

        # Language-aware adjustment
        lang_detected = None
        if config.ROUTER_LANG_DETECT_ENABLED:
            try:
                if detect is not None:
                    lang_detected = (detect(question) or "").lower()
            except Exception:
                lang_detected = None
            if config.ROUTER_LANG_ALLOW:
                if not lang_detected or lang_detected not in config.ROUTER_LANG_ALLOW:
                    decision["k"] = max(decision.get(
                        "k", config.RETRIEVAL_K_HIGH), config.ROUTER_LANG_MISMATCH_K)
                    decision["reason"] = (decision.get(
                        "reason", "") + "; lang_mismatch").strip("; ")

        if any(x in ql for x in config.CONTINUITY_KEYWORDS):
            decision["use_graph"] = True
        log_event("route", {**meta, "stage": "route", **decision,
                  "lang": lang_detected, "lang_allow": config.ROUTER_LANG_ALLOW})
        return decision
