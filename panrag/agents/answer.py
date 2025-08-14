import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..prompts import select_system_prompt
from ..logging_utils import log_event
from .. import config


class AnswerAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, question: str, docs: list, meta: dict) -> str:
        intent = meta.get("question_meta", {}).get("intent", "task")
        domain = meta.get("question_meta", {}).get("domain", "default")
        system = select_system_prompt(domain)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Question:\n{question}\n\nContext:\n{context}"),
        ])
        context = "\n\n".join(
            f"[{i+1}] {d.page_content}\nSOURCE: {d.metadata.get('source')}" for i, d in enumerate(docs)
        )
        max_tokens = config.ANSWER_MAX_TOKENS_CHITCHAT if intent == "chitchat" else config.ANSWER_MAX_TOKENS_TASK
        temperature = config.ANSWER_TEMPERATURE_CHITCHAT if intent == "chitchat" else config.ANSWER_TEMPERATURE_TASK
        gen_kwargs = {"max_tokens": max_tokens, "temperature": temperature}
        if config.ANSWER_STOP_SEQUENCES:
            gen_kwargs["stop"] = config.ANSWER_STOP_SEQUENCES
        t0 = time.perf_counter()
        out = (prompt | self.llm.bind(**gen_kwargs) | StrOutputParser()
               ).invoke({"question": question, "context": context})
        t1 = time.perf_counter()
        log_event("answer", {**meta, "stage": "answer",
                  "latency_ms": int((t1 - t0) * 1000), "gen": gen_kwargs})
        return out
