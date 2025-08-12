import os
from typing import List

import requests
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

DB_DIR = "chroma-db"


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

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        collection_name="docs",
        embedding_function=embeddings,
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

    def retrieve_with_graph(question: str):
        base_docs = retriever.get_relevant_documents(question)
        neighbor_ids = _expand_with_graph_context(base_docs)

        extra_docs = []
        for cid in neighbor_ids:
            # Pull neighbors by metadata filter. Use empty query + filter
            try:
                extra_docs.extend(
                    vectorstore.similarity_search(
                        "", k=1, filter={"chunk_id": cid}
                    )
                )
            except Exception:
                pass

        # Deduplicate by chunk_id
        seen = set()
        merged = []
        for d in base_docs + extra_docs:
            key = d.metadata.get("chunk_id") or (
                d.metadata.get("source"), d.metadata.get("chunk_index"))
            if key not in seen:
                seen.add(key)
                merged.append(d)
        return merged[:8]

    def retrieve_with_graph_formatted(question: str) -> str:
        return format_docs(retrieve_with_graph(question))

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


if __name__ == "__main__":
    chain = build_chain()
    print(chain.invoke("Cómo añado un certificado SSL en k3s?"))
