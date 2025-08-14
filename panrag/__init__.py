from .agent import AgenticRAG
from .embeddings import LMStudioEmbeddings
from . import config

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI


def build_agentic():
    embeddings = LMStudioEmbeddings(
        base_url=config.LMSTUDIO_BASE,
        api_key=config.LMSTUDIO_API_KEY,
        model=config.LMSTUDIO_EMBED_MODEL,
    )
    qdrant_client = QdrantClient(
        url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=config.QDRANT_COLLECTION,
        embedding=embeddings,
    )
    llm = ChatOpenAI(
        base_url=config.LMSTUDIO_BASE,
        api_key=config.LMSTUDIO_API_KEY,
        model=config.LMSTUDIO_CHAT_MODEL,
        temperature=0.1,
    )
    return AgenticRAG(vectorstore, embeddings, llm)
