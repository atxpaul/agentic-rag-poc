import os
from dotenv import load_dotenv

load_dotenv()

# Endpoints / models
LMSTUDIO_BASE = os.getenv("LMSTUDIO_BASE", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LMSTUDIO_EMBED_MODEL = os.getenv(
    "LMSTUDIO_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
LMSTUDIO_CHAT_MODEL = os.getenv("LMSTUDIO_CHAT_MODEL", "gpt-oss-20b")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Graph
GRAPH_ENABLED = os.getenv(
    "GRAPH_ENABLED", "false").lower() in ("1", "true", "yes")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j_password")

# Logging / versioning
RAG_INDEX_VERSION = os.getenv("RAG_INDEX_VERSION", "v1")
RAG_LOG_PATH = os.getenv("RAG_LOG_PATH", "logs/rag.log")

# Prompts
PROMPT_SYSTEM_GENERIC = os.getenv(
    "PROMPT_SYSTEM_GENERIC",
    "You are a precise assistant. Answer using ONLY the provided context. If the answer isn't in the context, say you don't know.",
)
PROMPT_SYSTEM_BY_DOMAIN_RAW = os.getenv("PROMPT_SYSTEM_BY_DOMAIN", "")

# Optional suffix appended to the system prompt of the AnswerAgent, configurable via .env.
ANSWER_PROMPT_SUFFIX = os.getenv("ANSWER_PROMPT_SUFFIX", "")
VERIFY_PROMPT_SYSTEM = os.getenv(
    "VERIFY_PROMPT_SYSTEM",
    "You are a strict verifier. Extract claims, count them, and map citations. Return JSON with: grounded (true/false), reason (string), claims_total (int), claims_supported (int), citations (array of objects with claim_id, source, rank).",
)
VERIFY_PROMPT_HUMAN = os.getenv(
    "VERIFY_PROMPT_HUMAN",
    "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}\n\nReturn JSON with fields: grounded (true/false), reason (string), claims_total (int), claims_supported (int), citations (list).",
)

# Router / retrieval thresholds
ROUTER_TOPSCORE_THRESHOLD = float(
    os.getenv("ROUTER_TOPSCORE_THRESHOLD", "0.35"))
ROUTER_MARGIN_THRESHOLD = float(os.getenv("ROUTER_MARGIN_THRESHOLD", "0.10"))
ROUTER_CONF_HIGH = float(os.getenv("ROUTER_CONF_HIGH", "0.70"))
ROUTER_CONF_MED = float(os.getenv("ROUTER_CONF_MED", "0.40"))
RETRIEVAL_K_HIGH = int(os.getenv("RETRIEVAL_K_HIGH", "6"))
RETRIEVAL_K_MED = int(os.getenv("RETRIEVAL_K_MED", "12"))
RETRIEVAL_K_LOW = int(os.getenv("RETRIEVAL_K_LOW", "20"))

# Optional override for router to force a fixed k regardless of confidence logic.
RETRIEVAL_K_OVERRIDE = int(
    os.getenv("RETRIEVAL_K_OVERRIDE", "0"))  # 0 means disabled

# Language-aware routing
ROUTER_LANG_DETECT_ENABLED = os.getenv(
    "ROUTER_LANG_DETECT_ENABLED", "true").lower() in ("1", "true", "yes")
ROUTER_LANG_ALLOW = [x.strip().lower() for x in os.getenv(
    "ROUTER_LANG_ALLOW", "").split(",") if x.strip()]
ROUTER_LANG_MISMATCH_K = int(
    os.getenv("ROUTER_LANG_MISMATCH_K", str(RETRIEVAL_K_LOW)))

# Reranker policy
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "")
RERANKER_ENABLE_BY_CONF = os.getenv(
    "RERANKER_ENABLE_BY_CONF", "true").lower() in ("1", "true", "yes")
RERANKER_CONF_THRESHOLD = float(os.getenv("RERANKER_CONF_THRESHOLD", "0.50"))
RERANKER_TOP_THRESHOLD = float(os.getenv("RERANKER_TOP_THRESHOLD", "0.80"))
RERANKER_MARGIN_THRESHOLD = float(
    os.getenv("RERANKER_MARGIN_THRESHOLD", "0.08"))

# Continuity / recovery
CONTINUITY_KEYWORDS = [k.strip().lower() for k in os.getenv(
    "CONTINUITY_KEYWORDS", "").split(",") if k.strip()]
RECOVERY_SYNONYMS = [s.strip() for s in os.getenv(
    "RECOVERY_SYNONYMS", "").split(",") if s.strip()]

# Domain detection mapping
DOMAIN_KEYWORDS_RAW = os.getenv("DOMAIN_KEYWORDS", "{}")

# Answer generation controls
ANSWER_STREAMING_ENABLED = os.getenv(
    "ANSWER_STREAMING_ENABLED", "false").lower() in ("1", "true", "yes")
ANSWER_MAX_TOKENS_TASK = int(os.getenv("ANSWER_MAX_TOKENS_TASK", "512"))
ANSWER_MAX_TOKENS_CHITCHAT = int(
    os.getenv("ANSWER_MAX_TOKENS_CHITCHAT", "256"))
ANSWER_TEMPERATURE_TASK = float(os.getenv("ANSWER_TEMPERATURE_TASK", "0.1"))
ANSWER_TEMPERATURE_CHITCHAT = float(
    os.getenv("ANSWER_TEMPERATURE_CHITCHAT", "0.6"))
ANSWER_STOP_SEQUENCES = [s.strip() for s in os.getenv(
    "ANSWER_STOP_SEQUENCES", "").split(",") if s.strip()]

# Policy
POLICY_CITATION_MIN_COVERAGE = float(
    os.getenv("POLICY_CITATION_MIN_COVERAGE", "0.9"))
