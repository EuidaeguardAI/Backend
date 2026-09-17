import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TRANSCRIBE_MODEL = os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
_embedding_dimensions = os.environ.get("OPENAI_EMBEDDING_DIMENSIONS", "").strip()
EMBEDDING_DIMENSIONS = int(_embedding_dimensions) if _embedding_dimensions else None

# 기본값 "*": 로컬 LAN에서 폰 등 다른 기기로 접속할 때 오리진(IP)이 매번 달라질 수 있어
# 와일드카드로 둔다. 쿠키/인증정보를 쓰지 않으므로(allow_credentials=False) 스펙상 안전하다.
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

KNOWLEDGE_SOURCES_DIR = BASE_DIR / "knowledge-sources"
LEGACY_VECTOR_STORE_PATH = BASE_DIR / "app" / "rag" / "vector_store.json"

_chroma_dir = Path(os.environ.get("CHROMA_PERSIST_DIR", "data/chroma"))
CHROMA_PERSIST_DIR = _chroma_dir if _chroma_dir.is_absolute() else BASE_DIR / _chroma_dir
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "est_ai_knowledge")
CHROMA_INDEX_VERSION = os.environ.get("CHROMA_INDEX_VERSION", "rag-chroma-v1")
REPORTS_PATH = BASE_DIR / "data" / "reports.jsonl"
