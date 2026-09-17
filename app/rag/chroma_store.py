"""Chroma 저장소 공통 구성과 메타데이터 호환 계층."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import (
    CHROMA_COLLECTION,
    CHROMA_INDEX_VERSION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

COLLECTION_SCHEMA_VERSION = "1"
DISTANCE_METRIC = "cosine"
_INTERNAL_METADATA_KEYS = {"industriesJson", "industryAll", "industriesMissing"}
_SAFE_COLLECTION = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$")


class ChromaStoreError(RuntimeError):
    """Chroma DB를 정상적으로 사용할 수 없을 때 발생한다."""


class ChromaCompatibilityError(ChromaStoreError):
    """색인과 런타임의 임베딩/스키마 구성이 다를 때 발생한다."""


@dataclass(frozen=True)
class ChromaStoreHandle:
    client: chromadb.ClientAPI
    collection: Collection
    vector_store: Chroma
    embeddings: OpenAIEmbeddings
    embedding_dimension: int


def make_embeddings() -> OpenAIEmbeddings:
    kwargs: dict[str, Any] = {"model": EMBEDDING_MODEL}
    if EMBEDDING_DIMENSIONS is not None:
        kwargs["dimensions"] = EMBEDDING_DIMENSIONS
    return OpenAIEmbeddings(**kwargs)


def validate_collection_name(name: str) -> str:
    if not _SAFE_COLLECTION.fullmatch(name):
        raise ValueError(
            "collection 이름은 3~63자의 영문·숫자·._-만 사용하고 "
            "영문/숫자로 시작·종료해야 합니다."
        )
    return name


@lru_cache(maxsize=4)
def get_chroma_client(path: str | None = None) -> chromadb.ClientAPI:
    persist_path = Path(path) if path is not None else CHROMA_PERSIST_DIR
    persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_path))


def clear_chroma_caches() -> None:
    """관리 명령으로 collection을 바꾼 뒤 동일 프로세스에서 다시 열 때 쓴다."""
    get_chroma_client.cache_clear()


def collection_metadata(embedding_model: str, dimension: int, index_version: str) -> dict[str, Any]:
    return {
        "embeddingModel": embedding_model,
        "embeddingDimension": dimension,
        "indexVersion": index_version,
        "distanceMetric": DISTANCE_METRIC,
        "schemaVersion": COLLECTION_SCHEMA_VERSION,
    }


def _metadata_value(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)):
        return value
    return None


def _industry_field(industry_id: str) -> str:
    digest = hashlib.sha256(industry_id.encode("utf-8")).hexdigest()[:16]
    return f"industry_{digest}"


def metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Document 메타데이터를 Chroma의 스칼라 호환 형식으로 변환한다.

    업종 목록은 JSON 문자열로 보존하고 필터용 bool 필드를 파생한다.
    이렇게 하면 배열 메타데이터를 지원하지 않던 Chroma 버전과도 호환된다.
    """
    result: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if key == "industries" or value is None:
            continue
        scalar = _metadata_value(value)
        if scalar is not None:
            result[key] = scalar

    raw_industries = metadata.get("industries")
    missing = not raw_industries
    industries = (
        [str(item) for item in raw_industries]
        if isinstance(raw_industries, (list, tuple, set))
        else []
    )
    result["industriesJson"] = json.dumps(industries, ensure_ascii=False, separators=(",", ":"))
    result["industriesMissing"] = missing
    # 기존 메타데이터가 없는 색인은 기존 정책대로 전 업종에 허용한다.
    result["industryAll"] = missing or "*" in industries
    for industry_id in industries:
        if industry_id != "*":
            result[_industry_field(industry_id)] = True
    return result


def metadata_from_chroma(metadata: dict[str, Any] | None) -> dict[str, Any]:
    restored = dict(metadata or {})
    encoded = restored.pop("industriesJson", "[]")
    try:
        industries = json.loads(encoded) if isinstance(encoded, str) else []
    except json.JSONDecodeError:
        industries = []
    restored["industries"] = industries if isinstance(industries, list) else []
    restored.pop("industryAll", None)
    restored.pop("industriesMissing", None)
    for key in list(restored):
        if key.startswith("industry_"):
            restored.pop(key)
    return restored


def industry_filter(industry_id: str | None) -> dict[str, Any] | None:
    if industry_id is None:
        return None
    return {
        "$or": [
            {"industryAll": {"$eq": True}},
            {_industry_field(industry_id): {"$eq": True}},
        ]
    }


def stable_document_id(source_path: str) -> str:
    return "doc-" + hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:24]


def stable_chunk_id(
    source_path: str, chunk_index: int, page_content: str, index_version: str = CHROMA_INDEX_VERSION
) -> str:
    content_hash = hashlib.sha256(page_content.encode("utf-8")).hexdigest()
    payload = f"{index_version}\0{source_path}\0{chunk_index}\0{content_hash}"
    return "chunk-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_collection(
    client: chromadb.ClientAPI,
    name: str,
    *,
    embedding_model: str,
    dimension: int,
    index_version: str = CHROMA_INDEX_VERSION,
) -> Collection:
    validate_collection_name(name)
    return client.create_collection(
        name=name,
        metadata=collection_metadata(embedding_model, dimension, index_version),
        configuration={"hnsw": {"space": DISTANCE_METRIC}},
        embedding_function=None,
    )


def validate_collection(collection: Collection) -> int:
    metadata = collection.metadata or {}
    expected = {
        "embeddingModel": EMBEDDING_MODEL,
        "indexVersion": CHROMA_INDEX_VERSION,
        "distanceMetric": DISTANCE_METRIC,
        "schemaVersion": COLLECTION_SCHEMA_VERSION,
    }
    mismatches = [
        f"{key}={metadata.get(key)!r} (기대: {value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    dimension = metadata.get("embeddingDimension")
    if not isinstance(dimension, int) or dimension <= 0:
        mismatches.append(f"embeddingDimension={dimension!r} (양의 정수가 필요)")
    elif EMBEDDING_DIMENSIONS is not None and dimension != EMBEDDING_DIMENSIONS:
        mismatches.append(
            f"embeddingDimension={dimension!r} (설정: {EMBEDDING_DIMENSIONS!r})"
        )
    if mismatches:
        raise ChromaCompatibilityError(
            "Chroma collection의 임베딩/색인 구성이 현재 설정과 다릅니다: "
            + "; ".join(mismatches)
            + ". 올바른 collection으로 CHROMA_COLLECTION을 전환하거나 새로 색인하세요."
        )
    return int(dimension)


def open_runtime_store() -> ChromaStoreHandle | None:
    """기존 collection을 열기만 한다. 요청 처리 중에 생성/색인하지 않는다."""
    if not CHROMA_PERSIST_DIR.exists():
        logger.warning(
            "Chroma DB 경로가 없습니다: %s. "
            "`python -m app.rag.manage status`로 확인하고 이전 또는 색인하세요.",
            CHROMA_PERSIST_DIR,
        )
        return None

    try:
        client = get_chroma_client(str(CHROMA_PERSIST_DIR))
        collection = client.get_collection(name=CHROMA_COLLECTION, embedding_function=None)
    except NotFoundError:
        logger.warning(
            "Chroma collection이 없습니다: %s (%s). "
            "`python -m app.rag.manage status`로 확인하고 이전 또는 색인하세요.",
            CHROMA_COLLECTION,
            CHROMA_PERSIST_DIR,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        raise ChromaStoreError(f"Chroma DB를 열지 못했습니다: {exc}") from exc

    dimension = validate_collection(collection)
    try:
        count = collection.count()
    except Exception as exc:  # noqa: BLE001
        raise ChromaStoreError(f"Chroma collection 상태를 읽지 못했습니다: {exc}") from exc
    if count == 0:
        logger.warning(
            "Chroma collection이 비어 있습니다: %s. 요청 중 자동 색인하지 않습니다.",
            CHROMA_COLLECTION,
        )
        return None

    embeddings = make_embeddings()
    vector_store = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
    )
    return ChromaStoreHandle(client, collection, vector_store, embeddings, dimension)
