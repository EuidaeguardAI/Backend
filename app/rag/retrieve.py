"""공통 Chroma 검색 계층.

실시간 추천과 물어보기 챗봇이 모두 이 함수를 사용한다.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document

from app.rag.chroma_store import (
    ChromaCompatibilityError,
    ChromaStoreError,
    ChromaStoreHandle,
    industry_filter,
    metadata_from_chroma,
    open_runtime_store,
)

MAX_PER_DOCUMENT = 2
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaStoreHandle | None:
    """PersistentClient, collection, LangChain Chroma wrapper를 프로세스 내에서 재사용한다."""
    return open_runtime_store()


def clear_vector_store_cache() -> None:
    get_vector_store.cache_clear()


def _matches(document: Document, sections: tuple[str, ...]) -> bool:
    section = document.metadata.get("section") or ""
    return any(pinned in section for pinned in sections)


def _matches_industry(document: Document, industry_id: str | None) -> bool:
    """업종을 모르거나 industries가 없는 예전 색인은 기존 정책대로 통과시킨다."""
    if industry_id is None:
        return True
    industries = document.metadata.get("industries")
    if not industries:
        return True
    return "*" in industries or industry_id in industries


def _query_by_vector(
    handle: ChromaStoreHandle,
    query_vector: list[float],
    *,
    n_results: int,
    industry_id: str | None,
    section_contains: str | None = None,
) -> list[tuple[Document, float | None]]:
    if n_results <= 0:
        return []
    where = industry_filter(industry_id)
    where_document = None
    if section_contains is not None:
        # Chroma는 스칼라 메타데이터 문자열의 부분 일치를 지원하지 않는다.
        # ingest가 page_content 머리글에 section을 포함하므로 문서 필터로 먼저
        # 좁힌 뒤, 아래에서 section 메타데이터를 다시 검증한다.
        where_document = {"$contains": section_contains}
    try:
        documents = handle.vector_store.similarity_search_by_vector(
            query_vector,
            k=n_results,
            filter=where,
            where_document=where_document,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chroma 검색 실패(collection=%s)", handle.collection.name)
        raise ChromaStoreError(f"Chroma 검색 중 DB 오류가 발생했습니다: {exc}") from exc
    return [
        (
            Document(
                id=document.id,
                page_content=document.page_content,
                metadata=metadata_from_chroma(document.metadata),
            ),
            None,
        )
        for document in documents
    ]


def retrieve_relevant_chunks_by_vector(
    query_vector: list[float],
    *,
    top_k: int = 5,
    pinned_sections: tuple[str, ...] = (),
    industry_id: str | None = None,
    store_handle: ChromaStoreHandle | None = None,
) -> list[Document]:
    """이미 계산한 하나의 질의 벡터를 pinned/일반 검색에 모두 재사용한다."""
    handle = store_handle if store_handle is not None else get_vector_store()
    if handle is None:
        return []
    if len(query_vector) != handle.embedding_dimension:
        raise ChromaCompatibilityError(
            f"질의 임베딩 차원({len(query_vector)})과 Chroma collection 차원"
            f"({handle.embedding_dimension})이 다릅니다. 임베딩 모델·dimensions·collection을 확인하세요."
        )

    try:
        collection_count = handle.collection.count()
    except Exception as exc:  # noqa: BLE001
        raise ChromaStoreError(f"Chroma collection 건수를 확인하지 못했습니다: {exc}") from exc
    if collection_count == 0:
        return []

    documents: list[Document] = []
    seen: set[str] = set()

    # pinned section별로 최소 1건을 보장한다. 겹치는 section 문자열이 있으므로
    # 전체 매칭 후보를 거리순으로 받고 중복이 아닌 첫 건을 선택한다.
    for section in pinned_sections:
        candidates = _query_by_vector(
            handle,
            query_vector,
            n_results=collection_count,
            industry_id=industry_id,
            section_contains=section,
        )
        for document, _distance in candidates:
            # where_document에 본문 우연 일치가 있을 수 있으므로 기존 section 조건을 재검증한다.
            if not _matches(document, (section,)) or not _matches_industry(document, industry_id):
                continue
            if document.page_content in seen:
                continue
            documents.append(document)
            seen.add(document.page_content)
            break

    per_document: dict[str, int] = {}
    candidate_pool = min(max(top_k * 8, 40), collection_count)
    for document, _distance in _query_by_vector(
        handle,
        query_vector,
        n_results=candidate_pool,
        industry_id=industry_id,
    ):
        if len(documents) >= top_k + len(pinned_sections):
            break
        if not _matches_industry(document, industry_id) or document.page_content in seen:
            continue
        title = document.metadata.get("documentTitle") or ""
        if per_document.get(title, 0) >= MAX_PER_DOCUMENT:
            continue
        per_document[title] = per_document.get(title, 0) + 1
        documents.append(document)
        seen.add(document.page_content)

    return documents


def retrieve_relevant_chunks(
    query_text: str,
    top_k: int = 5,
    pinned_sections: tuple[str, ...] = (),
    industry_id: str | None = None,
) -> list[Document]:
    """기존 호출 계약을 유지하며 Chroma에서 근거 청크를 검색한다.

    질의 임베딩은 이 호출에서 단 한 번만 계산하고 pinned 및 일반
    검색에 같은 벡터를 사용한다. Chroma의 cosine distance를 기존 similarity
    score와 같은 범위/방향으로 간주하지 않고, 순위만 사용한다.
    """
    handle = get_vector_store()
    if handle is None:
        return []
    query_vector = handle.embeddings.embed_query(query_text)
    return retrieve_relevant_chunks_by_vector(
        query_vector,
        top_k=top_k,
        pinned_sections=pinned_sections,
        industry_id=industry_id,
        store_handle=handle,
    )
