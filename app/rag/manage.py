"""Chroma 색인 이전·생성·갱신·삭제·상태 확인 CLI."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.load import load

from app.config import (
    CHROMA_COLLECTION,
    CHROMA_INDEX_VERSION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    LEGACY_VECTOR_STORE_PATH,
)
from app.rag.chroma_store import (
    ChromaCompatibilityError,
    ChromaStoreHandle,
    clear_chroma_caches,
    create_collection,
    get_chroma_client,
    make_embeddings,
    metadata_for_chroma,
    metadata_from_chroma,
    stable_chunk_id,
    stable_document_id,
    validate_collection_name,
)
from app.rag.ingest import build_documents
from app.rag.retrieve import MAX_PER_DOCUMENT, retrieve_relevant_chunks_by_vector

BATCH_SIZE = 100
DEFAULT_COMPARE_QUERIES = [
    "상한 식품의 환불 문의",
    "소비자가 보관을 잘못한 경우",
    "여러 개 구매 후 일부만 문제가 생긴 경우",
    "해당 업종과 관계없는 질문",
]


def _batches(items: list[dict[str, Any]], size: int = BATCH_SIZE):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _collection_or_none(client: Any, name: str):
    try:
        return client.get_collection(name=name, embedding_function=None)
    except NotFoundError:
        return None


def _assert_metadata(collection: Any, model: str, dimension: int, index_version: str) -> None:
    metadata = collection.metadata or {}
    expected = {
        "embeddingModel": model,
        "embeddingDimension": dimension,
        "indexVersion": index_version,
        "distanceMetric": "cosine",
        "schemaVersion": "1",
    }
    differences = [
        f"{key}={metadata.get(key)!r} (기대: {value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if differences:
        raise ChromaCompatibilityError(
            "대상 collection의 임베딩/색인 정보가 다릅니다: " + "; ".join(differences)
        )


def _target_collection(
    name: str,
    *,
    model: str,
    dimension: int,
    index_version: str,
    allow_existing: bool,
):
    validate_collection_name(name)
    client = get_chroma_client(str(CHROMA_PERSIST_DIR))
    collection = _collection_or_none(client, name)
    if collection is None:
        collection = create_collection(
            client,
            name,
            embedding_model=model,
            dimension=dimension,
            index_version=index_version,
        )
    else:
        _assert_metadata(collection, model, dimension, index_version)
        if not allow_existing and collection.count() > 0:
            raise RuntimeError(
                f"collection {name!r}이 이미 {collection.count()}개 레코드를 가집니다. "
                "전체 재색인은 새 collection 이름으로 실행하세요."
            )
    return collection


def _upsert(collection: Any, records: list[dict[str, Any]]) -> None:
    for batch in _batches(records):
        collection.upsert(
            ids=[item["id"] for item in batch],
            embeddings=[item["embedding"] for item in batch],
            documents=[item["document"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )


def _records_from_documents(documents: list[Document], embeddings: Any) -> tuple[list[dict[str, Any]], int]:
    vectors = embeddings.embed_documents([document.page_content for document in documents])
    if not vectors:
        raise RuntimeError("색인할 청크가 없습니다.")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise RuntimeError("임베딩 차원이 일정하지 않습니다.")

    records: list[dict[str, Any]] = []
    for document, vector in zip(documents, vectors, strict=True):
        metadata = dict(document.metadata)
        source_path = str(metadata.get("sourcePath") or metadata.get("documentTitle") or "unknown")
        chunk_index = int(metadata.get("chunkIndex", 0))
        metadata["documentId"] = stable_document_id(source_path)
        metadata["chunkIndex"] = chunk_index
        records.append(
            {
                "id": stable_chunk_id(source_path, chunk_index, document.page_content),
                "embedding": vector,
                "document": document.page_content,
                "metadata": metadata_for_chroma(metadata),
            }
        )
    return records, dimension


def _load_legacy_records(path: Path, index_version: str) -> tuple[list[dict[str, Any]], int]:
    """InMemoryVectorStore.dump JSON의 기존 벡터를 API 호출 없이 재사용한다."""
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    plain_store = isinstance(raw, dict) and all(
        isinstance(value, dict) and {"text", "vector"}.issubset(value)
        for value in raw.values()
    )
    if plain_store:
        decoded = raw
    else:
        try:
            decoded = load(raw, allowed_objects=[Document])
        except Exception:  # noqa: BLE001 - 예전 plain JSON 형식도 수용한다.
            decoded = raw
    if not isinstance(decoded, dict):
        raise ValueError("vector_store.json의 최상위가 dict가 아닙니다.")

    source_counts: defaultdict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    dimension: int | None = None
    for legacy_id, value in decoded.items():
        if not isinstance(value, dict):
            raise ValueError(f"레코드 {legacy_id!r}의 형식이 dict가 아닙니다.")
        text = value.get("text")
        vector = value.get("vector")
        metadata = dict(value.get("metadata") or {})
        if not isinstance(text, str) or not isinstance(vector, list) or not vector:
            raise ValueError(f"레코드 {legacy_id!r}에 text/vector가 없습니다.")
        if not all(isinstance(item, (int, float)) for item in vector):
            raise ValueError(f"레코드 {legacy_id!r}의 vector가 숫자 배열이 아닙니다.")
        current_dimension = len(vector)
        if dimension is None:
            dimension = current_dimension
        elif dimension != current_dimension:
            raise ValueError(
                f"임베딩 차원이 섞여 있습니다: {dimension}, {current_dimension} ({legacy_id})"
            )

        source_path = str(metadata.get("sourcePath") or metadata.get("documentTitle") or "legacy")
        chunk_index = int(metadata.get("chunkIndex", source_counts[source_path]))
        source_counts[source_path] = max(source_counts[source_path], chunk_index + 1)
        metadata.update(
            {
                "sourcePath": source_path,
                "documentId": stable_document_id(source_path),
                "chunkIndex": chunk_index,
                "legacyRecordId": str(legacy_id),
            }
        )
        records.append(
            {
                "id": stable_chunk_id(source_path, chunk_index, text, index_version),
                "embedding": [float(item) for item in vector],
                "document": text,
                "metadata": metadata_for_chroma(metadata),
            }
        )
    if dimension is None:
        raise ValueError("vector_store.json이 비어 있습니다.")
    return records, dimension


def command_migrate(args: argparse.Namespace) -> None:
    source = args.from_json.resolve()
    if not source.exists():
        raise FileNotFoundError(
            f"기존 JSON이 없습니다: {source}. 원본 문서 재색인은 `reindex --yes`를 사용하세요."
        )
    records, dimension = _load_legacy_records(source, args.index_version)
    print(
        f"이전 대상: {len(records)}개, {dimension}차원. JSON에 모델 정보가 없어 "
        f"{args.embedding_model!r}에서 만든 벡터로 간주합니다. 사실과 다르면 중단하세요."
    )
    collection = _target_collection(
        args.collection,
        model=args.embedding_model,
        dimension=dimension,
        index_version=args.index_version,
        allow_existing=True,
    )
    _upsert(collection, records)
    print(
        f"완료: {collection.count()}개. 같은 JSON을 다시 실행해도 안정 ID upsert로 중복되지 않습니다."
    )


def command_reindex(args: argparse.Namespace) -> None:
    documents = build_documents()
    collection_name = args.collection or (
        f"{CHROMA_COLLECTION}__{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    validate_collection_name(collection_name)
    print(f"재색인 대상: {len(documents)}개 청크, 모델={EMBEDDING_MODEL}, collection={collection_name}")
    if not args.yes:
        print(
            "이 명령은 전체 문서 임베딩 API를 호출하며 비용이 발생할 수 있습니다. "
            "확인 후 같은 명령에 --yes를 붙이세요. 기존 collection은 삭제하지 않습니다."
        )
        return

    client = get_chroma_client(str(CHROMA_PERSIST_DIR))
    existing = _collection_or_none(client, collection_name)
    if existing is not None and existing.count() > 0:
        raise RuntimeError(
            f"collection {collection_name!r}이 이미 {existing.count()}개 레코드를 가집니다. "
            "유료 임베딩 호출 전에 중단했습니다. 새 collection 이름을 사용하세요."
        )
    records, dimension = _records_from_documents(documents, make_embeddings())
    collection = _target_collection(
        collection_name,
        model=EMBEDDING_MODEL,
        dimension=dimension,
        index_version=CHROMA_INDEX_VERSION,
        allow_existing=False,
    )
    _upsert(collection, records)
    actual = collection.count()
    if actual != len(records):
        raise RuntimeError(f"저장 검증 실패: 기대 {len(records)}, 실제 {actual}")
    print(
        f"완료: {collection_name} ({actual}개). 검증 후 CHROMA_COLLECTION={collection_name}로 전환하세요."
    )


def _documents_for_source(source_path: str) -> list[Document]:
    matches = [doc for doc in build_documents() if doc.metadata.get("sourcePath") == source_path]
    if not matches:
        raise ValueError(f"색인 대상 문서를 찾지 못했습니다: {source_path}")
    return matches


def command_upsert_document(args: argparse.Namespace) -> None:
    documents = _documents_for_source(args.source_path)
    print(f"갱신 대상: {args.source_path} -> {len(documents)}개 청크")
    if not args.yes:
        print("임베딩 API 호출 후 기존 문서 청크를 교체합니다. 확인하면 --yes를 붙이세요.")
        return
    client = get_chroma_client(str(CHROMA_PERSIST_DIR))
    collection = _collection_or_none(client, args.collection)
    if collection is None:
        raise RuntimeError(f"collection이 없습니다: {args.collection}")
    stored_dimension = (collection.metadata or {}).get("embeddingDimension")
    if not isinstance(stored_dimension, int):
        raise ChromaCompatibilityError("collection에 embeddingDimension이 없습니다.")
    _assert_metadata(collection, EMBEDDING_MODEL, stored_dimension, CHROMA_INDEX_VERSION)
    records, dimension = _records_from_documents(documents, make_embeddings())
    _assert_metadata(collection, EMBEDDING_MODEL, dimension, CHROMA_INDEX_VERSION)
    # 새 벡터 생성에 성공한 뒤 이전 청크를 지워, 임베딩 실패로 정상 데이터가 먼저 사라지지 않게 한다.
    collection.delete(where={"sourcePath": {"$eq": args.source_path}})
    _upsert(collection, records)
    print(f"완료: {args.source_path}의 이전 청크를 제거하고 {len(records)}개로 갱신했습니다.")


def command_delete_document(args: argparse.Namespace) -> None:
    if not args.yes:
        print(f"삭제 대상: {args.collection} / sourcePath={args.source_path}. 확인하면 --yes를 붙이세요.")
        return
    client = get_chroma_client(str(CHROMA_PERSIST_DIR))
    collection = _collection_or_none(client, args.collection)
    if collection is None:
        raise RuntimeError(f"collection이 없습니다: {args.collection}")
    before = collection.count()
    collection.delete(where={"sourcePath": {"$eq": args.source_path}})
    removed = before - collection.count()
    print(f"완료: {removed}개 청크를 삭제했습니다.")


def command_status(args: argparse.Namespace) -> None:
    if not CHROMA_PERSIST_DIR.exists():
        print(f"Chroma DB 경로가 없습니다: {CHROMA_PERSIST_DIR}")
        return
    client = get_chroma_client(str(CHROMA_PERSIST_DIR))
    if args.collection:
        collection = _collection_or_none(client, args.collection)
        if collection is None:
            print(f"collection이 없습니다: {args.collection}")
            return
        print(f"{collection.name}: count={collection.count()}, metadata={collection.metadata}")
        return
    collections = client.list_collections()
    if not collections:
        print(f"collection이 없습니다: {CHROMA_PERSIST_DIR}")
        return
    for item in collections:
        name = item if isinstance(item, str) else item.name
        collection = client.get_collection(name=name, embedding_function=None)
        marker = " *active" if name == CHROMA_COLLECTION else ""
        print(f"{name}: count={collection.count()}{marker}, metadata={collection.metadata}")


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _legacy_retrieve(
    records: list[dict[str, Any]],
    query_vector: list[float],
    *,
    pinned_sections: tuple[str, ...],
    industry_id: str | None,
    top_k: int = 5,
) -> list[Document]:
    ranked: list[tuple[float, Document]] = []
    for item in records:
        metadata = metadata_from_chroma(item["metadata"])
        industries = metadata.get("industries")
        if industry_id is not None and industries and "*" not in industries and industry_id not in industries:
            continue
        ranked.append(
            (
                _cosine(query_vector, item["embedding"]),
                Document(page_content=item["document"], metadata=metadata),
            )
        )
    ranked.sort(key=lambda pair: pair[0], reverse=True)

    documents: list[Document] = []
    seen: set[str] = set()
    for section in pinned_sections:
        for _score, document in ranked:
            if section not in str(document.metadata.get("section") or ""):
                continue
            if document.page_content not in seen:
                documents.append(document)
                seen.add(document.page_content)
                break

    per_document: dict[str, int] = {}
    for _score, document in ranked[: max(top_k * 8, 40)]:
        if len(documents) >= top_k + len(pinned_sections):
            break
        if document.page_content in seen:
            continue
        title = str(document.metadata.get("documentTitle") or "")
        if per_document.get(title, 0) >= MAX_PER_DOCUMENT:
            continue
        per_document[title] = per_document.get(title, 0) + 1
        documents.append(document)
        seen.add(document.page_content)
    return documents


def command_compare(args: argparse.Namespace) -> None:
    if not args.from_json.exists():
        raise FileNotFoundError(f"기존 JSON이 없습니다: {args.from_json}")
    records, dimension = _load_legacy_records(args.from_json, args.index_version)
    client = get_chroma_client(str(CHROMA_PERSIST_DIR))
    collection = _collection_or_none(client, args.collection)
    if collection is None:
        raise RuntimeError(f"collection이 없습니다: {args.collection}")
    _assert_metadata(collection, args.embedding_model, dimension, args.index_version)
    queries = args.query or DEFAULT_COMPARE_QUERIES
    print(
        f"비교 대상: JSON {len(records)}개 / Chroma {collection.count()}개 / "
        f"{args.embedding_model} {dimension}차원"
    )
    if not args.yes:
        print(
            f"대표 질문 {len(queries)}개의 임베딩 API를 호출합니다. "
            "같은 질의 벡터를 두 저장소에 재사용하려면 --yes를 붙이세요."
        )
        return

    embeddings = make_embeddings()
    vector_store = Chroma(
        client=client,
        collection_name=args.collection,
        embedding_function=embeddings,
    )
    handle = ChromaStoreHandle(client, collection, vector_store, embeddings, dimension)
    for query in queries:
        pinned = (
            ("식료품(19개 업종)", "[별표 1]", "식료품")
            if any(hint in query for hint in ("환불", "보관", "상한", "상했", "문제"))
            else ()
        )
        query_vector = embeddings.embed_query(query)
        started = time.perf_counter()
        legacy = _legacy_retrieve(
            records,
            query_vector,
            pinned_sections=pinned,
            industry_id=args.industry,
        )
        legacy_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        chroma = retrieve_relevant_chunks_by_vector(
            query_vector,
            top_k=5,
            pinned_sections=pinned,
            industry_id=args.industry,
            store_handle=handle,
        )
        chroma_ms = (time.perf_counter() - started) * 1000
        print(f"\n질문: {query}\nJSON {legacy_ms:.2f}ms / Chroma {chroma_ms:.2f}ms")
        for label, documents in (("JSON", legacy), ("Chroma", chroma)):
            print(f"  {label}:")
            for index, document in enumerate(documents, 1):
                print(
                    f"    {index}. {document.metadata.get('documentTitle')} | "
                    f"{document.metadata.get('section')}"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate = subparsers.add_parser("migrate-json", help="기존 InMemoryVectorStore JSON 벡터를 Chroma로 이전")
    migrate.add_argument("--from-json", type=Path, default=LEGACY_VECTOR_STORE_PATH)
    migrate.add_argument("--collection", default=CHROMA_COLLECTION)
    migrate.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    migrate.add_argument("--index-version", default=CHROMA_INDEX_VERSION)
    migrate.set_defaults(handler=command_migrate)

    reindex = subparsers.add_parser("reindex", help="원본 문서 전체를 새 collection으로 색인")
    reindex.add_argument("--collection", help="생략하면 타임스탬프가 붙은 후보 collection 생성")
    reindex.add_argument("--yes", action="store_true", help="임베딩 API 비용 확인")
    reindex.set_defaults(handler=command_reindex)

    upsert = subparsers.add_parser("upsert-document", help="특정 원본 문서를 추가 또는 전체 교체")
    upsert.add_argument("source_path", help="knowledge-sources 기준 상대경로")
    upsert.add_argument("--collection", default=CHROMA_COLLECTION)
    upsert.add_argument("--yes", action="store_true")
    upsert.set_defaults(handler=command_upsert_document)

    delete = subparsers.add_parser("delete-document", help="sourcePath에 해당하는 청크 전체 삭제")
    delete.add_argument("source_path")
    delete.add_argument("--collection", default=CHROMA_COLLECTION)
    delete.add_argument("--yes", action="store_true")
    delete.set_defaults(handler=command_delete_document)

    status = subparsers.add_parser("status", help="collection 문서 수와 구성 확인")
    status.add_argument("--collection")
    status.set_defaults(handler=command_status)

    compare = subparsers.add_parser("compare", help="같은 질의 벡터로 JSON과 Chroma 검색 비교")
    compare.add_argument("--from-json", type=Path, default=LEGACY_VECTOR_STORE_PATH)
    compare.add_argument("--collection", default=CHROMA_COLLECTION)
    compare.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    compare.add_argument("--index-version", default=CHROMA_INDEX_VERSION)
    compare.add_argument("--industry", default="convenience_store")
    compare.add_argument(
        "--query",
        action="append",
        help="생략하면 문서의 대표 질문 4개 사용; 여러 번 지정 가능",
    )
    compare.add_argument("--yes", action="store_true", help="질의 임베딩 API 비용 확인")
    compare.set_defaults(handler=command_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (ChromaCompatibilityError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        clear_chroma_caches()


if __name__ == "__main__":
    main()
