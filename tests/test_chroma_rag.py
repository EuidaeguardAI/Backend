from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb
from langchain_chroma import Chroma

from app.graph.citations import ground_citations
from app.graph.ask_graph import run_ask_graph
from app.graph.recommendation_graph import run_recommendation_graph
from app.rag.chroma_store import (
    ChromaStoreHandle,
    create_collection,
    metadata_for_chroma,
    metadata_from_chroma,
    stable_chunk_id,
)
from app.rag.manage import _load_legacy_records, _upsert
from app.rag.retrieve import retrieve_relevant_chunks
from app.schemas import BusinessProfile, CitationDraft, SessionIntake


class CountingEmbeddings:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.query_calls = 0

    def embed_query(self, _text: str) -> list[float]:
        self.query_calls += 1
        return list(self.vector)


def _record(
    record_id: str,
    text: str,
    vector: list[float],
    *,
    title: str,
    section: str,
    industries: list[str] | None,
    source_path: str,
) -> dict:
    metadata = {
        "documentTitle": title,
        "sourceType": "standard",
        "section": section,
        "sourcePath": source_path,
        "documentId": "doc-test",
        "chunkIndex": 0,
    }
    if industries is not None:
        metadata["industries"] = industries
    return {
        "id": record_id,
        "embedding": vector,
        "document": text,
        "metadata": metadata_for_chroma(metadata),
    }


class ChromaRagTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.client = chromadb.PersistentClient(path=str(self.path))

    def tearDown(self) -> None:
        self.client.close()
        self.temp.cleanup()

    def test_persistence_can_be_read_by_new_process(self) -> None:
        collection = create_collection(
            self.client,
            "persistence-test",
            embedding_model="fake-model",
            dimension=3,
            index_version="test-v1",
        )
        collection.upsert(
            ids=["one"],
            embeddings=[[1.0, 0.0, 0.0]],
            documents=["persistent text"],
            metadatas=[metadata_for_chroma({"sourcePath": "one.md", "industries": ["*"]})],
        )
        code = (
            "import chromadb,sys; "
            "c=chromadb.PersistentClient(path=sys.argv[1]).get_collection('persistence-test'); "
            "print(c.count(), c.get(ids=['one'])['documents'][0])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.path)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "1 persistent text")

    def test_json_migration_is_idempotent_and_preserves_content(self) -> None:
        legacy_path = self.path / "vector_store.json"
        legacy = {
            "old-a": {
                "id": "old-a",
                "vector": [1.0, 0.0, 0.0],
                "text": "[문서 A · 식료품]\n원문 A",
                "metadata": {
                    "documentTitle": "문서 A",
                    "section": "식료품",
                    "sourceType": "standard",
                    "sourcePath": "a.md",
                    "industries": ["*"],
                },
            },
            "old-b": {
                "id": "old-b",
                "vector": [0.0, 1.0, 0.0],
                "text": "[문서 B · 보관]\n원문 B",
                "metadata": {
                    "documentTitle": "문서 B",
                    "section": "보관",
                    "sourceType": "guide",
                    "sourcePath": "b.md",
                    "industries": ["convenience_store"],
                },
            },
        }
        legacy_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        records, dimension = _load_legacy_records(legacy_path, "test-v1")
        collection = create_collection(
            self.client,
            "migration-test",
            embedding_model="fake-model",
            dimension=dimension,
            index_version="test-v1",
        )
        _upsert(collection, records)
        _upsert(collection, records)
        self.assertEqual(collection.count(), 2)
        stored = collection.get(include=["documents", "metadatas"])
        self.assertEqual(set(stored["documents"]), {legacy["old-a"]["text"], legacy["old-b"]["text"]})
        restored = [metadata_from_chroma(item) for item in stored["metadatas"]]
        self.assertEqual({item["sourcePath"] for item in restored}, {"a.md", "b.md"})
        self.assertIn(["*"], [item["industries"] for item in restored])

    def test_migrate_cli_can_be_rerun_without_duplicates(self) -> None:
        legacy_path = self.path / "cli-vector-store.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "old": {
                        "id": "old",
                        "vector": [1.0, 0.0, 0.0],
                        "text": "CLI 이전 원문",
                        "metadata": {
                            "documentTitle": "CLI 문서",
                            "section": "섹션",
                            "sourceType": "guide",
                            "sourcePath": "cli.md",
                            "industries": ["*"],
                        },
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        db_path = self.path / "cli-db"
        environment = os.environ.copy()
        environment.update(
            {
                "CHROMA_PERSIST_DIR": str(db_path),
                "CHROMA_COLLECTION": "cli-migration",
                "CHROMA_INDEX_VERSION": "test-v1",
                "OPENAI_EMBEDDING_MODEL": "fake-model",
                "PYTHONIOENCODING": "utf-8",
            }
        )
        command = [
            sys.executable,
            "-m",
            "app.rag.manage",
            "migrate-json",
            "--from-json",
            str(legacy_path),
            "--collection",
            "cli-migration",
            "--embedding-model",
            "fake-model",
            "--index-version",
            "test-v1",
        ]
        subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8", env=environment
        )
        subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8", env=environment
        )
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection("cli-migration")
        self.assertEqual(collection.count(), 1)
        self.assertEqual(collection.get()["documents"], ["CLI 이전 원문"])
        client.close()

    def test_filter_pinned_dedup_and_single_query_embedding(self) -> None:
        collection = create_collection(
            self.client,
            "retrieval-test",
            embedding_model="fake-model",
            dimension=3,
            index_version="test-v1",
        )
        records = [
            _record(
                "food",
                "[공통 기준 · 식료품(19개 업종)]\n부패 변질은 교환 또는 환급",
                [1.0, 0.0, 0.0],
                title="공통 기준",
                section="품목별 > 식료품(19개 업종)",
                industries=["*"],
                source_path="common.md",
            ),
            _record(
                "store",
                "[편의점 지침 · 보관]\n보관 상태를 확인한다",
                [0.9, 0.1, 0.0],
                title="편의점 지침",
                section="보관",
                industries=["convenience_store"],
                source_path="store.md",
            ),
            _record(
                "hospital",
                "[병원 지침 · 식료품(19개 업종)]\n관계없는 자료",
                [1.0, 0.0, 0.0],
                title="병원 지침",
                section="식료품(19개 업종)",
                industries=["hospital"],
                source_path="hospital.md",
            ),
            _record(
                "legacy",
                "[예전 자료 · 일반]\n업종 메타데이터 없음",
                [0.8, 0.2, 0.0],
                title="예전 자료",
                section="일반",
                industries=None,
                source_path="legacy.md",
            ),
        ]
        _upsert(collection, records)
        embeddings = CountingEmbeddings([1.0, 0.0, 0.0])
        vector_store = Chroma(
            client=self.client,
            collection_name="retrieval-test",
            embedding_function=embeddings,  # type: ignore[arg-type]
        )
        handle = ChromaStoreHandle(self.client, collection, vector_store, embeddings, 3)  # type: ignore[arg-type]
        with patch("app.rag.retrieve.get_vector_store", return_value=handle):
            documents = retrieve_relevant_chunks(
                "상한 식품 환불",
                top_k=3,
                pinned_sections=("식료품", "식료품(19개 업종)"),
                industry_id="convenience_store",
            )
        self.assertEqual(embeddings.query_calls, 1)
        self.assertEqual(len({doc.page_content for doc in documents}), len(documents))
        titles = {doc.metadata["documentTitle"] for doc in documents}
        self.assertIn("공통 기준", titles)
        self.assertIn("편의점 지침", titles)
        self.assertIn("예전 자료", titles)
        self.assertNotIn("병원 지침", titles)
        self.assertTrue(documents[0].metadata["section"].endswith("식료품(19개 업종)"))

    def test_document_replace_and_delete_leave_no_stale_chunks(self) -> None:
        collection = create_collection(
            self.client,
            "update-test",
            embedding_model="fake-model",
            dimension=3,
            index_version="test-v1",
        )
        old = [
            _record(
                stable_chunk_id("a.md", index, f"old-{index}", "test-v1"),
                f"old-{index}",
                [1.0, 0.0, 0.0],
                title="A",
                section="old",
                industries=["*"],
                source_path="a.md",
            )
            for index in range(2)
        ]
        _upsert(collection, old)
        collection.delete(where={"sourcePath": {"$eq": "a.md"}})
        new = _record(
            stable_chunk_id("a.md", 0, "new", "test-v1"),
            "new",
            [1.0, 0.0, 0.0],
            title="A",
            section="new",
            industries=["*"],
            source_path="a.md",
        )
        _upsert(collection, [new])
        self.assertEqual(collection.count(), 1)
        self.assertEqual(collection.get()["documents"], ["new"])
        collection.delete(where={"sourcePath": {"$eq": "a.md"}})
        self.assertEqual(collection.count(), 0)

    def test_fixed_safety_branch_skips_retrieval_and_llm(self) -> None:
        profile = BusinessProfile(
            industry="편의점",
            industryId="convenience_store",
            tasks=[],
            aiFeatures=[],
            onboardedAtMs=0,
        )
        intake = SessionIntake(
            inProgress=True,
            micAvailable=True,
            problemTypes=["refund_exchange"],
            behaviorTypes=["threat"],
            emergencyDeclared=False,
        )
        with (
            patch(
                "app.graph.recommendation_graph.retrieve_relevant_chunks",
                side_effect=AssertionError("안전 분기에서 검색하면 안 됨"),
            ),
            patch(
                "app.graph.recommendation_graph.ChatOpenAI",
                side_effect=AssertionError("안전 분기에서 LLM을 호출하면 안 됨"),
            ),
        ):
            result = run_recommendation_graph(
                profile,
                intake,
                recent_transcript=[],
                recent_situations=[],
                latest_text="죽여버린다, 칼 가져와",
            )
        self.assertTrue(result.isFixedSafetyScript)
        self.assertTrue(result.citations)

        with (
            patch(
                "app.graph.ask_graph.retrieve_relevant_chunks",
                side_effect=AssertionError("안전 분기에서 검색하면 안 됨"),
            ),
            patch(
                "app.graph.ask_graph.ChatOpenAI",
                side_effect=AssertionError("안전 분기에서 LLM을 호출하면 안 됨"),
            ),
        ):
            answer = run_ask_graph(profile, [], "죽여버린다, 칼 가져와")
        self.assertTrue(answer.isFixedSafetyScript)
        self.assertTrue(answer.citations)

    def test_citation_grounding_uses_retrieved_metadata(self) -> None:
        document = __import__("langchain_core.documents", fromlist=["Document"]).Document(
            page_content="[문서 A · 식료품]\n부패, 변질의 경우 제품교환 또는 구입가 환급",
            metadata={
                "documentTitle": "문서 A",
                "section": "식료품",
                "page": 8,
                "sourceType": "standard",
            },
        )
        grounded = ground_citations(
            [CitationDraft(label="지어낸 이름", section="지어낸 조항", quote="부패, 변질의 경우 제품교환 또는 구입가 환급")],
            [document],
        )
        self.assertEqual(grounded[0].label, "문서 A")
        self.assertEqual(grounded[0].section, "식료품 (p.8)")
        self.assertEqual(grounded[0].sourceType, "standard")
        rejected = ground_citations(
            [CitationDraft(label="지어낸 문서", section="지어낸 조항", quote="원문에 없는 인용문입니다")],
            [document],
        )
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
