# RAG 저장·검색 계층 Chroma 전환

작성일: 2026-09-17

## 1. 변경 전·후 구조

### 변경 전

```text
knowledge-sources/*.md
  -> ingest.py 청킹
  -> OpenAI 문서 임베딩
  -> LangChain InMemoryVectorStore
  -> app/rag/vector_store.json dump

API 요청
  -> JSON 전체 load
  -> 메모리에서 cosine similarity
```

### 변경 후

```text
[사전 관리 명령]
knowledge-sources/*.md
  -> 기존 ingest.py 청킹
  -> OpenAI 문서 임베딩
  -> 로컬 Chroma PersistentClient
  -> data/chroma/<collection>

[일반 상담]
업종·문제 유형 확인
  -> 질의 임베딩 1회
  -> pinned section별 Chroma 검색
  -> 일반 Chroma 검색
  -> 중복 제거·문서별 상한
  -> LLM 구조화 응답
  -> ground_citations()
  -> 검증된 citation만 반환

[긴급 상담]
위협 감지 -> 기존 고정 안전 절차 -> END
              (검색·LLM 미실행)
```

Chroma는 별도 서버 없이 프로세스 내 `PersistentClient`로 실행하고,
LangChain `Chroma` wrapper와 같은 client/collection을 재사용한다. 2026-09-17 검증
환경은 `chromadb 1.5.9`, `langchain-chroma 1.1.0`, `langchain-core 1.6.3`이다.

구현 기준은 [LangChain Chroma 통합 문서](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma),
[Chroma collection 구성](https://docs.trychroma.com/docs/collections/configure),
[Chroma metadata filter](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)를 확인했다.

## 2. 설정과 collection 호환성

`.env` 설정:

```dotenv
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# 임베딩을 축소 차원으로 생성했을 때만 설정
# OPENAI_EMBEDDING_DIMENSIONS=1536
CHROMA_PERSIST_DIR=data/chroma
CHROMA_COLLECTION=est_ai_knowledge
CHROMA_INDEX_VERSION=rag-chroma-v1
```

- 저장 경로와 활성 collection 이름은 `app/config.py`에서 관리한다.
- `data/chroma/`는 `.gitignore`에 포함되어 DB 파일이 Git에 들어가지 않는다.
- collection은 cosine 거리로 생성한다. Chroma distance를 기존 cosine similarity와
  같은 범위·방향의 점수로 간주하지 않고 순위에만 쓴다.
- collection metadata에 `embeddingModel`, `embeddingDimension`, `indexVersion`,
  `distanceMetric`, `schemaVersion`을 저장한다. 런타임 설정과 다르면 구체적인
  불일치 내용과 collection 전환 안내를 내고 검색을 중단한다.
- collection이 없거나 비어 있으면 경고를 남기고 빈 근거를 반환한다. 요청 처리
  중 자동 재색인하지 않는다. DB open/query 실패는 `ChromaStoreError`로 0건과 구분되어
  기존 API 502 처리로 연결된다.

## 3. 레코드 메타데이터와 ID

기존 `Document` 메타데이터를 유지한다.

```text
documentTitle, sourceType, section, page, industries, sourcePath,
documentId, chunkIndex
```

Chroma 내부에서 `industries`는 JSON 문자열로 보존하고, 필터용 `industryAll` 및
업종별 bool 필드를 파생한다. 이 방식은 배열 metadata를 받지 않던 Chroma 버전과도
호환되며, 검색 결과를 `Document`로 복원할 때 다시 목록으로 복원한다.

- `documentId = SHA-256(sourcePath)` 접두사/ucd95약값
- `chunkId = SHA-256(indexVersion + sourcePath + chunkIndex + contentHash)`

같은 문서·청킹·색인 버전은 같은 ID를 만들어 `upsert`를 반복해도 중복되지 않는다.
문서 갱신은 새 임베딩 생성이 성공한 뒤 해당 `sourcePath`의 예전 청크를 모두 지우고
새 청크를 넣어, 문서가 짧아졌을 때 끝의 예전 청크가 남지 않게 한다.

## 4. 검색 의미 보존

`retrieve_relevant_chunks(query_text, top_k=5, pinned_sections=(), industry_id=None)`의
호출 계약을 유지했다.

- 업종이 없으면 모든 자료를 허용한다.
- `*`는 전 업종 공통이다.
- 예전 색인처럼 `industries`가 없거나 비어 있으면 업종 공통으로 통과시킨다.
- Chroma는 scalar metadata 문자열 부분 일치를 지원하지 않는다. `section`이
  `page_content` 문맥 머리글에 포함되므로 `where_document $contains`로 pinned 후보를
  좁히고, 반환된 `section` metadata에 기존 부분 문자열 판정을 다시 적용한다.
- pinned section 하나당 중복이 아닌 최상위 청크 하나를 먼저 보장한 뒤 일반 검색
  결과를 합친다. pinned을 단순 일반 유사도 검색으로 대체하지 않았다.
- page content로 중복을 제거하고, 일반 결과에는 기존 `MAX_PER_DOCUMENT=2`와
  `candidate_pool=max(top_k*8, 40)` 정책을 유지한다.
- `embed_query()`는 호출당 한 번만 실행하고 같은 벡터를 모든 pinned 및 일반
  `query_embeddings`에 재사용한다.

실시간 추천 `recommendation_graph` 및 물어보기 `ask_graph`는 모두 같은 검색
함수를 계속 호출한다. full/compact 프롬프트와 구조화 출력 모델, 고정 안전
분기는 변경하지 않았다. `ground_citations()`는 인용문이 실제 검색 청크에 있을
때만 metadata의 문서명·조항·종류로 보정해 반환한다.

## 5. 설치와 초기 이전

```powershell
cd Backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 기존 JSON 벡터 재사용

`InMemoryVectorStore.dump` JSON은 `text`, `metadata`, `vector`를 갖고 있어 Chroma에 직접
upsert할 수 있다. 이 경로는 OpenAI 임베딩 API를 호출하지 않으며 기존 JSON을
삭제하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m app.rag.manage migrate-json `
  --from-json app/rag/vector_store.json `
  --collection est_ai_knowledge `
  --embedding-model text-embedding-3-small
```

JSON은 생성 모델 이름을 저장하지 않으므로 `--embedding-model`이 실제 모델인지
확인해야 한다. 벡터 차원은 JSON에서 검증한다. 차원이 섞여 있거나 collection의
모델·차원·색인 버전이 다르면 이전을 중단한다. 같은 JSON을 다시 실행해도
안정 ID `upsert`로 중복되지 않는다.

현재 체크아웃에는 `app/rag/vector_store.json`이 없어 실데이터 이전은 실행하지 못했다.
파일이 없거나 파일 형식/모델을 확신할 수 없으면 아래 재색인을 사용한다.

### 원본 문서 전체 색인

먼저 비용 발생 대상을 확인한다. 아래 명령은 임베딩을 호출하지 않는 preview다.

```powershell
.\.venv\Scripts\python.exe -m app.rag.manage reindex
```

현재 원본은 553개 청크다. 대상·모델·새 collection 이름을 확인한 뒤:

```powershell
.\.venv\Scripts\python.exe -m app.rag.manage reindex --yes
```

collection 이름을 생략하면 `est_ai_knowledge__YYYYMMDDHHMMSS` 형식의 새 후보를
만들고 저장 건수를 검증한다. 검색 테스트 후 `.env`의 `CHROMA_COLLECTION`을
후보 이름으로 바꾸고 API 프로세스를 재시작한다. 기존 정상 collection을 먼저 지우지 않는다.

## 6. 운영 명령

```powershell
# 특정 문서 추가/갱신 preview 및 실행
.\.venv\Scripts\python.exe -m app.rag.manage upsert-document "문서.md"
.\.venv\Scripts\python.exe -m app.rag.manage upsert-document "문서.md" --yes

# 특정 문서 청크 전체 삭제 preview 및 실행
.\.venv\Scripts\python.exe -m app.rag.manage delete-document "문서.md"
.\.venv\Scripts\python.exe -m app.rag.manage delete-document "문서.md" --yes

# 전체 collection 또는 특정 collection 건수/설정 확인
.\.venv\Scripts\python.exe -m app.rag.manage status
.\.venv\Scripts\python.exe -m app.rag.manage status --collection est_ai_knowledge
```

`source_path`는 `knowledge-sources/` 기준 `/` 구분자 상대경로로 `SOURCE_REGISTRY`의 키와 같다.

## 7. JSON 대 Chroma 검색 비교

다음 명령은 대표 질문 4개에 대해 질의 임베딩을 각 1번만 생성하고, 같은
벡터를 기존 JSON exact cosine 검색과 Chroma 검색에 넣어 문서·section·검색 시간을
출력한다.

```powershell
.\.venv\Scripts\python.exe -m app.rag.manage compare
# JSON/collection 건수와 질의 API 호출 안내 확인 후
.\.venv\Scripts\python.exe -m app.rag.manage compare --yes
```

기본 질문:

1. 상한 식품의 환불 문의
2. 소비자가 보관을 잘못한 경우
3. 여러 개 구매 후 일부만 문제가 생긴 경우
4. 해당 업종과 관계없는 질문

Chroma는 HNSW 근사 최근접 검색이므로 exact cosine인 기존 InMemory 순서와 다를 수
있다. 순서가 다르면 distance 공간, 업종/document 필터, HNSW 근사 검색 오차를
나눠 확인한다. 현재는 실제 JSON과 완성 collection이 없어 이 비교를 실행하지 못했다.

## 8. 검증 결과

2026-09-17 실행:

- Python 3.12.14 `compileall`: 통과
- `app.rag.chroma_store`, `retrieve`, `ingest`, `manage`, 두 graph, citations import: 통과
- Chroma 저장 후 별도 Python 프로세스에서 collection 재조회: 통과
- 샘플 InMemory JSON의 청크 수·원문·metadata 보존: 통과
- 샘플 JSON 이전 명령 로직 재실행 시 중복 없음: 통과
- 편의점 전용, `*` 공통, industries 없는 예전 자료 검색: 통과
- pinned section 검색, 겹치는 pinned 중복 제거: 통과
- 문서 교체 후 예전 청크 제거, 문서 삭제: 통과
- 질의 임베딩이 호출당 1회임: 통과
- 고정 안전 분기에서 검색·LLM 미실행: 통과
- citation quote 원문 대조 및 metadata 보정, 미검증 citation 제외: 통과
- 원본 청킹 preview: 553개 확인, 유료 임베딩은 미실행

실행 명령:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m app.rag.manage reindex
```

남은 실데이터 확인:

- 현재 체크아웃에 기존 `vector_store.json`이 없어 553개 실데이터 이전 전후 비교 불가
- 유료 API 승인 전이므로 553개 원본 재색인 및 대표 질문 실제 OpenAI 비교 미실행
- 샘플 벡터 DB로 기능 행동을 검증했으며, 운영 collection을 만든 후 `compare --yes`로
  근사 검색 순위와 실제 응답 citation을 추가 확인해야 한다.

## 9. 기존 JSON 방식으로 복구

이전 명령은 `vector_store.json`을 삭제하지 않는다. 빠른 롤백이 필요하면 Chroma
전환 커밋을 `git revert <commit>`하고, 기존 JSON을 `app/rag/vector_store.json`에 두며,
이전 `requirements.txt`을 재설치한 뒤 API를 재시작한다. 커밋 전 로컬 검증 중이면 다음
파일만 기준 브랜치 내용으로 복원한다.

```text
app/config.py
app/rag/ingest.py
app/rag/retrieve.py
app/graph/citations.py
requirements.txt
```

DB 디렉터리는 롤백 후에도 자동 삭제하지 않는다. 필요 시 별도 백업 후 삭제한다.
