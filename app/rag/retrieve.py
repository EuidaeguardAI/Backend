from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from app.config import EMBEDDING_MODEL, VECTOR_STORE_PATH


@lru_cache(maxsize=1)
def get_vector_store() -> InMemoryVectorStore | None:
    if not VECTOR_STORE_PATH.exists():
        return None
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return InMemoryVectorStore.load(str(VECTOR_STORE_PATH), embeddings)


def _matches(document: Document, sections: tuple[str, ...]) -> bool:
    section = document.metadata.get("section") or ""
    return any(pinned in section for pinned in sections)


def _matches_industry(document: Document, industry_id: str | None) -> bool:
    """온보딩에서 고른 업종의 사정권 안에 있는 청크인지 본다.

    ingest가 청크마다 넣어둔 industries를 그대로 쓴다("*"는 전 업종 공통). 편의점 실무
    매뉴얼이 병원·부동산 세션의 근거 후보로 올라오는 것을 막는 용도다. 업종을 모르거나
    industries가 없는 옛 색인은 통과시킨다 - 걸러서 후보가 통째로 비는 쪽이 더 나쁘다.
    """
    if industry_id is None:
        return True
    industries = document.metadata.get("industries")
    if not industries:
        return True
    return "*" in industries or industry_id in industries


MAX_PER_DOCUMENT = 2


def retrieve_relevant_chunks(
    query_text: str,
    top_k: int = 5,
    pinned_sections: tuple[str, ...] = (),
    industry_id: str | None = None,
) -> list[Document]:
    """유사도 검색 결과에, 반드시 함께 봐야 할 조항(pinned_sections)을 앞에 붙여 돌려준다.

    직원이 실제로 하는 말("요구르트가 빵빵해졌대요")과 고시 표 본문("2) 부패, 변질 o 제품교환
    또는 구입가 환급")은 어휘가 너무 달라서, 벡터 검색만으로는 정작 근거가 되는 조항이 밀려난다.
    실측에서 이 조항은 법률용어 질의("식료품 부패 변질 환불 기준")로는 1위였지만 구어체
    질의로는 39위였다. 게다가 별표Ⅱ에는 "제품교환 또는 구입가 환급"이라고만 적힌 비슷한
    조항이 업종별로 수십 개라, 질의를 보강해도 엉뚱한 업종 조항이 대신 올라온다.

    그래서 품목이 사전에 정해지는 경우(편의점의 환불 분쟁이면 별표Ⅱ 식료품)에는 검색에
    맡기지 않고 해당 섹션을 직접 지정한다. 지정된 섹션 안에서는 여전히 유사도로 고른다.

    industry_id를 주면 pinned/유사도 양쪽 모두 해당 업종용 문서와 전 업종 공통 문서로만
    후보를 좁힌다(_matches_industry 참고).
    """
    store = get_vector_store()
    if store is None:
        return []

    documents: list[Document] = []
    seen: set[str] = set()

    # 섹션마다 따로 top-1을 뽑는다. 하나의 필터로 묶어 k=2를 뽑으면, 질의와 어휘가 가까운
    # 섹션(예: "식료품(19개 업종)")이 두 자리를 다 차지해서 어휘가 먼 섹션(예: 일반기준의
    # "소비자 취급 잘못" 조항)이 후보에서 아예 밀려난다. 지정한 섹션은 모두 최소 1개씩
    # 근거 후보에 들어가야 한다.
    for section in pinned_sections:
        # k=1이면 앞선 섹션에서 이미 뽑힌 청크와 우연히 같은 청크가 다시 1등으로 나올 때
        # (예: "식료품"과 "식료품(19개 업종)"처럼 섹션 문자열이 겹치는 경우) 이 섹션 몫이
        # 통째로 비어버린다. k=3으로 여유를 두고 아직 안 뽑힌 첫 번째 결과를 쓴다.
        for document in store.similarity_search(
            query_text,
            k=3,
            filter=lambda d: _matches(d, (section,)) and _matches_industry(d, industry_id),
        ):
            if document.page_content in seen:
                continue
            documents.append(document)
            seen.add(document.page_content)
            break

    # 한 문서가 자리를 독식하지 않게 문서당 상한을 둔다. 식품등의 표시기준처럼 청크가 많은
    # 고시는 상위 5칸을 전부 차지해버려서 정작 필요한 매뉴얼·해결기준이 밀려난다.
    per_document: dict[str, int] = {}
    candidate_pool = max(top_k * 8, 40)
    for document, _score in store.similarity_search_with_score(
        query_text, k=candidate_pool, filter=lambda d: _matches_industry(d, industry_id)
    ):
        if len(documents) >= top_k + len(pinned_sections):
            break
        if document.page_content in seen:
            continue
        title = document.metadata.get("documentTitle") or ""
        if per_document.get(title, 0) >= MAX_PER_DOCUMENT:
            continue
        per_document[title] = per_document.get(title, 0) + 1
        documents.append(document)
        seen.add(document.page_content)

    return documents
