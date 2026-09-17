"""LLM이 적어낸 근거(citations)를 실제 검색 결과와 대조해 검증·보정한다.

화면에는 "답변 근거 · 편의점 고객응대 실무 매뉴얼 · 식료품"처럼 문서 제목만 떴다. 직원
입장에서는 그 매뉴얼의 '무엇'이 이렇게 응대하라고 했는지 알 수 없어서 근거 구실을 못 한다.
그래서 LLM에게 근거가 된 문장(quote)을 본문에서 그대로 옮기게 하고, 여기서 그 문장이 정말
검색된 청크 안에 있는지 확인한다.

확인된 문장은 그 청크의 메타데이터로 label·section·sourceType까지 다시 채운다. LLM이 옮겨
적은 위치 표기(실측에서 "식료품 식료품"처럼 같은 말을 두 번 적는 경우가 잦았다)보다 색인
당시의 메타데이터가 정확하기 때문이다.

문장이 확인되지 않으면 문서명·조항·인용문 전체를 응답에서 제외한다.
검색 결과에 없는 근거 표시를 남기는 것보다 근거 없음을 드러내는 편이 안전하다.
"""

import re

from langchain_core.documents import Document

from app.schemas import Citation, CitationDraft

# 카드 한 장에 들어갈 길이. 청크가 700자라 그대로 두면 근거 문장이 답변보다 길어진다.
MAX_QUOTE_CHARS = 200

# 이보다 짧은 인용은 검증이 의미가 없다. "환급" 두 글자는 아무 문서에나 들어 있다.
MIN_QUOTE_CHARS = 8

# 대조할 때 무시할 문자. 모델이 인용 부호를 붙이거나 줄바꿈·띄어쓰기를 바꿔 적는 정도는
# 원문을 옮긴 것으로 본다. 그 이상(단어를 바꾸거나 요약)은 통과시키지 않는다.
_IGNORED = re.compile(r"[\s\"'“”‘’「」『』()\[\]…·ㆍ‧,]+")

_MARKDOWN_EMPHASIS = re.compile(r"\*\*|__|`")


def _normalize(text: str) -> str:
    return _IGNORED.sub("", text)


def _shorten(quote: str) -> str:
    """줄바꿈을 펴고 카드 길이에 맞춰 자른다. 자를 때는 문장 끝을 우선한다."""
    # 화면에서는 형광펜 자체가 인용 표시라, 모델이 감싸 적은 따옴표는 떼고 보여준다.
    flattened = " ".join(quote.split()).strip("“”‘’\"'")
    # 매뉴얼이 마크다운이라 원문을 그대로 옮기면 강조 기호가 따라온다("**1차 경고:**").
    # 대조는 이미 끝났으므로 화면에 보일 때만 떼어낸다.
    flattened = _MARKDOWN_EMPHASIS.sub("", flattened)
    if len(flattened) <= MAX_QUOTE_CHARS:
        return flattened
    head = flattened[:MAX_QUOTE_CHARS]
    boundary = head.rfind(".")
    if boundary >= MAX_QUOTE_CHARS // 2:
        return head[: boundary + 1].strip()
    return head.strip() + "…"


def _describe_location(document: Document) -> str | None:
    """청크 메타데이터를 화면에 쓸 위치 표기로 만든다(예: "3. 용어의 정의 (p.4)")."""
    meta = document.metadata
    section = (meta.get("section") or "").strip() or None
    page = meta.get("page")
    if section and page:
        return f"{section} (p.{page})"
    if section:
        return section
    if page:
        return f"p.{page}"
    return None


def ground_citations(
    citations: list[CitationDraft], retrieved: list[Document]
) -> list[Citation]:
    """인용 문장이 검색된 청크에 실제로 있는 것만 quote로 남기고, 출처 표기를 메타데이터로 교정한다."""
    indexed = [(_normalize(document.page_content), document) for document in retrieved]
    grounded: list[Citation] = []
    seen: set[tuple] = set()

    for citation in citations:
        quote = (citation.quote or "").strip()
        needle = _normalize(quote)
        source: Document | None = None
        if len(needle) >= MIN_QUOTE_CHARS:
            source = next(
                (document for content, document in indexed if needle in content), None
            )

        if source is None:
            continue
        resolved = Citation(
            label=source.metadata.get("documentTitle") or citation.label,
            section=_describe_location(source) or citation.section,
            quote=_shorten(quote),
            sourceType=source.metadata.get("sourceType"),
        )

        key = (resolved.label, resolved.section, resolved.quote)
        if key in seen:
            continue
        seen.add(key)
        grounded.append(resolved)

    return grounded
