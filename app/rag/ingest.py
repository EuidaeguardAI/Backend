"""
지식베이스 임베딩 사전 계산 스크립트.
knowledge-sources/ 안의 문서를 재귀 순회하며 청크로 나누고 OpenAI 임베딩을 계산해
app/rag/vector_store.json 으로 저장한다 (LangChain InMemoryVectorStore.dump).
문서가 바뀔 때만 다시 실행하면 되고, 런타임(API)은 이 파일을 로드만 한다.

지원 형식: .md, .docx, .pdf
문서별 제목·분류·업종범위·인덱싱 여부는 아래 SOURCE_REGISTRY에서 관리한다.
레지스트리에 없는 파일도 인덱싱되지만 경고를 남기므로, 새 자료를 넣으면 레지스트리에 등록할 것.

실행: python -m app.rag.ingest
"""

import re
from dataclasses import dataclass
from pathlib import Path

import docx2txt
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import EMBEDDING_MODEL, KNOWLEDGE_SOURCES_DIR, VECTOR_STORE_PATH

CHUNK_CHARS = 700
CHUNK_OVERLAP = 120

# 온보딩(src/lib/mock/onboardingOptions.ts)의 industryId와 같은 값을 쓴다.
ALL_INDUSTRIES = ("*",)


@dataclass(frozen=True)
class SourceSpec:
    """문서 한 건의 인덱싱 정책."""

    title: str
    source_type: str  # manual | law | standard | notice | guide
    industries: tuple[str, ...] = ALL_INDUSTRIES
    indexed: bool = True
    reason: str = ""  # indexed=False 인 이유


# 키는 knowledge-sources/ 기준 상대경로(구분자는 "/").
SOURCE_REGISTRY: dict[str, SourceSpec] = {
    "편의점_고객응대_실무매뉴얼.md": SourceSpec(
        title="편의점 고객응대 실무 매뉴얼",
        source_type="manual",
        industries=("convenience_store",),
    ),
    "한국_고객응대근로자_보호_법령_가이드_조사.md": SourceSpec(
        title="한국 고객응대근로자 보호 법령 가이드",
        source_type="law",
    ),
    "한국_고객응대근로자_보호_법령_가이드_조사.docx": SourceSpec(
        title="한국 고객응대근로자 보호 법령 가이드 (docx)",
        source_type="law",
        indexed=False,
        reason="같은 문서의 .md 판이 더 완전하고 헤딩 구조가 있어 중복 제외",
    ),
    "refund/소비자기본법 시행령 [별표1] 일반적 소비자분쟁해결기준.pdf": SourceSpec(
        title="소비자기본법 시행령 [별표1] 일반적 소비자분쟁해결기준",
        source_type="standard",
    ),
    "refund/1. 소비자분쟁해결기준 개정안 전문(본문).pdf": SourceSpec(
        title="소비자분쟁해결기준 (고시 본문)",
        source_type="standard",
    ),
    "refund/[별표1] [별표 2] 품목별 해결기준(2024.12.27 시행).pdf": SourceSpec(
        title="소비자분쟁해결기준 [별표Ⅰ] 대상품목 · [별표Ⅱ] 품목별 해결기준",
        source_type="standard",
    ),
    "refund/식품등의 표시기준(제2025-60호, 2025.8.29.).pdf": SourceSpec(
        title="식품등의 표시기준 (식약처 고시 제2025-60호)",
        source_type="notice",
        industries=("convenience_store", "restaurant_cafe", "online_shopping"),
    ),
    "refund/고객응대근로자_건강보호_가이드(2019).pdf": SourceSpec(
        title="고객응대근로자 건강보호 가이드 (2019)",
        source_type="guide",
    ),
    "refund/[별표4] [별표3] 품목별 품질보증기간 및 부품보유기간(2024.12.27.시행).pdf": SourceSpec(
        title="소비자분쟁해결기준 [별표Ⅲ][별표Ⅳ] 품질보증기간 및 부품보유기간",
        source_type="standard",
        industries=(),
        indexed=False,
        reason="자동차·가전 등 공산품 품질보증기간표. 현재 온보딩 업종에 대응 업종이 없어 검색 노이즈가 됨",
    ),
    "refund/응대가이드_작성_프롬프트.md": SourceSpec(
        title="응대가이드 작성 프롬프트",
        source_type="manual",
        indexed=False,
        reason="지식이 아니라 응대가이드 작성용 프롬프트 템플릿",
    ),
    "README.md": SourceSpec(
        title="지식베이스 원본 자료 안내",
        source_type="manual",
        indexed=False,
        reason="인덱싱 정책 설명 문서",
    ),
}

SUPPORTED_SUFFIXES = {".md", ".docx", ".pdf"}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_CHARS,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "다. ", " ", ""],
    keep_separator=True,
)


def split_text(text: str) -> list[str]:
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


# ---------------------------------------------------------------- 형식별 로더


def strip_frontmatter(markdown: str) -> str:
    if not markdown.startswith("---"):
        return markdown
    end = markdown.find("---", 3)
    return markdown if end == -1 else markdown[end + 3 :]


def load_markdown(path: Path) -> list[dict]:
    """헤딩(#~####)을 섹션 경계로 삼아 나눈 뒤, 긴 섹션만 추가 분할한다."""
    lines = strip_frontmatter(path.read_text(encoding="utf-8")).split("\n")
    sections: list[dict] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush():
        content = "\n".join(buffer).strip()
        if content:
            sections.append({"section": heading, "content": content})
        buffer.clear()

    for line in lines:
        match = re.match(r"^(#{1,4})\s+(.*)$", line)
        if match:
            flush()
            heading = match.group(2).strip()
            continue
        buffer.append(line)
    flush()

    result: list[dict] = []
    for section in sections:
        for part in split_text(section["content"]):
            result.append({"section": section["section"], "content": part, "page": None})
    return result


def load_docx(path: Path) -> list[dict]:
    text = docx2txt.process(str(path))
    return [{"section": None, "content": part, "page": None} for part in split_text(text)]


# PDF 본문에서 섹션 제목으로 인정할 패턴.
# 문서마다 체계가 다르다. 소비자분쟁해결기준은 <별표 Ⅱ> + "2. 식료품(19개 업종)",
# 식품등의 표시기준은 조문이 아니라 "Ⅰ. 총 칙" + "3. 용어의 정의" 구조를 쓴다.
# "제N조(제목)"은 괄호 제목이 붙은 경우만 인정한다. 괄호 없는 "제7조제1항"은
# 다른 법령을 인용한 것이지 이 문서의 조문이 아니기 때문이다.
SECTION_PATTERNS = [
    re.compile(r"<\s*별\s*표\s*[ⅠⅡⅢⅣⅤIVX0-9]+\s*>"),
    re.compile(r"\[\s*별\s*표\s*[ⅠⅡⅢⅣⅤIVX0-9]+\s*\]"),
    re.compile(r"제\s*\d+\s*조\s*\([^)\n]{1,40}\)"),
    re.compile(r"^\s*\d{1,2}\.\s*[가-힣·ㆍ‧\s]{2,25}\(\s*\d+\s*개\s*업종\s*\)", re.MULTILINE),
    re.compile(r"^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧ]\s*\.\s*[^\n]{1,20}$", re.MULTILINE),
    # 짧은 제목만 인정한다. 길게 두면 "2. 일부 멸실된 때는 인도일의 인도" 같은 표 안의
    # 행이 제목으로 잡혀 섹션 표시가 오히려 틀려진다.
    re.compile(r"^\s*\d{1,2}\.\s*[가-힣][가-힣\s]{1,9}$", re.MULTILINE),
]


def find_headings(text: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for pattern in SECTION_PATTERNS:
        for match in pattern.finditer(text):
            found.append((match.start(), re.sub(r"\s+", " ", match.group()).strip()))
    return sorted(found, key=lambda item: item[0])


def load_pdf(path: Path) -> list[dict]:
    """페이지 단위로 읽되, 페이지 안에서 조문·별표 제목이 나오면 그 지점에서 섹션을 끊는다."""
    reader = PdfReader(str(path))
    result: list[dict] = []
    current_section: str | None = None

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        # (시작offset, 섹션명) 구간으로 페이지를 쪼갠다. 첫 제목 앞부분은 직전 섹션에 이어붙인다.
        boundaries = [(0, current_section)] + find_headings(text)
        for index, (start, section) in enumerate(boundaries):
            end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(text)
            segment = text[start:end].strip()
            if not segment:
                continue
            for part in split_text(segment):
                result.append({"section": section, "content": part, "page": page_number})
            current_section = section

    return result


LOADERS = {".md": load_markdown, ".docx": load_docx, ".pdf": load_pdf}


# ---------------------------------------------------------------- 수집


def discover_sources() -> list[tuple[Path, str, SourceSpec]]:
    """knowledge-sources/ 를 재귀 순회해 (경로, 상대키, 정책) 목록을 만든다."""
    discovered: list[tuple[Path, str, SourceSpec]] = []
    for path in sorted(KNOWLEDGE_SOURCES_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        key = path.relative_to(KNOWLEDGE_SOURCES_DIR).as_posix()
        spec = SOURCE_REGISTRY.get(key)
        if spec is None:
            print(f"  [경고] 레지스트리에 없는 문서: {key} -> 기본값으로 인덱싱합니다.")
            spec = SourceSpec(title=path.stem, source_type="unclassified")
        discovered.append((path, key, spec))
    return discovered


def contextualize(spec: SourceSpec, chunk: dict) -> str:
    """청크 앞에 문서명·섹션을 붙여서 임베딩한다.

    법령 표 본문은 "제품교환 또는 구입가 환급"처럼 문구가 건조해서, 직원이 실제로 말하는
    "요구르트가 상했다는데 환불해달래요" 같은 질의와 벡터 거리가 멀다. 문서명과 섹션
    ("소비자분쟁해결기준 ... 2. 식료품(19개 업종)")을 본문에 포함시키면 이 간극이 줄어든다.
    """
    header = spec.title
    if chunk["section"]:
        header = f"{header} · {chunk['section']}"
    return f"[{header}]\n{chunk['content']}"


def build_documents() -> list[Document]:
    documents: list[Document] = []
    for path, key, spec in discover_sources():
        if not spec.indexed:
            print(f"  [제외] {key} - {spec.reason}")
            continue

        chunks = LOADERS[path.suffix.lower()](path)
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=contextualize(spec, chunk),
                    metadata={
                        "documentTitle": spec.title,
                        "sourceType": spec.source_type,
                        "section": chunk["section"],
                        "page": chunk["page"],
                        "industries": list(spec.industries),
                        "sourcePath": key,
                    },
                )
            )
        print(f"  [포함] {key} -> {len(chunks)}개 청크 ({spec.source_type})")
    return documents


def main() -> None:
    print(f"지식베이스 수집: {KNOWLEDGE_SOURCES_DIR}")
    documents = build_documents()
    print(f"\n총 {len(documents)}개 청크에 대해 임베딩을 계산합니다...")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = InMemoryVectorStore.from_documents(documents, embeddings)
    store.dump(str(VECTOR_STORE_PATH))
    print(f"완료: {len(documents)}개 청크를 {VECTOR_STORE_PATH}에 저장했습니다.")


if __name__ == "__main__":
    main()
