"""
지식베이스 임베딩 사전 계산 스크립트.
knowledge-sources/ 안의 문서를 재귀 순회하며 청크로 나누고 OpenAI 임베딩을 계산해
app/rag/vector_store.json 으로 저장한다 (LangChain InMemoryVectorStore.dump).
문서가 바뀔 때만 다시 실행하면 되고, 런타임(API)은 이 파일을 로드만 한다.

지원 형식: .md 뿐이다. 원본 PDF·DOCX는 전부 Markdown으로 변환해서 넣는다(변환본만 저장소에
둔다). PDF 텍스트 추출은 표를 줄바꿈된 평문으로 뭉개서 "분쟁유형 / 해결기준 / 비고"의 열 대응이
사라졌는데, Markdown 표는 이 대응이 그대로 남아 LLM이 행·열을 읽을 수 있다.
문서별 제목·분류·업종범위·인덱싱 여부는 아래 SOURCE_REGISTRY에서 관리한다.
파일명은 원본 자료의 이름을 그대로 쓴다(별표 번호·고시 번호가 그대로 드러나는 편이 대조하기 쉽다).
레지스트리에 없는 파일도 인덱싱되지만 경고를 남기므로, 새 자료를 넣으면 레지스트리에 등록할 것.
"_" 로 시작하는 하위 폴더는 수집 대상에서 통째로 제외된다.

실행: python -m app.rag.ingest
"""

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


# 키는 knowledge-sources/ 기준 상대경로(구분자는 "/"). 파일명을 바꾸면 여기 키도 같이 바꾼다.
SOURCE_REGISTRY: dict[str, SourceSpec] = {
    "소비자분쟁해결기준_개정안_전문_본문.md": SourceSpec(
        title="소비자분쟁해결기준 (고시 본문)",
        source_type="standard",
    ),
    "[별표1] [별표 2] 품목별_해결기준_2024-12-27.md": SourceSpec(
        title="소비자분쟁해결기준 [별표Ⅰ] 대상품목 · [별표Ⅱ] 품목별 해결기준",
        source_type="standard",
    ),
    "식품등의_표시기준_제2025-60호_2025-08-29.md": SourceSpec(
        title="식품등의 표시기준 (식약처 고시 제2025-60호)",
        source_type="notice",
        industries=("convenience_store", "restaurant_cafe", "online_shopping"),
    ),
    "고객응대근로자_건강보호_가이드_2019.md": SourceSpec(
        title="고객응대근로자 건강보호 가이드 (2019)",
        source_type="guide",
    ),
    # 마트 계산대 응대를 전제로 쓰인 표준 매뉴얼이라 편의점·마트로 한정한다. 여기 담긴
    # 문제행동 고객 유형·건강보호 조치의 일반론은 위 건강보호 가이드(전 업종)가 이미 덮는다.
    "감정노동매뉴얼_마트계산원_2019_정리본.md": SourceSpec(
        title="고객응대근로자 건강보호 업종별 매뉴얼 - 마트계산원 (2019)",
        source_type="guide",
        industries=("convenience_store",),
    ),
    "[별표4] [별표3] 품목별 품질보증기간 및 부품보유기간(2024.12.27.시행).md": SourceSpec(
        title="소비자분쟁해결기준 [별표Ⅲ][별표Ⅳ] 품질보증기간 및 부품보유기간",
        source_type="standard",
        industries=(),
        indexed=False,
        reason="자동차·가전 등 공산품 품질보증기간표. 현재 온보딩 업종에 대응 업종이 없어 검색 노이즈가 됨",
    ),
    "README.md": SourceSpec(
        title="지식베이스 원본 자료 안내",
        source_type="manual",
        indexed=False,
        reason="인덱싱 정책 설명 문서",
    ),
}

# "_" 로 시작하는 폴더(_archive 등)는 통째로 건너뛴다. 변환 전 원본처럼
# 보관만 하는 자료를 레지스트리에 한 줄씩 제외 등록하지 않아도 되게 하기 위함이다.
IGNORED_DIR_PREFIX = "_"

SUPPORTED_SUFFIXES = {".md"}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_CHARS,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "다. ", " ", ""],
    keep_separator=True,
)


def split_text(text: str) -> list[str]:
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


# ---------------------------------------------------------------- Markdown 로더

# PDF를 변환할 때 남겨 둔 원문 쪽 표시. 두 가지 형태로 들어와 있다.
#   "## 원문 8쪽"          (소비자분쟁해결기준 별표Ⅰ·Ⅱ)
#   "<!-- 원문 PDF 3쪽 -->" (식품등의 표시기준)
# 이건 문서의 목차가 아니라 위치 정보다. 섹션 제목으로 쓰면 "2. 식료품(19개 업종)" 같은
# 진짜 제목을 덮어써서 근거 위치 표시가 "원문 8쪽"이 돼버리므로, 제목 대신 page로 뺀다.
PAGE_MARKER = re.compile(r"^\s*원문\s*(?:PDF\s*)?(\d+)\s*쪽\s*$")
PAGE_COMMENT = re.compile(r"^\s*<!--\s*원문\s*(?:PDF\s*)?(\d+)\s*쪽\s*-->\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_ROW = re.compile(r"^\s*\|")
# | --- | :---: | 형태의 구분행. 표의 헤더와 본문을 가르는 줄이다.
TABLE_DIVIDER = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
# 변환본은 원문 쪽 경계마다 수평선을 넣어 둔다. 내용이 없으므로 청크로 만들지 않는다.
HORIZONTAL_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")

# 표 조각 앞에 다시 붙일 도입부(예: "**청량음료, 과자류, 빙과류, ...**")의 길이 상한.
# 별표Ⅱ는 품종 목록을 표 바로 위에 굵은 글씨로 적어 두는데, 정작 "요구르트"·"빵" 같은
# 실제 품목명은 표 본문이 아니라 이 줄에만 나온다. 표만 따로 청크가 되면 그 단서가 끊긴다.
TABLE_LEAD_IN_CHARS = 200

# 이 길이 미만인 섹션만 "형제 섹션과 합쳐도 되는 조각"으로 본다(merge_units 참고).
MERGE_SMALL_CHARS = 300

# 근거 위치(citations[].section)에 담을 제목 한 마디의 길이 상한(heading_trail 참고).
HEADING_MAX_CHARS = 40


def strip_frontmatter(markdown: str) -> str:
    if not markdown.startswith("---"):
        return markdown
    end = markdown.find("---", 3)
    return markdown if end == -1 else markdown[end + 3 :]


def heading_trail(path: list[tuple[int, str]]) -> str | None:
    """헤딩 경로를 "상위 > 하위" 한 줄로 만든다.

    직전 헤딩 하나만 남기면 위치가 모호해진다. 별표Ⅱ의 "1) 완제품"이나 "가) 문구 게시 장소"는
    그 자체로는 어느 품목·어느 조치의 이야기인지 알 수 없고, citations[].section에 그대로
    실려서 사용자에게도 그렇게 보인다. 상위 제목을 함께 남기면 "5. 가전제품 ... > 1) 완제품"이 된다.

    제목이 길면 줄여서 담는다. 식품등의 표시기준 변환본은 원문에서 줄바꿈된 항목이 그대로
    헤딩이 돼서 ("다. “제조연월일”이라 함은 포장을 제외한 더 이상의 제조나 가공이") 문장
    중간에서 끊긴 제목이 많다. 근거 위치로 보여주기에는 길기만 하고, 본문에는 온전한
    헤딩 줄이 그대로 남아 있어 LLM이 읽는 내용은 줄지 않는다.
    """
    parts = [
        text if len(text) <= HEADING_MAX_CHARS else text[:HEADING_MAX_CHARS].rstrip() + "…"
        for _, text in path
    ]
    return " > ".join(parts) or None


def split_markdown_body(body: str) -> list[str]:
    """섹션 본문을 청크로 나눈다. Markdown 표는 행 경계를 지키고 헤더행을 반복한다.

    표를 일반 텍스트로 취급하면 700자에서 행 한가운데가 잘려서, 뒤 청크는 헤더행 없이
    "| 2) 부패, 변질 | o 제품교환 또는 구입가 환급 |"만 남는다. 열 이름이 사라지면 LLM은
    두 번째 칸이 해결기준인지 비고인지 알 수 없다. 그래서 표는 별도로 다뤄서,
    잘릴 때마다 헤더행(+구분행)과 도입부를 각 조각 앞에 다시 붙인다.
    """
    chunks: list[str] = []
    lines = body.split("\n")
    index = 0
    text_buffer: list[str] = []

    def flush_text(before_table: bool = False) -> str:
        """쌓인 평문을 청크로 내보낸다. 표 앞이면 마지막 문단을 도입부로 떼어 돌려준다."""
        text = "\n".join(text_buffer).strip()
        text_buffer.clear()
        if not text:
            return ""
        if before_table:
            head, _, lead_in = text.rpartition("\n\n")
            lead_in = lead_in.strip()
            if lead_in and len(lead_in) <= TABLE_LEAD_IN_CHARS:
                # 도입부는 표 청크마다 앞에 붙으므로 여기서 따로 내보내지 않는다(중복 방지).
                if head.strip():
                    chunks.extend(split_text(head))
                return lead_in
        chunks.extend(split_text(text))
        return ""

    while index < len(lines):
        if not TABLE_ROW.match(lines[index]):
            if not HORIZONTAL_RULE.match(lines[index]):
                text_buffer.append(lines[index])
            index += 1
            continue

        start = index
        while index < len(lines) and TABLE_ROW.match(lines[index]):
            index += 1
        table = lines[start:index]

        lead_in = flush_text(before_table=True)
        header = table[:2] if len(table) > 1 and TABLE_DIVIDER.match(table[1]) else []
        rows = table[len(header) :]
        prefix = ([lead_in] if lead_in else []) + header
        prefix_text = "\n".join(prefix)

        # 헤더행을 매번 붙이면서 CHUNK_CHARS를 넘지 않는 선까지 행을 모은다.
        # 행 하나가 이미 상한을 넘으면(별표Ⅱ 공산품 행은 1,600자가 넘는다) 자르지 않고
        # 통째로 한 청크에 둔다 — 행 중간을 자르면 표가 아니라 부서진 문자열이 된다.
        group: list[str] = []
        for row in rows:
            candidate = group + [row]
            length = len(prefix_text) + sum(len(line) + 1 for line in candidate)
            if group and length > CHUNK_CHARS:
                chunks.append("\n".join(prefix + group).strip())
                group = [row]
            else:
                group = candidate
        if group:
            chunks.append("\n".join(prefix + group).strip())

    flush_text()
    return [chunk for chunk in chunks if chunk]


def common_path(left: list[tuple[int, str]], right: list[tuple[int, str]]) -> list[tuple[int, str]]:
    shared: list[tuple[int, str]] = []
    for a, b in zip(left, right):
        if a != b:
            break
        shared.append(a)
    return shared


def merge_units(units: list[dict]) -> list[dict]:
    """형제 섹션을 CHUNK_CHARS까지 이어 붙인다.

    Markdown 변환본은 헤딩이 촘촘하다. 식품등의 표시기준은 용어 정의 한 항목("가. 제품명이라
    함은...")마다 #### 헤딩이 붙어서, 헤딩마다 청크를 끊으면 60~200자짜리 조각 수백 개가 된다.
    조각이 잘면 근거 후보 5칸을 채워도 LLM에게 가는 본문이 얼마 안 되고, 바로 옆 항목에 있는
    단서가 끊긴다. 그래서 같은 상위 제목 아래 이웃한 섹션은 한도까지 합친다.

    다만 아무 섹션이나 합치면 안 된다. 합쳐진 청크의 위치 표시는 공통 상위 제목이 되는데,
    "2. 식료품(19개 업종)"처럼 그 자체로 내용이 있는 섹션까지 형제와 합쳐버리면 위치 표시가
    "품목별 해결기준"으로 뭉개진다. 그러면 이 섹션을 겨냥한 PINNED_SECTIONS가 못 찾는다.
    그래서 형제끼리는 양쪽 다 MERGE_SMALL_CHARS 미만인 조각일 때만 합치고,
    하위 섹션(예: 식료품 표 아래 "【참고】 식료품 품목 관련 법령")은 위치 표시가 바뀌지
    않으므로 크기와 무관하게 상위 청크에 붙인다.

    원문 쪽 경계는 합치는 것을 막지 않는다. 별표Ⅱ는 표가 쪽을 넘어가면 "① 전자제품 (2-1)",
    "(2-2)"로 갈라져 있는데, 쪽마다 끊으면 이 둘이 영영 따로 논다. 합쳐진 청크의 page는
    첫 조각의 쪽수로 둔다(citations에 "(p.13)"으로 나가는 값이라 시작 쪽이면 충분하다).
    """
    merged: list[dict] = []
    group: list[dict] = []
    shared: list[tuple[int, str]] = []

    def render() -> None:
        if not group:
            return
        lines: list[str] = []
        previous: list[tuple[int, str]] = shared
        for unit in group:
            # 공통 상위 제목 아래로 내려간 부분은 본문에 헤딩 그대로 남긴다. 그래야 합쳐진
            # 청크 안에서도 "가. / 나. / 다."의 경계가 Markdown으로 보인다.
            for level, text in unit["path"][len(common_path(previous, unit["path"])) :]:
                lines.append(f"{'#' * level} {text}")
            lines.append(unit["content"])
            previous = unit["path"]
        merged.append(
            {
                "section": heading_trail(shared),
                "content": "\n\n".join(lines).strip(),
                "page": group[0]["page"],
            }
        )

    for unit in units:
        if group:
            candidate = common_path(shared, unit["path"])
            size = sum(len(item["content"]) for item in group) + len(unit["content"])
            # 공통 경로가 그대로면 unit은 현재 청크 제목의 하위 섹션이다(위치 표시가 안 바뀜).
            keeps_section = len(candidate) == len(shared)
            small_fragments = len(unit["content"]) < MERGE_SMALL_CHARS and all(
                len(item["content"]) < MERGE_SMALL_CHARS for item in group
            )
            if candidate and size <= CHUNK_CHARS and (keeps_section or small_fragments):
                group.append(unit)
                shared = candidate
                continue
            render()
        group = [unit]
        shared = unit["path"]
    render()
    return merged


def load_markdown(path: Path) -> list[dict]:
    """헤딩(#~######)을 섹션 경계로 삼고, 헤딩 경로와 원문 쪽수를 함께 남긴다."""
    lines = strip_frontmatter(path.read_text(encoding="utf-8")).split("\n")
    units: list[dict] = []
    heading_path: list[tuple[int, str]] = []
    page: int | None = None
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        buffer.clear()
        if not content:
            return
        for part in split_markdown_body(content):
            units.append({"path": heading_path, "content": part, "page": page})

    for line in lines:
        comment = PAGE_COMMENT.match(line)
        if comment:
            flush()
            page = int(comment.group(1))
            continue

        heading = HEADING.match(line)
        if not heading:
            buffer.append(line)
            continue

        flush()
        level, text = len(heading.group(1)), heading.group(2).strip()
        marker = PAGE_MARKER.match(text)
        if marker:
            page = int(marker.group(1))
            continue
        # 같거나 더 얕은 단계의 제목은 그 아래 경로를 대체한다(## 다음의 ##는 형제).
        heading_path = [item for item in heading_path if item[0] < level]
        heading_path.append((level, text))

    flush()
    return merge_units(units)


LOADERS = {".md": load_markdown}


# ---------------------------------------------------------------- 수집


def discover_sources() -> list[tuple[Path, str, SourceSpec]]:
    """knowledge-sources/ 를 재귀 순회해 (경로, 상대키, 정책) 목록을 만든다."""
    discovered: list[tuple[Path, str, SourceSpec]] = []
    for path in sorted(KNOWLEDGE_SOURCES_DIR.rglob("*")):
        if not path.is_file():
            continue
        key = path.relative_to(KNOWLEDGE_SOURCES_DIR).as_posix()
        if any(part.startswith(IGNORED_DIR_PREFIX) for part in key.split("/")[:-1]):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            print(f"  [건너뜀] {key} - .md 가 아님. Markdown으로 변환해서 넣을 것")
            continue
        spec = SOURCE_REGISTRY.get(key)
        if spec is None:
            print(f"  [경고] 레지스트리에 없는 문서: {key} -> 기본값으로 인덱싱합니다.")
            spec = SourceSpec(title=path.stem, source_type="unclassified")
        discovered.append((path, key, spec))

    # 레지스트리에만 남은 항목은 파일을 지우거나 이름을 바꾸고 레지스트리를 안 고친 흔적이다.
    # 조용히 두면 그 문서를 겨냥한 PINNED_SECTIONS가 아무 것도 못 찾는 채로 돌아간다.
    found = {key for _, key, _ in discovered}
    for key in SOURCE_REGISTRY:
        if key not in found:
            print(f"  [경고] 레지스트리에 있으나 파일이 없음: {key}")
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
