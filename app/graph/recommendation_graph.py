"""
응대 추천 LangGraph.

    START (조건 분기)
      ├─ 위협 키워드/반복 폭언 감지 → fixed_safety → END   (생성형 아닌 고정 안전 절차)
      └─ 그 외                     → retrieve → generate → END  (RAG 검색 → 구조화 생성)

기획서의 권장 흐름(음성 입력 → 위험도 분류 → 긴급/일반 분기 → RAG 검색 → 응대 생성)을
그대로 그래프로 옮긴 것이다.
"""

import time
from collections.abc import Iterator
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.config import CHAT_MODEL, OPENAI_API_KEY
from app.graph.citations import ground_citations
from app.http_client import shared_http_client
from app.rag.retrieve import retrieve_relevant_chunks
from app.safety.emergency_rules import (
    build_fixed_safety_recommendation,
    describe_detected_risk,
    should_trigger_fixed_safety,
)
from app.schemas import (
    BusinessProfile,
    Recommendation,
    RecommendationDraft,
    SessionIntake,
    StoreKnowledgeItem,
    TranscriptTurn,
)

SYSTEM_PROMPT = """당신은 "응대가드 AI"의 응대 보조 엔진입니다. 고객응대근로자(편의점·서비스업 종사자)가 폭언·환불 분쟁·위협 상황에서
안전하게 대응하도록 돕습니다. 다음 원칙을 반드시 지키세요.

1. 당신은 법률 판단 기관이 아닙니다. "범죄가 성립한다", "고소 가능하다" 같은 확정적 법률 판단을 하지 마세요.
   대신 "해당 가능성이 있어 관리자·전문가 확인이 필요합니다"처럼 운영 안내 톤을 사용하세요.
2. 답변은 반드시 제공된 근거 문서(citations 후보) 안에서만 만드세요. 근거로 삼을 만한 내용이 없으면
   citations를 빈 배열로 두고 needsHumanReview를 true로 설정하세요. 근거를 지어내지 마세요.
2-1. citations를 채울 때 label에는 근거 문서의 제목을, section에는 근거 문서에 표시된 위치
   (예: "별표Ⅱ 2. 식료품(19개 업종)", "3. 용어의 정의 (p.4)")를 그대로 옮기세요. 위치를 임의로 지어내지 마세요.
2-2. 근거 문서에는 공식 고시·법령과 사내 실무 매뉴얼이 섞여 있습니다. 사내 매뉴얼의 내용을 법령상 의무인 것처럼
   말하지 말고, "점포 기준으로는", "매장 절차상"처럼 구분해서 안내하세요.
2-3. citations의 quote에는 그 답변의 근거가 된 문장을 근거 문서 본문에서 **한 글자도 바꾸지 말고**
   그대로 1~2문장 옮기세요. 요약하거나 말을 다듬으면 서버가 원문 대조에 실패해 근거 표시가 사라집니다.
   본문에 그대로 옮길 만한 문장이 없는 문서는 아예 인용하지 마세요.
3. sayNow는 1~2문장, 정중하고 짧게. 고객을 자극하는 표현은 doNot에 포함하세요.
4. 대화에 위협·협박·반복 폭언이 있으면 situation을 threat 또는 emergency로, riskLevel을 높게 설정하세요.
5. 확신이 낮거나(소음, 불명확한 발화 등) 정보가 부족하면 confidence를 낮추고 needsHumanReview를 true로 하세요.
6. 실제로 일어난 발언만 근거로 삼고, 발화자가 하지 않은 말을 지어내지 마세요.
7. 모든 출력(sayNow, nextActions, doNot, citations)은 **반드시 한국어**로만 작성하세요.
   고객이 다른 언어로 말했거나 인식 결과에 외국어가 섞여 있어도 답변은 한국어로 씁니다.
   영어 단어나 로마자 표기를 섞지 말고, 고유명사를 빼면 우리말 표현을 쓰세요.
8. "자동 감지된 위험 표현"이 함께 주어지면 위험도 판단의 참고 신호로만 쓰세요.
   그 단어를 sayNow에 되풀이해 적지 말고, 오탐일 수 있으니 실제 대화 맥락을 우선하세요.
9. sayNow나 nextActions에 손님에게 사실 확인을 요청하는 내용(예: 소비기한 경과 여부, 영수증 유무,
   파손 정도, 환불 사유 확인)이 있다면, 손님이 답할 법한 아주 짧은 구어체 문장 2~4개를
   expectedReplies에 제시하세요. 서로 다른 결론(예: "상했어요" / "안 상했어요"처럼 반대되는 답)이
   나오도록 다양하게 구성하세요. 사실 확인을 요청하는 내용이 전혀 없으면 expectedReplies는
   빈 배열로 두세요. expectedReplies는 손님 입장의 말투로만 쓰고, 직원의 안내문이 되지 않게 하세요.
10. 변질·파손·하자를 이유로 한 환불·교환 요구에서는, 그 원인이 판매자·제조사 쪽 하자인지 아니면
    고객 본인의 보관·취급 부주의(예: 냉장 보관 표시 제품을 상온에 방치했다고 스스로 말한 경우) 때문인지
    먼저 구분하세요. 고객 발화에 보관·취급 부주의를 시사하는 내용이 있으면 곧바로 교환·환급을
    안내하지 말고, "소비자 귀책사유로 발생한 손상은 사업자가 책임지지 않을 수 있다"는 근거 문서
    내용을 함께 반영해 보관 방법·방치 시간 등 사실관계부터 확인하도록 안내하세요. 이때도
    "환불 안 해줘도 됩니다" 같은 확정적 판단은 하지 말고, "확인이 필요합니다"처럼 운영 안내 톤을
    유지하세요.
11. 여러 개를 구매해 그중 일부는 이미 먹거나 쓴 뒤 나머지에서 문제를 제기하는 경우, 전체 수량을
    자동으로 환불 대상에 포함하지 마세요. 실제로 문제가 확인된 수량이 몇 개인지, 이미 먹거나 쓴
    부분에서도 이상(맛·냄새·몸 상태 등)이 있었는지부터 확인하도록 nextActions에 넣으세요.
12. "우리 매장 내규"가 함께 주어지면 그 매장의 실제 운영 기준으로 존중해 반영하되, 반드시
    "저희 매장 기준으로는", "이 매장 규정상"처럼 법령·공식 고시와 구분되게 말하세요.
    내규가 법령·공식 고시와 충돌하면 법령이 우선입니다. 그리고 내규는 공식 문서가 아니므로
    **citations에는 절대 넣지 마세요** — citations는 아래 "근거 문서 후보"에서만 만듭니다."""

PROBLEM_TYPE_LABEL = {
    "refund_exchange": "환불·교환",
    "damage_contamination": "파손·오염",
    "payment_price": "결제·가격",
    "child_guardian": "아동·보호자 항의",
    "store_usage_exit": "매장 이용·퇴거 요청",
    "staff_complaint": "직원 응대 불만",
    "other": "기타·잘 모르겠음",
}

BEHAVIOR_TYPE_LABEL = {
    "shouting_abuse": "고성·욕설",
    "threat": "협박",
    "throwing_damage": "투척·파손",
    "physical_risk": "신체 접촉·폭행 위험",
    "repeated_demand": "요구 반복",
    "manager_request": "관리자 요구",
    "none": "위험 행동 없음",
}


class GraphState(TypedDict):
    profile: BusinessProfile
    intake: SessionIntake
    recent_transcript: list[TranscriptTurn]
    recent_situations: list[str]
    latest_text: str
    store_knowledge: list[StoreKnowledgeItem]
    retrieved: list[Document]
    recommendation: Recommendation | None


def _next_id() -> str:
    return f"rec-{int(time.time() * 1000)}"


def _now_ms() -> int:
    return int(time.time() * 1000)


def route_after_entry(state: GraphState) -> str:
    if should_trigger_fixed_safety(state["latest_text"], state["recent_situations"]):
        return "fixed_safety"
    return "retrieve"


def fixed_safety_node(state: GraphState) -> dict:
    draft = build_fixed_safety_recommendation(state["latest_text"])
    recommendation = Recommendation(
        **draft.model_dump(),
        id=_next_id(),
        createdAtMs=_now_ms(),
        isFixedSafetyScript=True,
    )
    return {"recommendation": recommendation}


# 업종·문제유형이 정해지면 품목도 사실상 정해진다. 그 경우 근거 조항을 검색에 맡기지 않고
# 직접 지정한다(이유는 retrieve_relevant_chunks 참고). 값은 ingest가 붙인 section 문자열의 일부다.
PINNED_SECTIONS: dict[tuple[str, str], tuple[str, ...]] = {
    ("convenience_store", "refund_exchange"): ("식료품(19개 업종)", "[별표 1]", "식료품"),
    ("convenience_store", "damage_contamination"): ("식료품(19개 업종)",),
    ("restaurant_cafe", "refund_exchange"): ("식료품(19개 업종)", "[별표 1]", "식료품"),
}

# 마이크 경로는 UI 개편 이후 intake.problemTypes를 더 이상 채우지 않는다("빠른 상황 입력" 생략).
# 그러면 위 PINNED_SECTIONS는 업종을 알아도 절대 걸리지 않는다. 그래서 손님 발화 자체에서
# 문제 유형을 짐작할 힌트를 두고(ask_graph의 PROBLEM_TYPE_HINTS와 같은 목적), problemTypes가
# 비어 있을 때의 안전망으로 쓴다.
#
# 주의(2026-09 재발 방지): 한국어 용언은 어간에 어미가 붙으며 음절 자체가 바뀐다
# ("상하다" + "ㄴ" → "상한", "상하다" + "았다" → "상했다"). "상했"처럼 활용형 하나만
# 문자열로 넣으면 "상한 냄새가 나요"처럼 아주 흔한 다른 활용형은 부분 문자열로도
# 걸리지 않아 pinned_sections가 통째로 비고, 순수 유사도 검색만으로는 이 조항이
# 콜로키얼한 질의에서 순위가 크게 밀린다(retrieve_relevant_chunks 주석 참고) — 그 결과
# citations가 비어 "근거 문서에서 확인되지 않아 일반적인 응대 원칙으로 답했습니다"가
# 나온다. 그래서 자주 쓰는 활용형·구어체 동의어를 개별 문자열로 나열해 둔다(형태소
# 분석기 없이 substring 매칭만 쓰는 한계를 우회하는 임시 방편이다).
PROBLEM_TYPE_HINTS: dict[str, tuple[str, ...]] = {
    "refund_exchange": (
        "환불",
        "교환",
        "반품",
        "영수증",
        "결제 취소",
        "돈 돌려",
        "물러주",
        "물러줘",
        "바꿔주",
        "바꿔줘",
        "취소해",
    ),
    "damage_contamination": (
        "상했",
        "상한",
        "상하고",
        "상해서",
        "상함",
        "상하네",
        "상하나",
        "상하는",
        "상할",
        "변질",
        "부패",
        "곰팡이",
        "유통기한",
        "소비기한",
        "이물",
        "오염",
        "파손",
        "쉰내",
        "쉬었",
        "썩",
        "구더기",
        "냄새",
    ),
}

# industryId조차 없거나(온보딩 전) PINNED_SECTIONS에 해당 업종 항목이 없을 때 쓰는 업종
# 무관 기본값. 식료품 조항은 특정 업종 전유물이 아니라 온보딩 여부와 무관하게 근거가 된다.
DEFAULT_PINNED_SECTIONS: dict[str, tuple[str, ...]] = {
    "refund_exchange": ("식료품(19개 업종)", "[별표 1]", "식료품"),
    "damage_contamination": ("식료품(19개 업종)", "식료품"),
}


def _infer_problem_types(state: GraphState) -> list[str]:
    """intake.problemTypes가 비어 있을 때 손님 발화에서 문제 유형을 짐작한다."""
    text = " ".join(
        [state["latest_text"]]
        + [turn.text for turn in state["recent_transcript"] if turn.speaker == "customer"]
    )
    return [
        problem_type
        for problem_type, hints in PROBLEM_TYPE_HINTS.items()
        if any(hint in text for hint in hints)
    ]


def _pinned_sections(state: GraphState) -> tuple[str, ...]:
    industry_id = state["profile"].industryId
    problem_types = state["intake"].problemTypes or _infer_problem_types(state)

    sections: list[str] = []
    for problem_type in problem_types:
        pinned = PINNED_SECTIONS.get((industry_id, problem_type)) or DEFAULT_PINNED_SECTIONS.get(
            problem_type, ()
        )
        for section in pinned:
            if section not in sections:
                sections.append(section)
    return tuple(sections)


def retrieve_node(state: GraphState) -> dict:
    # 지식베이스가 52 → 553 청크로 늘어나 top_k=3으로는 관련 조문이 밀려난다.
    docs = retrieve_relevant_chunks(
        state["latest_text"],
        top_k=5,
        pinned_sections=_pinned_sections(state),
        industry_id=state["profile"].industryId,
    )
    return {"retrieved": docs}


SOURCE_TYPE_LABEL = {
    "standard": "공식 고시(소비자분쟁해결기준)",
    "notice": "공식 고시(행정기관 고시)",
    "law": "법령 가이드",
    "guide": "공공기관 가이드",
    "manual": "사내 실무 매뉴얼(법령 아님)",
}


def _format_knowledge(index: int, doc: Document) -> str:
    """근거 후보 한 건을 출처·위치가 드러나게 렌더링한다. citations의 section은 여기서 나온다."""
    meta = doc.metadata
    where = meta.get("section") or "(위치 표시 없음)"
    if meta.get("page"):
        where = f"{where} (p.{meta['page']})"
    kind = SOURCE_TYPE_LABEL.get(meta.get("sourceType"), meta.get("sourceType") or "미분류")
    return (
        f"[근거 {index + 1}] {meta.get('documentTitle')}\n"
        f"  위치: {where}\n"
        f"  종류: {kind}\n"
        f"  본문(quote는 이 안에서 그대로 옮길 것):\n"
        f"{doc.page_content}"
    )


KNOWLEDGE_CATEGORY_LABEL = {
    "refund_policy": "환불·교환 기준",
    "product": "취급 상품",
    "frequent_claim": "자주 오는 클레임",
    "etc": "기타",
}


def _format_store_knowledge(items: list[StoreKnowledgeItem]) -> str:
    """직원이 등록한 매장 규정을 프롬프트 한 블록으로 만든다. 없으면 빈 문자열."""
    lines = []
    for item in items:
        label = KNOWLEDGE_CATEGORY_LABEL.get(item.category, item.category)
        title = item.title.strip()
        header = f"[{label}] {title}" if title else f"[{label}]"
        lines.append(f"{header}\n  {item.body.strip()}")
    return "\n".join(lines)


def _build_user_prompt(state: GraphState) -> str:
    intake = state["intake"]
    context_lines = [f"업종: {state['profile'].industry}"]
    if intake.problemTypes:
        labels = ", ".join(PROBLEM_TYPE_LABEL.get(t, t) for t in intake.problemTypes)
        context_lines.append(f"문제 유형: {labels}")
    if intake.behaviorTypes:
        labels = ", ".join(BEHAVIOR_TYPE_LABEL.get(t, t) for t in intake.behaviorTypes)
        context_lines.append(f"발생 행동: {labels}")
    if intake.priorAction:
        context_lines.append(f"이전 조치: {intake.priorAction}")

    transcript_lines = [
        f"{'고객' if turn.speaker == 'customer' else '직원'}: {turn.text}"
        for turn in state["recent_transcript"][-8:]
    ]

    knowledge_lines = [_format_knowledge(i, doc) for i, doc in enumerate(state["retrieved"])]

    detected = describe_detected_risk(state["latest_text"])

    sections = [
        "## 상황 정보",
        "\n".join(context_lines),
        "",
        "## 최근 대화",
        "\n".join(transcript_lines) or "(대화 없음)",
        "",
        "## 방금 들어온 발화",
        state["latest_text"],
    ]
    # 사전 매칭 결과를 참고 신호로 같이 넘긴다. 판단은 여전히 모델이 하되,
    # STT가 짧게 흘려 적은 폭언("ㅅㅂ", "뒤진다")을 놓치지 않게 하는 안전망이다.
    if detected:
        sections += [
            "",
            "## 자동 감지된 위험 표현 (참고 신호, 오탐 가능)",
            detected,
        ]
    # 직원이 앱에 직접 등록한 매장 규정. 공식 문서가 아니므로 "근거 문서 후보"와 섞지 않고
    # 별도 블록으로 둔다(규칙 12에서 citations 금지를 명시한다).
    store_rules = _format_store_knowledge(state.get("store_knowledge") or [])
    if store_rules:
        sections += [
            "",
            "## 우리 매장 내규 (직원이 직접 등록 · 법령 아님 · citations 금지)",
            store_rules,
        ]
    sections += [
        "",
        "## 근거 문서 후보 (이 안에서만 인용하세요)",
        "\n\n".join(knowledge_lines) or "(관련 근거 없음)",
        "",
        "위 내용을 바탕으로 한국어로만 답변을 작성하세요.",
    ]
    return "\n".join(sections)


def generate_node(state: GraphState) -> dict:
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.2,
        http_client=shared_http_client,
    )
    structured_llm = llm.with_structured_output(RecommendationDraft)

    draft: RecommendationDraft = structured_llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(state)),
        ]
    )

    recommendation = Recommendation(
        **draft.model_dump(exclude={"citations"}),
        # LLM이 적어낸 인용을 그대로 내보내지 않는다. 검색된 청크에 실제로 있는 문장만
        # 근거로 남기고 출처 표기도 메타데이터로 교정한다(graph/citations.py).
        citations=ground_citations(draft.citations, state["retrieved"]),
        id=_next_id(),
        createdAtMs=_now_ms(),
        isFixedSafetyScript=False,
    )
    return {"recommendation": recommendation}


def _build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("fixed_safety", fixed_safety_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_conditional_entry_point(
        route_after_entry, {"fixed_safety": "fixed_safety", "retrieve": "retrieve"}
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("fixed_safety", END)
    graph.add_edge("generate", END)
    return graph.compile()


recommendation_graph = _build_graph()


def _initial_state(
    profile: BusinessProfile,
    intake: SessionIntake,
    recent_transcript: list[TranscriptTurn],
    recent_situations: list[str],
    latest_text: str,
    store_knowledge: list[StoreKnowledgeItem] | None,
) -> GraphState:
    return {
        "profile": profile,
        "intake": intake,
        "recent_transcript": recent_transcript,
        "recent_situations": recent_situations,
        "latest_text": latest_text,
        "store_knowledge": store_knowledge or [],
        "retrieved": [],
        "recommendation": None,
    }


def run_recommendation_graph(
    profile: BusinessProfile,
    intake: SessionIntake,
    recent_transcript: list[TranscriptTurn],
    recent_situations: list[str],
    latest_text: str,
    store_knowledge: list[StoreKnowledgeItem] | None = None,
) -> Recommendation:
    result = recommendation_graph.invoke(
        _initial_state(
            profile,
            intake,
            recent_transcript,
            recent_situations,
            latest_text,
            store_knowledge,
        )
    )
    return result["recommendation"]


def stream_recommendation(
    profile: BusinessProfile,
    intake: SessionIntake,
    recent_transcript: list[TranscriptTurn],
    recent_situations: list[str],
    latest_text: str,
    store_knowledge: list[StoreKnowledgeItem] | None = None,
) -> Iterator[dict]:
    """추천 답변을 만드는 과정을 중간 결과째로 흘려보낸다.

    왜 그래프(invoke)를 쓰지 않는가: LangGraph는 노드 하나가 끝나야 결과를 준다. 그런데
    지연의 대부분은 generate 노드 **안에서** JSON이 다 써지기를 기다리는 시간이다. 전체
    JSON(근거·다음 행동·예상 답변까지)이 완성되기를 기다리면 3~5초가 그냥 흐른다. 직원이
    실제로 필요한 것은 그중 첫 문장(sayNow) 하나다. 그래서 여기서는 같은 노드 함수를 순서대로
    부르되 generate만 스트리밍으로 바꾼다 — 분기·검색·인용 검증 로직은 그래프와 공유한다.

    yield 하는 dict:
      {"type": "stage",   "stage": "retrieving" | "generating"}
      {"type": "partial", "situation": ..., "sayNow": "여기까지 도착한 문장"}
      {"type": "done",    "recommendation": Recommendation}
    """
    state = _initial_state(
        profile, intake, recent_transcript, recent_situations, latest_text, store_knowledge
    )

    # 위협·흉기·반복 폭언은 생성형을 거치지 않는다. 기다릴 것이 없으므로 바로 완성본을 준다.
    if route_after_entry(state) == "fixed_safety":
        yield {"type": "done", "recommendation": fixed_safety_node(state)["recommendation"]}
        return

    yield {"type": "stage", "stage": "retrieving"}
    state.update(retrieve_node(state))

    yield {"type": "stage", "stage": "generating"}

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.2,
        http_client=shared_http_client,
    )
    # 여기서만 pydantic 클래스 대신 JSON 스키마(dict)를 넘긴다. pydantic 클래스를 주면
    # LangChain이 전체 응답을 모아 한 번에 검증하므로 스트리밍을 걸어도 마지막에 완성본
    # 하나만 나온다. dict 스키마를 주면 부분 JSON을 관대하게 파싱해 자라나는 dict를 준다.
    structured_llm = llm.with_structured_output(RecommendationDraft.model_json_schema())
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(state)),
    ]

    # 스트리밍 중에는 부분 dict가 온다(아직 스키마를 만족하지 않으므로 pydantic 객체가 아니다).
    # 마지막 조각만이 완성된 draft다.
    last_chunk = None
    last_sent = ""
    for chunk in structured_llm.stream(messages):
        last_chunk = chunk
        partial = chunk if isinstance(chunk, dict) else chunk.model_dump()
        say_now = (partial.get("sayNow") or "").strip()
        # 같은 문장을 다시 보내지 않는다. 필드가 아직 비어 있는 동안에는 보낼 것이 없다.
        if not say_now or say_now == last_sent:
            continue
        last_sent = say_now
        yield {
            "type": "partial",
            # situation이 아직 안 왔으면 화면 색이 튀지 않게 중립값으로 둔다.
            "situation": partial.get("situation") or "normal",
            "sayNow": say_now,
        }

    if last_chunk is None:
        raise RuntimeError("모델이 아무 응답도 보내지 않았습니다.")

    draft = (
        last_chunk
        if isinstance(last_chunk, RecommendationDraft)
        else RecommendationDraft.model_validate(last_chunk)
    )

    recommendation = Recommendation(
        **draft.model_dump(exclude={"citations"}),
        # 인용 검증은 반드시 전체 JSON이 모인 뒤에 한다. 도중에 끊긴 문장으로 원문을 대조하면
        # 멀쩡한 근거도 전부 탈락한다.
        citations=ground_citations(draft.citations, state["retrieved"]),
        id=_next_id(),
        createdAtMs=_now_ms(),
        isFixedSafetyScript=False,
    )
    yield {"type": "done", "recommendation": recommendation}
