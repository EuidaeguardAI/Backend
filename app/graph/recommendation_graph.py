"""
응대 추천 LangGraph.

    START (조건 분기)
      ├─ 위협 키워드/반복 폭언 감지 → fixed_safety → END   (생성형 아닌 고정 안전 절차)
      └─ 그 외                     → retrieve → generate → END  (RAG 검색 → 구조화 생성)

기획서의 권장 흐름(음성 입력 → 위험도 분류 → 긴급/일반 분기 → RAG 검색 → 응대 생성)을
그대로 그래프로 옮긴 것이다.
"""

import time
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.config import CHAT_MODEL, OPENAI_API_KEY
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
    유지하세요."""

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


def _pinned_sections(state: GraphState) -> tuple[str, ...]:
    industry_id = state["profile"].industryId
    sections: list[str] = []
    for problem_type in state["intake"].problemTypes:
        for section in PINNED_SECTIONS.get((industry_id, problem_type), ()):
            if section not in sections:
                sections.append(section)
    return tuple(sections)


def retrieve_node(state: GraphState) -> dict:
    # 지식베이스가 52 → 714 청크로 늘어나 top_k=3으로는 관련 조문이 밀려난다.
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
        f"{doc.page_content}"
    )


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
    sections += [
        "",
        "## 근거 문서 후보 (이 안에서만 인용하세요)",
        "\n\n".join(knowledge_lines) or "(관련 근거 없음)",
        "",
        "위 내용을 바탕으로 한국어로만 답변을 작성하세요.",
    ]
    return "\n".join(sections)


def generate_node(state: GraphState) -> dict:
    llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)
    structured_llm = llm.with_structured_output(RecommendationDraft)

    draft: RecommendationDraft = structured_llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(state)),
        ]
    )

    recommendation = Recommendation(
        **draft.model_dump(),
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


def run_recommendation_graph(
    profile: BusinessProfile,
    intake: SessionIntake,
    recent_transcript: list[TranscriptTurn],
    recent_situations: list[str],
    latest_text: str,
) -> Recommendation:
    result = recommendation_graph.invoke(
        {
            "profile": profile,
            "intake": intake,
            "recent_transcript": recent_transcript,
            "recent_situations": recent_situations,
            "latest_text": latest_text,
            "retrieved": [],
            "recommendation": None,
        }
    )
    return result["recommendation"]
