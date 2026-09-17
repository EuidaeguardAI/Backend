"""
'물어보기' 탭(챗봇)의 질의응답 그래프.

    START (조건 분기)
      ├─ 위협·흉기 키워드 감지 → fixed_safety → END        (생성형 아닌 고정 안전 절차)
      └─ 그 외                 → retrieve → generate → END  (RAG 검색 → 구조화 답변)

analyze(recommendation_graph)와 같은 안전 원칙·같은 지식베이스를 쓰지만 입력이 다르다.
analyze는 "녹음된 손님 발화"를 받아 손님에게 읽어줄 문장을 만들고, 여기서는 "직원이 직접 던진
질문"을 받아 직원에게 설명한다. 그래서 설문(intake)이 없고, 답변 본문(answer)과 손님에게 읽어줄
문장(sayNow)이 분리돼 있다. 녹음을 켜고 설문을 채울 여유가 없을 때 쓰는 빠른 경로다.
"""

import time
from typing import Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.config import CHAT_MODEL, OPENAI_API_KEY
from app.graph.recommendation_graph import (
    DEFAULT_PINNED_SECTIONS,
    PINNED_SECTIONS,
    PROBLEM_TYPE_HINTS,
    _format_knowledge,
)
from app.graph.citations import ground_citations
from app.http_client import shared_http_client
from app.rag.retrieve import retrieve_relevant_chunks
from app.safety.emergency_rules import (
    build_fixed_safety_recommendation,
    contains_weapon_keyword,
    describe_detected_risk,
    should_trigger_fixed_safety,
)
from app.schemas import AskAnswer, AskAnswerDraft, AskMessage, BusinessProfile

SYSTEM_PROMPT = """당신은 "응대가드 AI"의 응대 상담 챗봇입니다. 고객응대근로자(편의점·서비스업 종사자)가
응대 도중 또는 응대 직후에 "이럴 땐 어떻게 해야 하나요?"라고 물으면, 바로 실행할 수 있는 형태로 답합니다.
다음 원칙을 반드시 지키세요.

1. 당신은 법률 판단 기관이 아닙니다. "범죄가 성립한다", "고소하면 이긴다" 같은 확정적 법률 판단을 하지 마세요.
   대신 "해당 가능성이 있어 관리자·전문가 확인이 필요합니다"처럼 운영 안내 톤을 사용하세요.
2. answer는 제공된 근거 문서 안에서만 만드세요. 근거로 삼을 만한 내용이 없으면 citations를 빈 배열로 두고
   needsHumanReview를 true로 설정한 뒤, 일반적인 응대 원칙 수준에서만 조심스럽게 안내하세요. 근거를 지어내지 마세요.
2-1. citations의 label에는 근거 문서의 제목을, section에는 근거 문서에 표시된 위치
   (예: "별표Ⅱ 2. 식료품(19개 업종)", "3. 용어의 정의 (p.4)")를 그대로 옮기세요. 위치를 임의로 지어내지 마세요.
2-2. 근거 문서에는 공식 고시·법령과 사내 실무 매뉴얼이 섞여 있습니다. 사내 매뉴얼의 내용을 법령상 의무인 것처럼
   말하지 말고, "점포 기준으로는", "매장 절차상"처럼 구분해서 안내하세요.
2-3. citations의 quote에는 그 답변의 근거가 된 문장을 근거 문서 본문에서 **한 글자도 바꾸지 말고**
   그대로 1~2문장 옮기세요. 요약하거나 말을 다듬으면 서버가 원문 대조에 실패해 근거 표시가 사라집니다.
   본문에 그대로 옮길 만한 문장이 없는 문서는 아예 인용하지 마세요.
3. answer는 질문에 대한 설명입니다. 2~4문장으로 짧게, 결론부터 쓰세요. 인사말·사과·서론을 붙이지 마세요.
4. sayNow는 손님에게 그대로 소리 내어 읽을 수 있는 1~2문장입니다. 손님에게 할 말이 필요한 질문일 때만 채우고,
   절차나 규정만 묻는 질문이면 비워 두세요(null).
5. nextActions는 직원이 지금 순서대로 할 일을 짧은 구로 2~4개, doNot은 하지 말아야 할 것을 0~3개 적으세요.
   answer에 쓴 문장을 그대로 반복하지 마세요.
6. 질문에 위협·협박·흉기·반복 폭언 정황이 있으면 situation을 threat 또는 emergency로, 그 밖에는
   normal(일반 문의), dispute(규정 분쟁), abuse(폭언) 중에서 고르세요.
7. 직원의 안전이 규정 준수보다 우선입니다. 신체적 위험이 의심되면 규정 설명보다 대피·관리자 호출·112 신고를 먼저 안내하세요.
8. 모든 출력(answer, sayNow, nextActions, doNot, citations)은 **반드시 한국어**로만 작성하세요.
   영어 단어나 로마자 표기를 섞지 말고, 고유명사를 빼면 우리말 표현을 쓰세요.
9. 이전 대화가 함께 주어지면 이어지는 질문("그럼 영수증이 없으면요?")의 생략된 주어를 그 맥락에서 채워 이해하세요.
10. "자동 감지된 위험 표현"이 함께 주어지면 위험도 판단의 참고 신호로만 쓰세요. 오탐일 수 있으니 질문의 실제 맥락을 우선하세요.
11. 여러 개를 구매해 일부는 이미 먹거나 쓴 뒤 나머지에서 문제를 제기하는 질문에는, 전체 수량을
    자동으로 환불 대상에 포함해 답하지 마세요. 실제로 문제가 확인된 수량과 이미 먹거나 쓴 부분의
    이상 여부부터 확인하라고 nextActions에 넣으세요."""


# 폭언·위협 질문도 환불 질의와 같은 문제를 겪는다. 직원이 쓰는 말("계속 소리를 질러요")과
# 조문·매뉴얼의 표제("고객의 폭언 등으로 인한 건강장해 예방조치")가 너무 달라, 유사도 검색만으로는
# 정작 근거가 되는 항목이 밀려나고 citations가 빈 채로 나온다(retrieve_relevant_chunks 주석 참고).
# 업종과 무관하게 적용한다 — 폭언 보호 조항은 전 업종 공통이다.
SAFETY_HINTS = (
    "욕",
    "폭언",
    "고성",
    "소리",
    "모욕",
    "반말",
    "협박",
    "위협",
    "난동",
    "시비",
    "폭행",
    "무섭",
)
SAFETY_SECTIONS = ("B. 욕설·모욕·고성", "제41조(고객의 폭언")


class AskState(TypedDict):
    profile: Optional[BusinessProfile]
    history: list[AskMessage]
    question: str
    retrieved: list[Document]
    answer: Optional[AskAnswer]


def _next_id() -> str:
    return f"ask-{int(time.time() * 1000)}"


def _now_ms() -> int:
    return int(time.time() * 1000)


def route_after_entry(state: AskState) -> str:
    # 챗봇에는 누적 상황(recent_situations)이 없으므로 키워드 경로만 본다.
    if should_trigger_fixed_safety(state["question"], []):
        return "fixed_safety"
    return "retrieve"


def fixed_safety_node(state: AskState) -> dict:
    draft = build_fixed_safety_recommendation(state["question"])
    lead = (
        "흉기로 볼 수 있는 표현이 있습니다. 규정을 따지기 전에 몸부터 피하세요."
        if contains_weapon_keyword(state["question"])
        else "위협으로 볼 수 있는 표현이 있습니다. 설득하거나 따지지 말고 아래 순서대로 움직이세요."
    )
    answer = AskAnswer(
        id=_next_id(),
        situation=draft.situation,
        answer=f"{lead} 이 안내는 AI가 만든 문장이 아니라 고정된 안전 절차입니다.",
        sayNow=draft.sayNow,
        nextActions=draft.nextActions,
        doNot=draft.doNot,
        citations=draft.citations,
        needsHumanReview=draft.needsHumanReview,
        isFixedSafetyScript=True,
        createdAtMs=_now_ms(),
    )
    return {"answer": answer}


def _retrieval_query(state: AskState) -> str:
    """검색에 쓸 질의. 짧은 후속 질문은 앞 질문을 붙여야 검색이 맥락을 잃지 않는다."""
    question = state["question"]
    if len(question) >= 15:
        return question
    previous = [message.content for message in state["history"] if message.role == "user"]
    return f"{previous[-1]} {question}" if previous else question


def _pinned_sections(state: AskState) -> tuple[str, ...]:
    question = state["question"]
    profile = state["profile"]
    sections: list[str] = []

    if any(hint in question for hint in SAFETY_HINTS):
        sections.extend(SAFETY_SECTIONS)

    # '물어보기'는 설문 없이 바로 묻는 빠른 경로라 업종(profile)이 아예 없는 채로 오는 경우가
    # 흔하다. profile이 없거나 그 업종에 대한 PINNED_SECTIONS가 없으면 업종 무관 기본값으로
    # 대체한다 — 그렇지 않으면 힌트가 정확히 맞아도 근거가 하나도 안 잡힌다.
    for problem_type, hints in PROBLEM_TYPE_HINTS.items():
        if not any(hint in question for hint in hints):
            continue
        pinned = ()
        if profile is not None:
            pinned = PINNED_SECTIONS.get((profile.industryId, problem_type), ())
        if not pinned:
            pinned = DEFAULT_PINNED_SECTIONS.get(problem_type, ())
        for section in pinned:
            if section not in sections:
                sections.append(section)

    return tuple(sections)


def retrieve_node(state: AskState) -> dict:
    profile = state["profile"]
    docs = retrieve_relevant_chunks(
        _retrieval_query(state),
        top_k=5,
        pinned_sections=_pinned_sections(state),
        industry_id=profile.industryId if profile else None,
    )
    return {"retrieved": docs}


def _build_user_prompt(state: AskState) -> str:
    profile = state["profile"]
    knowledge_lines = [_format_knowledge(i, doc) for i, doc in enumerate(state["retrieved"])]
    detected = describe_detected_risk(state["question"])

    sections = [
        "## 상황 정보",
        f"업종: {profile.industry}" if profile else "업종: (설정 안 됨)",
        "",
        "## 직원의 질문",
        state["question"],
    ]
    # analyze와 같은 안전망. STT가 아니라 타이핑이지만, 급할 때 줄여 쓴 표현을 놓치지 않게 한다.
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


def _history_messages(history: list[AskMessage]) -> list:
    """직전 대화를 그대로 넘긴다. 답변 본문만 남기므로 근거·조치는 매 턴 다시 검색된다."""
    messages: list = []
    for message in history[-6:]:
        if message.role == "user":
            messages.append(HumanMessage(content=message.content))
        else:
            messages.append(AIMessage(content=message.content))
    return messages


def generate_node(state: AskState) -> dict:
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.2,
        http_client=shared_http_client,
    )
    structured_llm = llm.with_structured_output(AskAnswerDraft)

    draft: AskAnswerDraft = structured_llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *_history_messages(state["history"]),
            HumanMessage(content=_build_user_prompt(state)),
        ]
    )

    answer = AskAnswer(
        **draft.model_dump(exclude={"citations"}),
        # analyze와 같은 검증을 거친다. 지어낸 문장을 근거라며 강조해 보여주지 않기 위해서다.
        citations=ground_citations(draft.citations, state["retrieved"]),
        id=_next_id(),
        createdAtMs=_now_ms(),
        isFixedSafetyScript=False,
    )
    return {"answer": answer}


def _build_graph():
    graph = StateGraph(AskState)
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


ask_graph = _build_graph()


def run_ask_graph(
    profile: Optional[BusinessProfile],
    history: list[AskMessage],
    question: str,
) -> AskAnswer:
    result = ask_graph.invoke(
        {
            "profile": profile,
            "history": history,
            "question": question,
            "retrieved": [],
            "answer": None,
        }
    )
    return result["answer"]
