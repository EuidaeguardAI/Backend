"""
반복 폭언·위협 감지 시 생성형 답변 대신 고정 안전 절차를 우선 표시한다.
(기획서 안전 원칙: "긴급 상황은 고정된 안전 절차를 우선 표시한다")
프론트엔드 src/lib/safety/emergencyRules.ts와 동일한 규칙을 유지한다.

키워드 목록 자체는 app/safety/abusive_lexicon.py로 분리했다.
"""

from app.safety.abusive_lexicon import (
    THREAT,
    WEAPON,
    detect_categories,
    match_severe_profanity,
    match_threat,
    match_weapon,
)
from app.schemas import Citation, RecommendationDraft, RiskLevel

# 하위 호환용 별칭. 실제 매칭은 abusive_lexicon의 정규화 매칭을 쓴다.
THREAT_KEYWORDS = THREAT + WEAPON

ESCALATING_SITUATIONS: set[RiskLevel] = {"abuse", "threat"}


def contains_threat_keyword(text: str) -> bool:
    return bool(match_threat(text))


def contains_weapon_keyword(text: str) -> bool:
    return bool(match_weapon(text))


def is_repeated_escalation(recent_situations: list[str]) -> bool:
    last_two = recent_situations[-2:]
    return len(last_two) == 2 and all(
        situation in ESCALATING_SITUATIONS for situation in last_two
    )


def should_trigger_fixed_safety(latest_text: str, recent_situations: list[str]) -> bool:
    return contains_threat_keyword(latest_text) or is_repeated_escalation(
        recent_situations
    )


def describe_detected_risk(text: str) -> str:
    """감지된 위험 표현을 LLM 프롬프트에 넣을 한 줄 요약으로 만든다. 없으면 빈 문자열."""
    categories = detect_categories(text)
    if not categories:
        return ""
    parts = [
        f"{category}({', '.join(words[:5])})" for category, words in categories.items()
    ]
    return " / ".join(parts)


def build_fixed_safety_recommendation(latest_text: str = "") -> RecommendationDraft:
    """고정 안전 절차. 흉기·강한 위협이 감지되면 112 신고를 첫 조치로 올린다."""
    weapon_detected = contains_weapon_keyword(latest_text)
    severe = weapon_detected or bool(match_severe_profanity(latest_text))

    if weapon_detected:
        next_actions = [
            "즉시 112 신고",
            "고객과 거리 확보 후 대피",
            "응대 중단",
            "관리자 즉시 호출",
        ]
    else:
        next_actions = [
            "고객과 거리 확보",
            "응대 중단",
            "관리자 즉시 호출",
            "위험 시 112 신고",
        ]

    do_not = ["논쟁하지 않기", "혼자 제지하려 하지 않기"]
    if severe:
        do_not.append("등을 보이거나 좁은 공간으로 이동하지 않기")

    return RecommendationDraft(
        situation="emergency",
        riskLevel=5,
        confidence=0.95,
        sayNow="지금은 대응하지 않고 거리를 확보하겠습니다.",
        nextActions=next_actions,
        doNot=do_not,
        citations=[Citation(label="산업안전보건법 제41조", section="건강장해 예방조치")],
        needsHumanReview=True,
    )
