"""
Next.js 프론트엔드(src/lib/types.ts, src/lib/api/schemas.ts)와 1:1로 대응하는 계약.
필드명은 프론트엔드와 그대로 맞추기 위해 의도적으로 camelCase를 사용한다.
"""

from typing import Literal, Optional

from pydantic import BaseModel

RiskLevel = Literal["normal", "dispute", "abuse", "threat", "emergency"]
Speaker = Literal["customer", "staff", "ai"]


class Citation(BaseModel):
    label: str
    section: Optional[str] = None


class Recommendation(BaseModel):
    id: str
    situation: RiskLevel
    riskLevel: int
    confidence: float
    sayNow: str
    nextActions: list[str]
    doNot: list[str]
    citations: list[Citation]
    needsHumanReview: bool
    isFixedSafetyScript: bool
    # 손님이 다음에 할 법한 짧은 답변 후보(예: "상했어요" / "안 상했어요"). 사실 확인이 필요 없으면 빈 배열.
    expectedReplies: list[str] = []
    createdAtMs: int


class RecommendationDraft(BaseModel):
    """LLM이 직접 채우는 필드만. id/createdAtMs/isFixedSafetyScript는 서버가 나중에 채운다."""

    situation: RiskLevel
    riskLevel: int
    confidence: float
    sayNow: str
    nextActions: list[str]
    doNot: list[str]
    citations: list[Citation]
    needsHumanReview: bool
    expectedReplies: list[str] = []


class BusinessProfile(BaseModel):
    industry: str
    industryId: str
    tasks: list[str]
    aiFeatures: list[str]
    onboardedAtMs: int


class SessionIntake(BaseModel):
    inProgress: bool
    micAvailable: bool
    problemTypes: list[str]
    behaviorTypes: list[str]
    priorAction: Optional[str] = None
    emergencyDeclared: bool


class TranscriptTurn(BaseModel):
    speaker: Speaker
    text: str


class AnalyzeRequest(BaseModel):
    profile: BusinessProfile
    intake: SessionIntake
    recentTranscript: list[TranscriptTurn]
    recentSituations: list[RiskLevel]
    latestText: str


class AnalyzeResponse(BaseModel):
    recommendation: Recommendation


class SttResponse(BaseModel):
    text: str


class ReportRequest(BaseModel):
    sessionId: str


class ReportResponse(BaseModel):
    success: bool


class AskMessage(BaseModel):
    """'물어보기' 탭의 대화 한 턴. role은 화면에 찍히는 말풍선의 주인이다."""

    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    # 온보딩을 건너뛴 사용자도 물어볼 수 있어야 하므로 profile은 선택이다.
    profile: Optional[BusinessProfile] = None
    history: list[AskMessage] = []
    question: str


class AskAnswerDraft(BaseModel):
    """LLM이 직접 채우는 필드만. id/createdAtMs/isFixedSafetyScript는 서버가 채운다."""

    situation: RiskLevel
    answer: str
    sayNow: Optional[str] = None
    nextActions: list[str]
    doNot: list[str]
    citations: list[Citation]
    needsHumanReview: bool


class AskAnswer(BaseModel):
    id: str
    situation: RiskLevel
    answer: str
    # 손님에게 그대로 읽어줄 문장이 있으면 채운다. 설명만 필요한 질문이면 비어 있다.
    sayNow: Optional[str] = None
    nextActions: list[str]
    doNot: list[str]
    citations: list[Citation]
    needsHumanReview: bool
    isFixedSafetyScript: bool
    createdAtMs: int


class AskResponse(BaseModel):
    answer: AskAnswer
