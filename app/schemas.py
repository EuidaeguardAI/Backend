"""
Next.js 프론트엔드(src/lib/types.ts, src/lib/api/schemas.ts)와 1:1로 대응하는 계약.
필드명은 프론트엔드와 그대로 맞추기 위해 의도적으로 camelCase를 사용한다.
"""

from typing import Literal, Optional

from pydantic import BaseModel

RiskLevel = Literal["normal", "dispute", "abuse", "threat", "emergency"]
Speaker = Literal["customer", "staff", "ai"]


class CitationDraft(BaseModel):
    """LLM이 직접 적어내는 인용. sourceType은 서버가 검색 결과에서 채운다."""

    label: str
    section: Optional[str] = None
    # 답변의 근거가 된 문장을 근거 문서 본문에서 그대로 옮긴 것.
    # 검색된 청크 안에 실제로 있는 문장인지 서버가 확인한 뒤에만 살아남는다(graph/citations.py).
    quote: Optional[str] = None


class Citation(CitationDraft):
    # 색인 당시의 분류(manual | law | standard | notice | guide). 화면에서 사내 매뉴얼과
    # 공식 고시를 구분해 표시하는 데 쓴다.
    sourceType: Optional[str] = None


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
    """LLM이 직접 채우는 필드만. id/createdAtMs/isFixedSafetyScript는 서버가 나중에 채운다.

    필드 순서가 곧 모델이 JSON을 써 내려가는 순서다. 스트리밍(/analyze/stream)에서 직원이
    가장 먼저 봐야 하는 것은 "지금 뭐라고 말할지"(sayNow)이므로 그것을 앞쪽에 둔다.
    근거(citations)는 가장 길고 화면에서는 접혀 있으므로 맨 뒤로 보낸다 — 순서를 바꾸면
    첫 문장이 뜨는 시점이 그만큼 늦어진다.
    """

    situation: RiskLevel
    riskLevel: int
    sayNow: str
    confidence: float
    nextActions: list[str]
    doNot: list[str]
    expectedReplies: list[str] = []
    citations: list[CitationDraft]
    needsHumanReview: bool


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


class StoreKnowledgeItem(BaseModel):
    """직원이 앱에 직접 등록한 매장 규정 한 건.

    프론트엔드의 localStorage에만 있는 데이터로, 요청마다 실려 온다. 서버는 이것을
    저장하지 않고 프롬프트에만 쓴다(사용자 기기 밖으로 나가지 않는다는 화면의 안내와
    어긋나지 않게 하려는 것이다).
    """

    category: str
    title: str
    body: str


class AnalyzeRequest(BaseModel):
    profile: BusinessProfile
    intake: SessionIntake
    recentTranscript: list[TranscriptTurn]
    recentSituations: list[RiskLevel]
    latestText: str
    storeKnowledge: list[StoreKnowledgeItem] = []


class AnalyzeResponse(BaseModel):
    recommendation: Recommendation


class SttResponse(BaseModel):
    text: str
    # 무음 구간에서 지어낸 문장으로 보여 서버가 걸러낸 경우 True (text는 빈 문자열).
    # 프론트엔드 동작에는 영향이 없고, 왜 인식 결과가 안 뜨는지 확인할 때 쓴다.
    filtered: bool = False


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
    citations: list[CitationDraft]
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
