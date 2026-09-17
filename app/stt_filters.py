"""
무음·잡음 구간에서 음성 인식 모델이 "지어낸" 문장을 걸러내는 필터.

배경: 음성 인식 모델은 말소리가 없는 구간을 받으면 빈 문자열을 돌려주는 대신 학습 데이터에서
자주 본 문장을 뱉는 성질이 있다. 한국어에서는 영상 자막 데이터의 영향으로
"시청해주셔서 감사합니다", "구독과 좋아요 부탁드립니다" 같은 문장이 특히 자주 나온다.

이게 이 서비스에서 위험한 이유는, 그 문장이 그대로 "손님이 한 말"이 되어 추천 답변과 위험도가
만들어지기 때문이다. 아무도 말하지 않았는데 상담이 진행되는 것처럼 보인다.

1차 방어는 프론트엔드의 무음 판정(VAD, src/lib/mic/voiceActivity.ts)이라 대부분의 무음 구간은
여기까지 오지도 않는다. 이 파일은 그 그물을 빠져나온 것(작은 생활소음이 섞여 VAD를 통과한 구간)을
걸러내는 2차 방어다.
"""

import re

# 비교할 때는 띄어쓰기와 문장부호를 모두 지운 형태로 맞춘다.
# 같은 환각이 "감사합니다." / "감사 합니다" 처럼 조금씩 다르게 나오기 때문이다.
_PUNCTUATION = re.compile(r"[\s.,!?~…·\"'“”‘’()\[\]\-ㅡ_]+")

# 인식 결과가 이 글자 수 미만이면 내용이 없다고 본다(정규화 후 기준).
MIN_MEANINGFUL_CHARS = 3

# 결과 전체가 이것과 정확히 일치할 때만 버린다.
# 부분 일치로 하면 "시청해주셔서 감사합니다만 환불은 해주세요" 같은 진짜 발화까지 날아간다.
_EXACT_HALLUCINATIONS = {
    "시청해주셔서감사합니다",
    "시청해주셔서감사드립니다",
    "지금까지시청해주셔서감사합니다",
    "오늘도시청해주셔서감사합니다",
    "끝까지시청해주셔서감사합니다",
    "구독과좋아요부탁드립니다",
    "구독좋아요알림설정부탁드립니다",
    "구독과좋아요알림설정부탁드립니다",
    "다음영상에서만나요",
    "다음영상에서뵙겠습니다",
    "다음시간에만나요",
    "다음에또만나요",
    "여러분의시청이큰힘이됩니다",
    "이영상은유료광고를포함하고있습니다",
    "감사합니다",
    "고맙습니다",
    "안녕하세요",
    "안녕히계세요",
    "수고하셨습니다",
}

# 사전에 없는 변형까지 잡는 패턴. 모두 "결과 전체가 그 형태일 때"만 걸리도록 앞뒤를 묶어 둔다.
# 부분 일치로 두면 "시청해주셔서 감사합니다만 환불은 해주셔야죠" 같은 진짜 발화가 통째로 사라진다.
_HALLUCINATION_PATTERNS = [
    re.compile(r"한글자막.{0,20}"),
    re.compile(r".{0,10}자막(제공|제작|by).{0,15}"),
    re.compile(r"(지금까지|오늘도|끝까지|정말)?시청해주.{0,8}감사.{0,8}"),
    re.compile(r"구독.{0,3}좋아요.{0,3}(알림설정)?.{0,3}(부탁|눌러|구독)?.{0,8}"),
    re.compile(r"(MBC|KBS|SBS|JTBC|YTN|채널A)뉴스.{0,15}"),
    re.compile(r".{0,12}영상(편집|제작).{0,15}"),
]

# "감사합니다감사합니다감사합니다"처럼 짧은 조각이 세 번 이상 반복되는 형태.
_REPEAT_PATTERN = re.compile(r"^(.{1,10}?)\1{2,}$")


def normalize(text: str) -> str:
    return _PUNCTUATION.sub("", text)


def is_hallucination(text: str) -> bool:
    """무음 구간에서 지어낸 문장으로 보이면 True."""
    normalized = normalize(text)
    if len(normalized) < MIN_MEANINGFUL_CHARS:
        return True
    if normalized in _EXACT_HALLUCINATIONS:
        return True
    if _REPEAT_PATTERN.match(normalized):
        return True
    return any(pattern.fullmatch(normalized) for pattern in _HALLUCINATION_PATTERNS)


def filter_transcript(text: str) -> tuple[str, bool]:
    """
    인식 결과를 검사해 (내보낼 텍스트, 걸러냈는지) 를 돌려준다.
    걸러낸 경우 텍스트는 빈 문자열이며, 프론트엔드는 이를 "이번 구간은 아무 말도 없었다"로 다룬다.
    """
    stripped = (text or "").strip()
    if not stripped:
        return "", False
    if is_hallucination(stripped):
        return "", True
    return stripped, False
