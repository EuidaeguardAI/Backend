"""OpenAI 호출이 공유하는 HTTP 클라이언트.

왜 필요한가: httpx의 기본 keepalive 만료가 5초다. 상시 녹음에서는 손님과 손님 사이에
몇 분씩 비므로, 새 손님의 첫 발화는 거의 항상 연결이 끊긴 상태에서 시작한다. 그러면
TLS 핸드셰이크부터 다시 하느라 임베딩·채팅 호출 하나하나가 1초 안팎을 더 쓴다.

실측(이 프로젝트 개발 환경, 2026-09):
    검색 연속 호출        0.30s
    20초 쉬었다가 호출    1.38s   ← 차이 대부분이 연결 재수립

연결을 오래 살려 두면 이 비용이 사라진다. 손님 한 명에 한 번씩 붙던 지연이라
체감 차이가 크다.
"""

import httpx

# 상담과 상담 사이의 공백을 넉넉히 덮는 값. 이보다 더 오래 비면 어차피 연결이 끊겨도
# 상관없는 한가한 시간대다.
KEEPALIVE_EXPIRY_SECONDS = 600

_limits = httpx.Limits(
    max_connections=20,
    max_keepalive_connections=10,
    keepalive_expiry=KEEPALIVE_EXPIRY_SECONDS,
)

# 하나를 여러 클라이언트가 공유한다. OpenAI SDK와 langchain-openai 모두 http_client를 받는다.
# 타임아웃은 SDK 기본값(무한 대기 없음)을 덮어쓰지 않도록 넉넉하게만 잡는다.
shared_http_client = httpx.Client(limits=_limits, timeout=httpx.Timeout(60.0, connect=10.0))
