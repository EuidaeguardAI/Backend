from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.rag.retrieve import retrieve_relevant_chunks
from app.routers import analyze, ask, report, stt


def _warm_up() -> None:
    """지식베이스를 읽어 두고 OpenAI 연결까지 미리 맺어 둔다.

    두 가지가 첫 요청에만 붙는다.
      - 24MB짜리 vector_store.json 로딩 (2~3초, lru_cache라 한 번뿐)
      - api.openai.com까지의 TLS 핸드셰이크 (1초 안팎)
    상담의 첫 발화가 이 둘을 다 물게 되는데, 하필 그게 직원이 처음 보는 응답이다.
    그래서 버리는 질의를 한 번 돌려 둘 다 끝내 놓는다(임베딩 호출 1회 비용).
    """
    retrieve_relevant_chunks("예열", top_k=1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await anyio.to_thread.run_sync(_warm_up)
    except Exception as exc:  # noqa: BLE001
        # 색인이 없거나 네트워크가 안 되는 상태에서도 서버는 떠야 한다.
        print("[startup] 예열 실패(첫 요청이 느릴 수 있음):", exc)
    yield


app = FastAPI(title="응대가드 AI 백엔드", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    # 쿠키/세션을 쓰지 않으므로 credentials는 끈다 (allow_origins=["*"]는 credentials=True와 함께 쓸 수 없음).
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stt.router)
app.include_router(analyze.router)
app.include_router(ask.router)
app.include_router(report.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
