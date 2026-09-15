from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.routers import analyze, ask, report, stt

app = FastAPI(title="응대가드 AI 백엔드")

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
