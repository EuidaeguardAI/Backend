import json
from collections.abc import AsyncIterator

import anyio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.graph.recommendation_graph import (
    run_recommendation_graph,
    stream_recommendation,
)
from app.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    if not body.latestText.strip():
        raise HTTPException(status_code=400, detail="latestText가 필요합니다.")
    try:
        recommendation = run_recommendation_graph(
            profile=body.profile,
            intake=body.intake,
            recent_transcript=body.recentTranscript,
            recent_situations=body.recentSituations,
            latest_text=body.latestText,
            store_knowledge=body.storeKnowledge,
        )
        return AnalyzeResponse(recommendation=recommendation)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="분석에 실패했습니다.") from exc


def _sse(event: str, payload: dict) -> str:
    """SSE 프레임 한 장. data는 한 줄이어야 하므로 JSON에 줄바꿈을 넣지 않는다."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/analyze/stream")
async def analyze_stream(body: AnalyzeRequest) -> StreamingResponse:
    """분석 결과를 SSE로 흘려보낸다.

    /analyze와 결과는 같지만, 전체 JSON이 완성되기를 기다리지 않고 첫 문장(sayNow)이
    써지는 대로 내보낸다. 손님 앞에 서 있는 직원에게는 3~5초를 기다리는 것과 1초 안에
    첫 문장을 보는 것이 전혀 다른 경험이다.

    /analyze는 폴백 경로로 그대로 남아 있다.
    """
    if not body.latestText.strip():
        raise HTTPException(status_code=400, detail="latestText가 필요합니다.")

    async def event_stream() -> AsyncIterator[str]:
        # LangChain 호출은 동기(블로킹)다. 그대로 돌리면 이벤트 루프가 멈춰 같은 순간의
        # 다른 요청(/stt 등)까지 함께 막힌다. 그래서 제너레이터를 한 조각씩 꺼내는 일
        # 자체를 워커 스레드에 맡긴다 — 조각을 기다리는 동안 루프는 계속 돈다.
        events = stream_recommendation(
            profile=body.profile,
            intake=body.intake,
            recent_transcript=body.recentTranscript,
            recent_situations=body.recentSituations,
            latest_text=body.latestText,
            store_knowledge=body.storeKnowledge,
        )
        done = object()

        def next_event() -> object:
            try:
                return next(events)
            except StopIteration:
                return done

        while True:
            try:
                event = await anyio.to_thread.run_sync(next_event)
            except Exception as exc:  # noqa: BLE001
                print("[analyze/stream] failed:", exc)
                yield _sse("error", {"type": "error", "detail": "분석에 실패했습니다."})
                return
            if event is done:
                return

            assert isinstance(event, dict)
            if event["type"] == "done":
                yield _sse(
                    "done",
                    {"type": "done", "recommendation": event["recommendation"].model_dump()},
                )
            else:
                yield _sse(event["type"], event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # 프록시(Next dev 서버의 rewrite 포함)가 응답을 모아 두면 스트리밍이 의미가 없다.
            "X-Accel-Buffering": "no",
        },
    )
