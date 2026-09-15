from fastapi import APIRouter, HTTPException

from app.graph.recommendation_graph import run_recommendation_graph
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
        )
        return AnalyzeResponse(recommendation=recommendation)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="분석에 실패했습니다.") from exc
