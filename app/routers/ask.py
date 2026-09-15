from fastapi import APIRouter, HTTPException

from app.graph.ask_graph import run_ask_graph
from app.schemas import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question이 필요합니다.")
    try:
        answer = run_ask_graph(
            profile=body.profile,
            history=body.history,
            question=body.question.strip(),
        )
        return AskResponse(answer=answer)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="답변 생성에 실패했습니다.") from exc
