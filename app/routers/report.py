import json
from datetime import datetime, timezone

from fastapi import APIRouter

from app.config import REPORTS_PATH
from app.schemas import ReportRequest, ReportResponse

router = APIRouter()


@router.post("/report", response_model=ReportResponse)
def submit_report(body: ReportRequest) -> ReportResponse:
    try:
        REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "sessionId": body.sessionId,
            "reportedAt": datetime.now(timezone.utc).isoformat(),
        }
        with REPORTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # 저장 실패해도 사용자 흐름은 막지 않는다.
    return ReportResponse(success=True)
