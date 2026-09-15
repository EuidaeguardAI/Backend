from fastapi import APIRouter, File, HTTPException, UploadFile
from openai import OpenAI

from app.config import OPENAI_API_KEY, TRANSCRIBE_MODEL
from app.schemas import SttResponse
from app.stt_hints import STT_PROMPT

router = APIRouter()
_client = OpenAI(api_key=OPENAI_API_KEY)


@router.post("/stt", response_model=SttResponse)
def transcribe(audio: UploadFile = File(...)) -> SttResponse:
    try:
        file_bytes = audio.file.read()
        transcription = _client.audio.transcriptions.create(
            file=(
                audio.filename or "chunk.webm",
                file_bytes,
                audio.content_type or "audio/webm",
            ),
            model=TRANSCRIBE_MODEL,
            # 한국어로 고정한다. 자동 감지에 맡기면 짧은 구간(감탄사, 욕설 한 마디)에서
            # 엉뚱한 언어로 잡혀 영어·일본어 문장이 섞여 들어온다.
            language="ko",
            # 매장 용어·폭언 표현을 미리 알려 인식률을 올린다. (app/stt_hints.py)
            prompt=STT_PROMPT,
            temperature=0,
        )
        return SttResponse(text=transcription.text or "")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="음성 인식에 실패했습니다.") from exc
