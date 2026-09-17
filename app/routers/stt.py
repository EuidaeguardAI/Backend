from fastapi import APIRouter, File, HTTPException, UploadFile
from openai import OpenAI

from app.config import OPENAI_API_KEY, TRANSCRIBE_MODEL
from app.schemas import SttResponse
from app.stt_filters import filter_transcript
from app.stt_hints import STT_PROMPT

router = APIRouter()
_client = OpenAI(api_key=OPENAI_API_KEY)

# whisper-1은 구간마다 "여기엔 말이 없었을 확률"(no_speech_prob)을 같이 준다.
# 이 값이 높고 인식 신뢰도(avg_logprob)가 낮으면 무음에 대고 지어낸 문장으로 본다.
# gpt-4o-transcribe 계열은 이 값을 주지 않으므로, 확률 기반 컷이 필요하면
# OPENAI_TRANSCRIBE_MODEL=whisper-1 로 바꿔 쓴다(대신 일반 인식 품질은 조금 떨어진다).
_NO_SPEECH_PROB_MAX = 0.6
_AVG_LOGPROB_MIN = -1.0


def _supports_no_speech_prob(model: str) -> bool:
    return model.startswith("whisper")


def _drop_silent_segments(transcription) -> str:  # noqa: ANN001
    """whisper-1의 verbose 응답에서 '말이 없었던' 구간을 빼고 남은 텍스트만 잇는다."""
    segments = getattr(transcription, "segments", None) or []
    if not segments:
        return getattr(transcription, "text", "") or ""

    kept: list[str] = []
    for segment in segments:
        no_speech_prob = getattr(segment, "no_speech_prob", 0.0) or 0.0
        avg_logprob = getattr(segment, "avg_logprob", 0.0) or 0.0
        if no_speech_prob > _NO_SPEECH_PROB_MAX and avg_logprob < _AVG_LOGPROB_MIN:
            continue
        kept.append((getattr(segment, "text", "") or "").strip())
    return " ".join(part for part in kept if part)


@router.post("/stt", response_model=SttResponse)
def transcribe(audio: UploadFile = File(...)) -> SttResponse:
    try:
        file_bytes = audio.file.read()
        use_verbose = _supports_no_speech_prob(TRANSCRIBE_MODEL)
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
            response_format="verbose_json" if use_verbose else "json",
        )
        raw = (
            _drop_silent_segments(transcription)
            if use_verbose
            else (transcription.text or "")
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="음성 인식에 실패했습니다.") from exc

    # 무음 구간에서 모델이 지어낸 문장을 거른다. 걸러낸 경우 빈 문자열을 돌려주면
    # 프론트엔드는 "이번 구간에는 아무 말도 없었다"로 처리한다. (app/stt_filters.py)
    text, filtered = filter_transcript(raw)
    return SttResponse(text=text, filtered=filtered)
