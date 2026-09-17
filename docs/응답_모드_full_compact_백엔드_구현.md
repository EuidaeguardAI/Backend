# 실시간 응답 모드 백엔드 구현

## 목적

`responseMode: "full" | "compact"`에 따라 화면 표현만 바꾸는 것이 아니라 LLM 시스템 프롬프트와 구조화 출력 스키마를 실제로 분리한다. 필드가 없는 기존 `/analyze` 요청은 `full`로 처리한다.

## 변경 파일

- `app/schemas.py`
- `app/routers/analyze.py`
- `app/graph/recommendation_graph.py`
- `app/safety/emergency_rules.py`

## API 계약과 하위 호환

- `AnalyzeRequest.responseMode`의 기본값은 `full`이다.
- `Recommendation.responseMode`에도 실제 생성 모드를 기록하며 기본값은 `full`이다.
- 공통 응답의 `glanceSummary`, `ttsText`는 선택값이다.
- full 응답은 두 필드를 `null`로 직렬화하고 compact 응답은 두 필드를 채운다.

따라서 새 필드가 없는 기존 요청과 저장 추천은 full로 해석할 수 있다.

## 프롬프트 선택

`recommendation_graph.py`에 완전한 독립 문자열 두 개를 둔다.

- `FULL_SYSTEM_PROMPT`: `f37cb9b`의 기존 프롬프트를 복원했다. `sayNow는 1~2문장, 정중하고 짧게. 고객을 자극하는 표현은 doNot에 포함하세요.` 문장을 유지하며 `glanceSummary`와 `ttsText` 지시는 없다.
- `COMPACT_SYSTEM_PROMPT`: `d0c4f04`의 요약·TTS 규칙을 유지하고, 최대 3단계·35자 목표 요약, 50자 목표 내부 코칭, 상호 모순 금지, 제공된 근거 밖 사실 추가 금지와 무근거 법적 단정 금지를 명시한다.

문자열 조각을 조합하지 않으므로 full 프롬프트에 compact 규칙이 섞이지 않는다.

## 구조화 출력 스키마

- `FullRecommendationDraft`: 기존 필드만 포함한다.
- `CompactRecommendationDraft`: full 필드를 상속하고 `glanceSummary`, `ttsText`를 필수로 추가한다.

`generate_node`는 `response_mode`를 한 번 읽어 프롬프트와 모델 클래스를 함께 선택하고 `with_structured_output()`을 한 번 호출한다. full의 JSON 스키마에는 compact 전용 필드가 노출되지 않는다.

## LangGraph와 RAG

```text
START
  ├─ 고정 안전 조건 → fixed_safety → END
  └─ 일반 상황 → retrieve → generate(full 또는 compact) → END
```

`GraphState.response_mode`가 라우터에서 그래프 끝까지 전달된다. 모드 분기는 `generate_node` 내부에서만 일어나므로 `retrieve_node`, 위험 감지와 `ground_citations()`는 추천당 한 번만 실행된다. compact 근거를 위해 별도 검색이나 요약 LLM을 추가하지 않았다.

## 고정 안전 추천

고정 안전 분기는 기존처럼 RAG와 LLM을 건너뛴다. 거리 확보, 관리자 호출, 필요 시 신고 절차와 자동 TTS 금지 판단용 `isFixedSafetyScript`를 유지한다.

- compact: 고정 `glanceSummary`, `ttsText`를 포함한다.
- full: `sayNow`와 안전 절차만 반환하며 두 compact 필드는 비어 있다.
- 두 모드 모두 선택된 `responseMode`를 기록한다.
- 안전 절차 citation은 검색 근거가 아니므로 `sourceType`을 잃지 않게 그대로 공통 응답으로 전달한다.

## 검증 결과

- `uv run --with-requirements requirements.txt python -m compileall -q app`: 통과
- Pydantic 필드 검사: full에는 9개 기존 필드만 있고 compact에만 `glanceSummary`, `ttsText`가 추가됨
- `AnalyzeRequest`와 `Recommendation`의 모드 기본값 `full`: 확인
- 프롬프트 검사: full에 compact 필드명 없음, compact에 요약·TTS·근거 제한 규칙 있음
- 고정 안전 생성 검사: full은 보조 필드 없음, compact는 고정 보조 필드 포함, 양쪽 모두 모드 기록

## 남은 수동 검증

실제 OpenAI 호출이 가능한 운영 환경에서 같은 환불 문의를 두 모드로 보내 문장 품질, compact 길이 목표와 인용 grounding 결과를 확인한다.
