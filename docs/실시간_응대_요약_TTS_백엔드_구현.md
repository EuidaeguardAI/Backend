# 실시간 응대 요약·TTS 백엔드 구현

> 후속 변경: `full`/`compact` 모드 분리로 두 보조 필드는 이제 compact 출력에서만 생성한다. 최신 계약과 흐름은 `응답_모드_full_compact_백엔드_구현.md`를 기준으로 한다.

## 목적

기존 고객 응대 문장인 `sayNow`의 역할과 길이를 유지하면서, 한 번의 LLM 구조화 응답에서 직원용 요약과 음성 코칭 문장을 함께 생성한다.

- `sayNow`: 고객에게 직접 말하는 완전한 응대 문장
- `glanceSummary`: 직원이 2초 안에 파악할 수 있는 최대 3단계 행동 요약
- `ttsText`: 직원에게 행동 순서를 알려주는 자연스러운 내부 코칭 문장

별도의 요약 LLM 호출이나 중복 RAG 검색은 추가하지 않았다.

## 변경 파일

- `app/schemas.py`
- `app/graph/recommendation_graph.py`
- `app/safety/emergency_rules.py`

## 데이터 계약

`Recommendation`과 LLM 구조화 출력용 `RecommendationDraft`에 다음 필드를 추가했다.

```python
glanceSummary: str
ttsText: str
```

백엔드에서 생성되는 새 추천은 두 필드를 항상 포함한다. 기존 `sayNow`, 인용, 후속 행동, 금지 행동, 검토 필요 여부와 안전 스크립트 표시는 변경하지 않았다.

## LLM 생성 규칙

`recommendation_graph.py`의 기존 시스템 프롬프트에 다음 규칙을 추가했다.

1. `sayNow`는 기존처럼 정중한 고객 응대용 1~2문장으로 작성한다.
2. `glanceSummary`는 최대 3개의 짧은 행동 단위를 ` → `로 연결한다.
3. `glanceSummary`는 전체 35자 이내를 목표로 하며, 법률 설명이나 긴 부연을 넣지 않는다.
4. `ttsText`는 50자 이내의 자연스러운 직원 코칭 한 문장을 목표로 한다.
5. `ttsText`에는 목록 기호, 화살표, 괄호를 사용하지 않는다.
6. 세 문구는 같은 대응 순서와 내용을 가리키며 서로 모순되지 않아야 한다.
7. 세 문구를 포함한 모든 출력은 한국어로 작성한다.

구조화 출력 모델 자체에 새 필드를 포함했으므로 기존 `generate_node`의 단일 LLM 요청에서 함께 생성된다. 검색 노드와 인용 근거 검증 흐름은 그대로 유지된다.

## 고정 안전 추천

위협 또는 반복적인 위험 상황에서 사용하는 고정 안전 추천에는 다음 값을 명시적으로 추가했다.

```json
{
  "glanceSummary": "거리 확보 → 관리자 호출 → 필요 시 신고",
  "ttsText": "고객과 거리를 확보하고 관리자 또는 경찰의 도움을 요청하세요."
}
```

기존 고정 `sayNow`, 위험도 5, 신고·거리 확보·관리자 호출 절차와 인용 정보는 변경하지 않았다.

## 처리 흐름

```text
실시간 발화
  ├─ 고정 안전 조건 충족 → fixed_safety → 세 문구가 포함된 고정 Recommendation
  └─ 일반 상황 → 기존 RAG 검색 → 단일 구조화 LLM 요청 → 세 문구가 포함된 Recommendation
```

## 검증

- `uv run --with-requirements requirements.txt python -m compileall -q app`: 통과
- 고정 안전 추천을 실제로 생성하고 `glanceSummary`, `ttsText` 존재 여부 확인: 통과
- Git diff whitespace 검사: 통과

실제 LLM의 글자 수와 표현 품질은 모델 지시를 통해 유도한다. 운영 환경에서는 일반 환불·교환 대화를 사용해 세 문구의 일관성과 길이를 추가로 관찰해야 한다.
