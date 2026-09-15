"""
app/safety/abusive_lexicon.py의 어휘 목록을 프론트엔드 TS 파일로 내보낸다.

같은 사전을 두 언어로 손으로 관리하면 반드시 어긋나므로, 파이썬 쪽을 원본으로 두고
아래 명령으로 TS를 다시 생성한다.

    python scripts/export_lexicon_ts.py
"""

import io
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.safety import abusive_lexicon as lex  # noqa: E402

TARGET = (
    BASE_DIR.parent
    / "euidaeguard-web"
    / "src"
    / "lib"
    / "safety"
    / "abusiveLexicon.ts"
)

GROUPS = [
    ("PROFANITY", lex.PROFANITY, "욕설·인신공격"),
    ("SEVERE_PROFANITY", lex.SEVERE_PROFANITY, "강한 욕설"),
    ("CHOSUNG_PROFANITY", lex.CHOSUNG_PROFANITY, "초성 욕설"),
    ("CONTEMPT", lex.CONTEMPT, "인격 모독·직업 비하"),
    ("SEXUAL_HARASSMENT", lex.SEXUAL_HARASSMENT, "성희롱"),
    ("THREAT", lex.THREAT, "위협·협박"),
    ("WEAPON", lex.WEAPON, "흉기 언급"),
]


def _format_list(words: list[str]) -> str:
    """한 줄에 6개씩 끊어서 보기 좋게 출력한다."""
    lines = []
    for i in range(0, len(words), 6):
        chunk = ", ".join(json.dumps(word, ensure_ascii=False) for word in words[i : i + 6])
        lines.append("  " + chunk + ",")
    return "\n".join(lines)


HEADER = """// 이 파일은 자동 생성됩니다. 직접 수정하지 마세요.
// 원본: euidaeguard-backend/app/safety/abusive_lexicon.py
// 재생성: cd euidaeguard-backend && python scripts/export_lexicon_ts.py
//
// 욕설·위협 감지 전용 어휘 사전. 화면에 보여 주거나 생성 프롬프트에 넣지 않는다.
"""

BODY = """
// 공백·문장부호를 지워 "씨 발!!"과 "씨발"을 같게 만든다.
const NON_WORD = /[^0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ]+/g;

export function normalize(text: string): string {
  return text.toLowerCase().replace(NON_WORD, "");
}

function normalizedSet(...groups: string[][]): string[] {
  const seen = new Set<string>();
  for (const group of groups) {
    for (const word of group) {
      const normalized = normalize(word);
      if (normalized) seen.add(normalized);
    }
  }
  return [...seen];
}

const PROFANITY_N = normalizedSet(PROFANITY, SEVERE_PROFANITY, CHOSUNG_PROFANITY);
const SEVERE_N = normalizedSet(SEVERE_PROFANITY);
const CONTEMPT_N = normalizedSet(CONTEMPT);
const SEXUAL_N = normalizedSet(SEXUAL_HARASSMENT);
const WEAPON_N = normalizedSet(WEAPON);
const THREAT_ONLY_N = normalizedSet(THREAT);
// 고정 안전 절차 트리거는 위협 + 흉기를 함께 본다.
const THREAT_N = normalizedSet(THREAT, WEAPON);

function match(text: string, words: string[]): string[] {
  const haystack = normalize(text);
  if (!haystack) return [];
  return words.filter((word) => haystack.includes(word));
}

export const matchProfanity = (text: string) => match(text, PROFANITY_N);
export const matchSevereProfanity = (text: string) => match(text, SEVERE_N);
export const matchContempt = (text: string) => match(text, CONTEMPT_N);
export const matchSexualHarassment = (text: string) => match(text, SEXUAL_N);
export const matchWeapon = (text: string) => match(text, WEAPON_N);
export const matchThreat = (text: string) => match(text, THREAT_N);

/** 감지된 카테고리별 매칭 단어. 비어 있는 카테고리는 넣지 않는다. */
export function detectCategories(text: string): Record<string, string[]> {
  const found: Record<string, string[]> = {
    흉기: matchWeapon(text),
    위협: match(text, THREAT_ONLY_N),
    성희롱: matchSexualHarassment(text),
    욕설: matchProfanity(text),
    인격모독: matchContempt(text),
  };
  return Object.fromEntries(
    Object.entries(found).filter(([, words]) => words.length > 0),
  );
}
"""


if __name__ == "__main__":
    chunks = [HEADER]
    for name, words, label in GROUPS:
        chunks.append(f"\n// {label}\nexport const {name}: string[] = [\n{_format_list(words)}\n];\n")
    chunks.append(BODY)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    io.open(TARGET, "w", encoding="utf-8", newline="\n").write("".join(chunks))
    print(f"generated {TARGET}")
