import os, glob, re, random, datetime
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

try:
    import orjson as json
    def jloads(b: bytes): return json.loads(b)
except Exception:
    import json as _json
    def jloads(b: bytes): return _json.loads(b.decode("utf-8"))

IN_DIR   = r"data/raw/ham"
OUT_CSV  = r"data/processed/ham3000.csv"   # <- 원하는 파일명
HAM_N    = 3000
SEED     = 42

TEXT_KEYS = [
    "text","message","msg","body","content","contents",
    "utterance","utter_text","utter","desc","description","value"
]

SPEAKER_PREFIX = re.compile(r"^\s*(?:\d+|[A-Za-z가-힣._-]+)\s*[:：]\s*")
MULTISPACE = re.compile(r"\s+")

# URL / PHONE 정규식
URL_REGEX = re.compile(
    r"(https?://\S+|www\.\S+|[a-z0-9.-]+\.(?:com|net|org|co\.kr|kr|io|ai|shop|me)\S*)",
    re.IGNORECASE,
)
PHONE_REGEX = re.compile(
    r"(?:(?:\+?82[-\s]?)?0?1[0-9][-.\s]?\d{3,4}[-.\s]?\d{4}|\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})"
)

def take_first_utterance(raw: str) -> Optional[str]:
    """여러 줄 텍스트에서 첫 발화만 추출 + 화자 prefix 제거 + 공백 정리"""
    if not isinstance(raw, str):
        return None
    s = raw.replace("\\n", "\n").replace("\r", "\n")
    for line in s.split("\n"):
        line = line.strip().strip('"').strip("'")
        if not line:
            continue
        line = SPEAKER_PREFIX.sub("", line)
        line = MULTISPACE.sub(" ", line).strip()
        if len(line) >= 3:
            return line
    return None

def load_json(fp: str):
    with open(fp, "rb") as f:
        try:
            return jloads(f.read())
        except Exception:
            return None

def get_text_from_dict(d: Dict[str, Any]) -> Optional[str]:
    for k in TEXT_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None

def collect_texts(obj: Any, acc: List[str]):
    """annotations 내부에서 text 후보 수집 (재귀)."""
    if isinstance(obj, dict):
        t = get_text_from_dict(obj)
        if t:
            acc.append(t)
        for v in obj.values():
            collect_texts(v, acc)
    elif isinstance(obj, list):
        for it in obj:
            collect_texts(it, acc)

def normalize_text(s: str) -> str:
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = SPEAKER_PREFIX.sub("", s)
    s = MULTISPACE.sub(" ", s).strip().strip('"').strip("'")
    return s

def first_line_from_lines(anns: Dict[str, Any]) -> Optional[str]:
    """가능하면 lines[0].norm_text -> lines[0].text 순으로 첫 줄을 뽑는다."""
    lines = anns.get("lines")
    if isinstance(lines, list) and lines:
        cand = lines[0].get("norm_text") or lines[0].get("text")
        if isinstance(cand, str) and cand.strip():
            cand = SPEAKER_PREFIX.sub("", cand)
            cand = MULTISPACE.sub(" ", cand).strip()
            if len(cand) >= 3:
                return cand
    return None

def extract_first(fp: str) -> Optional[Tuple[str, str]]:
    """
    반환: (text, sample_id)
      - text: 첫 발화
      - sample_id: info[0].id 또는 filename(확장자 제거)
    """
    j = load_json(fp)
    if not isinstance(j, dict):
        return None

    info = j.get("info")
    if not isinstance(info, list) or not info:
        return None

    block = info[0]
    anns = block.get("annotations")
    if not isinstance(anns, dict):
        return None

    # id 우선순위: info.id -> filename(확장자 제거)
    sample_id = str(block.get("id")) if block.get("id") is not None else None
    if not sample_id:
        filename = block.get("filename") or os.path.basename(fp)
        sample_id = os.path.splitext(str(filename))[0]

    # 1) 가장 신뢰도 높은 방법: lines[0]에서 바로 추출
    first = first_line_from_lines(anns)
    if first:
        return first, sample_id

    # 2) fallback: annotations 전체 텍스트에서 첫 줄 추출
    acc: List[str] = []
    collect_texts(anns, acc)
    for raw in acc:
        t = take_first_utterance(raw)
        if t and len(t) >= 3:
            return t, sample_id

    return None

def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "**", "*.json"), recursive=True))
    print(f"Found {len(files)} json files in {IN_DIR}")

    rows = []
    skipped = 0
    for fp in files:
        res = extract_first(fp)
        if res is None:
            skipped += 1
            continue
        text, sample_id = res
        text_norm = normalize_text(text)

        # has_url / has_phone
        has_url = 1 if URL_REGEX.search(text_norm) else 0
        has_phone = 1 if PHONE_REGEX.search(text_norm) else 0

        rows.append({
            "text": text_norm,
            "id": sample_id,
            "length": len(text_norm),
            "has_url": has_url,
            "has_phone": has_phone,
            "label": 0,  # 햄
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"⚠️ No rows collected. skipped_files={skipped}")
        return

    # 텍스트 완전 중복 제거 (동일 문장 중복 방지)
    df = df.drop_duplicates(subset=["text"])

    # 3000개 샘플링
    random.seed(SEED)
    if len(df) > HAM_N:
        df = df.sample(n=HAM_N, random_state=SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # 원하는 컬럼 순서로 저장
    df = df[["text", "id", "length", "has_url", "has_phone", "label"]]
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"✅ saved: {OUT_CSV}  shape={df.shape}  skipped_files={skipped}")

if __name__ == "__main__":
    main()
