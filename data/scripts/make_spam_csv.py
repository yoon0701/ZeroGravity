# make_spam_csv.py  (LLM-only, folder -> 파일당 1개, 목표 3000개)
import os, re, json, glob, random, argparse, time
from typing import Any, Dict, List
import pandas as pd
from openai import OpenAI

# ---------- CLI ----------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=r"data/raw/spamInstruct", help="파일 또는 폴더 경로")
    ap.add_argument("--out", dest="out_csv", default=r"data/processed/spam3000.csv")
    ap.add_argument("--limit", type=int, default=10000, help="처리할 JSON 파일 개수(폴더일 때 상한)")
    ap.add_argument("--nper", type=int, default=1, help="instruct 하나당 생성 개수 (파일당 1개 로딩과 조합 권장: 1)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", type=int, default=3000, help="최종 생성 목표 개수")
    ap.add_argument("--retries", type=int, default=3, help="LLM 호출 재시도 횟수")
    ap.add_argument("--fallback", action="store_true", help="LLM 실패 시 간단 템플릿으로 대체 생성")
    return ap.parse_args()

# ---------- Regex / Filters ----------
URL_RX   = re.compile(r"(?:https?://\S+|www\.\S+|[a-z0-9.-]+\.(?:com|net|org|co\.kr|kr|io|ai|shop|me)\S*)", re.I)
PHONE_RX = re.compile(r"(?:(?:\+?82[-\s]?)?0?1[0-9][-.\s]?\d{3,4}[-.\s]?\d{4}|\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})")
FORBIDDEN_PII = [r"\b주민등록번호\b", r"\b여권번호\b", r"\b운전면허\b", r"카드번호", r"계좌번호", r"\b주소\b", r"\b실명\b"]

# ---------- Loaders ----------
def list_json_files(path: str, limit: int) -> List[str]:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")))
        return files[:limit] if limit and limit > 0 else files
    return [path]

def load_one_item_per_file(files: List[str]) -> List[Dict[str, Any]]:
    picked = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            data = obj.get("data") or obj.get("Data") or []
            if isinstance(data, list) and data:
                # 파일당 1개: 첫 번째 (원하면 random.choice(data)로 변경)
                item = data[0]
                picked.append(item)
        except Exception as e:
            print(f"⚠️ load fail: {fp} ({e})")
    return picked

def get_instruct_id(item: Dict[str, Any]) -> str:
    return (item.get("instruct_id") or item.get("id") or item.get("UID") or
            f"UNK{random.randint(0,999999)}")

def get_instruct_text(item: Dict[str, Any]) -> str:
    return (item.get("instruct_text") or item.get("instruction") or
            item.get("prompt") or item.get("query") or "")

# ---------- LLM ----------
def build_system_prompt() -> str:
    return (
        "당신은 '스팸/피싱 탐지 모델' 학습용 합성 데이터 생성기입니다.\n"
        "- 실제 개인정보/브랜드/회사명/실전화번/실주소 금지. 반드시 <URL>, <PHONE> placeholder만 사용.\n"
        "- 폭력/혐오/차별/불법 실행 방법 금지. 과장광고/사칭 톤만 허용.\n"
        "- 채널: SMS. 1~2문장, 전체 60~120자. 문장 끝에 마침표/느낌표 사용.\n"
        "- 한국어. 자연스럽고 현실적인 구어체로.\n"
        "- 항목마다 bait 단어(예: 무료/한정/선착순/긴급/혜택/당첨/환급/본인인증/계정정지) 1~2개 포함."
    )

def template_fallback(instruct_text: str) -> str:
    hints = [w for w in ["박람회","이벤트","쿠폰","환급","인증","계정","배송","투자"] if w in instruct_text]
    base = random.choice([
        "지금 확인하지 않으면 이용이 제한될 수 있습니다.",
        "오늘만 한정 혜택이 제공됩니다.",
        "본인 확인 후 절차를 완료해 주세요."
    ])
    tail = random.choice(["확인: <URL>", "상세: <URL>", "문의: <PHONE>"])
    msg = f"{base} {tail}"
    if hints and random.random() < 0.6:
        msg += f" {random.choice(hints)}"
    msg = re.sub(r"\s+", " ", msg).strip()
    if "<URL>" not in msg:
        msg += " 확인: <URL>"
    return msg

def llm_generate(client: OpenAI, model: str, instruct_text: str, n_per: int, retries: int = 3, use_fallback: bool = False) -> List[str]:
    sys = build_system_prompt()
    usr = (
        f"[instruct_text]\n{instruct_text}\n\n"
        "- 반드시 <URL> 1개 포함. <PHONE>은 30~50% 확률로 포함.\n"
        "- JSON만 출력하세요. 스키마: {{\"items\":[{{\"text\":\"...\"}}, ...]}}"
    )
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": usr}],
                temperature=0.8,
                top_p=0.9,
                response_format={"type": "json_object"},
                timeout=40
            )
            raw = resp.choices[0].message.content  # JSON 문자열
            data = json.loads(raw)
            items = data.get("items", [])
            texts = [it["text"] for it in items if isinstance(it, dict) and isinstance(it.get("text"), str)]
            if texts:
                return texts[:n_per]
            last_err = "empty_items"
        except Exception as e:
            last_err = str(e)
            # 지수 백오프
            time.sleep(1.5 * (attempt + 1))
    if use_fallback:
        return [template_fallback(instruct_text) for _ in range(n_per)]
    raise RuntimeError(f"LLM failed after retries: {last_err}")

# ---------- Validation ----------
def sanitize(msg: str) -> str:
    s = msg.strip()
    if "<URL>" not in s:
        s += (" 확인: <URL>" if not s.endswith((".", "!", "…")) else " 확인: <URL>")
    for pat in FORBIDDEN_PII:
        s = re.sub(pat, "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_valid(msg: str) -> bool:
    if not (40 <= len(msg) <= 160): return False
    if "<URL>" not in msg and not URL_RX.search(msg): return False
    # 진짜 전화번호 탐지되면(placeholder 아닌) 제외
    if PHONE_RX.search(msg) and "<PHONE>" not in msg: return False
    for pat in FORBIDDEN_PII:
        if re.search(pat, msg): return False
    return True

# ---------- Main ----------
def main():
    args = get_args()
    random.seed(args.seed)

    files = list_json_files(args.in_path, args.limit)
    print(f"📁 Using {len(files)} file(s):")
    for fp in files[:10]:  # 너무 길면 앞 10개만 표시
        print(" -", os.path.basename(fp))
    if len(files) > 10:
        print(f"   ... (+{len(files)-10} more)")

    items = load_one_item_per_file(files)
    print(f"🧾 Loaded instruct items (1 per file): {len(items)}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=40)
    rows = []
    skipped = 0

    for item in items:
        if len(rows) >= args.target:
            break  # 목표 개수 도달 시 조기 종료

        iid   = get_instruct_id(item)
        itext = get_instruct_text(item).strip()
        if not itext:
            skipped += 1
            continue
        try:
            texts = llm_generate(client, args.model, itext, args.nper, retries=args.retries, use_fallback=args.fallback)
        except Exception as e:
            print(f"⚠️ LLM fail for {iid}: {e}")
            skipped += 1
            continue

        gen_cnt = 0
        for i, t in enumerate(texts, 1):
            if len(rows) >= args.target:
                break
            t2 = sanitize(t)
            if not is_valid(t2):
                continue
            rows.append({
                "text": t2,
                "id": f"{iid}" if args.nper == 1 else f"{iid}-{i:02d}",
                "length": len(t2),
                "has_url": 1 if "<URL>" in t2 or URL_RX.search(t2) else 0,
                "has_phone": 1 if "<PHONE>" in t2 or PHONE_RX.search(t2) else 0,
                "label": 1
            })
            gen_cnt += 1
            if gen_cnt >= args.nper:
                break

    if not rows:
        print("⚠️ No rows generated.")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)

    # 목표 수량 맞추기 (넘치면 샘플링)
    if len(df) > args.target:
        df = df.sample(n=args.target, random_state=args.seed).reset_index(drop=True)

    df = df[["text","id","length","has_url","has_phone","label"]]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"✅ saved: {args.out_csv}  rows={len(df)}  skipped_items={skipped}")

if __name__ == "__main__":
    main()
