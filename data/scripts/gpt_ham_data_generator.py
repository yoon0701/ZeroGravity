import os
import pandas as pd
import random
import time
from tqdm import tqdm
from openai import OpenAI
from openai import RateLimitError
import re

# ✅ OpenAI 클라이언트 설정


# ✅ 경로 설정
HAM_CSV_PATH = "../processed/ham3000.csv"
OUTPUT_CSV_PATH = "../processed/gpt_augmented_ham500.csv"
TARGET_SAMPLE_SIZE = 500

CONDITIONS = [
    {"has_url": 1, "has_phone": 0},
    {"has_url": 1, "has_phone": 1},
    {"has_url": 0, "has_phone": 1}
]

# ✅ 자연스러운 톤 유도 프롬프트
PROMPT_TEMPLATE = """
너는 평범한 20~30대 한국인이야. 친구나 지인과 메시지를 주고받듯, 아래 조건에 맞는 자연스러운 톡/문자 메시지 한 문장을 만들어줘.

🧩 조건:
- has_url={has_url}, has_phone={has_phone}
- 문장 안에 <URL> 또는 <PHONE>이 자연스럽게 포함되어야 해
- 문장은 20자 이상이며, 짧거나 부자연스러운 표현은 피하고 상황이 느껴지는 문장으로 작성해
- 스팸처럼 보이면 안 되고, 정말 누가 보냈을 법한 일상적인 톤으로 써 줘
- 너무 형식적이거나 정보 전달만 하는 문장은 피하고, 감탄사, 말 줄임, 이모티콘, 반복 문자(ㅋㅋ, ㅎㅎ, ~~ 등)도 자유롭게 사용해
- 다양한 말투(예: 반말, 존댓말, 장난스러운 말투 등)를 섞어서 매번 다르게 써줘

📝 예시:
- 사진 여기 올려놨어 <URL> 한 번 봐봐 ㅋㅋㅋ
- 오늘 저녁 약속 그거 <PHONE>으로 전화 와도 받아줭
- 이거 진짜 재밌닼ㅋㅋㅋ 링크 여기 <URL>
- 혹시 길 헷갈리면 이 번호로 연락주셈 <PHONE>

👉 출력은 문장 하나만! 따옴표 없이 자연스럽게 작성해줘.
"""

# ✅ GPT 호출 함수 (rate limit 자동 처리 + 전처리 포함)
def generate_text_with_gpt(has_url, has_phone):
    user_prompt = PROMPT_TEMPLATE.format(has_url=has_url, has_phone=has_phone)
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "너는 자연스러운 한국어 대화 문장을 만드는 AI야."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0
            )
            gen_text = response.choices[0].message.content.strip()

            # 큰따옴표 제거
            if gen_text.startswith('"') and gen_text.endswith('"'):
                gen_text = gen_text[1:-1].strip()
            gen_text = re.sub(r'^"(.*)"$', r'\1', gen_text).strip()

            return gen_text
        except RateLimitError:
            print("⏳ Rate limit에 걸림. 5초 후 재시도...")
            time.sleep(5)
        except Exception as e:
            print(f"[!] GPT 호출 실패: {e}")
            return None

# ✅ 기존 데이터 불러오기 (중복 방지용)
existing_df = pd.DataFrame()
if os.path.exists(OUTPUT_CSV_PATH):
    existing_df = pd.read_csv(OUTPUT_CSV_PATH)
    print(f"📂 기존 생성된 샘플: {len(existing_df)}개")

# ✅ 원본 ham 중에서 추출
df = pd.read_csv(HAM_CSV_PATH)

# ✅ 이미 사용한 ID는 제외
used_ids = set(existing_df["id"]) if not existing_df.empty else set()
available_df = df[~df["id"].isin(used_ids)].reset_index(drop=True)

# ✅ 남은 수 만큼만 생성
remaining = TARGET_SAMPLE_SIZE - len(existing_df)
if remaining <= 0:
    print("✅ 이미 충분한 샘플이 존재합니다. 생성할 필요 없음.")
    exit()

print(f"🚀 생성할 샘플 수: {remaining}개")

# ✅ 생성 시작
generated_data = existing_df.to_dict("records")  # 이어쓰기
start_index = len(existing_df)

for i in tqdm(range(remaining)):
    row = available_df.loc[i]
    cond = random.choice(CONDITIONS)
    gen_text = generate_text_with_gpt(**cond)

    if gen_text is None:
        continue

    new_row = {
        "text": gen_text,
        "id": row["id"],
        "length": len(gen_text),
        "has_url": cond["has_url"],
        "has_phone": cond["has_phone"],
        "label": 0
    }
    generated_data.append(new_row)

    # ✅ 중간 저장: 10개 단위로 저장
    if (i + 1) % 10 == 0 or (i + 1) == remaining:
        temp_df = pd.DataFrame(generated_data)
        temp_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"💾 중간 저장됨: {len(temp_df)}개")

print(f"\n✅ 최종 저장 완료: {len(generated_data)}개 → {OUTPUT_CSV_PATH}")
