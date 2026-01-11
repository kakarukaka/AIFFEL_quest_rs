import time
import re
import pandas as pd
from google import genai
from google.genai import types
from src.config import GOOGLE_API_KEY
from src.prompts import LLM_JUDGE_PROMPT_TEMPLATE
from src.rag import get_dynamic_drug_context
from src.silver_llm import generate_silver_response_stable, ask_ai_grandchild

client = genai.Client(api_key=GOOGLE_API_KEY)

def evaluate_with_fastchat(user_query, context, ai_response):
    """
    FastChat 스타일 프롬프트를 사용한 LLM-as-a-Judge 평가 함수
    """
    
    # 1. FastChat 구조에 맞게 입력 데이터 구성
    combined_question = f"""[User Query]
{user_query}

[Context/Reference]
{context}"""

    # 2. FastChat Prompt Template (Single-v1)
    prompt_template = LLM_JUDGE_PROMPT_TEMPLATE

    # 3. 프롬프트 완성
    final_prompt = prompt_template.format(
        question=combined_question,
        answer=ai_response
    )

    # 4. LLM 호출 (Judge)
    try:
        model = "gemini-2.5-flash"
        response = client.models.generate_content(
            model=model,
            contents=final_prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Evaluation Error: {e}"

def extract_score(judge_text):
    match = re.search(r"\[\[(\d+)\]\]", judge_text)
    return int(match.group(1)) if match else None

def normalize_judge_query(q):
    if q.strip() == "":
        return "Initial turn: The assistant provides a general medication overview based on the given context."
    return q

if __name__ == "__main__":
    drugs = [["에스부펜정",
    "몬테시움정10mg",
    "슈다페드정",
    "엘도란트캡슐",
    "에스원엠프정20mg",
    "무코스타서방정150mg"],

    ["시너젯세미정",
    "에이스타정",
    "록소론정",
    "올페드린정35mg"],

    ["소론도정",
    "아제틴정",
    "액티피드정",
    "휴텍스파모티딘정20mg"]]

    query = ["", "타이레놀과 같이 먹어도 돼?", "", "이 약 속이 울렁거려", "이 약 계속 먹어도 괜찮아?","이 약 언제 먹는 거야?"]

    real_conditions = ["고혈압, 당뇨"]

    results = []

    for drug_idx, real_drugs in enumerate(drugs):
        print(f"\n==============================")
        print(f"💊 Drug Set {drug_idx+1}")
        print(f"약 목록: {real_drugs}")
        print(f"기저질환: {real_conditions}")
        print(f"==============================\n")

        for i, q in enumerate(query):
            print(f"🔹 Query {i+1}/{len(query)}")

            # 1️⃣ RAG Context
            real_context = get_dynamic_drug_context(real_drugs, q, real_conditions)

            # 2️⃣ AI 응답 생성
            real_response = generate_silver_response_stable(
                real_context,
                q,
                real_conditions
            )

            print(f"🤖 AI 응답: {real_response}")

            # 3️⃣ Judge 평가
            judge_query = normalize_judge_query(q)
            judge_result = evaluate_with_fastchat(
                judge_query,
                real_context,
                real_response
            )

            score = extract_score(judge_result)

            print(f"⚖️ Judge 평가: {judge_result}")
            print(f"📊 Score: {score}\n")

            # 4️⃣ 결과 저장
            results.append({
                "drug_set": drug_idx + 1,
                "drugs": real_drugs,
                "query": q if q else "[INITIAL]",
                "response": real_response,
                "judge_result": judge_result,
                "score": score
            })

            # 5️⃣ Rate limit 대응
            if (i + 1) % 2 == 0 and (i + 1) < len(query):
                print("⏸️ API rate limit 방어: 20초 대기...\n")
                time.sleep(20)

        print("✅ 해당 drug set 완료\n")

    # judge_result와 score만 추출
    df_scores = pd.DataFrame([{"judge_result": r["judge_result"], "score": r["score"]} for r in results])
    df_scores.to_excel("judge_scores.xlsx", index=False, engine='openpyxl')
