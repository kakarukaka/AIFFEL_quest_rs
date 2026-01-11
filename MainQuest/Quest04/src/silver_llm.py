import time
import re
from google import genai
from google.genai import types
from src.config import GOOGLE_API_KEY
from src.prompts import SYSTEM_INSTRUCTION
from src.rag import get_dynamic_drug_context

client = genai.Client(api_key=GOOGLE_API_KEY)

# 전역 변수로 대화 세션 관리 (기억력 유지)
chat_session = None
FIXED_DRUG_CONTEXT = ""

def set_fixed_drug_context(initial_context):
    global FIXED_DRUG_CONTEXT
    FIXED_DRUG_CONTEXT = initial_context

# [수정] user_conditions 파라미터 추가
def generate_silver_response_stable(final_context, user_question="", user_conditions=[]):
    global chat_session
    target_model = "gemini-2.5-flash-lite"
    #target_model = "gemini-flash-latest"

    # [개선] 2. 기억력 추가: 세션이 없으면 새로 생성
    if chat_session is None:
        # [최적화] 시스템 프롬프트에 '사투리 이해' 능력 통합
        system_instruction = SYSTEM_INSTRUCTION
        chat_session = client.chats.create(
            model=target_model,
            config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.6)
        )

    # [신규] Latency 방어 코드
    try:
        if len(chat_session.history) > 20:
            chat_session.history = chat_session.history[-20:]
            print("뇌 용량 확보: 오래된 기억을 정리했습니다. (Latency 최적화)")
    except:
        pass

    start_time = time.time()
    try:
        # 문맥 정보를 포함하여 메시지 전송
        msg = ""
        
        # ===== 고정 약봉투 컨텍스트 =====
        if FIXED_DRUG_CONTEXT:
            msg += f"[기존 복용 중인 약 정보]\n{FIXED_DRUG_CONTEXT}\n\n"
            
        # [신규] 기저질환 정보 주입
        if user_conditions:
            condition_str = ", ".join(user_conditions)
            msg += f"[어르신 건강 상태: {condition_str}]\n이 약들이 건강 상태에 나쁜 영향을 주는지 [주의사항, 부작용] 정보를 통해 체크해서 알려줄 것. 다만 나쁜 영향을 주는 약이 없다면 언급하지 말 것.\n\n"

        if final_context and final_context.strip() != FIXED_DRUG_CONTEXT.strip():
            msg += f"[새로 찾은 약 정보]\n{final_context}\n\n"
        
        msg += f"[질문]\n{user_question}"
        print(f"\n\n\n {msg} \n\n\n")
        response = chat_session.send_message(msg)

        # [정제] 마크다운 기호 제거
        clean_text = re.sub(r'[\*\#\-\_\>]', '', response.text).strip()
        clean_text = " ".join(clean_text.split())

        print(f"최적화된 응답 지연 시간: {time.time() - start_time:.3f}초")
        return clean_text
    except Exception as e:
        return f"죄송해요 할머니, 다시 한번만 말씀해 주셔요. (오류: {e})"

def ask_ai_grandchild(user_query, drug_list, user_conditions=[]):
    start_time = time.time()

    if not user_query or not user_query.strip():
        user_query = "이 약들에 대해 알기 쉽게 요약해서 설명해줘."
        print("[System] 빈 질문이 감지되어 '요약 요청'으로 자동 변환했습니다.")

    # Step 1 & 2
    current_context = get_dynamic_drug_context(drug_list, user_query, user_conditions)

    # ✅ 최초 1회만 약봉투 고정 저장
    if FIXED_DRUG_CONTEXT == "" and current_context:
        set_fixed_drug_context(current_context)

    # Step 3
    response = generate_silver_response_stable(
        current_context,
        user_question=user_query,
        user_conditions=user_conditions
    )

    print(f"총 지연 시간: {time.time() - start_time:.3f}초")
    return response
