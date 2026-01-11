import os
import time
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
from src.config import DRUG_CSV_FILE, FAISS_INDEX_DIR, GOOGLE_API_KEY
from src.prompts import ROUTER_PROMPT_TEMPLATE

# 전역 변수 설정
client = genai.Client(api_key=GOOGLE_API_KEY)
df_filtered = None
vector_db = None

def build_or_load_faiss(df, faiss_dir=FAISS_INDEX_DIR):
    global df_filtered
    selected_cols = ['제품명', '효능', '복용방법', '주의사항', '상호작용', '부작용', '보관방법']
    df_filtered = df[selected_cols].fillna("정보 없음")

    embeddings = HuggingFaceEmbeddings(
        model_name="Yoonyoul/fine-tuned-e5-small-drugproduct",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(faiss_dir):
        print("기존 FAISS 벡터 DB를 불러옵니다...")
        return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)

    print("FAISS 벡터 DB를 새로 생성합니다 (검색 최적화 구조)...")
    documents = []
    for _, row in df_filtered.iterrows():
        # [수정] 검색용 텍스트는 제품명과 핵심 효능만 (독해 속도 향상)
        search_content = f"제품명: {row['제품명']}"

        # [수정] 모든 상세 정보를 메타데이터로 각각 저장
        metadata = {
            "item_name": row['제품명'],
            "효능": row['효능'],
            "복용방법": row['복용방법'],
            "주의사항": row['주의사항'],
            "상호작용": row['상호작용'],
            "부작용": row['부작용'],
            "보관방법": row['보관방법']
        }
        documents.append(Document(page_content=search_content, metadata=metadata))

    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local(faiss_dir)
    return vector_db

# 초기화 (import 시 실행)
if os.path.exists(DRUG_CSV_FILE):
    df_drug = pd.read_csv(DRUG_CSV_FILE)
    vector_db = build_or_load_faiss(df_drug, FAISS_INDEX_DIR)
else:
    print(f"Warning: {DRUG_CSV_FILE} not found. Please run data_fetcher.py first.")


def hybrid_search(ocr_text, threshold=0.6):
    """
    약 이름 전용 Hybrid Search

    1. 키워드 매칭 → 후보군 생성
       - 1개면 즉시 반환
       - 여러 개면 점수화 후 확신 있을 때만 반환
    2. 애매하거나 실패 시 → 벡터 유사도 검색
    """
    global vector_db, df_filtered
    if vector_db is None or df_filtered is None:
         print("DB not initialized.")
         return None

    start_time = time.time()

    # =========================
    # A. 키워드 기반 후보군 생성
    # =========================
    candidates = df_filtered[
        df_filtered['제품명'].str.contains(ocr_text, na=False, case=False)
    ].copy()

    # -------------------------
    # A-1. 후보가 1개인 경우
    # -------------------------
    if len(candidates) == 1:
        row = candidates.iloc[0]
        target_name = row['제품명']

        metadata = {col: row[col] for col in df_filtered.columns}
        metadata['item_name'] = target_name

        doc = Document(
            page_content=f"제품명: {target_name}",
            metadata=metadata
        )

        print(f"hybrid_search Latency: {time.time() - start_time:.4f}초")
        print(f"[키워드 매칭 단일 후보] '{target_name}' 즉시 반환")
        return doc

    # -------------------------
    # A-2. 후보가 여러 개인 경우 → 점수화
    # -------------------------
    if len(candidates) > 1:
        important_tokens = ['mg', '정', '캡슐', '시럽', '서방', 'ER', 'CR']

        def token_score(name, query):
            score = 0
            for t in important_tokens:
                if t in query and t in name:
                    score += 1
            return score

        candidates['token_score'] = candidates['제품명'].apply(
            lambda x: token_score(x, ocr_text)
        )
        candidates['len_diff'] = candidates['제품명'].str.len().sub(len(ocr_text)).abs()

        # 점수 기반 정렬
        candidates = candidates.sort_values(
            ['token_score', 'len_diff'],
            ascending=[False, True]
        )

        best = candidates.iloc[0]
        second = candidates.iloc[1]

        # -------------------------
        # A-3. "확신 조건"
        #  - 토큰 점수 차이 존재
        #  - 길이 차이가 충분히 작음
        # -------------------------
        confident = (
            best['token_score'] > second['token_score'] or
            best['len_diff'] <= 2
        )

        if confident:
            target_name = best['제품명']
            metadata = {col: best[col] for col in df_filtered.columns}
            metadata['item_name'] = target_name

            doc = Document(
                page_content=f"제품명: {target_name}",
                metadata=metadata
            )

            print(f"hybrid_search Latency: {time.time() - start_time:.4f}초")
            print(f"[키워드 매칭 다중 후보 → 확신] '{target_name}' 반환")
            return doc

        else:
            print("[키워드 매칭 다중 후보] 확신 부족 → 벡터 검색으로 전환")

    # =========================
    # B. 벡터 유사도 검색
    # =========================
    print(f"[벡터 검색 실행] '{ocr_text}'와 비슷한 약을 찾는 중...")
    results_with_score = vector_db.similarity_search_with_score(ocr_text, k=1)

    if not results_with_score:
        print(f"hybrid_search Latency: {time.time() - start_time:.4f}초")
        return None

    doc, score = results_with_score[0]
    print(f"유사도 점수(Distance): {score:.4f}")

    if score > threshold:
        print(f"hybrid_search Latency: {time.time() - start_time:.4f}초")
        print(f"[주의] 유사도 점수({score:.4f})가 임계값 초과 → 제외")
        return None

    print(f"hybrid_search Latency: {time.time() - start_time:.4f}초")
    return doc

# [신규] 질문 의도 분석 (라우터) + 타겟팅 기능 추가
def route_drug_intent(user_query, drug_candidates):
    start_time = time.time()

    # 질문이 없거나 약 목록이 없으면 기본값 반환 (전체 약, 기본 설명)
    if not user_query or not drug_candidates:
        return ["ALL"], ["효능", "복용방법", "주의사항"]

    # 프롬프트에 약 목록 주입
    candidates_str = ", ".join(drug_candidates)

    router_prompt = ROUTER_PROMPT_TEMPLATE.format(
        user_query=user_query,
        candidates_str=candidates_str
    )

    try:
        #model = "gemini-flash-latest"
        model = "gemini-2.5-flash-lite"
        res = client.models.generate_content(
            model=model,
            contents=router_prompt
        )

        text = res.text.strip()
        print(f"[Router Output] {text}\n") # 디버깅용

        # '|' 기준으로 파싱
        parts = text.split("|")
        if len(parts) != 2:
            # 파싱 실패 시 안전하게 전체/기본 반환
            return ["ALL"], ["효능", "복용방법"]

        targets_str, fields_str = parts

        # 1. 타겟 약 파싱
        targets = [t.strip() for t in targets_str.split(",")]

        # 2. 필요 정보 파싱
        ALLOWED_FIELDS = ["효능", "복용방법", "주의사항", "상호작용", "부작용", "보관방법", "그외"]
        intents = [i.strip() for i in fields_str.split(",")]
        intents = [i for i in intents if i in ALLOWED_FIELDS]

        # 필드가 하나도 안 잡히면 '그외' 처리
        if not intents:
            intents = ["그외"]

        latency = (time.time() - start_time)
        print(f"[Latency] route_drug_intent: {latency:.4f} 초")

        return targets, intents

    except Exception as e:
        # LLM 실패 시 안전하게 전체/기본 반환
        print(f"Router Error: {e}")
        return ["ALL"], ["효능", "복용방법"]


# [수정] 동적 컨텍스트 생성 함수 (기저질환 로직 추가)
def get_dynamic_drug_context(ocr_result_list, user_query="", user_conditions=[]):
    start_time = time.time()

    # 1. 라우터 호출
    target_drugs, required_fields = route_drug_intent(user_query, ocr_result_list)

    # [신규] 기저질환이 선택된 경우, 안전 점검을 위해 '주의사항'과 '부작용' 정보를 강제로 가져옴
    if user_conditions:
        print(f"[Safety Check] 기저질환({user_conditions}) 감지 -> 주의사항/부작용 정보 자동 추가")
        required_fields.extend(["주의사항", "부작용"])
        required_fields = list(set(required_fields)) # 중복 제거

    # Case A. "그외" 질문이면 RAG 생략 (단, 기저질환 체크가 아닐 때만)
    if "그외" in required_fields and not user_conditions:
        print("'그외' 질문 -> RAG 컨텍스트 생성 안 함")
        return ""

    # 2. 검색 대상 설정 (Targeting)
    search_targets = []

    if "ALL" in target_drugs:
        print("검색 대상: 전체 목록 (OCR 결과)")
        search_targets = ocr_result_list
    else:
        print(f"검색 대상: 사용자 지정 약물 {target_drugs}")
        search_targets = target_drugs

    new_context_parts = []

    for drug_name in search_targets:
        doc = hybrid_search(drug_name)

        if doc is None:
            print(f"'{drug_name}' 정보를 DB에서 찾을 수 없습니다.")
            continue

        meta = doc.metadata
        drug_info = f"제품명: {meta['item_name']}\n"

        for field in required_fields:
            if field in meta:
                drug_info += f"{field}: {meta[field]}\n"

        new_context_parts.append(drug_info)

    latency = (time.time() - start_time)
    print(f"[Latency] get_dynamic_drug_context: {latency:.4f} 초")
    return "\n\n".join(new_context_parts)
