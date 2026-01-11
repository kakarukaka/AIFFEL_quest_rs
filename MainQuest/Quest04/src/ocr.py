import cv2
import numpy as np
import pandas as pd
import re
from rapidfuzz import process, fuzz
import easyocr
import time
from src.config import DRUG_NAME_CSV

# [최적화] OCR 모델을 함수 밖에서 미리 로드 (전역 변수)
# 이렇게 하면 함수 실행 때마다 모델을 다시 로드하지 않아 속도가 빨라집니다.
print("⏳ EasyOCR 모델 로딩 중... (최초 1회만 실행)")
shared_ocr_reader = easyocr.Reader(['ko'], gpu=True)
print("✅ EasyOCR 모델 로딩 완료!")

# ==========================================
# 핵심 함수: 약품명 추출 + latency 측정
# ==========================================
def extract_drugs_from_image(img_path, reader=None, db_csv=DRUG_NAME_CSV, score_threshold=70):

    # 1. DB 로드 및 정규화
    df = pd.read_csv(db_csv, header=None)
    db_names_raw = df[0].dropna().astype(str).tolist()
    db_norm = list(set([re.split(r"[\(\[]", str(x).replace(" ", ""))[0] for x in db_names_raw]))

    # 2. OCR Reader 설정
    if reader is None:
        # 파라미터로 받지 않았을 경우 전역 모델 사용 (안전장치)
        reader = shared_ocr_reader

    start_time = time.time()  # 시작 시간 측정
    # 3. 이미지 로드
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4. OCR 전처리 함수
    def preprocess(img, strong=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if strong:
            clahe = cv2.createCLAHE(2.0, (8,8))
            gray = clahe.apply(gray)
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 5)
        return gray

    # 5. 후보군 필터링 함수
    def normalize(s):
        return re.split(r"[\(\[]", str(s).replace(" ", ""))[0]

    def decompose(s):
        out = []
        for ch in s:
            if '가' <= ch <= '힣':
                c = ord(ch) - 0xAC00
                out.extend([c//588, (c%588)//28, c%28])
            else:
                out.append(ch)
        return out

    def jamo_score(a, b):
        return fuzz.ratio(decompose(a), decompose(b))

    DROP_KW = re.compile(
        r"(복약|주의|투약|횟수|일수|보관|용법|용량|식후|식전|안내|사항|약국|조제|약품명|제품명)"
    )

    # 6. OCR 수행
    # reader 객체 재사용
    res_weak = reader.readtext(preprocess(img_rgb), detail=1)
    res_strong = reader.readtext(preprocess(img_rgb, strong=True), detail=1)
    ocr_blocks = res_weak + res_strong

    # 7. 후보군 선정
    candidates = []
    for bbox, text, conf in ocr_blocks:
        t = normalize(text)
        if len(t) >= 2 and not DROP_KW.search(t):
            candidates.append({'text': t, 'bbox': bbox})

    # 8. DB 매칭 및 Crop
    results = []
    seen_names = set()

    for item in candidates:
        cand = item['text']
        match = process.extractOne(cand, db_names_raw, scorer=fuzz.ratio, score_cutoff=50)
        if match:
            best_name, score, _ = match
            final_score = max(score, jamo_score(cand, best_name))
            if final_score >= score_threshold and best_name not in seen_names:
                # Crop
                pts = np.array(item['bbox'], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                H, W = img.shape[:2]
                x, y, w, h = max(0,x), max(0,y), min(W-x,w), min(H-y,h)
                crop_img = img[y:y+h, x:x+w]

                results.append({'name': best_name, 'crop': crop_img})
                seen_names.add(best_name)

    # 함수 종료 시간 계산
    end_time = time.time()
    print(f"▶️ 약품명 추출 완료. 소요 시간: {end_time - start_time:.2f}초")

    # name만 리스트로 추출
    res_names = [item['name'] for item in results if item['name'] != "제품명"]
    print("✅ 검출된 약품명:", res_names)

    return results
