# SilverTalk-Medi (말하는 약봉투)

어르신들을 위해 복잡한 약 정보를 쉽고 친절하게 설명해주는 **AI 말하는 약봉투** 서비스입니다.
약봉투를 사진으로 찍으면, AI가 약 이름을 인식하고(OCR), 효능/복용법/주의사항을 읽어줍니다(TTS).
또한 궁금한 점을 음성으로 물어보면, 할머니/할아버지의 눈높이에 맞춰 친절하게 대답해줍니다.

## 시연 영상 (Demo)

<video src="test.MP4" controls="controls" width="100%"></video>

> **참고**: 로컬 환경(VS Code 등)에서는 위 플레이어로 바로 재생이 가능합니다.
> 만약 GitHub 웹사이트에서 보실 예정이라면, 파일을 이슈(Issue) 입력창에 드래그하여 업로드한 뒤 생성된 링크를 이곳에 붙여넣어야 웹에서 바로 재생됩니다.

[영상 파일 직접 열기 (test.MP4)](test.MP4)

## 주요 기능

- **약봉투 인식 (OCR)**: 약봉투 사진에서 약 이름을 자동으로 추출합니다.
- **음성 브리핑 (TTS)**: 인식된 약의 효능과 복용법을 또박또박 읽어드립니다.
- **실버 케어 페르소나 (LLM)**: "SilverTalk-Medi" 페르소나가 적용되어, 어르신이 이해하기 쉬운 구어체로 답변합니다.
- **안전 점검 (RAG + Router)**: 기저질환(고혈압, 당뇨 등)을 고려하여 나쁜 영향을 줄 수 있는 약을 경고해줍니다.
- **음성 대화 (STT)**: 타자 치기 어려워하시는 분들을 위해 음성으로 질문을 받고 대답합니다.

## 설치 방법

### 1. 필수 환경 설정
Python 3.8 이상이 필요합니다.

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows
```

### 2. 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 3. API 키 설정
`src/config.py` 파일에 Google Gemini API Key와 공공데이터포털 API Key가 설정되어 있습니다.

## 실행 방법

### 웹 애플리케이션 (Gradio)
```bash
python app.py
```
실행 후 터미널에 표시되는 로컬 URL(예: `http://127.0.0.1:7860`)로 접속하세요.

### 성능 평가 (LLM-as-a-Judge)
```bash
python evaluation.py
```

## 프로젝트 구조

```
MQ04/
├── app.py                  # 메인 실행 파일 (Gradio 앱)
├── evaluation.py           # 성능 평가 스크립트
├── requirements.txt        # 의존성 패키지 목록
├── README.md               # 프로젝트 설명서
├── test.MP4                # 시연 영상
├── src/                    # 소스 코드 디렉토리
│   ├── config.py           # 설정 및 API 키
│   ├── data_fetcher.py     # 약품 정보 수집 (공공데이터 API)
│   ├── ocr.py              # OCR (이미지 텍스트 추출)
│   ├── rag.py              # RAG (벡터 검색 및 라우팅)
│   ├── silver_llm.py       # LLM (답변 생성 및 페르소나)
│   └── prompts.py          # 프롬프트 모음
└── ...
```

## 주의사항
- 본 서비스는 보조적인 정보 제공 수단이며, 의사나 약사의 전문적인 진단을 대체할 수 없습니다.
- 공공데이터포털의 의약품 개요 정보를 기반으로 답변합니다.
