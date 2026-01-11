import gradio as gr
import ast
import numpy as np
from google.cloud import texttospeech
import time
import os
import io

# Import from modules
from src.ocr import extract_drugs_from_image, shared_ocr_reader
from src.silver_llm import ask_ai_grandchild
from src.config import DRUG_JSON_CREDENTIALS
from google.cloud import speech

# 인증 키 설정 (기존 설정 유지)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = DRUG_JSON_CREDENTIALS

# [최적화] STT 클라이언트 전역 로드 (미리 연결)
print("⏳ STT 클라이언트 연결 중...")
shared_stt_client = speech.SpeechClient()
print("✅ STT 클라이언트 연결 완료!")

def speech_to_text(audio_path, client=None):
    # 클라이언트가 파라미터로 안 넘어오면 전역 변수 사용
    if client is None:
        client = shared_stt_client

    # 오디오 파일 로드
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    audio = speech.RecognitionAudio(content=audio_bytes)

    # 한국어 설정
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,
        language_code="ko-KR",
        enable_automatic_punctuation=True
    )

    start_time = time.time()  # 시작 시간 기록

    # STT 호출 (client 객체 재사용)
    response = client.recognize(
        config=config,
        audio=audio
    )

    end_time = time.time()
    latency = end_time - start_time
    print(f"음성 인식 지연 시간(Latency): {latency:.3f}초")

    full_text = ""
    for result in response.results:
        full_text += result.alternatives[0].transcript

    return full_text.strip()


# [최적화] TTS 클라이언트 전역 로드 (미리 연결)
print("TTS 클라이언트 연결 중...")
shared_tts_client = texttospeech.TextToSpeechClient()
print("TTS 클라이언트 연결 완료!")

# -------------------------
# TTS (메모리 + latency 로그)
# -------------------------
def tts_to_memory(text, client=None):
    if client is None:
        client = shared_tts_client

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Neural2-B"
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000,
        speaking_rate=1
    )

    start = time.time()
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    print(f"[TTS Latency] {time.time() - start:.3f}s")

    audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
    return (24000, audio_data)

# -------------------------
# Logic 0: OCR Wrapper
# -------------------------
def run_ocr_logic(img_path):
    if img_path is None:
        return "[]"
    
    print(f"[OCR] 이미지 분석 시작: {img_path}")
    try:
        results = extract_drugs_from_image(img_path, reader=shared_ocr_reader)
        drug_names = [item['name'] for item in results if item['name'] != "제품명"] 
        return str(drug_names)
    except Exception as e:
        print(f"[OCR Error] {e}")
        return "['에러가 발생했습니다']"

# -------------------------
# Logic 1: 초기 브리핑
# -------------------------
# [수정] condition_list 추가
def start_briefing(drug_list_str, condition_list):
    start_total_briefing = time.time() 

    try:
        drug_list = ast.literal_eval(drug_list_str)
    except:
        return [{"role": "assistant", "content": "약 목록 형식이 잘못되었습니다."}], None
    
    if not drug_list:
         return [{"role": "assistant", "content": "인식된 약이 없습니다."}], None

    # [수정] 기저질환 전달
    briefing_text = ask_ai_grandchild("", drug_list, user_conditions=condition_list)

    audio_data = tts_to_memory(briefing_text, client=shared_tts_client)

    chat_history = [{"role": "assistant", "content": briefing_text}]

    print(f"[Total Briefing Latency] {time.time() - start_total_briefing:.3f}s")

    return chat_history, audio_data

# -------------------------
# Logic 2: 음성 대화
# -------------------------
# [수정] condition_list 추가
def process_voice_chat(audio_path, drug_list_str, condition_list, chat_history):
    start_total_process = time.time()
    print(f"[Process Voice Chat] Function started.")

    if audio_path is None:
        return chat_history, None

    # 1. STT
    stt_start = time.time()
    user_text = speech_to_text(audio_path, client=shared_stt_client)
    print(f"[STT Latency] {time.time() - stt_start:.3f}s")

    if not user_text:
        return chat_history, None
    print(f"[Process Voice Chat] User text: {user_text}")

    # 2. RAG & LLM
    try:
        drug_list = ast.literal_eval(drug_list_str)
    except:
        drug_list = []

    llm_start = time.time()
    # [수정] 기저질환 전달
    ai_text = ask_ai_grandchild(user_text, drug_list, user_conditions=condition_list)
    print(f"[LLM Latency] {time.time() - llm_start:.3f}s")

    # 3. TTS
    audio_data = tts_to_memory(ai_text, client=shared_tts_client)

    # 4. History update
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": ai_text})

    print(f"[Total Process Latency] {time.time() - start_total_process:.3f}s")

    return chat_history, audio_data

# -------------------------
# UI
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 어르신을 위한 SilverTalk-Medi (말하는 약봉투)")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="약봉투 사진 찍기/올리기")
            btn_ocr = gr.Button("약 찾기 (이미지 분석)", variant="secondary")
            
            drug_input = gr.Textbox(
                label="인식된 약 목록 (OCR 결과)",
                value="[]",
                lines=2,
                interactive=True
            )

            # [신규] 건강 상태 체크박스
            condition_input = gr.CheckboxGroup(
                choices=["고혈압", "당뇨","심장질환","신장질환","위장질환"],
                label="앓고 계신 지병을 체크",
                info="체크하시면 해당 질환에 나쁜 약인지 알려드려요."
            )

            btn_start = gr.Button("설명 듣기 시작", variant="primary")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="대화 내용")
            audio_output = gr.Audio(label="목소리 변환", autoplay=True)

    gr.Markdown("### 궁금한 점을 물어보세요")
    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="마이크"
    )

    # 1. OCR 실행
    btn_ocr.click(
        fn=run_ocr_logic,
        inputs=[image_input],
        outputs=[drug_input]
    )

    # 2. 설명 듣기 (TTS 브리핑) - condition_input 추가
    btn_start.click(
        fn=start_briefing,
        inputs=[drug_input, condition_input],
        outputs=[chatbot, audio_output]
    )

    # 3. 음성 대화 - condition_input 추가
    audio_input.stop_recording(
        fn=process_voice_chat,
        inputs=[audio_input, drug_input, condition_input, chatbot],
        outputs=[chatbot, audio_output]
    )


if __name__ == "__main__":
    demo.launch(debug=True, share=True)
