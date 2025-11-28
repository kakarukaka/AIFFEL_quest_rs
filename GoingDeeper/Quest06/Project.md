# OCR Project

CRNN(Convolutional Recurrent Neural Network)을 사용하여 이미지 내 텍스트를 인식하는 Text Recognition 모델을 구현

## 프로젝트 구조

```
.
├── main.ipynb          # 프로젝트 실행을 위한 메인 노트북
├── src/                # 소스 코드 모듈 디렉토리
│   ├── dataset.py      # 데이터셋 로드 (Lazy Loading), 전처리, Augmentation
│   ├── model.py        # CRNN 모델 아키텍처 정의
│   ├── trainer.py      # 학습 루프, 메트릭 기록, 체크포인트 저장
│   ├── inference.py    # 추론, 시각화, End-to-End OCR 파이프라인
│   ├── analysis.py     # 데이터셋 통계 분석 및 샘플 시각화
│   └── utils.py        # 라벨 변환, CTC 디코딩, 메트릭 계산(Edit Distance) 등
└── Project.md           # 프로젝트 설명 파일
```

## 주요 기능 (Features)
 
1.  **Data Augmentation**: `albumentations` 라이브러리를 사용하여 회전, 블러, 색상 변형 등 다양한 증강 기법을 적용, 모델의 강건성(Robustness)을 높여보고자 함
2.  **데이터 분석 및 시각화**:
    *   텍스트 길이 분포 및 이미지 해상도/종횡비 분석 (`src/analysis.py`)
    *   데이터셋 샘플 및 증강된 이미지 시각화 기능 제공
3.  **평가 지표 (Metrics)**:
    *   **Accuracy**: 정확도
    *   **Edit Distance (Levenshtein Distance)**: 예측값과 정답 간의 편집 거리 (유사도 측정)
4.  **End-to-End OCR**: `EasyOCR`로 텍스트 영역을 검출(Detection)하고, 학습된 `CRNN` 모델로 인식(Recognition)하는 통합 파이프라인을 제공

## 요구 사항 (Requirements)

```bash
pip install torch torchvision numpy matplotlib opencv-python lmdb tqdm six easyocr python-Levenshtein albumentations
```

## 사용 방법 (Usage)

1.  **데이터 준비**: MJSynth 데이터셋(`data_lmdb_release`)이 준비되어 있어야 합니다. 경로는 `main.ipynb` 내에서 `HOME` 환경변수를 기준으로 설정됩니다.
2.  **실행**: `main.ipynb` 파일을 열고 셀을 순차적으로 실행합니다.
    *   **데이터셋 분석**: 텍스트 길이, 이미지 크기 분포 확인 및 샘플 시각화
    *   **데이터 로드**: Augmentation이 적용된 학습 데이터 로더 생성
    *   **모델 학습**: CRNN 모델 학습 및 검증 (Best Model 자동 저장)
    *   **결과 시각화**: 학습 과정의 Loss, Accuracy, Edit Distance 그래프 확인
    *   **추론 테스트**: 검증 데이터셋에 대한 모델 성능 확인
    *   **End-to-End OCR**: 실제 이미지(`sample.jpg`)에서 텍스트 검출 및 인식 수행

## 모듈 상세 설명

*   **`src/dataset.py`**: `MJDataset` 클래스는 LMDB 데이터를 로드하며, `albumentations`를 이용한 이미지 증강을 지원합니다. `ctc_collate_fn`은 가변 길이의 텍스트 라벨을 배치 단위로 처리합니다.
*   **`src/model.py`**: VGG 스타일의 CNN 특징 추출기와 BiLSTM 순환 신경망을 결합한 CRNN 모델입니다. 최종 출력은 CTC Loss를 위해 `(Time, Batch, Class)` 형태를 가집니다.
*   **`src/trainer.py`**: `run_training` 함수는 학습 및 검증 루프를 담당합니다. `Adadelta` 옵티마이저(기본값)를 사용하며, `Adam`으로 변경 가능합니다. 학습 중 Best Validation Loss를 갱신할 때마다 모델을 저장합니다.
*   **`src/inference.py`**: `check_inference`는 모델의 추론 결과를 시각화합니다. `detect_text`와 `recognize_img` 함수는 각각 텍스트 검출과 인식을 수행하여 End-to-End OCR을 구현합니다.
*   **`src/analysis.py`**: `analyze_text_length_distribution`, `analyze_image_distribution` 함수로 데이터셋의 통계적 특성을 파악하고, `visualize_dataset_samples`, `visualize_augmented_samples`로 데이터를 시각적으로 확인합니다.
*   **`src/utils.py`**: `LabelConverter`는 문자와 인덱스 간의 변환을 담당합니다. `compute_metric`은 Accuracy와 Edit Distance를 계산하여 모델 성능을 평가합니다.
