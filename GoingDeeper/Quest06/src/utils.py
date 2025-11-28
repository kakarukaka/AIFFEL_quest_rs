import numpy as np
import matplotlib.pyplot as plt
import torch
import os

NUMBERS = "0123456789"
ENG_CHAR_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TARGET_CHARACTERS = ENG_CHAR_UPPER + NUMBERS

class LabelConverter(object):
     def __init__(self, character):
         self.character = "-" + character
         self.label_map = dict()
         for i, char in enumerate(self.character):
             self.label_map[char] = i

     def encode(self, text):
         encoded_label = []
         for i, char in enumerate(text):
             encoded_label.append(self.label_map[char])
         return np.array(encoded_label, dtype=np.int32)

     def decode(self, encoded_label):
         target_characters = list(self.character)
         decoded_label = ""
         for encode in encoded_label:
             decoded_label += self.character[encode]
         return decoded_label

def levenshtein_distance(s1, s2):
    """두 문자열 간의 Levenshtein Distance(편집 거리)를 계산"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compute_metric(output, labels_padded, label_lengths, label_converter):
    """
    Batch 단위로 Accuracy와 Edit Distance를 계산
    """
    # 1. 예측값 디코딩 (Greedy)
    # (T, B, C) -> (B, T)
    out = output.detach().cpu().numpy()
    argmax = out.argmax(axis=2).transpose()

    preds_str = []
    for seq in argmax:
        decoded = []
        prev = None
        for idx in seq:
            if idx != 0 and idx != prev: # blank(0) 제거 및 중복 제거
                decoded.append(idx)
            prev = idx
        preds_str.append(label_converter.decode(decoded))

    # 2. 정답 라벨 디코딩
    labels_str = []
    for i in range(len(labels_padded)):
        length = label_lengths[i]
        # 패딩 제거 후 디코딩
        label_seq = labels_padded[i][:length].cpu().numpy()
        labels_str.append(label_converter.decode(label_seq))

    # 3. Metric 계산
    correct_count = 0
    total_distance = 0
    batch_size = len(preds_str)

    for pred, gt in zip(preds_str, labels_str):
        if pred == gt:
            correct_count += 1
        total_distance += levenshtein_distance(pred, gt)

    return correct_count, total_distance, batch_size

def decode_greedy(output, label_converter):
    # (T,B,C) -> (B,T) index
    out = output.detach().cpu().numpy()  # (T,B,C)
    argmax = out.argmax(axis=2).transpose()  # (B,T)

    results = []
    for seq in argmax:
        decoded = []
        prev = None
        for idx in seq:
            if idx != 0 and idx != prev:
                decoded.append(idx)
            prev = idx
        decoded_str = label_converter.decode(decoded)
        results.append(decoded_str)
    return results

import os
import torch
import matplotlib.pyplot as plt

def plot_training_metrics(checkpoint_path, start_epoch=None):
    """
    체크포인트 파일의 학습 기록을 시각화합니다.
    
    Parameters:
    - checkpoint_path: .pt 또는 .pth 파일 경로
    - start_epoch: None이면 처음부터, 정수 k이면 k번째 epoch부터 그리기 (1-based indexing)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 필요한 키들이 있는지 확인
    if 'train_losses' not in checkpoint:
        print("Checkpoint에 train_losses가 없습니다.")
        return
    
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_accs = checkpoint['train_accuracies']
    val_accs = checkpoint['val_accuracies']
    train_dists = checkpoint.get('train_edit_dists', [])
    val_dists = checkpoint.get('val_edit_dists', [])
    
    # 전체 epoch 범위
    total_epochs = len(train_losses)
    epochs_all = list(range(1, total_epochs + 1))
    
    # start_epoch 처리
    if start_epoch is None or start_epoch < 1:
        start_idx = 0
        displayed_epochs = epochs_all
    else:
        start_epoch = int(start_epoch)
        start_idx = start_epoch - 1  # 1-based -> 0-based 인덱스
        if start_idx >= total_epochs:
            print(f"start_epoch={start_epoch}이 총 epoch 수({total_epochs})보다 큽니다. 전체를 표시합니다.")
            start_idx = 0
        displayed_epochs = epochs_all[start_idx:]
    
    # 슬라이싱
    train_losses = train_losses[start_idx:]
    val_losses = val_losses[start_idx:]
    train_accs = train_accs[start_idx:]
    val_accs = val_accs[start_idx:]
    
    if train_dists:
        train_dists = train_dists[start_idx:]
        val_dists = val_dists[start_idx:]
    
    # 그래프 그리기
    plt.figure(figsize=(20, 6))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(displayed_epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(displayed_epochs, val_losses, label='Validation Loss', marker='o', markersize=3)
    plt.title(f'Loss over Epochs (from epoch {displayed_epochs[0]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(displayed_epochs, train_accs, label='Train Accuracy', marker='o', markersize=3)
    plt.plot(displayed_epochs, val_accs, label='Validation Accuracy', marker='o', markersize=3)
    plt.title(f'Accuracy over Epochs (from epoch {displayed_epochs[0]})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Edit Distance (있을 경우에만)
    if train_dists:
        plt.subplot(1, 3, 3)
        plt.plot(displayed_epochs, train_dists, label='Train Edit Dist', marker='o', markersize=3)
        plt.plot(displayed_epochs, val_dists, label='Validation Edit Dist', marker='o', markersize=3)
        plt.title(f'Edit Distance over Epochs (from epoch {displayed_epochs[0]})')
        plt.xlabel('Epoch')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # Edit Distance가 없을 때 빈 서브플롯 대신 메시지 표시
        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.5, 'No Edit Distance data', 
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()