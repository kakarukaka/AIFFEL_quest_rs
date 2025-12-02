import os
import torch
import numpy as np
from src.utils import calculate_pixel_accuracy, calculate_mask_iou, calculate_boundary_iou
import time

def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs=5, device='cuda', experiment_dir=None, accumulation_steps=1):
    print(f"\n{'='*40}")
    print(f"=== Training Start: {model_name} ===")
    print(f"{'='*40}")

    # 경로 설정
    base_path = "model"
    if experiment_dir:
        save_path = os.path.join(base_path, experiment_dir)
    else:
        save_path = base_path

    os.makedirs(save_path, exist_ok=True)

    checkpoint_path = os.path.join(save_path, f"{model_name}_checkpoint.pth")
    best_model_path = os.path.join(save_path, f"{model_name}_best.pth")

    start_epoch = 0
    best_loss = float('inf')

    # AMP Scaler 초기화
    scaler = torch.amp.GradScaler('cuda')

    # History Dictionary 초기화
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_iou': [], 'val_iou': [],
        'train_boundary_iou': [], 'val_boundary_iou': [],
        'train_class_iou': {k: [] for k in range(5)},
        'train_class_acc': {k: [] for k in range(5)},
        'val_class_iou': {k: [] for k in range(5)},
        'val_class_acc': {k: [] for k in range(5)}
    }

    # Checkpoint Resume Logic
    if os.path.exists(checkpoint_path):
        print(f"[Info] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

        # Load History if exists
        if 'history' in checkpoint:
            history = checkpoint['history']

            # [Bug Fix] Check for length mismatch if resuming from old checkpoint
            current_len = len(history['train_loss'])

            # 이전 버전 체크포인트 호환성을 위해 키가 없으면 생성 및 Padding
            for key in ['train_class_iou', 'train_class_acc', 'val_class_iou', 'val_class_acc', 'train_boundary_iou', 'val_boundary_iou']:
                if key not in history:
                    # 기존 History 길이만큼 None으로 채워서 길이 맞춤
                    if isinstance(history['train_loss'], list):
                         history[key] = [None] * current_len
                    else: # dict case
                         history[key] = {k: [None] * current_len for k in range(5)}

        print(f"[Info] Resuming from epoch {start_epoch+1} (Best Validation Loss so far: {best_loss:.4f})")

    # Helper function for class-wise metrics
    def get_class_metrics(output, target, num_classes=5):
        with torch.no_grad():
            if isinstance(output, (list, tuple)):
                output = output[-1]
            prediction = torch.argmax(output, dim=1)

            ious = {}
            accs = {}

            for cls in range(num_classes):
                pred_mask = (prediction == cls)
                target_mask = (target == cls)

                intersection = (pred_mask & target_mask).sum().item()
                union = (pred_mask | target_mask).sum().item()
                target_count = target_mask.sum().item()

                # IoU Calculation
                if union > 0:
                    ious[cls] = intersection / union
                else:
                    ious[cls] = None

                # Accuracy (Recall) Calculation
                if target_count > 0:
                    accs[cls] = intersection / target_count
                else:
                    accs[cls] = None
            return ious, accs

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        
        # --- 1. Training Loop ---
        model.train()
        train_loss, train_acc, train_iou, train_boundary_iou = 0.0, 0.0, 0.0, 0.0

        train_cls_iou_sum = {k: 0.0 for k in range(5)}
        train_cls_iou_cnt = {k: 0 for k in range(5)}
        train_cls_acc_sum = {k: 0.0 for k in range(5)}
        train_cls_acc_cnt = {k: 0 for k in range(5)}

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            # AMP Context Manager
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets.long())
                loss = loss / accumulation_steps # Normalize loss

            # Backward Pass with Scaler
            scaler.scale(loss).backward()
            # loss.backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                #optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

            metric_outputs = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
            
            # Metric 계산 (Global)
            train_acc += calculate_pixel_accuracy(metric_outputs, targets)
            train_iou += calculate_mask_iou(metric_outputs, targets)
            train_boundary_iou += calculate_boundary_iou(metric_outputs, targets)

            # Metric 계산 (Class-wise)
            cls_ious, cls_accs = get_class_metrics(outputs, targets)
            for k in range(5):
                if cls_ious[k] is not None:
                    train_cls_iou_sum[k] += cls_ious[k]
                    train_cls_iou_cnt[k] += 1
                if cls_accs[k] is not None:
                    train_cls_acc_sum[k] += cls_accs[k]
                    train_cls_acc_cnt[k] += 1

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_train_boundary_iou = train_boundary_iou / len(train_loader)

        # --- 2. Validation Loop ---
        model.eval()
        val_loss, val_acc, val_iou, val_boundary_iou = 0.0, 0.0, 0.0, 0.0

        val_cls_iou_sum = {k: 0.0 for k in range(5)}
        val_cls_iou_cnt = {k: 0 for k in range(5)}
        val_cls_acc_sum = {k: 0.0 for k in range(5)}
        val_cls_acc_cnt = {k: 0 for k in range(5)}

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                if isinstance(outputs, (list, tuple)):
                    # Deep Supervision인 경우: Metric 계산을 위해 마지막 출력만 사용
                    val_outputs = outputs[-1]
                    # Loss도 마지막 것만 계산하여 Standard U-Net과 스케일 맞춤 (선택사항)
                    # 만약 학습 Loss와 비교하고 싶다면 그대로 두고, 모델 간 비교라면 아래처럼 분리하세요.
                    loss = criterion(val_outputs, targets.long()) 
                else:
                    val_outputs = outputs
                    loss = criterion(outputs, targets.long())
                    
                val_loss += loss.item()

                val_acc += calculate_pixel_accuracy(val_outputs, targets)
                val_iou += calculate_mask_iou(val_outputs, targets)
                val_boundary_iou += calculate_boundary_iou(val_outputs, targets)

                # Metric 계산 (Class-wise)
                cls_ious, cls_accs = get_class_metrics(outputs, targets)
                for k in range(5):
                    if cls_ious[k] is not None:
                        val_cls_iou_sum[k] += cls_ious[k]
                        val_cls_iou_cnt[k] += 1
                    if cls_accs[k] is not None:
                        val_cls_acc_sum[k] += cls_accs[k]
                        val_cls_acc_cnt[k] += 1

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_boundary_iou = val_boundary_iou / len(val_loader)

        # --- 3. Logging & History Update ---
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        history['train_iou'].append(avg_train_iou)
        history['val_iou'].append(avg_val_iou)
        history['train_boundary_iou'].append(avg_train_boundary_iou)
        history['val_boundary_iou'].append(avg_val_boundary_iou)

        # Class-wise History Update
        for k in range(5):
            t_iou = train_cls_iou_sum[k] / train_cls_iou_cnt[k] if train_cls_iou_cnt[k] > 0 else 0.0
            t_acc = train_cls_acc_sum[k] / train_cls_acc_cnt[k] if train_cls_acc_cnt[k] > 0 else 0.0
            v_iou = val_cls_iou_sum[k] / val_cls_iou_cnt[k] if val_cls_iou_cnt[k] > 0 else 0.0
            v_acc = val_cls_acc_sum[k] / val_cls_acc_cnt[k] if val_cls_acc_cnt[k] > 0 else 0.0

            history['train_class_iou'][k].append(t_iou)
            history['train_class_acc'][k].append(t_acc)
            history['val_class_iou'][k].append(v_iou)
            history['val_class_acc'][k].append(v_acc)

        end_time = time.time()  # epoch 종료 시간 기록
        epoch_time = end_time - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"Acc: {avg_train_acc:.4f}/{avg_val_acc:.4f} | "
              f"IoU: {avg_train_iou:.4f}/{avg_val_iou:.4f} | "
              f"BoundIoU: {avg_train_boundary_iou:.4f}/{avg_val_boundary_iou:.4f} | "
              f"Time: {epoch_time:.2f}s")

        # --- 4. Checkpoint Save ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'history': history
        }, checkpoint_path)

        # --- 5. Best Model Save ---
        if avg_val_loss < best_loss:
            print(f" -> Best model updated! ({best_loss:.4f} -> {avg_val_loss:.4f})")
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"=== Training Finished: {model_name} ===\n")
    return history
