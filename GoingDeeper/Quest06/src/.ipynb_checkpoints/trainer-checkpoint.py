import os
import time
import torch
from src.utils import compute_metric

def run_training(
        train_loader,
        valid_loader,
        model,
        optimizer,
        criterion,
        device,
        label_converter,
        epochs=1,
        checkpoint_path="model_checkpoint.pth",
        best_model_path="best_model.pth"
        ):

    checkpoint_path = os.path.expanduser(checkpoint_path)
    best_model_path = os.path.expanduser(best_model_path)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    start_epoch = 1
    best_val_loss = float('inf')

    # 기록용 리스트 초기화
    train_losses = []
    train_accuracies = [] 
    train_edit_dists = [] 

    val_losses = []
    val_accuracies = []
    val_edit_dists = []

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        # 기존 기록 불러오기
        train_losses = checkpoint.get('train_losses', [])
        train_accuracies = checkpoint.get('train_accuracies', [])
        train_edit_dists = checkpoint.get('train_edit_dists', [])

        val_losses = checkpoint.get('val_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        val_edit_dists = checkpoint.get('val_edit_dists', [])

        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming training from epoch {start_epoch}...")
    else:
        print("No checkpoint found. Starting training from scratch...")

    for epoch in range(start_epoch, epochs+1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        processed_train_batches = 0

        # Train Metric 집계 변수
        train_correct = 0
        train_total_dist = 0
        train_total_samples = 0

        for idx, (imgs, labels_padded, input_lengths, label_lengths, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels_padded = labels_padded.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_padded, input_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            processed_train_batches += 1

            # Train Metric 계산 (모든 배치 수행)
            with torch.no_grad():
                t_correct, t_dist, t_batch_sz = compute_metric(outputs, labels_padded, label_lengths, label_converter)
                train_correct += t_correct
                train_total_dist += t_dist
                train_total_samples += t_batch_sz

            if idx % 200 == 0:
                current_acc = train_correct / train_total_samples * 100
                print(f"[Epoch {epoch}][Batch {idx}] Loss: {loss.item():.4f} | Avg Acc: {current_acc:.2f}%")

        avg_train_loss = train_loss / processed_train_batches if processed_train_batches > 0 else 0.0
        avg_train_acc = train_correct / train_total_samples * 100
        avg_train_dist = train_total_dist / train_total_samples

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        train_edit_dists.append(avg_train_dist)

        # --- Valid ---
        print("Validating...")
        model.eval()
        valid_loss = 0.0
        processed_val_batches = 0

        # Valid Metric 집계 변수
        val_correct = 0
        val_total_dist = 0
        val_total_samples = 0

        with torch.no_grad():
            for idx, (imgs, labels_padded, input_lengths, label_lengths, _) in enumerate(valid_loader):
                imgs = imgs.to(device)
                labels_padded = labels_padded.to(device)
                input_lengths = input_lengths.to(device)
                label_lengths = label_lengths.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels_padded, input_lengths, label_lengths)
                valid_loss += loss.item()
                processed_val_batches += 1

                # Metric 계산
                correct, dist, batch_sz = compute_metric(outputs, labels_padded, label_lengths, label_converter)
                val_correct += correct
                val_total_dist += dist
                val_total_samples += batch_sz

        avg_val_loss = valid_loss / processed_val_batches if processed_val_batches > 0 else 0.0
        val_losses.append(avg_val_loss)

        # Metric 결과 계산
        avg_val_acc = val_correct / val_total_samples * 100
        avg_val_dist = val_total_dist / val_total_samples

        val_accuracies.append(avg_val_acc)
        val_edit_dists.append(avg_val_dist)

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start

        print(f"[Epoch {epoch}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"               Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}%")
        print(f"               Train Dist: {avg_train_dist:.4f} | Val Dist: {avg_val_dist:.4f} | Time: {epoch_duration:.2f}s")

        # 체크포인트 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'train_edit_dists': train_edit_dists,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'val_edit_dists': val_edit_dists,
            'best_val_loss': best_val_loss
        }, checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Model improved. Best model saved at {best_model_path}")

    return model, train_losses, train_accuracies, train_edit_dists, val_losses, val_accuracies, val_edit_dists
