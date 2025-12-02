import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from PIL import Image
from skimage.io import imread
from src.utils import calculate_boundary_iou


def plot_training_history(history, model_name):
    """학습 결과를 시각화하는 함수"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig = plt.figure(figsize=(16, 10))  

    fig.suptitle(f"\n{model_name}", fontsize=24)

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title(f'{model_name} - Pixel Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # IoU
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    plt.plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    plt.title(f'{model_name} - Mean IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    # Boundary IoU
    if 'train_boundary_iou' in history:
        plt.subplot(2, 2, 4)
        plt.plot(epochs, history['train_boundary_iou'], 'b-', label='Train Bound IoU')
        plt.plot(epochs, history['val_boundary_iou'], 'r-', label='Val Bound IoU')
        plt.title(f'{model_name} - Boundary IoU')
        plt.xlabel('Epochs')
        plt.ylabel('Boundary IoU')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Class Names
CLASS_NAMES = ["Background", "Vehicle", "Nature", "Safety", "Road"]

def plot_class_metrics(history, model_name):
    """
    Visualizes class-wise Accuracy and IoU for Train and Validation sets.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Retrieve metrics dictionary
    t_acc = history.get('train_class_acc', {})
    v_acc = history.get('val_class_acc', {})
    t_iou = history.get('train_class_iou', {})
    v_iou = history.get('val_class_iou', {})

    if not t_acc:
        print(f"No class-wise metrics found for {model_name}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"\nClass-wise Metrics: {model_name}", fontsize=24)

    # Calculate global min/max for Accuracy to share Y-axis
    all_acc_values = []
    for k in range(5):
        all_acc_values.extend([x for x in t_acc.get(k, []) if x is not None])
        all_acc_values.extend([x for x in v_acc.get(k, []) if x is not None])

    if all_acc_values:
        min_acc, max_acc = min(all_acc_values), max(all_acc_values)
        margin = (max_acc - min_acc) * 0.05
        acc_ylim = (max(0, min_acc - margin), min(1.0, max_acc + margin))
    else:
        acc_ylim = (0, 1)

    # (1,1) Train Accuracy
    for k in range(5):
        axes[0, 0].plot(epochs, t_acc[k], label=f"{CLASS_NAMES[k]}")
    axes[0, 0].set_title("Train - Class Accuracy")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(acc_ylim)
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # (1,2) Valid Accuracy
    for k in range(5):
        axes[0, 1].plot(epochs, v_acc[k], label=f"{CLASS_NAMES[k]}")
    axes[0, 1].set_title("Valid - Class Accuracy")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(acc_ylim)
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # Calculate global min/max for IoU to share Y-axis
    all_iou_values = []
    for k in range(5):
        all_iou_values.extend([x for x in t_iou.get(k, []) if x is not None])
        all_iou_values.extend([x for x in v_iou.get(k, []) if x is not None])

    if all_iou_values:
        min_iou, max_iou = min(all_iou_values), max(all_iou_values)
        margin = (max_iou - min_iou) * 0.05
        iou_ylim = (max(0, min_iou - margin), min(1.0, max_iou + margin))
    else:
        iou_ylim = (0, 1)

    # (2,1) Train IoU
    for k in range(5):
        axes[1, 0].plot(epochs, t_iou[k], label=f"{CLASS_NAMES[k]}")
    axes[1, 0].set_title("Train - Class IoU")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("IoU")
    axes[1, 0].set_ylim(iou_ylim)
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # (2,2) Valid IoU
    for k in range(5):
        axes[1, 1].plot(epochs, v_iou[k], label=f"{CLASS_NAMES[k]}")
    axes[1, 1].set_title("Valid - Class IoU")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("IoU")
    axes[1, 1].set_ylim(iou_ylim)
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# Class Definitions
CLASS_INFO = {
    0: {'name': 'Background', 'color': (0, 0, 0)},
    1: {'name': 'Vehicle',    'color': (64, 0, 128)},
    2: {'name': 'Nature',     'color': (0, 128, 0)},
    3: {'name': 'Safety',     'color': (255, 255, 0)},
    4: {'name': 'Road',       'color': (255, 0, 0)}
}

def decode_segmap(image, num_classes=5):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, num_classes):
        idx = image == l
        color = CLASS_INFO[l]['color']
        r[idx] = color[0]
        g[idx] = color[1]
        b[idx] = color[2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def encode_mask_target(mask):
    """시각화 함수 내에서 라벨 처리를 위해 사용하는 헬퍼 함수"""
    mask = mask.astype(np.int64)
    new_mask = np.zeros_like(mask, dtype=np.int64)

    # 1: Vehicle
    vehicle_ids = [1, 26, 27, 28, 29, 30, 31, 32, 33]
    for vid in vehicle_ids: new_mask[mask == vid] = 1
    # 2: Nature
    nature_ids = [21, 22]
    for nid in nature_ids: new_mask[mask == nid] = 2
    # 3: Safety
    safety_ids = [14, 19, 20]
    for sid in safety_ids: new_mask[mask == sid] = 3
    # 4: Road
    new_mask[mask == 7] = 4
    return new_mask

def calculate_detailed_metrics(output, target, num_classes=5):
    """
    Global Acc, Mean IoU 및 각 클래스별 IoU, Acc(Recall)를 계산합니다.
    """
    with torch.no_grad():
        if isinstance(output, (list, tuple)):
            output = output[-1]

        prediction = torch.argmax(output, dim=1)

        # 1. Global Accuracy
        correct = (prediction == target).sum().item()
        total = target.numel()
        global_acc = correct / total if total > 0 else 0.0

        # 2. Class-wise IoU & Accuracy (Recall)
        class_ious = []
        class_accs = []

        for cls in range(num_classes):
            pred_mask = (prediction == cls)
            target_mask = (target == cls)

            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            target_count = target_mask.sum().item()

            # IoU
            if union == 0:
                iou = float('nan') # 해당 클래스가 예측/정답 어디에도 없음
            else:
                iou = intersection / union

            # Class Accuracy (Recall)
            if target_count == 0:
                acc = float('nan') # 정답에 해당 클래스가 없음
            else:
                acc = intersection / target_count

            class_ious.append(iou)
            class_accs.append(acc)

        # Mean IoU (NaN 제외 평균)
        valid_ious = [x for x in class_ious if not np.isnan(x)]
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0

        # Boundary IoU
        boundary_iou = calculate_boundary_iou(output, target, num_classes)

        return global_acc, mean_iou, class_ious, class_accs, boundary_iou

def visualize_comparison(models, preproc, image_path):
    # # Print Legend
    # print("\n" + "="*40)
    # print(" [Class Color Legend] ")
    # for idx, info in CLASS_INFO.items():
    #     print(f"  {info['name']} ({idx}): RGB {info['color']}")
    # print("="*40 + "\n")

    # 원본 이미지 로드
    origin_img = imread(image_path)
    h, w, _ = origin_img.shape

    # 정답 라벨 로드
    label_path = image_path.replace("image_2", "semantic")
    target_tensor = None
    processed_target_mask = None

    if os.path.exists(label_path):
        label_img = imread(label_path)
        encoded_mask = encode_mask_target(label_img)
        processed_target_mask = encoded_mask

    # 전처리
    data = {"image": origin_img}
    if processed_target_mask is not None:
        data["mask"] = processed_target_mask

    processed = preproc(**data)

    input_tensor = torch.tensor(processed["image"] / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    if "mask" in processed:
        target_tensor = torch.tensor(processed["mask"], dtype=torch.long).unsqueeze(0)

    # Plot 설정
    fig, axes = plt.subplots(2, 2, figsize=(16, 10)) # 세로 길이 약간 더 증가 (텍스트 공간 확보)
    axes = axes.flatten()

    axes[0].imshow(origin_img)
    axes[0].set_title("Original Image", fontsize=15)
    axes[0].axis('off')

    model_keys = ["Standard_UNet", "NestedUNet_v1", "NestedUNet_v2"]
    SHORT_NAMES = ["Bg", "Veh", "Nat", "Saf", "Rd"]

    for i, key in enumerate(model_keys):
        if key not in models: continue

        model = models[key]
        try:
            model_device = next(model.parameters()).device
        except:
            model_device = torch.device('cpu')

        input_on_device = input_tensor.to(model_device)

        model.eval()
        with torch.no_grad():
            output = model(input_on_device)
            if isinstance(output, (list, tuple)): output = output[-1]

        # Metric Calculation
        score_text = "No Ground Truth"

        if target_tensor is not None:
            target_on_device = target_tensor.to(model_device)
            g_acc, m_iou, c_ious, c_accs, b_iou = calculate_detailed_metrics(output, target_on_device)

            # 라벨에 넣을 텍스트 생성 (한 줄씩)
            lines = [f"Total Acc: {g_acc:.4f} | mIoU: {m_iou:.4f} | BoundIoU: {b_iou:.4f}"]

            for idx in range(5):
                iou_s = f"{c_ious[idx]:.2f}" if not np.isnan(c_ious[idx]) else "-"
                acc_s = f"{c_accs[idx]:.2f}" if not np.isnan(c_accs[idx]) else "-"
                lines.append(f"{SHORT_NAMES[idx]}: IoU {iou_s} / Acc {acc_s}")

            score_text = "\n".join(lines)

        # Visualization
        output_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        rgb_mask = decode_segmap(output_mask)

        mask_img = Image.fromarray(rgb_mask).resize((w, h), resample=Image.NEAREST).convert('RGBA')
        bg_img = Image.fromarray(origin_img).convert('RGBA')
        blended = Image.blend(bg_img, mask_img, alpha=0.5)

        ax_idx = i + 1
        axes[ax_idx].imshow(np.array(blended))
        axes[ax_idx].set_title(f"Model: {key}", fontsize=15)
        axes[ax_idx].set_xlabel(score_text, fontsize=10, fontweight='bold')
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

    plt.tight_layout()
    plt.show()