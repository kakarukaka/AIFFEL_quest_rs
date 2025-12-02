import torch
import torch
import numpy as np
import torch.nn.functional as F

def calculate_pixel_accuracy(output, target):
    """
    Pixel Accuracy: 전체 픽셀 중 맞춘 픽셀의 비율
    output: (B, C, H, W) 또는 List[(B, C, H, W), ...]
    target: (B, H, W)
    """
    with torch.no_grad():
        # Deep Supervision(List)일 경우, 마지막(가장 정교한) Output만 사용
        if isinstance(output, (list, tuple)):
            output = output[-1]

        # (Batch, Class, H, W) -> (Batch, H, W) 클래스 인덱스 추출
        prediction = torch.argmax(output, dim=1)

        correct = (prediction == target).sum().item()
        total = target.numel() # 전체 픽셀 수

        return correct / total

def calculate_mask_iou(output, target, num_classes=5):
    """
    Mask IoU (Mean IoU): 클래스별 IoU의 평균
    output: (B, C, H, W) 또는 List[(B, C, H, W), ...]
    target: (B, H, W)
    """
    with torch.no_grad():
        if isinstance(output, (list, tuple)):
            output = output[-1]

        prediction = torch.argmax(output, dim=1)

        ious = []
        # 배치 내 모든 픽셀을 통합하여 계산 (Batch-wise IoU)
        for cls in range(num_classes):
            pred_mask = (prediction == cls)
            target_mask = (target == cls)

            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()

            if union == 0:
                # 해당 클래스가 정답에도 없고 예측에도 없으면 IoU 계산에서 제외 (NaN 방지)
                continue

            ious.append(intersection / union)

        # 계산된 클래스들의 IoU 평균 반환
        return np.mean(ious) if ious else 0.0

def calculate_boundary_iou(output, target, num_classes=5, dilation_ratio=0.02):
    """
    Boundary IoU: 객체 경계(Boundary) 영역에서의 IoU 계산
    output: (B, C, H, W)
    target: (B, H, W)
    """
    with torch.no_grad():
        if isinstance(output, (list, tuple)):
            output = output[-1]
        
        prediction = torch.argmax(output, dim=1) # (B, H, W)
        
        # One-hot encoding
        pred_onehot = F.one_hot(prediction, num_classes).permute(0, 3, 1, 2).float() # (B, C, H, W)
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Kernel size for dilation/erosion
        h, w = target.shape[-2:]
        boundary_width = int(np.ceil(np.sqrt(h * w) * dilation_ratio))
        kernel_size = 2 * boundary_width + 1
        padding = boundary_width
        
        def get_boundary(mask_onehot):
            # Dilation: MaxPool2d
            dilated = F.max_pool2d(mask_onehot, kernel_size=kernel_size, stride=1, padding=padding)
            # Erosion: -MaxPool2d(-x) or 1 - MaxPool2d(1-x)
            eroded = 1 - F.max_pool2d(1 - mask_onehot, kernel_size=kernel_size, stride=1, padding=padding)
            return dilated - eroded

        pred_boundary = get_boundary(pred_onehot)
        target_boundary = get_boundary(target_onehot)
        
        ious = []
        for cls in range(num_classes):
            p_b = pred_boundary[:, cls]
            t_b = target_boundary[:, cls]
            
            intersection = (p_b * t_b).sum().item()
            union = (p_b + t_b).clamp(0, 1).sum().item()
            
            # 해당 클래스의 경계가 정답/예측 모두에 없으면 제외
            if union == 0:
                continue
                
            ious.append(intersection / union)
            
        return np.mean(ious) if ious else 0.0
