import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) - Raw Logits from model
        # targets: (B, H, W) - LongTensor with class indices

        inputs = F.softmax(inputs, dim=1)

        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        intersection = (inputs * targets_one_hot).sum(dim=(0, 2, 3))
        union = inputs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class HybridLoss(nn.Module):
    """Cross Entropy + Dice Loss"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, inputs, targets):
        # Deep Supervision: inputs가 리스트(또는 튜플)로 들어올 경우 처리
        if isinstance(inputs, (list, tuple)):
            loss = 0
            for x in inputs:
                loss += self.ce(x, targets) + self.dice(x, targets)
            return loss / len(inputs) # 평균 Loss 반환
        else:
            # 단일 Output일 경우
            return self.ce(inputs, targets) + self.dice(inputs, targets)
