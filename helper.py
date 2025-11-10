import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

class DiceLoss(nn.Module):
    """Function to compute Dice Loss"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)

        # Flatten
        preds = preds.view(-1) # Flatten for easier calculation
        targets = targets.view(-1) # Flatten for easier calculation
        intersection = (preds * targets).sum() # Intersection is overlapping between prediction and ground truth
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth) # Dice Value

        # Dice Loss = 1 - Dice Value
        return 1 - dice


class FocalLoss(nn.Module):
    """Function to compute Focal Loss. Helps with imbalanced dataset"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha >= 0:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce_loss
        return loss.mean()

class CombinedLoss(nn.Module):
    """Function to combine Focal Loss and Dice Loss"""
    def __init__(self, dice_weight=0.7, focal_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth=1.0)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return self.dice_weight * dice + self.focal_weight * focal


def compute_metrics(preds, targets, threshold=0.5, eps=1e-7):
    preds = torch.sigmoid(preds) # Pushdown to [0,1]

    # Flatten to ensure smooth calculation
    preds = (preds > threshold).float().cpu().numpy().flatten()
    targets = (targets > 0.5).float().cpu().numpy().flatten()

    # Intersection covers case when both prediction and target value is 1
    intersection = np.sum(preds * targets)
    union = np.sum(preds) + np.sum(targets) - intersection # (A U B) = A + B - Intersection

    # Calculate metrics
    iou = (intersection + eps) / (union + eps)
    dice = (2 * (intersection + eps))/ (np.sum(preds) + np.sum(targets) + eps)
    acc = accuracy_score(targets, preds)

    return float(iou), float(dice), float(acc)