"""Loss functions for NCCT→CTA + Vessel Segmentation multi-task training.

Round 2: Replaces pure BCE with DiceLoss + BCE combination for segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation.

    Dice = 1 - (2 * |A ∩ B| + smooth) / (|A| + |B| + smooth)

    Uses continuous probabilities (no threshold) for differentiability.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """Compute soft Dice loss.

        Args:
            pred: (B, 1, H, W) logits (before sigmoid)
            target: (B, 1, H, W) binary labels

        Returns:
            scalar loss = 1 - Dice
        """
        # Sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)
        batch_size = pred.shape[0]

        # Flatten spatial dims
        pred_flat = pred.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice_per_sample = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_per_sample

        return dice_loss.mean()


class CombinedLoss(nn.Module):
    """Multi-task loss: L1 for CTA regression + Dice+BCE for segmentation.

    Loss = lambda_cta * L1(CTA) + lambda_seg * (alpha * DiceLoss + (1-alpha) * BCE)

    Round 2 changes:
    - DiceLoss replaces pure BCE as the primary segmentation loss
    - BCE retained with small weight for training stability
    - Loss weights biased toward segmentation (config: 0.3/0.7)
    """

    def __init__(self, lambda_cta=0.3, lambda_seg=0.7, dice_weight=0.8, dice_smooth=1.0):
        """
        Args:
            lambda_cta: weight for CTA generation loss
            lambda_seg: weight for segmentation loss
            dice_weight: weight of DiceLoss within segmentation loss (1-dice_weight = BCE weight)
            dice_smooth: smooth factor for DiceLoss
        """
        super().__init__()
        self.lambda_cta = lambda_cta
        self.lambda_seg = lambda_seg
        self.dice_weight = dice_weight
        self.l1 = nn.L1Loss()
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, cta_pred, cta_gt, seg_pred, seg_gt):
        """
        Args:
            cta_pred: (B, 1, H, W) CTA output (tanh, range ~[-1, 1])
            cta_gt:   (B, 1, H, W) CTA ground truth (normalized)
            seg_pred: (B, 1, H, W) segmentation logits (before sigmoid)
            seg_gt:   (B, 1, H, W) binary mask {0, 1}

        Returns:
            total_loss, l1_loss_value, seg_loss_value
        """
        l1_val = self.l1(cta_pred, cta_gt)

        dice_val = self.dice_loss(seg_pred, seg_gt)
        bce_val = self.bce(seg_pred, seg_gt)
        seg_loss = self.dice_weight * dice_val + (1.0 - self.dice_weight) * bce_val

        total = self.lambda_cta * l1_val + self.lambda_seg * seg_loss

        return total, l1_val.item(), seg_loss.item()
