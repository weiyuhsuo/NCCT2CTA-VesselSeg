"""Evaluation metrics for CTA generation and vessel segmentation."""

import torch
import torch.nn.functional as F


def dice_coeff(seg_pred_logits, seg_gt, threshold=0.5, smooth=1e-6):
    """Dice coefficient for binary segmentation.

    Args:
        seg_pred_logits: (B, 1, H, W) raw logits
        seg_gt: (B, 1, H, W) binary labels {0, 1}
        threshold: binarization threshold on sigmoid output
        smooth: small constant for numerical stability

    Returns:
        scalar: mean Dice over the batch
    """
    batch_size = seg_pred_logits.shape[0]
    pred = (torch.sigmoid(seg_pred_logits) > threshold).float()
    pred_flat = pred.view(batch_size, -1)
    gt_flat = seg_gt.view(batch_size, -1)
    intersection = (pred_flat * gt_flat).sum(dim=1)
    dice_per_sample = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + smooth)
    return dice_per_sample.mean().item()


def iou_score(seg_pred_logits, seg_gt, threshold=0.5, smooth=1e-6):
    """Intersection over Union.

    Args:
        seg_pred_logits: (B, 1, H, W) raw logits
        seg_gt: (B, 1, H, W) binary labels {0, 1}

    Returns:
        scalar: mean IoU over the batch
    """
    batch_size = seg_pred_logits.shape[0]
    pred = (torch.sigmoid(seg_pred_logits) > threshold).float()
    pred_flat = pred.view(batch_size, -1)
    gt_flat = seg_gt.view(batch_size, -1)
    intersection = (pred_flat * gt_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - intersection
    iou_per_sample = (intersection + smooth) / (union + smooth)
    return iou_per_sample.mean().item()


def pixel_accuracy(seg_pred_logits, seg_gt, threshold=0.5):
    """Pixel-wise accuracy.

    Args:
        seg_pred_logits: (B, 1, H, W) raw logits
        seg_gt: (B, 1, H, W) binary labels

    Returns:
        scalar: accuracy
    """
    pred = (torch.sigmoid(seg_pred_logits) > threshold).float()
    correct = (pred == seg_gt).float().mean().item()
    return correct


def psnr(cta_pred, cta_gt, data_range=1.0):
    """Peak Signal-to-Noise Ratio.

    Args:
        cta_pred: (B, 1, H, W) predicted CTA
        cta_gt: (B, 1, H, W) ground truth CTA

    Returns:
        scalar: PSNR in dB, mean over batch
    """
    batch_size = cta_pred.shape[0]
    mse = F.mse_loss(cta_pred, cta_gt, reduction='none').view(batch_size, -1).mean(dim=1)
    # Handle zero MSE
    psnr_per_sample = torch.where(
        mse > 0,
        20.0 * torch.log10(torch.tensor(data_range, device=mse.device)) - 10.0 * torch.log10(mse),
        torch.tensor(100.0, device=mse.device)  # cap for perfect match
    )
    return psnr_per_sample.mean().item()


def ssim(cta_pred, cta_gt, data_range=1.0, window_size=11):
    """Structural Similarity Index (simplified, differentiable approximation).

    Uses a Gaussian-weighted local window for structural comparison.

    Args:
        cta_pred: (B, 1, H, W)
        cta_gt:   (B, 1, H, W)
        data_range: dynamic range of the data
        window_size: local window size for SSIM computation

    Returns:
        scalar: mean SSIM over the batch
    """
    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Gaussian window
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)

    device = cta_pred.device
    window = gaussian_window(window_size).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, W, W)

    # Mean and variance via convolution
    mu1 = F.conv2d(cta_pred, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(cta_gt, window, padding=window_size // 2, groups=1)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(cta_pred ** 2, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(cta_gt ** 2, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(cta_pred * cta_gt, window, padding=window_size // 2, groups=1) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-8)

    return ssim_map.mean().item()
