"""Round 2 Training: Dice Loss + 512x512 + segmentation-biased weights.

Key changes from Round 1:
- DiceLoss (80%) + BCE (20%) instead of pure BCE
- 512×512 input resolution
- Loss weights: 0.7 seg / 0.3 CTA
- Longer early-stop patience (20)
- CosineAnnealing scheduler option
"""
import os, sys, json
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent))
from config_v2 import *
from model import MultiTaskUNet
from losses import CombinedLoss, DiceLoss
from metrics import dice_coeff, iou_score, psnr, ssim
from dataset_v3 import PreprocessedDataset


def load_split():
    """Load the pre-computed train/val/test split from preprocessed_512."""
    with open(PREPROC_DIR / "split.json", 'r') as f:
        split_info = json.load(f)
    return split_info['train'], split_info['val'], split_info['test']


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path, is_best=False):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }, path)
    if is_best:
        best_path = path.parent / "best_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
        }, best_path)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_seg = 0.0
    for ncct, cta_gt, seg_gt in loader:
        ncct, cta_gt, seg_gt = ncct.to(device), cta_gt.to(device), seg_gt.to(device)
        cta_pred, seg_pred = model(ncct)
        loss, l1, seg = criterion(cta_pred, cta_gt, seg_pred, seg_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_l1 += l1
        total_seg += seg
    n = len(loader)
    return total_loss / n, total_l1 / n, total_seg / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    for ncct, cta_gt, seg_gt in loader:
        ncct, cta_gt, seg_gt = ncct.to(device), cta_gt.to(device), seg_gt.to(device)
        cta_pred, seg_pred = model(ncct)
        loss, _, _ = criterion(cta_pred, cta_gt, seg_pred, seg_gt)
        total_loss += loss.item()
        total_dice += dice_coeff(seg_pred, seg_gt)
        total_iou += iou_score(seg_pred, seg_gt)
        total_psnr += psnr(cta_pred, cta_gt, data_range=1.0)
        total_ssim += ssim(cta_pred, cta_gt, data_range=1.0)
    n = len(loader)
    return {
        'loss': total_loss / n,
        'dice': total_dice / n,
        'iou': total_iou / n,
        'psnr': total_psnr / n,
        'ssim': total_ssim / n,
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"Round 2: Dice Loss + {IMAGE_SIZE}×{IMAGE_SIZE} + seg-biased weights")
    print(f"Config: λ_cta={LAMBDA_CTA}, λ_seg={LAMBDA_SEG}, dice_w={DICE_WEIGHT}, bs={BATCH_SIZE}")

    # Load split
    train_ids, val_ids, test_ids = load_split()
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Save test IDs for later
    with open(OUTPUT_DIR / "test_ids.json", 'w') as f:
        json.dump(test_ids, f, indent=2)

    # Datasets
    train_ds = PreprocessedDataset(train_ids, PREPROC_DIR, augment=True)
    val_ds = PreprocessedDataset(val_ids, PREPROC_DIR, augment=False)

    print(f"Train slices: {len(train_ds)}, Val slices: {len(val_ds)}")
    print("Creating DataLoader...", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print("DataLoader ready, starting training...", flush=True)

    # Model
    model = MultiTaskUNet(in_channels=IN_CHANNELS, features=FEATURES).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & Scheduler
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # CosineAnnealing for smoother LR decay (Round 2 improvement)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    # Round 2: Dice-based combined loss
    criterion = CombinedLoss(
        lambda_cta=LAMBDA_CTA,
        lambda_seg=LAMBDA_SEG,
        dice_weight=DICE_WEIGHT,
        dice_smooth=1.0,
    )

    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': [],
        'val_psnr': [], 'val_ssim': [],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nTraining started at {timestamp}")
    print(f"Checkpoints → {CHECKPOINT_DIR}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_l1, train_seg = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = validate(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_psnr'].append(val_metrics['psnr'])
        history['val_ssim'].append(val_metrics['ssim'])

        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{EPOCHS} | LR: {lr_now:.2e} | "
              f"Train Loss: {train_loss:.4f} (L1:{train_l1:.4f} Seg:{train_seg:.4f}) | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f} | "
              f"PSNR: {val_metrics['psnr']:.2f} | SSIM: {val_metrics['ssim']:.4f}")

        ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth"
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            patience_counter = 0
            print(f"  *** New best! Dice: {best_dice:.4f} | IoU: {val_metrics['iou']:.4f}")
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, ckpt_path, is_best)

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
            break

    with open(OUTPUT_DIR / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    print(f"Best model saved to {CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
