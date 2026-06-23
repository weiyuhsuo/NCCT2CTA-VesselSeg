"""Training script v2 - uses preprocessed .npz data for fast loading."""
import os, sys, json, random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from model import MultiTaskUNet
from losses import CombinedLoss
from metrics import dice_coeff, iou_score, psnr, ssim

# Import the new dataset
from dataset_v2 import PreprocessedDataset

def load_split():
    """Load the pre-computed train/val/test split."""
    import json
    with open(PREPROC_DIR / "split.json", 'r') as f:
        split_info = json.load(f)
    return split_info['train'], split_info['val'], split_info['test']

def save_checkpoint(model, optimizer, epoch, metrics, path, is_best=False):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)
    if is_best:
        best_path = path.parent / "best_model.pth"
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': metrics}, best_path)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_bce = 0.0
    for ncct, cta_gt, seg_gt in loader:
        ncct, cta_gt, seg_gt = ncct.to(device), cta_gt.to(device), seg_gt.to(device)
        cta_pred, seg_pred = model(ncct)
        loss, l1, bce = criterion(cta_pred, cta_gt, seg_pred, seg_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_l1 += l1
        total_bce += bce
    n = len(loader)
    return total_loss/n, total_l1/n, total_bce/n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    for ncct, cta_gt, seg_gt in loader:
        ncct, cta_gt, seg_gt = ncct.to(device), cta_gt.to(device), seg_gt.to(device)
        cta_pred, seg_pred = model(ncct)
        loss, _, _ = criterion(cta_pred, cta_gt, seg_pred, seg_gt)
        total_loss += loss.item()
        total_dice += dice_coeff(seg_pred, seg_gt)
        total_psnr += psnr(cta_pred, cta_gt)
        total_ssim += ssim(cta_pred, cta_gt, data_range=1.0)
    n = len(loader)
    return {'loss': total_loss/n, 'dice': total_dice/n, 'psnr': total_psnr/n, 'ssim': total_ssim/n}

def main():
    print(f"Device: {DEVICE}")

    # Load split from preprocessing
    train_ids, val_ids, test_ids = load_split()
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Save test IDs for later
    with open(OUTPUT_DIR / "test_ids.json", 'w') as f:
        json.dump(test_ids, f, indent=2)

    # Datasets using preprocessed data
    train_ds = PreprocessedDataset(train_ids, PREPROC_DIR / "train", augment=True)
    val_ds = PreprocessedDataset(val_ids, PREPROC_DIR / "val", augment=False)

    print(f"Train slices: {len(train_ds)}, Val slices: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = MultiTaskUNet(in_channels=IN_CHANNELS, features=FEATURES).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=SCHEDULER_FACTOR,
                                   patience=SCHEDULER_PATIENCE)
    criterion = CombinedLoss(lambda_cta=LAMBDA_CTA, lambda_seg=LAMBDA_SEG)

    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_psnr': [], 'val_ssim': []}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nTraining started at {timestamp}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_l1, train_bce = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = validate(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_psnr'].append(val_metrics['psnr'])
        history['val_ssim'].append(val_metrics['ssim'])

        scheduler.step(val_metrics['dice'])

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} (L1:{train_l1:.4f} BCE:{train_bce:.4f}) | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Dice: {val_metrics['dice']:.4f} | PSNR: {val_metrics['psnr']:.2f} | SSIM: {val_metrics['ssim']:.4f}")

        ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth"
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            patience_counter = 0
            print(f"  *** New best! Dice: {best_dice:.4f}")
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, epoch, val_metrics, ckpt_path, is_best)

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
            break

    with open(OUTPUT_DIR / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    print(f"Best model saved to {CHECKPOINT_DIR / 'best_model.pth'}")

if __name__ == "__main__":
    main()
