"""Round 2 Test: 512×512 evaluation with Dice Loss model.

Evaluates both slice-level and per-case 3D metrics.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config_v2 import *
from model import MultiTaskUNet
from losses import CombinedLoss
from metrics import dice_coeff, iou_score, psnr, ssim, pixel_accuracy
from utils import inference_volume, save_nii
from dataset_v3 import PreprocessedDataset


@torch.no_grad()
def test_slices(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = total_acc = total_psnr = total_ssim = 0.0
    for ncct, cta_gt, seg_gt in loader:
        ncct, cta_gt, seg_gt = ncct.to(device), cta_gt.to(device), seg_gt.to(device)
        cta_pred, seg_pred = model(ncct)
        loss, _, _ = criterion(cta_pred, cta_gt, seg_pred, seg_gt)
        total_loss += loss.item()
        total_dice += dice_coeff(seg_pred, seg_gt)
        total_iou += iou_score(seg_pred, seg_gt)
        total_acc += pixel_accuracy(seg_pred, seg_gt)
        total_psnr += psnr(cta_pred, cta_gt)
        total_ssim += ssim(cta_pred, cta_gt)
    n = len(loader)
    return {k: round(v / n, 4) for k, v in {
        'loss': total_loss, 'dice': total_dice, 'iou': total_iou,
        'accuracy': total_acc, 'psnr': total_psnr, 'ssim': total_ssim,
    }.items()}


@torch.no_grad()
def test_per_case(model, test_ids, device):
    results = []
    model.eval()
    for pid in tqdm(test_ids, desc="Testing per case"):
        ncct_path = NCCT_DIR / f"patient{pid}.nii"
        cta_gt_path = CTA_DIR / f"patient{pid}.nii"
        seg_gt_path = SEG_DIR / f"patient{pid}.nii.gz"

        # Run full-volume inference
        cta_pred_vol, seg_pred_vol, affine, header = inference_volume(
            model, ncct_path, device, IMAGE_SIZE, NCCT_CLIP, CTA_CLIP)

        import nibabel as nib
        cta_gt = nib.load(str(cta_gt_path), mmap=True).get_fdata().astype(np.float32)
        seg_gt = nib.load(str(seg_gt_path), mmap=True).get_fdata().astype(np.float32)

        # CTA metrics: compute PSNR on normalized, clipped range
        cta_min, cta_max = CTA_CLIP
        cta_pred_clipped = np.clip(cta_pred_vol, cta_min, cta_max)
        cta_pred_norm = (cta_pred_clipped - cta_min) / (cta_max - cta_min)
        cta_gt_clipped = np.clip(cta_gt, cta_min, cta_max)
        cta_gt_norm = (cta_gt_clipped - cta_min) / (cta_max - cta_min)

        p_t = torch.from_numpy(cta_pred_norm).unsqueeze(0).unsqueeze(0).float()
        g_t = torch.from_numpy(cta_gt_norm).unsqueeze(0).unsqueeze(0).float()
        mse = torch.nn.functional.mse_loss(p_t, g_t).item()
        psnr_val = 20 * np.log10(1.0) - 10 * np.log10(mse) if mse > 0 else float('inf')

        # Segmentation metrics
        seg_p_bin = (seg_pred_vol > 0.5).astype(np.float32)
        intersection = (seg_p_bin * seg_gt).sum()
        dice_val = (2.0 * intersection + 1e-6) / (seg_p_bin.sum() + seg_gt.sum() + 1e-6)
        iou_val = (intersection + 1e-6) / (seg_p_bin.sum() + seg_gt.sum() - intersection + 1e-6)
        acc_val = (seg_p_bin == seg_gt).mean()

        results.append({
            'patient_id': pid,
            'psnr': round(float(psnr_val), 2),
            'dice': round(float(dice_val), 4),
            'iou': round(float(iou_val), 4),
            'accuracy': round(float(acc_val), 4),
        })

        # Save prediction volumes
        save_nii(seg_pred_vol, affine, header, PRED_DIR / f"patient{pid}_seg_pred.nii.gz")
        save_nii(cta_pred_vol.astype(np.float32), affine, header, PRED_DIR / f"patient{pid}_cta_syn.nii.gz")

    return results


def main():
    print(f"Device: {DEVICE}")
    print(f"Round 2 Evaluation — {IMAGE_SIZE}×{IMAGE_SIZE}")

    # Load model
    model = MultiTaskUNet(in_channels=IN_CHANNELS, features=FEATURES).to(DEVICE)
    best_path = CHECKPOINT_DIR / "best_model.pth"
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded best model (epoch {ckpt.get('epoch', '?')})")

    # Load test IDs
    with open(OUTPUT_DIR / "test_ids.json", 'r') as f:
        test_ids = json.load(f)
    print(f"Test patients: {len(test_ids)}")

    # Slice-level evaluation (on preprocessed 512×512 slices)
    print("\n=== Slice-level evaluation (512×512) ===")
    test_ds = PreprocessedDataset(test_ids, PREPROC_DIR, augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)
    criterion = CombinedLoss(lambda_cta=LAMBDA_CTA, lambda_seg=LAMBDA_SEG, dice_weight=DICE_WEIGHT)

    slice_metrics = test_slices(model, test_loader, criterion, DEVICE)
    print(f"  Loss: {slice_metrics['loss']} | Dice: {slice_metrics['dice']} | IoU: {slice_metrics['iou']}")
    print(f"  PSNR: {slice_metrics['psnr']} | SSIM: {slice_metrics['ssim']} | Acc: {slice_metrics['accuracy']}")

    # Per-case 3D evaluation
    print("\n=== Per-case 3D evaluation ===")
    case_results = test_per_case(model, test_ids, DEVICE)

    dice_vals = [r['dice'] for r in case_results]
    psnr_vals = [r['psnr'] for r in case_results]
    iou_vals = [r['iou'] for r in case_results]
    acc_vals = [r['accuracy'] for r in case_results]

    print(f"\n=== 3D Summary (n={len(case_results)}) ===")
    print(f"  PSNR:     {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f}")
    print(f"  Dice:     {np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}")
    print(f"  IoU:      {np.mean(iou_vals):.4f} ± {np.std(iou_vals):.4f}")
    print(f"  Accuracy: {np.mean(acc_vals):.4f} ± {np.std(acc_vals):.4f}")

    # Per-case detail
    print(f"\n{'ID':<8} {'PSNR':>8} {'Dice':>8} {'IoU':>8} {'Acc':>8}")
    print("-" * 40)
    for r in case_results:
        print(f"  {r['patient_id']:<6} {r['psnr']:>8.2f} {r['dice']:>8.4f} {r['iou']:>8.4f} {r['accuracy']:>8.4f}")

    # Rank worst cases
    print(f"\n=== Bottom 5 Cases (Dice) ===")
    sorted_by_dice = sorted(case_results, key=lambda r: r['dice'])
    for r in sorted_by_dice[:5]:
        print(f"  {r['patient_id']}: Dice={r['dice']:.4f}, IoU={r['iou']:.4f}")

    # Save results
    output = {
        'round': 2,
        'config': {
            'image_size': IMAGE_SIZE,
            'lambda_cta': LAMBDA_CTA,
            'lambda_seg': LAMBDA_SEG,
            'dice_weight': DICE_WEIGHT,
            'batch_size': BATCH_SIZE,
        },
        'slice_metrics': slice_metrics,
        'per_case': case_results,
        'summary': {
            'psnr_mean': round(float(np.mean(psnr_vals)), 2),
            'psnr_std': round(float(np.std(psnr_vals)), 2),
            'dice_mean': round(float(np.mean(dice_vals)), 4),
            'dice_std': round(float(np.std(dice_vals)), 4),
            'iou_mean': round(float(np.mean(iou_vals)), 4),
            'iou_std': round(float(np.std(iou_vals)), 4),
            'acc_mean': round(float(np.mean(acc_vals)), 4),
            'acc_std': round(float(np.std(acc_vals)), 4),
        },
    }
    with open(OUTPUT_DIR / "test_results.json", 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults → {OUTPUT_DIR / 'test_results.json'}")
    print(f"Predictions → {PRED_DIR}/")


if __name__ == "__main__":
    main()
