"""Round 2 visualization: generates per-case prediction images and summary charts."""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config_v2 import *
from model import MultiTaskUNet
from utils import inference_volume


def make_figure(ncct_slice, cta_gt_slice, cta_pred_slice, seg_gt_slice, seg_pred_slice,
                pid, z, metrics, save_path):
    """Create a 1×5 comparison figure for one slice."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    titles = ['NCCT (Input)', 'CTA (GT)', 'CTA (Pred)', 'Vessel (GT)', 'Vessel (Pred)']
    images = [ncct_slice, cta_gt_slice, cta_pred_slice, seg_gt_slice, seg_pred_slice]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    info = f"Patient {pid} | Slice {z} | Dice={metrics['dice']:.3f} | PSNR={metrics['psnr']:.1f}"
    fig.suptitle(info, fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print(f"Device: {DEVICE}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = MultiTaskUNet(in_channels=IN_CHANNELS, features=FEATURES).to(DEVICE)
    best_path = CHECKPOINT_DIR / "best_model.pth"
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded model (epoch {ckpt.get('epoch', '?')})")

    # Load test results
    with open(OUTPUT_DIR / "test_results.json", 'r') as f:
        results = json.load(f)

    test_ids = [r['patient_id'] for r in results['per_case']]
    print(f"Visualizing {len(test_ids)} test cases...")

    # Representative cases: best, median, worst (by Dice)
    sorted_cases = sorted(results['per_case'], key=lambda r: r['dice'])
    worst = sorted_cases[0]
    median = sorted_cases[len(sorted_cases) // 2]
    best = sorted_cases[-1]
    showcase = [
        (best['patient_id'], 'best'),
        (median['patient_id'], 'median'),
        (worst['patient_id'], 'worst'),
    ]

    for pid, label in showcase:
        print(f"  Rendering {label}: patient {pid}")

        ncct_path = NCCT_DIR / f"patient{pid}.nii"
        cta_gt_path = CTA_DIR / f"patient{pid}.nii"
        seg_gt_path = SEG_DIR / f"patient{pid}.nii.gz"

        # Get predictions
        cta_pred_vol, seg_pred_vol, affine, header = inference_volume(
            model, ncct_path, DEVICE, IMAGE_SIZE, NCCT_CLIP, CTA_CLIP)

        # Load ground truth
        ncct_nii = nib.load(str(ncct_path))
        ncct_vol = ncct_nii.get_fdata().astype(np.float32)
        cta_gt_vol = nib.load(str(cta_gt_path), mmap=True).get_fdata().astype(np.float32)
        seg_gt_vol = nib.load(str(seg_gt_path), mmap=True).get_fdata().astype(np.float32)

        # Normalize NCCT for display
        ncct_disp = np.clip(ncct_vol, *NCCT_CLIP)
        ncct_disp = (ncct_disp - NCCT_CLIP[0]) / (NCCT_CLIP[1] - NCCT_CLIP[0])

        # Normalize CTA for display
        cta_gt_disp = np.clip(cta_gt_vol, *CTA_CLIP)
        cta_gt_disp = (cta_gt_disp - CTA_CLIP[0]) / (CTA_CLIP[1] - CTA_CLIP[0])
        cta_pred_disp = np.clip(cta_pred_vol, *CTA_CLIP)
        cta_pred_disp = (cta_pred_disp - CTA_CLIP[0]) / (CTA_CLIP[1] - CTA_CLIP[0])

        # Find a slice with visible vessels (middle + where seg has content)
        Z = ncct_vol.shape[2]
        z_mid = Z // 2
        seg_mid = seg_gt_vol[:, :, z_mid]
        if seg_mid.sum() > 100:
            best_z = z_mid
        else:
            # Scan for slice with most vessels
            vessel_counts = [seg_gt_vol[:, :, z].sum() for z in range(Z)]
            best_z = int(np.argmax(vessel_counts))

        # Get case metrics
        case_metrics = next(r for r in results['per_case'] if r['patient_id'] == pid)

        # Create figure
        make_figure(
            ncct_disp[:, :, best_z],
            cta_gt_disp[:, :, best_z],
            cta_pred_disp[:, :, best_z],
            seg_gt_vol[:, :, best_z],
            (seg_pred_vol[:, :, best_z] > 0.5).astype(np.float32),
            pid, best_z, case_metrics,
            FIG_DIR / f"case_{pid}_{label}.png"
        )

    # Summary bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ids = [r['patient_id'] for r in results['per_case']]
    dice_vals = [r['dice'] for r in results['per_case']]
    psnr_vals = [r['psnr'] for r in results['per_case']]

    colors_dice = ['#2ecc71' if d >= 0.85 else '#f39c12' if d >= 0.75 else '#e74c3c'
                   for d in dice_vals]
    ax1.bar(range(len(ids)), dice_vals, color=colors_dice, edgecolor='white')
    ax1.axhline(y=np.mean(dice_vals), color='blue', linestyle='--', label=f'Mean={np.mean(dice_vals):.3f}')
    ax1.set_xticks(range(len(ids)))
    ax1.set_xticklabels(ids, rotation=45, ha='right', fontsize=7)
    ax1.set_ylabel('3D Dice')
    ax1.set_title(f'Vessel Segmentation Dice (3D) — μ={np.mean(dice_vals):.3f}, σ={np.std(dice_vals):.3f}')
    ax1.set_ylim(0, 1.0)
    ax1.legend()

    colors_psnr = ['#2ecc71' if p >= 28 else '#f39c12' if p >= 24 else '#e74c3c'
                   for p in psnr_vals]
    ax2.bar(range(len(ids)), psnr_vals, color=colors_psnr, edgecolor='white')
    ax2.axhline(y=np.mean(psnr_vals), color='blue', linestyle='--', label=f'Mean={np.mean(psnr_vals):.1f}')
    ax2.set_xticks(range(len(ids)))
    ax2.set_xticklabels(ids, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title(f'CTA Generation PSNR — μ={np.mean(psnr_vals):.1f}, σ={np.std(psnr_vals):.1f}')
    ax2.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "summary_metrics.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nVisualizations saved to {FIG_DIR}/")
    print(f"  - case_*_best.png")
    print(f"  - case_*_median.png")
    print(f"  - case_*_worst.png")
    print(f"  - summary_metrics.png")


if __name__ == "__main__":
    main()
