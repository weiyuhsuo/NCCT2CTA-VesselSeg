import os, sys, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from model import MultiTaskUNet

def visualize_case(model, pid, device, save_dir):
    """Generate comparison figure for one patient."""
    from utils import inference_volume
    
    ncct_path = NCCT_DIR / f"patient{pid}.nii"
    cta_path = CTA_DIR / f"patient{pid}.nii"
    seg_path = SEG_DIR / f"patient{pid}.nii.gz"
    
    cta_pred_vol, seg_pred_vol, affine, header = inference_volume(
        model, ncct_path, device, IMAGE_SIZE, NCCT_CLIP)
    
    ncct_data = nib.load(str(ncct_path)).get_fdata().astype(np.float32)
    cta_gt = nib.load(str(cta_path)).get_fdata().astype(np.float32)
    seg_gt = nib.load(str(seg_path)).get_fdata().astype(np.float32)
    
    # Find slices with visible vessels (middle of vessel range)
    z_indices = np.where(seg_gt.sum(axis=(0,1)) > 500)[0]
    if len(z_indices) < 3:
        z_indices = np.arange(seg_gt.shape[2])
    
    # Pick 3 representative slices: mid-vessel region
    step = max(1, len(z_indices) // 3)
    selected_z = [z_indices[i] for i in [len(z_indices)//4, len(z_indices)//2, 3*len(z_indices)//4]]
    
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    titles = ['NCCT Input', 'Syn CTA', 'Real CTA', 'CTA Diff (x5)', 'Pred SEG / GT']
    
    for row, z in enumerate(selected_z):
        ncct_sl = np.clip(ncct_data[:,:,z], -200, 500)
        ncct_sl = (ncct_sl - (-200)) / (500 - (-200))
        
        cta_p_sl = np.clip(cta_pred_vol[:,:,z], -200, 800)
        cta_p_sl = (cta_p_sl - (-200)) / (800 - (-200))
        
        cta_g_sl = np.clip(cta_gt[:,:,z], -200, 800)
        cta_g_sl = (cta_g_sl - (-200)) / (800 - (-200))
        
        diff = np.abs(cta_p_sl - cta_g_sl) * 5
        diff = np.clip(diff, 0, 1)
        
        seg_p = seg_pred_vol[:,:,z]
        seg_g = seg_gt[:,:,z]
        seg_overlay = np.zeros((*seg_p.shape, 3))
        seg_overlay[:,:,1] = seg_p  # green: prediction
        seg_overlay[:,:,0] = seg_g  # red: ground truth
        # Yellow = overlap
        
        axes[row,0].imshow(ncct_sl, cmap='gray')
        axes[row,0].set_title(titles[0] if row == 0 else '')
        axes[row,1].imshow(cta_p_sl, cmap='gray')
        axes[row,1].set_title(titles[1] if row == 0 else '')
        axes[row,2].imshow(cta_g_sl, cmap='gray')
        axes[row,2].set_title(titles[2] if row == 0 else '')
        axes[row,3].imshow(diff, cmap='hot')
        axes[row,3].set_title(titles[3] if row == 0 else '')
        axes[row,4].imshow(seg_overlay)
        axes[row,4].set_title(titles[4] if row == 0 else '')
        
        for ax in axes[row]:
            ax.axis('off')
    
    plt.suptitle(f'Patient {pid} - Slice {selected_z}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / f'case_{pid}.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print(f"Device: {DEVICE}")
    
    # Load model
    model = MultiTaskUNet(in_channels=IN_CHANNELS, features=FEATURES).to(DEVICE)
    best_path = CHECKPOINT_DIR / "best_model.pth"
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded best model from epoch {ckpt.get('epoch', '?')}")
    
    # Load test results to find best/worst cases
    with open(OUTPUT_DIR / "test_results.json", 'r') as f:
        results = json.load(f)
    
    cases = results['per_case']
    cases_sorted = sorted(cases, key=lambda x: x['dice'])
    
    # Select: worst 2, median 1, best 2
    selected = [cases_sorted[0], cases_sorted[1],
                cases_sorted[len(cases_sorted)//2],
                cases_sorted[-2], cases_sorted[-1]]
    
    print(f"Visualizing {len(selected)} cases: {[c['patient_id'] for c in selected]}")
    
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for case in selected:
        pid = case['patient_id']
        print(f"  Processing patient {pid} (Dice={case['dice']:.4f})...")
        visualize_case(model, pid, DEVICE, FIG_DIR)
    
    # Also create a summary bar chart of per-case metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    all_pids = [c['patient_id'] for c in cases]
    all_dice = [c['dice'] for c in cases]
    all_psnr = [c['psnr'] for c in cases]
    
    axes[0].bar(range(len(all_dice)), all_dice, color='steelblue')
    axes[0].set_xlabel('Case')
    axes[0].set_ylabel('Dice')
    axes[0].set_title('Per-Case Dice Coefficient')
    axes[0].axhline(y=np.mean(all_dice), color='red', linestyle='--', label=f'Mean: {np.mean(all_dice):.4f}')
    axes[0].legend()
    
    axes[1].bar(range(len(all_psnr)), all_psnr, color='coral')
    axes[1].set_xlabel('Case')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('Per-Case PSNR')
    axes[1].axhline(y=np.mean(all_psnr), color='red', linestyle='--', label=f'Mean: {np.mean(all_psnr):.1f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'summary_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figures saved to {FIG_DIR}/")

if __name__ == "__main__":
    main()