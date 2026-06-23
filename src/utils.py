"""Utility functions for inference and data I/O."""

import numpy as np
import torch
import nibabel as nib
import cv2


def inference_volume(model, ncct_path, device, image_size=512, ncct_clip=(-200, 500), cta_clip=(-200, 800)):
    """Run inference on a full 3D NCCT volume, produce CTA + segmentation predictions.

    Process:
    1. Load NCCT .nii file
    2. For each axial slice: preprocess → model inference → collect
    3. Stack predictions into 3D volumes
    4. Resize predictions to original resolution
    5. Return prediction volumes + affine/header for saving

    Args:
        model: trained MultiTaskUNet
        ncct_path: Path to NCCT .nii file
        device: torch device
        image_size: (H, W) for model input
        ncct_clip: (min, max) HU clipping range for NCCT
        cta_clip: (min, max) HU clipping range for CTA (for inverse normalization)

    Returns:
        cta_pred_vol: (orig_H, orig_W, Z) predicted CTA in HU
        seg_pred_vol: (orig_H, orig_W, Z) vessel probability map [0, 1]
        affine: NIfTI affine matrix
        header: NIfTI header
    """
    # Load NCCT
    ncct_nii = nib.load(str(ncct_path))
    ncct = ncct_nii.get_fdata().astype(np.float32)
    affine = ncct_nii.affine
    header = ncct_nii.header

    Z = ncct.shape[2]
    orig_H, orig_W = ncct.shape[0], ncct.shape[1]

    # Normalize NCCT once
    ncct_min, ncct_max = ncct_clip
    ncct_clipped = np.clip(ncct, ncct_min, ncct_max)
    ncct_norm = (ncct_clipped - ncct_min) / (ncct_max - ncct_min)

    cta_pred_slices = []
    seg_pred_slices = []

    for z in range(Z):
        # Preprocess slice
        sl = ncct_norm[:, :, z].astype(np.float32)
        sl = cv2.resize(sl, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        # Model forward
        inp = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
        with torch.no_grad():
            cta_out, seg_out = model(inp)
        # cta_out: (1, 1, H, W) tanh output
        # seg_out: (1, 1, H, W) logits

        cta_sl = cta_out.squeeze().cpu().numpy()  # (H, W)
        seg_sl = torch.sigmoid(seg_out).squeeze().cpu().numpy()  # (H, W) probability

        # Resize back to original resolution
        cta_sl = cv2.resize(cta_sl, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
        seg_sl = cv2.resize(seg_sl, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)

        cta_pred_slices.append(cta_sl)
        seg_pred_slices.append(seg_sl)

    cta_pred_vol = np.stack(cta_pred_slices, axis=2)  # (H, W, Z)
    seg_pred_vol = np.stack(seg_pred_slices, axis=2)  # (H, W, Z)

    # Inverse normalize CTA: tanh output is normalized to [0,1] range after clipping,
    # but model output is tanh → need to map back to HU
    # Model outputs are in normalized [0,1] space (matching training target)
    # Denormalize: HU = pred * (cta_max - cta_min) + cta_min
    cta_min, cta_max = cta_clip
    cta_pred_vol = cta_pred_vol * (cta_max - cta_min) + cta_min

    return cta_pred_vol, seg_pred_vol, affine, header


def save_nii(data, affine, header, save_path):
    """Save a 3D numpy array as a NIfTI file.

    Args:
        data: (H, W, Z) numpy array
        affine: NIfTI affine matrix
        header: NIfTI header
        save_path: output path
    """
    img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(img, str(save_path))
