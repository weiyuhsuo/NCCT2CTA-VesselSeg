"""Preprocess all 3D nii volumes into 2D slices.
Round 2: 512×512 resolution, memory-mappable .npy per modality.

Format per patient:
    patient{pid}_ncct.npy  — (H, W, Z) float32, memory-mappable
    patient{pid}_cta.npy   — (H, W, Z) float32
    patient{pid}_seg.npy   — (H, W, Z) float32
    patient{pid}_meta.npz  — orig_shape, ncct_clip, cta_clip

Run once before training:  python preprocess.py
"""
import os, sys, json, random
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

ROOT = Path("/cpfs01/projects-SSD/cfff-71d5b2895244_SSD/hyb_24110860026/weiyushuo/NCCT2CTA-VesselSeg")
DATA_DIR = ROOT / "data" / "data_100"
PREPROC_DIR = Path("/tmp/preprocessed_512")
NCCT_CLIP = (-200, 500)
CTA_CLIP = (-200, 800)
IMAGE_SIZE = 512
SEED = 42


def get_patient_ids():
    ncct_dir = DATA_DIR / "NCCT"
    cta_dir = DATA_DIR / "CTA"
    seg_dir = DATA_DIR / "SEG"
    ncct_files = sorted([
        f.name.split(".")[0].replace("patient", "")
        for f in ncct_dir.iterdir() if f.suffix == ".nii"
    ])
    cta_files = sorted([
        f.name.split(".")[0].replace("patient", "")
        for f in cta_dir.iterdir() if f.suffix == ".nii"
    ])
    seg_files = sorted([
        f.name.split(".")[0].replace("patient", "")
        for f in seg_dir.iterdir() if f.suffix == ".gz"
    ])
    return sorted(set(ncct_files) & set(cta_files) & set(seg_files))


def preprocess_patient(pid):
    """Resize and normalize one patient's volumes to IMAGE_SIZE×IMAGE_SIZE."""
    ncct_path = DATA_DIR / "NCCT" / f"patient{pid}.nii"
    cta_path = DATA_DIR / "CTA" / f"patient{pid}.nii"
    seg_path = DATA_DIR / "SEG" / f"patient{pid}.nii.gz"

    ncct = nib.load(str(ncct_path), mmap=True).get_fdata().astype(np.float32)
    cta = nib.load(str(cta_path), mmap=True).get_fdata().astype(np.float32)
    seg = nib.load(str(seg_path), mmap=True).get_fdata().astype(np.float32)

    Z = ncct.shape[2]
    orig_shape = (ncct.shape[0], ncct.shape[1])

    # Clip & normalize full volumes
    ncct_clipped = np.clip(ncct, *NCCT_CLIP)
    cta_clipped = np.clip(cta, *CTA_CLIP)
    ncct_min, ncct_max = NCCT_CLIP
    cta_min, cta_max = CTA_CLIP
    ncct_norm = (ncct_clipped - ncct_min) / (ncct_max - ncct_min)
    cta_norm = (cta_clipped - cta_min) / (cta_max - cta_min)

    # Pre-allocate output arrays
    ncct_resized = np.zeros((IMAGE_SIZE, IMAGE_SIZE, Z), dtype=np.float32)
    cta_resized = np.zeros((IMAGE_SIZE, IMAGE_SIZE, Z), dtype=np.float32)
    seg_resized = np.zeros((IMAGE_SIZE, IMAGE_SIZE, Z), dtype=np.float32)

    for z in range(Z):
        ncct_resized[:, :, z] = cv2.resize(
            ncct_norm[:, :, z].astype(np.float32),
            (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        cta_resized[:, :, z] = cv2.resize(
            cta_norm[:, :, z].astype(np.float32),
            (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        seg_resized[:, :, z] = cv2.resize(
            seg[:, :, z].astype(np.float32),
            (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    return ncct_resized, cta_resized, seg_resized, orig_shape


def main():
    patient_ids = get_patient_ids()
    print(f"Total patients: {len(patient_ids)}")
    print(f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"Output dir: {PREPROC_DIR}")

    # Estimate storage
    avg_z = 350
    per_patient_gb = IMAGE_SIZE * IMAGE_SIZE * avg_z * 4 * 3 / 1e9
    print(f"Estimated storage: ~{per_patient_gb * len(patient_ids):.1f} GB total")
    print(f"Estimated per-patient RAM (mmap): ~0 GB (on-demand)")

    # Split (same seed = same split as Round 1)
    random.seed(SEED)
    random.shuffle(patient_ids)
    n_test = int(len(patient_ids) * 0.20)
    n_val = int(len(patient_ids) * 0.10)
    test_ids = sorted(patient_ids[:n_test])
    val_ids = sorted(patient_ids[n_test:n_test + n_val])
    train_ids = sorted(patient_ids[n_test + n_val:])
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    PREPROC_DIR.mkdir(parents=True, exist_ok=True)

    # Save split
    split_info = {'train': train_ids, 'val': val_ids, 'test': test_ids, 'seed': SEED}
    with open(PREPROC_DIR / "split.json", 'w') as f:
        json.dump(split_info, f, indent=2)

    # Process all patients
    for pid in tqdm(patient_ids, desc="Preprocessing patients"):
        ncct_resized, cta_resized, seg_resized, orig_shape = preprocess_patient(pid)

        # Save as individual .npy files (memory-mappable)
        np.save(PREPROC_DIR / f"patient{pid}_ncct.npy", ncct_resized)
        np.save(PREPROC_DIR / f"patient{pid}_cta.npy", cta_resized)
        np.save(PREPROC_DIR / f"patient{pid}_seg.npy", seg_resized)

        # Save metadata
        np.savez(PREPROC_DIR / f"patient{pid}_meta.npz",
                 orig_shape=orig_shape, image_size=IMAGE_SIZE,
                 ncct_clip=NCCT_CLIP, cta_clip=CTA_CLIP)

    print("Preprocessing complete!")
    print(f"Data saved to {PREPROC_DIR}")

    # Print stats
    total_gb = sum(
        f.stat().st_size for f in PREPROC_DIR.glob("*.npy")
    ) / 1e9
    print(f"Total .npy storage: {total_gb:.1f} GB")


if __name__ == "__main__":
    main()
