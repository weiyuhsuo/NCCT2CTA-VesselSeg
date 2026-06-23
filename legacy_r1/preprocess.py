"""Preprocess all 3D nii volumes into 2D slice .pt files.
Run once before training to avoid slow NFS reads during training."""
import os, sys, json, random
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

ROOT = Path("/cpfs01/projects-SSD/cfff-71d5b2895244_SSD/hyb_24110860026/weiyushuo/NCCT2CTA-VesselSeg")
DATA_DIR = ROOT / "data" / "data_100"
PREPROC_DIR = ROOT / "data" / "preprocessed"
NCCT_CLIP = (-200, 500)
CTA_CLIP = (-200, 800)
IMAGE_SIZE = 256
SEED = 42

def get_patient_ids():
    ncct_dir = DATA_DIR / "NCCT"
    cta_dir = DATA_DIR / "CTA"
    seg_dir = DATA_DIR / "SEG"
    ncct_files = sorted([f.name.split(".")[0].replace("patient","") for f in ncct_dir.iterdir() if f.suffix == ".nii"])
    cta_files = sorted([f.name.split(".")[0].replace("patient","") for f in cta_dir.iterdir() if f.suffix == ".nii"])
    seg_files = sorted([f.name.split(".")[0].replace("patient","") for f in seg_dir.iterdir() if f.suffix == ".gz"])
    return sorted(set(ncct_files) & set(cta_files) & set(seg_files))

def preprocess_patient(pid):
    """Preprocess one patient's data and save as train/val/test slices."""
    ncct_path = DATA_DIR / "NCCT" / f"patient{pid}.nii"
    cta_path = DATA_DIR / "CTA" / f"patient{pid}.nii"
    seg_path = DATA_DIR / "SEG" / f"patient{pid}.nii.gz"

    ncct = nib.load(str(ncct_path), mmap=True).get_fdata().astype(np.float32)
    cta = nib.load(str(cta_path), mmap=True).get_fdata().astype(np.float32)
    seg = nib.load(str(seg_path), mmap=True).get_fdata().astype(np.float32)

    Z = ncct.shape[2]
    ncct_clipped = np.clip(ncct, *NCCT_CLIP)
    cta_clipped = np.clip(cta, *CTA_CLIP)
    ncct_min, ncct_max = NCCT_CLIP
    cta_min, cta_max = CTA_CLIP
    ncct_norm = (ncct_clipped - ncct_min) / (ncct_max - ncct_min)
    cta_norm = (cta_clipped - cta_min) / (cta_max - cta_min)

    slices_data = []
    for z in range(Z):
        ncct_sl = ncct_norm[:, :, z].astype(np.float32)
        cta_sl = cta_norm[:, :, z].astype(np.float32)
        seg_sl = seg[:, :, z].astype(np.float32)

        ncct_sl = cv2.resize(ncct_sl, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        cta_sl = cv2.resize(cta_sl, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        seg_sl = cv2.resize(seg_sl, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        slices_data.append({
            'ncct': ncct_sl, 'cta': cta_sl, 'seg': seg_sl, 'z': z
        })
    return slices_data, (ncct.shape[0], ncct.shape[1])  # original H, W

def main():
    patient_ids = get_patient_ids()
    print(f"Total patients: {len(patient_ids)}")

    # Split
    random.seed(SEED)
    random.shuffle(patient_ids)
    n_test = int(len(patient_ids) * 0.20)
    n_val = int(len(patient_ids) * 0.10)
    test_ids = sorted(patient_ids[:n_test])
    val_ids = sorted(patient_ids[n_test:n_test+n_val])
    train_ids = sorted(patient_ids[n_test+n_val:])
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Save split
    split_info = {'train': train_ids, 'val': val_ids, 'test': test_ids, 'seed': SEED}
    with open(PREPROC_DIR / "split.json", 'w') as f:
        json.dump(split_info, f, indent=2)

    for split_name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        out_dir = PREPROC_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for pid in tqdm(ids, desc=f"Preprocessing {split_name}"):
            slices_data, orig_shape = preprocess_patient(pid)
            np.savez_compressed(
                out_dir / f"patient{pid}.npz",
                ncct=np.stack([s['ncct'] for s in slices_data], axis=2),
                cta=np.stack([s['cta'] for s in slices_data], axis=2),
                seg=np.stack([s['seg'] for s in slices_data], axis=2),
                orig_shape=orig_shape,
            )

    print("Preprocessing complete!")
    print(f"Data saved to {PREPROC_DIR}")

if __name__ == "__main__":
    main()
