"""Dataset with memory-mapped .npy access for 512×512 training.

Round 2: Uses np.load(mmap_mode='r') to access large 512×512 volumes
without loading everything into RAM. Only the accessed slices are read from disk.

Data format expected (from preprocess.py):
    patient{pid}_ncct.npy  — (512, 512, Z) float32
    patient{pid}_cta.npy   — (512, 512, Z) float32
    patient{pid}_seg.npy   — (512, 512, Z) float32
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """Memory-efficient dataset using mmap'd .npy files.

    Opens each patient's data as a memory-mapped array. Only the accessed
    slices are read into RAM. Works for both 256×256 and 512×512.
    """

    def __init__(self, patient_ids, data_dir, augment=False):
        """
        Args:
            patient_ids: list of patient ID strings (e.g., ['0533', '0544', ...])
            data_dir: Path to directory containing patient{pid}_*.npy files
            augment: whether to apply data augmentation (training only)
        """
        self.augment = augment
        self.data_dir = data_dir

        # Open all volumes as mmap arrays — negligible RAM cost
        self.volumes = {}
        for pid in patient_ids:
            self.volumes[pid] = {
                'ncct': np.load(data_dir / f"patient{pid}_ncct.npy", mmap_mode='r'),
                'cta': np.load(data_dir / f"patient{pid}_cta.npy", mmap_mode='r'),
                'seg': np.load(data_dir / f"patient{pid}_seg.npy", mmap_mode='r'),
            }

        # Build slice index
        self.slices = []
        for pid in patient_ids:
            n_slices = self.volumes[pid]['ncct'].shape[2]
            for z in range(n_slices):
                self.slices.append((pid, z))

    def __len__(self):
        return len(self.slices)

    def _augment(self, ncct, cta, seg):
        """Random horizontal/vertical flips (in-place copies)."""
        if random.random() < 0.5:
            ncct = np.fliplr(ncct).copy()
            cta = np.fliplr(cta).copy()
            seg = np.fliplr(seg).copy()
        if random.random() < 0.5:
            ncct = np.flipud(ncct).copy()
            cta = np.flipud(cta).copy()
            seg = np.flipud(seg).copy()
        return ncct, cta, seg

    def __getitem__(self, idx):
        pid, z = self.slices[idx]
        # mmap: only this slice is read from disk into RAM
        ncct_slice = self.volumes[pid]['ncct'][:, :, z].copy().astype(np.float32)
        cta_slice = self.volumes[pid]['cta'][:, :, z].copy().astype(np.float32)
        seg_slice = self.volumes[pid]['seg'][:, :, z].copy().astype(np.float32)

        if self.augment:
            ncct_slice, cta_slice, seg_slice = self._augment(ncct_slice, cta_slice, seg_slice)

        ncct_t = torch.from_numpy(ncct_slice).unsqueeze(0)  # (1, H, W)
        cta_t = torch.from_numpy(cta_slice).unsqueeze(0)
        seg_t = torch.from_numpy(seg_slice).unsqueeze(0)
        return ncct_t, cta_t, seg_t
