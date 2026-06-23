"""Dataset that loads 256x256 preprocessed data and resizes to 512x512.

Strategy: use existing 256 data (4.1 GB, fits in RAM) + cv2 on-the-fly resize.
Avoids 110 GB storage and mmap memory pressure of native 512 data.
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class PreprocessedDataset(Dataset):
    """Loads all preprocessed 256x256 .npz into memory. Resizes to target_size."""

    def __init__(self, patient_ids, data_dir, augment=False, target_size=512):
        self.augment = augment
        self.target_size = target_size
        self.ncct_data = {}
        self.cta_data = {}
        self.seg_data = {}

        for pid in patient_ids:
            data = np.load(data_dir / f"patient{pid}.npz")
            self.ncct_data[pid] = np.ascontiguousarray(data["ncct"])
            self.cta_data[pid] = np.ascontiguousarray(data["cta"])
            self.seg_data[pid] = np.ascontiguousarray(data["seg"])
            data.close()

        self.slices = []
        for pid in patient_ids:
            n_slices = self.ncct_data[pid].shape[2]
            for z in range(n_slices):
                self.slices.append((pid, z))

    def __len__(self):
        return len(self.slices)

    def _augment(self, ncct, cta, seg):
        if random.random() < 0.5:
            ncct, cta, seg = np.fliplr(ncct).copy(), np.fliplr(cta).copy(), np.fliplr(seg).copy()
        if random.random() < 0.5:
            ncct, cta, seg = np.flipud(ncct).copy(), np.flipud(cta).copy(), np.flipud(seg).copy()
        return ncct, cta, seg

    def __getitem__(self, idx):
        pid, z = self.slices[idx]
        ncct_sl = self.ncct_data[pid][:, :, z].copy()
        cta_sl = self.cta_data[pid][:, :, z].copy()
        seg_sl = self.seg_data[pid][:, :, z].copy()

        # Resize 256 -> 512 on-the-fly
        if self.target_size != 256:
            ncct_sl = cv2.resize(ncct_sl, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            cta_sl = cv2.resize(cta_sl, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            seg_sl = cv2.resize(seg_sl, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            ncct_sl, cta_sl, seg_sl = self._augment(ncct_sl, cta_sl, seg_sl)

        ncct_t = torch.from_numpy(ncct_sl).unsqueeze(0)
        cta_t = torch.from_numpy(cta_sl).unsqueeze(0)
        seg_t = torch.from_numpy(seg_sl).unsqueeze(0)
        return ncct_t, cta_t, seg_t
