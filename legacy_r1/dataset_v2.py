"""Dataset that loads ALL preprocessed data into memory. Fast, no NFS bottleneck."""
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):
    """Loads all preprocessed .npz into memory at init. Each __getitem__ returns one slice."""

    def __init__(self, patient_ids, data_dir, augment=False):
        self.augment = augment

        # Load all data into memory
        self.ncct_data = {}  # pid -> (H, W, Z) np.ndarray
        self.cta_data = {}
        self.seg_data = {}

        for pid in patient_ids:
            data = np.load(data_dir / f"patient{pid}.npz")
            self.ncct_data[pid] = np.ascontiguousarray(data['ncct'])  # (H, W, Z)
            self.cta_data[pid] = np.ascontiguousarray(data['cta'])
            self.seg_data[pid] = np.ascontiguousarray(data['seg'])
            data.close()

        # Build slice index
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
        ncct_slice = self.ncct_data[pid][:, :, z].copy()
        cta_slice = self.cta_data[pid][:, :, z].copy()
        seg_slice = self.seg_data[pid][:, :, z].copy()

        if self.augment:
            ncct_slice, cta_slice, seg_slice = self._augment(ncct_slice, cta_slice, seg_slice)

        ncct_t = torch.from_numpy(ncct_slice).unsqueeze(0)
        cta_t = torch.from_numpy(cta_slice).unsqueeze(0)
        seg_t = torch.from_numpy(seg_slice).unsqueeze(0)
        return ncct_t, cta_t, seg_t
