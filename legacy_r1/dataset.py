import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import cv2

class NiiDataset(Dataset):
    """2D slice dataset from 3D nii volumes. Caches up to 5 patient volumes in memory."""

    def __init__(self, patient_ids, ncct_dir, cta_dir, seg_dir,
                 image_size=256, ncct_clip=(-200,500), cta_clip=(-200,800),
                 augment=False):
        self.patient_ids = list(patient_ids)
        self.ncct_dir = ncct_dir
        self.cta_dir = cta_dir
        self.seg_dir = seg_dir
        self.image_size = image_size
        self.ncct_clip = ncct_clip
        self.cta_clip = cta_clip
        self.augment = augment
        self._cache = {}
        self.slices = []
        for pid in self.patient_ids:
            ncct_path = self.ncct_dir / f"patient{pid}.nii"
            img = nib.load(str(ncct_path))
            n_slices = img.shape[2]
            for s in range(n_slices):
                self.slices.append((pid, s))

    def __len__(self):
        return len(self.slices)

    def _get_volumes(self, pid):
        if pid not in self._cache:
            # mmap=True avoids loading full data until accessed
            ncct = nib.load(str(self.ncct_dir / f"patient{pid}.nii"), mmap=True).get_fdata().astype(np.float32)
            cta = nib.load(str(self.cta_dir / f"patient{pid}.nii"), mmap=True).get_fdata().astype(np.float32)
            seg = nib.load(str(self.seg_dir / f"patient{pid}.nii.gz"), mmap=True).get_fdata().astype(np.float32)
            self._cache[pid] = (ncct, cta, seg)
            if len(self._cache) > 5:
                oldest = next(iter(self._cache))
                if oldest != pid:
                    del self._cache[oldest]
        return self._cache[pid]

    def _preprocess_slice(self, ncct_slice, cta_slice, seg_slice):
        ncct_slice = np.clip(ncct_slice, *self.ncct_clip)
        cta_slice = np.clip(cta_slice, *self.cta_clip)
        ncct_min, ncct_max = self.ncct_clip
        cta_min, cta_max = self.cta_clip
        ncct_slice = (ncct_slice - ncct_min) / (ncct_max - ncct_min)
        cta_slice = (cta_slice - cta_min) / (cta_max - cta_min)
        ncct_slice = cv2.resize(ncct_slice, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        cta_slice = cv2.resize(cta_slice, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        seg_slice = cv2.resize(seg_slice, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return ncct_slice, cta_slice, seg_slice

    def _augment(self, ncct, cta, seg):
        if random.random() < 0.5:
            ncct, cta, seg = np.fliplr(ncct).copy(), np.fliplr(cta).copy(), np.fliplr(seg).copy()
        if random.random() < 0.5:
            ncct, cta, seg = np.flipud(ncct).copy(), np.flipud(cta).copy(), np.flipud(seg).copy()
        return ncct, cta, seg

    def __getitem__(self, idx):
        pid, slice_idx = self.slices[idx]
        ncct_vol, cta_vol, seg_vol = self._get_volumes(pid)
        ncct_slice = ncct_vol[:, :, slice_idx]
        cta_slice = cta_vol[:, :, slice_idx]
        seg_slice = seg_vol[:, :, slice_idx]
        ncct_slice, cta_slice, seg_slice = self._preprocess_slice(ncct_slice, cta_slice, seg_slice)
        if self.augment:
            ncct_slice, cta_slice, seg_slice = self._augment(ncct_slice, cta_slice, seg_slice)
        ncct_t = torch.from_numpy(ncct_slice.copy()).unsqueeze(0)
        cta_t = torch.from_numpy(cta_slice.copy()).unsqueeze(0)
        seg_t = torch.from_numpy(seg_slice.copy()).unsqueeze(0)
        return ncct_t, cta_t, seg_t
