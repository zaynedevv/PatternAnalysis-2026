"""Data loading and pre-processing.

Images are cropped, then converted to CIELAB with skimage and normalised:
L/100 -> [0,1] (model input), ab/128 -> [-1,1] (target).

Splits (leakage-free, deterministic):
  train  DIV2K_train_HR, sorted files[:-80]   (720 images)
  val    DIV2K_train_HR, sorted files[-80:]   (80 images)  - lambda / ab_scale selection
  test   Kodak24                              (24 images)  - scored once, at the end
"""

import os

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import Dataset


NUM_VAL_IMAGES = 80
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def list_images(folder):
    """Sorted full paths of every image in `folder` (sorting makes the split reproducible)."""
    names = [name for name in os.listdir(folder) if name.lower().endswith(IMAGE_EXTENSIONS)]
    return sorted(os.path.join(folder, name) for name in names)


def split_train_val(folder, num_val=NUM_VAL_IMAGES):
    """Fixed slice of the sorted file list, so every run and every lambda sees the same split."""
    all_files = list_images(folder)

    train_files = all_files[:-num_val]
    val_files = all_files[-num_val:]

    assert not set(train_files) & set(val_files)
    return train_files, val_files


def normalize(lab_image):
    """numpy Lab (H,W,3) -> lightness (H,W,1) in [0,1] and ab (H,W,2) in [-1,1], both float32."""
    lightness = lab_image[:, :, 0:1] / 100.0
    ab = lab_image[:, :, 1:3] / 128.0
    return lightness.astype('float32'), ab.astype('float32')


def load_rgb(path):
    """Image file -> numpy RGB (H,W,3) float32 in [0,1]."""
    image = Image.open(path).convert('RGB')
    return np.asarray(image, dtype=np.float32) / 255.0


def to_tensor(array):
    """numpy (H,W,C) -> torch (C,H,W)."""
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)


class LabDataset(Dataset):
    """Returns (lightness, ab, rgb) tensors for one square crop of one image.

    random_crop=True   random crop + 50% horizontal flip (training augmentation)
    random_crop=False  centre crop (validation / test, deterministic)
    """

    def __init__(self, files, crop_size, random_crop):
        self.files = files
        self.crop_size = crop_size
        self.random_crop = random_crop

    def __len__(self):
        return len(self.files)

    def name(self, index):
        """File name without folder or extension, e.g. 'kodim01'."""
        return os.path.splitext(os.path.basename(self.files[index]))[0]

    def __getitem__(self, index):
        rgb = load_rgb(self.files[index])
        height, width, _ = rgb.shape
        size = self.crop_size

        # Choose the crop position
        if self.random_crop:
            top = np.random.randint(0, height - size + 1)
            left = np.random.randint(0, width - size + 1)
        else:
            top = (height - size) // 2
            left = (width - size) // 2

        # Crop before rgb2lab: converting the crop is much cheaper than the full image
        rgb = rgb[top : top + size, left : left + size]

        # Horizontal flip, training only
        if self.random_crop and np.random.rand() < 0.5:
            rgb = rgb[:, ::-1]

        rgb = np.ascontiguousarray(rgb)
        lightness, ab = normalize(rgb2lab(rgb))

        return to_tensor(lightness), to_tensor(ab), to_tensor(rgb)
