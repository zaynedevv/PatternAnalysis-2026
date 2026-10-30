"""Mac-only sanity test (no GPU, no mamba_ssm).

python test_pipeline.py data/DIV2K_train_HR

1. lab_to_rgb (torch) matches skimage and has finite gradients
2. LabDataset returns the right shapes, dtypes and ranges
3. RGB -> Lab -> RGB round trip survives saving as PNG
4. train/val split is disjoint; MeanMap baseline fits and predicts the right shape
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

from dataset import LabDataset, normalize, split_train_val, load_rgb

# MambaIRv2 is imported lazily inside Colorizer, so this works without mamba_ssm
from modules import MeanMap, lab_to_rgb


data_dir = sys.argv[1]
train_files, val_files = split_train_val(data_dir)

output_dir = 'test_out'
os.makedirs(output_dir, exist_ok=True)


def centre_crop(path, size=256):
    rgb = load_rgb(path)
    height, width, _ = rgb.shape

    top = (height - size) // 2
    left = (width - size) // 2
    return np.ascontiguousarray(rgb[top : top + size, left : left + size])


def to_batch_tensor(array):
    """numpy (H,W,C) -> torch (1,C,H,W)."""
    return torch.from_numpy(array).permute(2, 0, 1)[None]


def to_image(tensor):
    """torch (1,C,H,W) -> numpy (H,W,C)."""
    return tensor[0].permute(1, 2, 0).detach().numpy()


# ---------- 1. lab_to_rgb vs skimage ----------
rgb = centre_crop(train_files[0])
lab = rgb2lab(rgb)
lightness, ab = normalize(lab)

ours = to_image(lab_to_rgb(to_batch_tensor(lightness), to_batch_tensor(ab)))
reference = lab2rgb(lab)

max_error = np.abs(ours - reference).max()
assert max_error < 1e-2, f'lab_to_rgb differs from skimage by {max_error:.4f}'

# Gradients must be finite, including for all-zero ab (the step-0 grey output)
ab_real = to_batch_tensor(ab).clone().requires_grad_(True)
ab_zero = torch.zeros_like(ab_real, requires_grad=True)

for ab_input in (ab_real, ab_zero):
    lab_to_rgb(to_batch_tensor(lightness), ab_input).mean().backward()
    assert torch.isfinite(ab_input.grad).all(), 'non-finite gradient through lab_to_rgb'

print(f'OK  lab_to_rgb matches skimage (max err {max_error:.5f}), gradients finite')


# ---------- 2. dataset shapes / ranges ----------
for random_crop in (True, False):
    dataset = LabDataset(train_files[:4], crop_size=128, random_crop=random_crop)
    sample_lightness, sample_ab, sample_rgb = dataset[0]

    assert sample_lightness.shape == (1, 128, 128)
    assert sample_ab.shape == (2, 128, 128)
    assert sample_rgb.shape == (3, 128, 128)
    assert sample_lightness.dtype == sample_ab.dtype == sample_rgb.dtype == torch.float32

    assert 0 <= sample_lightness.min() and sample_lightness.max() <= 1, 'L out of [0,1]'
    assert -1 <= sample_ab.min() and sample_ab.max() <= 1, 'ab out of [-1,1]'
    assert 0 <= sample_rgb.min() and sample_rgb.max() <= 1, 'rgb out of [0,1]'

print(
    f'OK  dataset shapes/dtypes/ranges '
    f'(L {sample_lightness.min():.2f}..{sample_lightness.max():.2f}, '
    f'ab {sample_ab.min():.2f}..{sample_ab.max():.2f})'
)


# ---------- 3. round trip through PNG ----------
reconstructed = to_image(lab_to_rgb(to_batch_tensor(lightness), to_batch_tensor(ab)))
greyscale = np.repeat(lightness, 3, axis=2)

for tag, image in [('gt', rgb), ('recon', reconstructed), ('gray', greyscale)]:
    Image.fromarray((image * 255).round().astype(np.uint8)).save(f'{output_dir}/roundtrip_{tag}.png')

original_png = np.asarray(Image.open(f'{output_dir}/roundtrip_gt.png'), dtype=np.float32)
reconstructed_png = np.asarray(Image.open(f'{output_dir}/roundtrip_recon.png'), dtype=np.float32)

mean_difference = np.abs(original_png - reconstructed_png).mean()
assert mean_difference < 1.0, f'round trip mean abs diff {mean_difference:.2f} (0-255 scale)'

print(f'OK  PNG round trip (mean abs diff {mean_difference:.3f}/255), images in {output_dir}/')


# ---------- 4. split + MeanMap ----------
assert len(val_files) == 80
assert not set(train_files) & set(val_files)

fit_dataset = LabDataset(train_files[:4], crop_size=128, random_crop=False)
mean_map = MeanMap().fit((sample_l[None], sample_ab[None]) for sample_l, sample_ab, _ in fit_dataset)

sample_lightness, _, _ = fit_dataset[0]
assert mean_map(sample_lightness[None]).shape == (1, 2, 128, 128)

print(f'OK  split {len(train_files)}/{len(val_files)} disjoint, MeanMap fits and predicts')
