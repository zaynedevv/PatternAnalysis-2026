"""Mac-only sanity test (no GPU, no mamba_ssm).

python test_pipeline.py data/DIV2K_train_HR

1. lab_to_rgb (torch) matches skimage and has finite gradients
2. LabDataset returns the right shapes, dtypes and ranges
3. RGB -> Lab -> RGB round trip survives saving as PNG
"""
import os, sys, types
import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

# train.py imports model (needs mamba_ssm) and lpips; stub them so LabDataset imports on a Mac
sys.modules.setdefault('model', types.SimpleNamespace(Colorizer=None))
sys.modules.setdefault('lpips', types.ModuleType('lpips'))

from lab_utils import normalize, lab_to_rgb
from train import LabDataset

data_dir = sys.argv[1]
files = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir))
out_dir = 'test_out'
os.makedirs(out_dir, exist_ok=True)


def centre_crop(path, c=256):
    img = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    H, W, _ = img.shape
    y, x = (H - c) // 2, (W - c) // 2
    return np.ascontiguousarray(img[y:y + c, x:x + c])


def to_tensor(arr):
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


# ---------- 1. lab_to_rgb vs skimage ----------
rgb = centre_crop(files[0])
lab = rgb2lab(rgb)
L, ab = normalize(lab)

ours = lab_to_rgb(to_tensor(L), to_tensor(ab))[0].permute(1, 2, 0).numpy()
ref = lab2rgb(lab)
err = np.abs(ours - ref).max()
assert err < 1e-2, f'lab_to_rgb differs from skimage by {err:.4f}'

ab_t = to_tensor(ab).clone().requires_grad_(True)
ab_zero = torch.zeros_like(ab_t, requires_grad=True)          # grey output, the step-0 case
for a in (ab_t, ab_zero):
    lab_to_rgb(to_tensor(L), a).mean().backward()
    assert torch.isfinite(a.grad).all(), 'non-finite gradient through lab_to_rgb'
print(f'OK  lab_to_rgb matches skimage (max err {err:.5f}), gradients finite')


# ---------- 2. dataset shapes / ranges ----------
for random_crop in (True, False):
    ds = LabDataset(files[:4], crop=128, random_crop=random_crop)
    L_t, ab_t, rgb_t = ds[0]
    assert L_t.shape == (1, 128, 128) and ab_t.shape == (2, 128, 128) and rgb_t.shape == (3, 128, 128)
    assert L_t.dtype == ab_t.dtype == rgb_t.dtype == torch.float32
    assert 0 <= L_t.min() and L_t.max() <= 1, f'L out of [0,1]: {L_t.min():.3f}..{L_t.max():.3f}'
    assert -1 <= ab_t.min() and ab_t.max() <= 1, f'ab out of [-1,1]: {ab_t.min():.3f}..{ab_t.max():.3f}'
    assert 0 <= rgb_t.min() and rgb_t.max() <= 1
print(f'OK  dataset shapes/dtypes/ranges (L {L_t.min():.2f}..{L_t.max():.2f}, ab {ab_t.min():.2f}..{ab_t.max():.2f})')


# ---------- 3. round trip through PNG ----------
recon = lab_to_rgb(to_tensor(L), to_tensor(ab))[0].permute(1, 2, 0).numpy()
gray = np.repeat(L, 3, axis=2)
for tag, arr in [('gt', rgb), ('recon', recon), ('gray', gray)]:
    Image.fromarray((arr * 255).round().astype(np.uint8)).save(f'{out_dir}/roundtrip_{tag}.png')

gt_png = np.asarray(Image.open(f'{out_dir}/roundtrip_gt.png'), dtype=np.float32)
recon_png = np.asarray(Image.open(f'{out_dir}/roundtrip_recon.png'), dtype=np.float32)
diff = np.abs(gt_png - recon_png).mean()
assert diff < 1.0, f'round trip mean abs diff {diff:.2f} (0-255 scale)'
print(f'OK  PNG round trip (mean abs diff {diff:.3f}/255), images in {out_dir}/')
