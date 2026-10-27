"""Evaluate one method on Kodak24.

python eval.py --method finetuned --ckpt runs/lam0.1/ckpt_5000.pt --out results/finetuned
python eval.py --method frozen                                      --out results/frozen
python eval.py --method meanmap --train_data data/DIV2K_train_HR    --out results/meanmap

Writes per-image PNGs (gray, pred, gt), metrics.csv, and prints the means.
"""
import os, csv, argparse
import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from lab_utils import normalize, lab_to_rgb

parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=['finetuned', 'frozen', 'meanmap'], required=True)
parser.add_argument('--ckpt', default=None)
parser.add_argument('--pretrained', default='experiments/mambairv2_ColorDN_15.pth')
parser.add_argument('--test_data', default='data/kodak24')
parser.add_argument('--train_data', default='data/DIV2K_train_HR')   # meanmap only
parser.add_argument('--out', required=True)
parser.add_argument('--crop', type=int, default=256)
parser.add_argument('--ab_scale', type=float, default=1.0)   # post-hoc saturation boost; pick on val, not Kodak
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.out, exist_ok=True)


# ---------- loading ----------
def load_centre_crop(path, c):
    img = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    H, W, _ = img.shape
    y, x = (H - c) // 2, (W - c) // 2
    return np.ascontiguousarray(img[y:y + c, x:x + c])


def to_tensor(arr):
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def to_numpy(t):
    return t[0].permute(1, 2, 0).cpu().numpy()


# ---------- the three methods ----------
if args.method in ('finetuned', 'frozen'):
    from model import Colorizer
    model = Colorizer(args.pretrained if args.method == 'frozen' else None, args.crop).to(device)
    if args.method == 'finetuned':
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model.eval()

    @torch.no_grad()
    def predict_ab(L):
        return model(L).float()

else:  # meanmap: for each L bin, the mean ab over training crops
    bins = 64
    sums = np.zeros((bins, 2)); counts = np.zeros(bins)
    files = sorted(os.path.join(args.train_data, f) for f in os.listdir(args.train_data))
    for f in files:
        L, ab = normalize(rgb2lab(load_centre_crop(f, 512)))
        idx = np.clip((L[..., 0] * bins).astype(int), 0, bins - 1).ravel()
        np.add.at(sums, idx, ab.reshape(-1, 2))
        np.add.at(counts, idx, 1)
    table = torch.from_numpy(sums / np.maximum(counts, 1)[:, None]).float().to(device)

    def predict_ab(L):
        idx = (L[:, 0] * bins).long().clamp(0, bins - 1)        # (B,H,W)
        return table[idx].permute(0, 3, 1, 2)                   # (B,2,H,W)


# ---------- metrics ----------
perceptual = lpips.LPIPS(net='alex').to(device).eval()


def colourfulness(rgb):
    """Hasler & Süsstrunk 2003. rgb numpy (H,W,3) in [0,1]."""
    r, g, b = [rgb[..., i] * 255 for i in range(3)]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)


def chroma_bleed(L, ab):
    """Mean chroma gradient at strong luminance edges. Compare pred vs gt as a ratio."""
    gy, gx = np.gradient(L[..., 0])
    edge = np.hypot(gx, gy)
    mask = edge > np.percentile(edge, 95)
    ab_grad = sum(np.hypot(*np.gradient(ab[..., i])) for i in range(2))
    return float(ab_grad[mask].mean())


# ---------- main loop ----------
rows = []
test_files = sorted(os.path.join(args.test_data, f) for f in os.listdir(args.test_data))

for path in test_files:
    name = os.path.splitext(os.path.basename(path))[0]
    rgb_gt = load_centre_crop(path, args.crop)
    L_np, ab_gt = normalize(rgb2lab(rgb_gt))

    L = to_tensor(L_np)
    ab_pred = (predict_ab(L) * args.ab_scale).clamp(-1, 1)
    rgb_pred = to_numpy(lab_to_rgb(L, ab_pred))
    ab_pred_np = to_numpy(ab_pred)

    row = {
        'image': name,
        'psnr': peak_signal_noise_ratio(rgb_gt, rgb_pred, data_range=1.0),
        'ssim_rgb': structural_similarity(rgb_gt, rgb_pred, channel_axis=2, data_range=1.0),
        'ssim_L': structural_similarity(L_np[..., 0], L_np[..., 0], data_range=1.0),   # identity by construction
        'lpips': perceptual(to_tensor(rgb_pred) * 2 - 1, to_tensor(rgb_gt) * 2 - 1).item(),
        'colourfulness_pred': colourfulness(rgb_pred),
        'colourfulness_gt': colourfulness(rgb_gt),
        'bleed_pred': chroma_bleed(L_np, ab_pred_np),
        'bleed_gt': chroma_bleed(L_np, ab_gt),
    }
    rows.append(row)

    gray = np.repeat(L_np, 3, axis=2)
    for tag, arr in [('gray', gray), ('pred', rgb_pred), ('gt', rgb_gt)]:
        Image.fromarray((arr * 255).astype(np.uint8)).save(f'{args.out}/{name}_{tag}.png')

# ---------- write results ----------
with open(f'{args.out}/metrics.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

print(f'\n{args.method}  ab_scale {args.ab_scale}  ({len(rows)} images)')
for k in rows[0]:
    if k != 'image':
        print(f'  {k:20s} {np.mean([r[k] for r in rows]):.4f}')
cf_ratio = np.mean([r['colourfulness_pred'] for r in rows]) / np.mean([r['colourfulness_gt'] for r in rows])
bl_ratio = np.mean([r['bleed_pred'] for r in rows]) / np.mean([r['bleed_gt'] for r in rows])
print(f'  colourfulness ratio  {cf_ratio:.3f}   (target 0.75-1.25)')
print(f'  bleed ratio          {bl_ratio:.3f}   (target <= 1.20)')