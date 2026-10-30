"""Run a trained colouriser (or a baseline) on the test set: metrics, images and a figure.

python predict.py --method finetuned --ckpt runs/lam0.1/ckpt_1500.pt --ab_scale 2.0 --out results/final
python predict.py --method frozen                                                  --out results/frozen
python predict.py --method meanmap --train_data data/DIV2K_train_HR                --out results/meanmap

Writes per-image PNGs (gray, pred, gt), metrics.csv, examples.png, and prints the means.
--test_data data/div2k_val scores the validation split instead (used to pick ab_scale).
"""

import os
import csv
import argparse

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import LabDataset, list_images
from modules import Colorizer, MeanMap, lab_to_rgb


MEANMAP_FIT_CROP = 512


# ---------- args ----------
parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=['finetuned', 'frozen', 'meanmap'], required=True)
parser.add_argument('--ckpt', default=None)  # fine-tuned weights
parser.add_argument('--pretrained', default='experiments/mambairv2_ColorDN_15.pth')
parser.add_argument('--test_data', default='data/kodak24')
parser.add_argument('--train_data', default='data/DIV2K_train_HR')  # meanmap only
parser.add_argument('--out', required=True)
parser.add_argument('--crop', type=int, default=256)
parser.add_argument('--ab_scale', type=float, default=1.0)  # saturation boost; pick on val, not test
parser.add_argument('--n_examples', type=int, default=4)  # rows in examples.png
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.out, exist_ok=True)


# ---------- the three methods ----------
if args.method in ('finetuned', 'frozen'):
    # frozen = pretrained body + untrained zero head, i.e. it outputs the greyscale input
    pretrained_path = args.pretrained if args.method == 'frozen' else None
    model = Colorizer(pretrained_path, args.crop)

    if args.method == 'finetuned':
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))

else:
    # meanmap: fitted on 512² centre crops of every DIV2K training image
    fit_files = list_images(args.train_data)
    fit_dataset = LabDataset(fit_files, MEANMAP_FIT_CROP, random_crop=False)
    model = MeanMap(num_bins=64).fit((lightness[None], ab[None]) for lightness, ab, _ in fit_dataset)

model = model.to(device).eval()


# ---------- metrics ----------
lpips_metric = lpips.LPIPS(net='alex', verbose=False).to(device).eval()


def colourfulness(rgb):
    """Hasler & Süsstrunk 2003 colourfulness index. rgb numpy (H,W,3) in [0,1]."""
    red, green, blue = [rgb[..., channel] * 255 for channel in range(3)]

    red_green = red - green
    yellow_blue = 0.5 * (red + green) - blue

    spread = np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2)
    average = np.sqrt(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
    return spread + 0.3 * average


def chroma_bleed(lightness, ab):
    """Mean colour gradient on the 5% strongest lightness edges.

    Compare predicted vs ground truth as a ratio: a higher ratio means colour changes more
    than it should at edges.
    """
    gradient_y, gradient_x = np.gradient(lightness[..., 0])
    edge_strength = np.hypot(gradient_x, gradient_y)
    strong_edges = edge_strength > np.percentile(edge_strength, 95)

    colour_gradient = sum(np.hypot(*np.gradient(ab[..., channel])) for channel in range(2))
    return float(colour_gradient[strong_edges].mean())


def to_numpy(tensor):
    """torch (1,C,H,W) or (C,H,W) -> numpy (H,W,C)."""
    return tensor.reshape(-1, *tensor.shape[-2:]).permute(1, 2, 0).cpu().numpy()


# ---------- inference loop ----------
test_dataset = LabDataset(list_images(args.test_data), args.crop, random_crop=False)
rows = []
examples = []

for index in range(len(test_dataset)):
    image_name = test_dataset.name(index)
    lightness, ab_true, rgb_true = (tensor[None].to(device) for tensor in test_dataset[index])

    # Predict ab, boost saturation if asked, and rebuild the RGB image
    with torch.no_grad():
        ab_predicted = (model(lightness).float() * args.ab_scale).clamp(-1, 1)
        rgb_predicted = lab_to_rgb(lightness, ab_predicted)

        # LPIPS expects images in [-1, 1]
        lpips_value = lpips_metric(rgb_predicted * 2 - 1, rgb_true * 2 - 1).item()

    lightness_np = to_numpy(lightness)
    true_np = to_numpy(rgb_true)
    predicted_np = to_numpy(rgb_predicted)

    row = {
        'image': image_name,
        'psnr': peak_signal_noise_ratio(true_np, predicted_np, data_range=1.0),
        'ssim_rgb': structural_similarity(true_np, predicted_np, channel_axis=2, data_range=1.0),
        # Identity by construction: the output reuses the input's lightness
        'ssim_L': structural_similarity(lightness_np[..., 0], lightness_np[..., 0], data_range=1.0),
        'lpips': lpips_value,
        'colourfulness_pred': colourfulness(predicted_np),
        'colourfulness_gt': colourfulness(true_np),
        'bleed_pred': chroma_bleed(lightness_np, to_numpy(ab_predicted)),
        'bleed_gt': chroma_bleed(lightness_np, to_numpy(ab_true)),
    }
    rows.append(row)

    # Save greyscale input, prediction and ground truth
    greyscale = np.repeat(lightness_np, 3, axis=2)
    for tag, image in [('gray', greyscale), ('pred', predicted_np), ('gt', true_np)]:
        Image.fromarray((image * 255).astype(np.uint8)).save(f'{args.out}/{image_name}_{tag}.png')

    if len(examples) < args.n_examples:
        examples.append((image_name, greyscale, predicted_np, true_np, lpips_value))


# ---------- write results ----------
with open(f'{args.out}/metrics.csv', 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


# examples.png: one row per image, columns greyscale | prediction | ground truth
figure, axes = plt.subplots(len(examples), 3, figsize=(7.5, 2.6 * len(examples)), squeeze=False)

for row_index, (image_name, greyscale, predicted, ground_truth, lpips_value) in enumerate(examples):
    panels = [
        (greyscale, 'greyscale input'),
        (predicted, f'{args.method} (LPIPS {lpips_value:.3f})'),
        (ground_truth, 'ground truth'),
    ]

    for column_index, (image, title) in enumerate(panels):
        axis = axes[row_index, column_index]
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])

        # Column titles on the first row; the prediction column shows LPIPS on every row
        if row_index == 0 or column_index == 1:
            axis.set_title(title, fontsize=9)

    axes[row_index, 0].set_ylabel(image_name, fontsize=9)

figure.tight_layout()
figure.savefig(f'{args.out}/examples.png', dpi=100)


# ---------- print summary ----------
def mean_of(metric):
    return np.mean([row[metric] for row in rows])


print(f'\n{args.method}  ab_scale {args.ab_scale}  ({len(rows)} images)')

for metric in rows[0]:
    if metric != 'image':
        print(f'  {metric:20s} {mean_of(metric):.4f}')

colourfulness_ratio = mean_of('colourfulness_pred') / mean_of('colourfulness_gt')
bleed_ratio = mean_of('bleed_pred') / mean_of('bleed_gt')

print(f'  colourfulness ratio  {colourfulness_ratio:.3f}   (target 0.75-1.25)')
print(f'  bleed ratio          {bleed_ratio:.3f}   (target <= 1.20)')
