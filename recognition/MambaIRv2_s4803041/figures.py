"""Build the README figures from predict.py / train.py outputs. CPU only.

python figures.py --results outputs/results --val outputs/val --runs outputs/runs --out figures

comparison.png  grey | meanmap | fine-tuned | fine-tuned x2 | ground truth, on chosen Kodak images
failures.png    same columns, on the autopsy cases
curves.png      validation L1 per lambda, and the LPIPS / colourfulness trade-off over ab_scale
"""

import os
import re
import csv
import argparse

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


# ---------- args ----------
parser = argparse.ArgumentParser()
parser.add_argument('--results', default='outputs/results')
parser.add_argument('--val', default='outputs/val')
parser.add_argument('--runs', default='outputs/runs')
parser.add_argument('--out', default='figures')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

LAMS = ['0', '0.1', '0.5', '1.0']
SCALES = ['1.0', '1.5', '2.0', '2.5']

# (column title, results folder, image tag) for the image grids
COLUMNS = [
    ('Greyscale input', 'frozen', 'gray'),
    ('Baseline (meanmap)', 'meanmap', 'pred'),
    ('Fine-tuned λ=0.1', 'lam0.1', 'pred'),
    ('Fine-tuned λ=0.1, ab×2', 'final_lam0.1_s2.0', 'pred'),
    ('Ground truth', 'frozen', 'gt'),
]


# ---------- helpers ----------
def read_metrics(folder):
    with open(f'{folder}/metrics.csv') as csv_file:
        return list(csv.DictReader(csv_file))


def mean_metrics(folder):
    """Average of every metric column in folder/metrics.csv."""
    rows = read_metrics(folder)
    metrics = [name for name in rows[0] if name != 'image']
    return {name: np.mean([float(row[name]) for row in rows]) for name in metrics}


def lpips_per_image(folder):
    return {row['image']: float(row['lpips']) for row in read_metrics(folder)}


# ---------- image grids ----------
def grid(image_names, path):
    """One row per Kodak image, one column per method; LPIPS under each prediction."""
    lpips_by_method = {
        folder: lpips_per_image(f'{args.results}/{folder}')
        for _, folder, tag in COLUMNS
        if tag == 'pred'
    }

    num_rows, num_columns = len(image_names), len(COLUMNS)
    figure, axes = plt.subplots(num_rows, num_columns, figsize=(2.4 * num_columns, 2.5 * num_rows))

    for row_index, name in enumerate(image_names):
        for column_index, (title, folder, tag) in enumerate(COLUMNS):
            axis = axes[row_index, column_index]
            axis.imshow(Image.open(f'{args.results}/{folder}/{name}_{tag}.png'))
            axis.set_xticks([])
            axis.set_yticks([])

            if row_index == 0:
                axis.set_title(title, fontsize=9)

            if column_index == 0:
                axis.set_ylabel(name, fontsize=9)

            if tag == 'pred':
                axis.set_xlabel(f'LPIPS {lpips_by_method[folder][name]:.3f}', fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


grid(['kodim21', 'kodim07', 'kodim16'], f'{args.out}/comparison.png')
grid(['kodim03', 'kodim23', 'kodim17', 'kodim13'], f'{args.out}/failures.png')


# ---------- curves ----------
figure, (loss_axis, tradeoff_axis) = plt.subplots(1, 2, figsize=(11, 4))

# Left: validation L1 over training, one line per lambda
for lam in LAMS:
    with open(f'{args.runs}/lam{lam}.log') as log_file:
        val_lines = re.findall(r'VAL step (\d+)\s+l1 ([\d.]+)', log_file.read())

    steps = [int(step) for step, _ in val_lines]
    val_l1 = [float(l1) for _, l1 in val_lines]
    loss_axis.plot(steps, val_l1, marker='o', label=f'λ={lam}')

loss_axis.set_xlabel('training step (batch 8)')
loss_axis.set_ylabel('validation L1 (ab)')
loss_axis.set_title('DIV2K validation (80 images)')
loss_axis.grid(alpha=0.3)
loss_axis.legend()

# Right: LPIPS against colourfulness as ab_scale grows, one line per lambda
for lam in LAMS:
    per_scale = [mean_metrics(f'{args.val}/lam{lam}_s{scale}') for scale in SCALES]
    colourfulness_ratio = [m['colourfulness_pred'] / m['colourfulness_gt'] for m in per_scale]
    lpips_values = [m['lpips'] for m in per_scale]
    tradeoff_axis.plot(colourfulness_ratio, lpips_values, marker='o', label=f'λ={lam} (val)')

# Kodak24 reference points
kodak_points = [
    ('final_lam0.1_s2.0', 'chosen model, Kodak24', '*'),
    ('meanmap', 'meanmap, Kodak24', 's'),
    ('meanmap_s2.0', 'meanmap ab×2, Kodak24', 'D'),
]
for folder, label, marker in kodak_points:
    means = mean_metrics(f'{args.results}/{folder}')
    tradeoff_axis.scatter(
        means['colourfulness_pred'] / means['colourfulness_gt'],
        means['lpips'],
        marker=marker,
        s=90,
        color='k',
        label=label,
        zorder=3,
    )

tradeoff_axis.axvspan(0.75, 1.25, color='green', alpha=0.08, label='colourfulness target')
tradeoff_axis.set_xlabel('colourfulness / ground truth (points: ab_scale 1, 1.5, 2, 2.5)')
tradeoff_axis.set_ylabel('LPIPS (lower is better)')
tradeoff_axis.set_title('Saturation vs perceptual error')
tradeoff_axis.grid(alpha=0.3)
tradeoff_axis.legend(fontsize=7)

figure.tight_layout()
figure.savefig(f'{args.out}/curves.png', dpi=110)

print(f'wrote {args.out}/comparison.png, failures.png, curves.png')
