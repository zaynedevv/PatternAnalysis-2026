"""Fine-tune MambaIRv2 for colourisation (L -> ab). Train, validate, test, save.

torchrun --standalone --nproc_per_node=1 train.py --iters 1500 --lam 0.1 --out runs/lam0.1

Every 500 steps (and at the end): validation L1/LPIPS, checkpoint runs/<name>/ckpt_<step>.pt,
and runs/<name>/curves.png (training + validation losses). After the last step the model is
scored once on the test set (--test_data). Full test metrics and images: predict.py.

Loss = L1(ab) + lam * LPIPS(RGB). L1 alone predicts the average colour (brown); LPIPS pushes
toward perceptually realistic colour.
"""

import os
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import lpips
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import LabDataset, split_train_val, list_images
from modules import Colorizer, lab_to_rgb


VALIDATE_EVERY = 500  # steps
LOG_EVERY = 50  # steps
WARMUP_STEPS = 20  # throughput and memory are measured after this


def plot_curves(history, path):
    """history: {'train': [(step, l1, lpips)], 'val': [...]} -> two-panel PNG."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    panels = [(axes[0], 1, 'L1 (ab)'), (axes[1], 2, 'LPIPS')]

    for axis, column, title in panels:
        for split, line_style in [('train', '-'), ('val', 'o-')]:
            if not history[split]:
                continue
            steps = [entry[0] for entry in history[split]]
            values = [entry[column] for entry in history[split]]
            axis.plot(steps, values, line_style, label=split)

        axis.set_xlabel('step')
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


if __name__ == '__main__':

    # ---------- args ----------
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/DIV2K_train_HR')  # split into train / val
    parser.add_argument('--test_data', default='data/kodak24')
    parser.add_argument('--ckpt', default='experiments/mambairv2_ColorDN_15.pth')  # pretrained
    parser.add_argument('--out', default='runs/run0')
    parser.add_argument('--crop', type=int, default=128)
    parser.add_argument('--test_crop', type=int, default=256)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lam', type=float, default=0.0)  # LPIPS weight in the loss
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    # ---------- distributed setup (torchrun sets LOCAL_RANK) ----------
    dist.init_process_group('nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    is_main_process = dist.get_rank() == 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------- data ----------
    train_files, val_files = split_train_val(args.data)

    train_dataset = LabDataset(train_files, args.crop, random_crop=True)
    val_dataset = LabDataset(val_files, args.crop, random_crop=False)

    # Each GPU gets a different shard of the training set
    train_sampler = DistributedSampler(train_dataset, seed=args.seed)

    train_loader = DataLoader(
        train_dataset,
        args.batch,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, args.batch, num_workers=2)

    # ---------- model ----------
    model = Colorizer(args.ckpt, args.crop).to(device)
    model = DDP(model, device_ids=[local_rank])

    # LPIPS is always built for monitoring; it only enters the loss when lam > 0
    lpips_metric = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    for parameter in lpips_metric.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iters)

    # ---------- loss ----------
    def compute_loss(lightness, ab_target, rgb_target):
        """Returns (total loss tensor, L1 value, LPIPS value)."""
        ab_predicted = model(lightness)
        l1_loss = F.l1_loss(ab_predicted, ab_target)

        lpips_loss = torch.tensor(0.0, device=device)
        if args.lam > 0:
            rgb_predicted = lab_to_rgb(lightness.float(), ab_predicted.float())
            # LPIPS expects images in [-1, 1]
            lpips_loss = lpips_metric(rgb_predicted * 2 - 1, rgb_target * 2 - 1).mean()

        total_loss = l1_loss + args.lam * lpips_loss
        return total_loss, l1_loss.item(), lpips_loss.item()

    # ---------- validation / test ----------
    @torch.no_grad()
    def evaluate(loader):
        """Mean L1(ab) and LPIPS(RGB) over a loader. Used for both validation and test."""
        # Unwrapped model: no DDP sync, so rank 0 can run the test set alone
        network = model.module
        network.eval()

        l1_values, lpips_values = [], []
        for lightness, ab_target, rgb_target in loader:
            lightness = lightness.to(device)
            ab_target = ab_target.to(device)
            rgb_target = rgb_target.to(device)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                ab_predicted = network(lightness)
            ab_predicted = ab_predicted.float()

            rgb_predicted = lab_to_rgb(lightness, ab_predicted)
            lpips_value = lpips_metric(rgb_predicted * 2 - 1, rgb_target * 2 - 1).mean()

            l1_values.append(F.l1_loss(ab_predicted, ab_target).item())
            lpips_values.append(lpips_value.item())

        network.train()
        return float(np.mean(l1_values)), float(np.mean(lpips_values))

    history = {'train': [], 'val': []}

    def validate(step):
        l1_value, lpips_value = evaluate(val_loader)
        history['val'].append((step, l1_value, lpips_value))

        if is_main_process:
            print(f'VAL step {step}  l1 {l1_value:.4f}  lpips {lpips_value:.4f}', flush=True)
            plot_curves(history, f'{args.out}/curves.png')

    # ---------- training loop ----------
    os.makedirs(args.out, exist_ok=True)

    # Step-0 baseline: the zero head outputs the grey input
    validate(0)

    step = 0
    timer_start = time.time()
    images_since_timer = 0
    model.train()

    while step < args.iters:
        # Reshuffle on each pass over the data
        train_sampler.set_epoch(step)

        for lightness, ab_target, rgb_target in train_loader:
            lightness = lightness.to(device)
            ab_target = ab_target.to(device)
            rgb_target = rgb_target.to(device)

            # Forward in bf16, then a normal backward / optimiser step
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, l1_value, lpips_value = compute_loss(lightness, ab_target, rgb_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            images_since_timer += lightness.shape[0]

            # Restart the timer and memory counter after warm-up, so the numbers are honest
            if step == WARMUP_STEPS:
                torch.cuda.reset_peak_memory_stats()
                timer_start = time.time()
                images_since_timer = 0

            if is_main_process and step % LOG_EVERY == 0:
                images_per_second = images_since_timer / (time.time() - timer_start)
                peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
                history['train'].append((step, l1_value, lpips_value))

                print(
                    f'step {step}  l1 {l1_value:.4f}  lpips {lpips_value:.4f}  '
                    f'{images_per_second:.1f} img/s  peak {peak_memory_gb:.2f} GB',
                    flush=True,
                )

            if step % VALIDATE_EVERY == 0 or step == args.iters:
                validate(step)

                if is_main_process:
                    torch.save(model.module.state_dict(), f'{args.out}/ckpt_{step}.pt')

            if step >= args.iters:
                break

    # ---------- test (once, after training) ----------
    if is_main_process and os.path.isdir(args.test_data):
        test_files = list_images(args.test_data)
        test_dataset = LabDataset(test_files, args.test_crop, random_crop=False)

        l1_value, lpips_value = evaluate(DataLoader(test_dataset, batch_size=1))

        print(
            f'TEST ({len(test_dataset)} images, {args.test_crop}² centre crops)  '
            f'l1 {l1_value:.4f}  lpips {lpips_value:.4f}',
            flush=True,
        )

    dist.destroy_process_group()
