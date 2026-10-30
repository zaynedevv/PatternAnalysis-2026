"""Time one training step and list the slowest GPU ops. Single GPU, no torchrun.

python bench.py                      # default settings
python bench.py --cudnn_benchmark    # let cuDNN pick faster conv kernels
python bench.py --fp32               # no bf16 autocast
python bench.py --profile            # also print the top-15 CUDA ops
"""

import time
import argparse

import torch
import torch.nn.functional as F

from modules import Colorizer


# ---------- args ----------
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='experiments/mambairv2_ColorDN_15.pth')
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--crop', type=int, default=128)
parser.add_argument('--steps', type=int, default=5)
parser.add_argument('--cudnn_benchmark', action='store_true')
parser.add_argument('--fp32', action='store_true')
parser.add_argument('--profile', action='store_true')
args = parser.parse_args()


# ---------- setup ----------
torch.backends.cudnn.benchmark = args.cudnn_benchmark

model = Colorizer(args.ckpt, args.crop).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Random inputs: timing doesn't depend on image content
lightness = torch.rand(args.batch, 1, args.crop, args.crop, device='cuda')
ab_target = torch.rand(args.batch, 2, args.crop, args.crop, device='cuda') * 2 - 1


def training_step():
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not args.fp32):
        loss = F.l1_loss(model(lightness), ab_target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# ---------- timing ----------
# Warm-up steps (cuDNN autotune, memory allocator) are not timed
for _ in range(2):
    training_step()

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

start = time.time()
for _ in range(args.steps):
    training_step()
torch.cuda.synchronize()

seconds_per_step = (time.time() - start) / args.steps
images_per_second = args.batch / seconds_per_step
peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

print(f'{seconds_per_step:.2f} s/step   {images_per_second:.1f} img/s   peak {peak_memory_gb:.2f} GB')


# ---------- profile ----------
if args.profile:
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        training_step()
        torch.cuda.synchronize()

    print(profiler.key_averages().table(sort_by='self_device_time_total', row_limit=15))
