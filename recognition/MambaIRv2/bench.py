"""Time one training step and list the slowest GPU ops. Single GPU, no torchrun.

python bench.py                      # default settings
python bench.py --cudnn_benchmark    # let cuDNN pick faster conv kernels
python bench.py --fp32               # no bf16 autocast
python bench.py --profile            # also print the top-15 CUDA ops
"""
import time, argparse, torch, torch.nn.functional as F
from model import Colorizer

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='experiments/mambairv2_ColorDN_15.pth')
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--crop', type=int, default=128)
parser.add_argument('--steps', type=int, default=5)
parser.add_argument('--cudnn_benchmark', action='store_true')
parser.add_argument('--fp32', action='store_true')
parser.add_argument('--profile', action='store_true')
args = parser.parse_args()

torch.backends.cudnn.benchmark = args.cudnn_benchmark
model = Colorizer(args.ckpt, args.crop).cuda().train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
L = torch.rand(args.batch, 1, args.crop, args.crop, device='cuda')
ab = torch.rand(args.batch, 2, args.crop, args.crop, device='cuda') * 2 - 1


def step():
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not args.fp32):
        loss = F.l1_loss(model(L), ab)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


for _ in range(2):                                # warm-up (cuDNN autotune, allocator)
    step()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

t = time.time()
for _ in range(args.steps):
    step()
torch.cuda.synchronize()
dt = (time.time() - t) / args.steps
print(f'{dt:.2f} s/step   {args.batch / dt:.1f} img/s   peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')

if args.profile:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by='self_device_time_total', row_limit=15))
