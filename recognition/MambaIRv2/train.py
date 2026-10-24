import os, time, argparse, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np, lpips
from PIL import Image
from skimage.color import rgb2lab
from lab_utils import to_model, lab_to_rgb_torch
from model import Colorizer

# ---------- args ----------
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/DIV2K_train_HR')
parser.add_argument('--ckpt', default='experiments/mambairv2_ColorDN_15.pth')
parser.add_argument('--out', default='runs/run0')
parser.add_argument('--crop', type=int, default=128)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--iters', type=int, default=500)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lam', type=float, default=0.0)   # LPIPS weight
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()


# ---------- distributed ----------
dist.init_process_group('nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)
is_main = dist.get_rank() == 0
torch.manual_seed(args.seed)
np.random.seed(args.seed)


#---------- data --------------
class LabDataset(Dataset):
    def __init__(self, files, crop, random_crop):
        self.files = files
        self.crop = crop
        self.random_crop = random_crop

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert('RGB')
        img = np.asarray(img, dtype=np.float32) / 255.0
        H, W, _ = img.shape
        c = self.crop

        if self.random_crop:
            y = np.random.randint(0, H - c + 1)
            x = np.random.randint(0, W - c + 1)
        else:
            y, x = (H - c) // 2, (W - c) // 2
        img = img[y:y+c, x:x+c]

        if self.random_crop and np.random.rand() < 0.5:
            img = img[:, ::-1]
        img = np.ascontiguousarray(img)

        L, ab = normalize(rgb2lab(img))

        to_tensor = lambda arr: torch.from_numpy(arr).permute(2, 0, 1)
        return to_tensor(L), to_tensor(ab), to_tensor(img)


files = sorted(os.path.join(args.data, f) for f in os.listdir(args.data))
train_files, val_files = files[:-80], files[-80:]

train_ds = LabDataset(train_files, args.crop, random_crop=True)
val_ds = LabDataset(val_files, args.crop, random_crop=False)

sampler = DistributedSampler(train_ds, seed=args.seed)
train_loader = DataLoader(train_ds, args.batch, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, args.batch, num_workers=2)


# ---------- model ----------
model = Colorizer(args.ckpt, args.crop).to(device)
model = DDP(model, device_ids=[local_rank])


perceptual = None
if args.lam > 0:
    perceptual = lpips.LPIPS(net='alex').to(device).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iters)


# ---------- loss ----------
def compute_loss(L, ab_true, rgb_true):
    ab_pred = model(L)
    loss_l1 = F.l1_loss(ab_pred, ab_true)
    loss_lp = torch.tensor(0.0, device=device)
    if perceptual is not None:
        rgb_pred = lab_to_rgb(L.float(), ab_pred.float())
        loss_lp = perceptual(rgb_pred * 2 - 1, rgb_true * 2 - 1).mean()
    return loss_l1 + args.lam * loss_lp, loss_l1.item(), loss_lp.item()


# ---------- validation ----------
@torch.no_grad()
def validate(step):
    model.eval()
    l1_vals, lp_vals = [], []
    for L, ab_true, rgb_true in val_loader:
        L, ab_true, rgb_true = L.to(device), ab_true.to(device), rgb_true.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ab_pred = model(L)
        l1_vals.append(F.l1_loss(ab_pred.float(), ab_true).item())
        if perceptual is not None:
            rgb_pred = lab_to_rgb(L, ab_pred.float())
            lp_vals.append(perceptual(rgb_pred * 2 - 1, rgb_true * 2 - 1).mean().item())
    if is_main:
        lp = np.mean(lp_vals) if lp_vals else 0.0
        print(f'VAL step {step}  l1 {np.mean(l1_vals):.4f}  lpips {lp:.4f}', flush=True)
    model.train()


# ---------- training loop ----------
os.makedirs(args.out, exist_ok=True)
step = 0
start_time = time.time()
images_seen = 0
model.train()

while step < args.iters:
    sampler.set_epoch(step)
    for L, ab_true, rgb_true in train_loader:
        L, ab_true, rgb_true = L.to(device), ab_true.to(device), rgb_true.to(device)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, l1, lp = compute_loss(L, ab_true, rgb_true)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        step += 1
        images_seen += L.shape[0]

        if step == 20:   # reset after warmup so numbers are honest
            torch.cuda.reset_peak_memory_stats()
            start_time, images_seen = time.time(), 0

        if is_main and step % 50 == 0:
            throughput = images_seen / (time.time() - start_time)
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f'step {step}  l1 {l1:.4f}  lpips {lp:.4f}  {throughput:.1f} img/s  peak {peak_gb:.2f} GB', flush=True)

        if step % 500 == 0 or step == args.iters:
            validate(step)
            if is_main:
                torch.save(model.module.state_dict(), f'{args.out}/ckpt_{step}.pt')

        if step >= args.iters:
            break

dist.destroy_process_group()