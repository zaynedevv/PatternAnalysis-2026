import os, time, argparse, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np, lpips
from PIL import Image
from skimage.color import rgb2lab
from lab_utils import to_model, lab_to_rgb_torch
from analysis.model_zoo.mambairv2 import MambaIRv2

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
