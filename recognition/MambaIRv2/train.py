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
