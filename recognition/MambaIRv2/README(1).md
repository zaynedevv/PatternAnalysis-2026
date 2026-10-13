# Feasibility Review: MambaIRv2 Image Colourisation

Recommendation: proceed with a bounded MambaIRv2 fine-tune (about 9 GPU-hours), gated on a one-hour smoke test. Plausibility is judged by distribution, boundary and human tests, not pixel error alone.

## 1. User need and acceptance criteria

For archivists and analysts who need greyscale images colourised for human viewing, not measurement. Measured on the supplied unseen test set against a colour-mapping baseline:

1. LPIPS at least 15% lower than the baseline.
2. Colourfulness within ±25% of the ground-truth mean (no desaturation collapse).
3. Luminance SSIM against the input at least 0.98; PSNR and RGB SSIM reported, not gated.
4. Chroma bleed across strong luminance edges no worse than ground truth +20%.
5. Peak VRAM at most 15 GB per GPU (128×128 crops, batch 8); at most 8 GPU-hours of training.

## 2. Model and course concepts

**Dataset.** Test: Kodak24 (24 PNGs, 256×256 centre crops). Train: DIV2K + Flickr2K in CIELAB, random 128×128 crops and flips, 10% held out for validation; the test set is used once, at the end.

**Architecture and band.** MambaIRv2 (Hard), fine-tuned from the official mambairv2\_ColorDN\_15.pth denoising checkpoint, built with upscale = 1 and no upsampler (transfer learning). It takes L and predicts only ab, so structure is preserved by construction and only colour is learned.

**Why it suits the dilemma.** Its linear-time global receptive field lets distant context choose consistent hues (Gu and Dao, 2023; ref \[18\]), while semantic-guided neighbouring should limit bleed across edges, which criterion 4 tests.

**Loss and baselines.** Colourisation is ill-posed, so L1 alone predicts the conditional mean and gives brown output (Zhang et al., 2016). We add an LPIPS perceptual loss (Zhang et al., 2018), sweeping its weight λ over {0, 0.1, 0.5}, and compare against the frozen model and a per-brightness mean-colour mapping.

## 3. Feasibility evidence

Data audit: the test set is Kodak24, 24 lossless 768×512 RGB PNGs, a standard colour-restoration benchmark that the pretrained checkpoint was tested on but not trained on. Each image is centre-cropped to 256×256, and inputs are generated with a fixed seed per image and saved to disk, so every model is scored on identical inputs. The course notebook already shows the environment builds on current Colab (Python 3.13, torch 2.11, latest causal\_conv1d and mamba\_ssm) and a MambaIRv2-Base checkpoint loads with all keys matched. Remaining results are to be filled from the Colab run. Smoke test: `torchrun --standalone --nproc_per_node=1 train.py --crop 128 --batch 8 --iters 500 --seed 0`.

| Check | Pass condition | Result |
| --- | --- | --- |
| Kernels build, checkpoint loads | forward pass runs | Passed for SR checkpoint (course notebook); ColorDN to confirm |
| Peak VRAM | at most 15 GB |  |
| Throughput | at least 20 img/s |  |
| Validation L1(ab) after 500 iterations | down at least 10% |  |
| Baselines on Kodak24 | all metrics recorded |  |

Plausibility will be shown to the PM through colourfulness and ab-histogram match, the bleed score, a blind real-versus-colourised test (5–10 raters, 50 pairs), and an autopsy of 3–5 failure cases.

## 4. Risks, budget and fallback

Main risks: mamba\_ssm kernels failing to build (lowered: they built in the course notebook; fallback causal\_conv1d 1.5.2, mamba\_ssm 2.2.5); desaturation (LPIPS term); out-of-memory, as the authors trained at batch 2 per GPU (bf16, gradient checkpointing, smaller batch); hallucinated skin or organ colour (human review, "synthetic colour" label, never for diagnosis or evidence).

**Compute budget.** 1 GPU-hour smoke test, 6 GPU-hours for the three-run λ sweep, 2 GPU-hours for final training and evaluation: about 9 GPU-hours on one 16–40 GB GPU. Stop and escalate if the smoke test exceeds 15 GB or falls below 10 img/s.

**Next and fallback.** Run the smoke test, then λ = 0 versus 0.1 at 5,000 iterations. If MambaIRv2 fails to build or fit, fall back to MambaIR v1, then a U-Net, with the same protocol and criteria.

## Reproduce (Colab, GPU runtime)

```bash
git clone https://github.com/csguoh/MambaIR.git && cd MambaIR
pip install causal_conv1d mamba_ssm lpips scikit-image   # fallback: causal_conv1d==1.5.2 mamba_ssm==2.2.5
wget https://github.com/csguoh/MambaIR/releases/download/v1.0/mambairv2_ColorDN_15.pth -P experiments
```

```python
import torch
from analysis.model_zoo.mambairv2 import MambaIRv2
model = MambaIRv2(img_size=128, in_chans=3, embed_dim=174, d_state=16, depths=(6,)*6,
    num_heads=[6]*6, window_size=16, inner_rank=64, num_tokens=128, convffn_kernel_size=5,
    mlp_ratio=2., upscale=1, img_range=1., upsampler='', resi_connection='1conv').cuda()
model.load_state_dict(torch.load('experiments/mambairv2_ColorDN_15.pth')['params'])
```
