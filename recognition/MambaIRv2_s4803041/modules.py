"""Model components (PyTorch only, no NumPy).

Colorizer     pretrained MambaIRv2 ColorDN body + new zero-init ab head
MeanMap       baseline: mean ab per lightness bin, fitted on training images
lab_to_rgb    differentiable Lab -> sRGB, so LPIPS can sit inside the loss

Shapes use (batch, channels, height, width). In CIELAB, "L" is lightness and "a", "b" are the
two colour channels (a: green-red, b: blue-yellow). The model sees L and predicts ab.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# Normalisation used everywhere (see dataset.normalize): L/100 -> [0,1], ab/128 -> [-1,1]
L_SCALE = 100
AB_SCALE = 128

# CIE D65 reference white, and the matrix from linear XYZ to linear sRGB
D65_WHITE = (0.95047, 1.00000, 1.08883)
XYZ_TO_RGB = (
    (3.2406, -1.5372, -0.4986),
    (-0.9689, 1.8758, 0.0415),
    (0.0557, -0.2040, 1.0570),
)


def lab_to_rgb(lightness, ab):
    """Convert normalised Lab to sRGB, differentiably.

    lightness  (B,1,H,W) in [0,1]
    ab         (B,2,H,W) in [-1,1]
    returns    (B,3,H,W) sRGB in [0,1]
    """
    L = lightness * L_SCALE
    a = ab[:, 0:1] * AB_SCALE
    b = ab[:, 1:2] * AB_SCALE

    # Lab -> XYZ (CIE 1976 formulas)
    f_y = (L + 16) / 116
    f_x = f_y + a / 500
    f_z = f_y - b / 200

    def inverse_lab_curve(t):
        """Undo Lab's cube-root compression (linear segment near black)."""
        return torch.where(t > 6 / 29, t**3, 3 * (6 / 29) ** 2 * (t - 4 / 29))

    X = D65_WHITE[0] * inverse_lab_curve(f_x)
    Y = D65_WHITE[1] * inverse_lab_curve(f_y)
    Z = D65_WHITE[2] * inverse_lab_curve(f_z)

    # XYZ -> linear RGB
    red, green, blue = (row[0] * X + row[1] * Y + row[2] * Z for row in XYZ_TO_RGB)
    linear_rgb = torch.cat([red, green, blue], dim=1).clamp(0, 1)

    # Linear RGB -> sRGB gamma curve.
    # The clamp keeps the gradient of x^(1/2.4) finite at 0.
    srgb = torch.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * linear_rgb.clamp(min=1e-6) ** (1 / 2.4) - 0.055,
    )
    return srgb.clamp(0, 1)


class Colorizer(nn.Module):
    """Pretrained MambaIRv2 ColorDN body + new zero-init ab head.

    pretrained_path  the official mambairv2_ColorDN_15.pth, or None to skip loading it
                     (predict.py then loads a whole fine-tuned state dict instead)
    crop_size        training crop size; must be a multiple of the 16-pixel attention window

    The attribute names `body` and `head` are the keys in saved checkpoints; renaming them
    breaks loading of existing weights.
    """

    FEATURE_CHANNELS = 174

    def __init__(self, pretrained_path, crop_size):
        super().__init__()

        # From the csguoh/MambaIR repository. Imported here because it needs mamba_ssm (GPU),
        # and MeanMap / lab_to_rgb should still work without it.
        from analysis.model_zoo.mambairv2 import MambaIRv2

        self.body = MambaIRv2(
            img_size=crop_size,
            in_chans=3,
            embed_dim=self.FEATURE_CHANNELS,
            d_state=16,
            depths=(6,) * 6,
            num_heads=[6] * 6,
            window_size=16,
            inner_rank=64,
            num_tokens=128,
            convffn_kernel_size=5,
            mlp_ratio=2.0,
            upscale=1,
            img_range=1.0,
            upsampler='',
            resi_connection='1conv',
        )

        if pretrained_path is not None:
            pretrained_weights = torch.load(pretrained_path, map_location='cpu')['params']
            print(self.body.load_state_dict(pretrained_weights, strict=True))

        # The original RGB output layer is bypassed by the new ab head.
        # Freeze it, otherwise DDP errors on parameters that never get a gradient.
        self.body.conv_last.requires_grad_(False)

        # New head: 174 features -> 2 channels (a, b).
        # Zero init means step 0 outputs ab = 0, i.e. exactly the greyscale input.
        self.head = nn.Conv2d(self.FEATURE_CHANNELS, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, lightness):
        """lightness (B,1,H,W) in [0,1] -> predicted ab (B,2,H,W), roughly in [-1,1]."""
        height, width = lightness.shape[-2:]
        window = self.body.window_size
        assert height % window == 0 and width % window == 0, 'crop must be a multiple of 16'

        # Extra inputs the attention blocks need, built the same way as MambaIRv2.forward
        attention_inputs = {
            'attn_mask': self.body.calculate_mask([height, width]).to(lightness.device),
            'rpi_sa': self.body.relative_position_index_SA,
        }

        # Grey -> 3 identical channels, so the pretrained first conv can be reused.
        # Unlike MambaIRv2.forward, no RGB mean is subtracted and the input is not added back
        # at the end: the output is ab, not a cleaned-up copy of the input.
        three_channel_input = lightness.repeat(1, 3, 1, 1)
        shallow_features = self.body.conv_first(three_channel_input)

        deep_features = self.forward_features(shallow_features, attention_inputs)
        deep_features = self.body.conv_after_body(deep_features)

        # Long skip connection, then predict ab
        return self.head(deep_features + shallow_features)

    def forward_features(self, features, attention_inputs):
        """MambaIRv2.forward_features, with gradient checkpointing per block.

        MambaIRv2 accepts a use_checkpoint flag but never uses it. Each block keeps ~3.5 GB of
        attention/scan activations at batch 8 @ 128², so even one 6-block group doesn't fit in
        22 GB without this. Checkpointing recomputes each block's forward pass during backward
        instead of storing it.
        """
        body = self.body
        use_checkpointing = self.training and torch.is_grad_enabled()
        image_size = (features.shape[2], features.shape[3])

        # (B,C,H,W) -> (B,H*W,C) token sequence
        tokens = body.patch_embed(features)

        # 6 residual groups (ASSB.forward, unrolled), each of 6 attention + selective-scan blocks
        for group in body.layers:
            group_input = tokens

            for block in group.residual_group.layers:
                if use_checkpointing:
                    tokens = checkpoint(
                        block, tokens, image_size, attention_inputs, use_reentrant=False
                    )
                else:
                    tokens = block(tokens, image_size, attention_inputs)

            # Conv at the end of each group, plus the group's residual connection
            group_output = group.conv(group.patch_unembed(tokens, image_size))
            tokens = group.patch_embed(group_output) + group_input

        tokens = body.norm(tokens)

        # Back to (B,C,H,W)
        return body.patch_unembed(tokens, image_size)


class MeanMap(nn.Module):
    """Colour-mapping baseline.

    Split lightness into `num_bins` equal bins and give every pixel the average ab of its bin
    over the fitting images. A lookup table, no learning.
    """

    def __init__(self, num_bins=64):
        super().__init__()
        self.num_bins = num_bins

        # Mean (a, b) per bin
        self.register_buffer('table', torch.zeros(num_bins, 2))

    def _bin_index(self, lightness):
        """(B,1,H,W) in [0,1] -> integer bin per pixel, (B,H,W)."""
        bins = (lightness[:, 0] * self.num_bins).long()
        return bins.clamp(0, self.num_bins - 1)

    @torch.no_grad()
    def fit(self, lightness_ab_pairs):
        """lightness_ab_pairs: iterable of (lightness (B,1,H,W), ab (B,2,H,W)) tensors."""
        ab_sums = torch.zeros(self.num_bins, 2, dtype=torch.float64)
        pixel_counts = torch.zeros(self.num_bins, dtype=torch.float64)

        for lightness, ab in lightness_ab_pairs:
            bin_of_pixel = self._bin_index(lightness).flatten()
            ab_per_pixel = ab.permute(0, 2, 3, 1).reshape(-1, 2).double()
            ones = torch.ones_like(bin_of_pixel, dtype=torch.float64)

            ab_sums.index_add_(0, bin_of_pixel, ab_per_pixel)
            pixel_counts.index_add_(0, bin_of_pixel, ones)

        mean_ab = ab_sums / pixel_counts.clamp(min=1)[:, None]
        self.table.copy_(mean_ab.float())
        return self

    def forward(self, lightness):
        """lightness (B,1,H,W) -> ab (B,2,H,W) looked up from the table."""
        ab = self.table[self._bin_index(lightness)]
        return ab.permute(0, 3, 1, 2)
