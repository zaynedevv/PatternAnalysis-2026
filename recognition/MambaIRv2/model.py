import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
from analysis.model_zoo.mambairv2 import MambaIRv2


class Colorizer(nn.Module):
    """Pretrained MambaIRv2 ColorDN body + new zero-init ab head.
    ckpt_path=None skips the pretrained load (eval.py then loads a fine-tuned state dict)."""
    def __init__(self, ckpt_path, crop):
        super().__init__()
        self.body = MambaIRv2(
            img_size=crop, in_chans=3, embed_dim=174, d_state=16, depths=(6,)*6,
            num_heads=[6]*6, window_size=16, inner_rank=64, num_tokens=128,
            convffn_kernel_size=5, mlp_ratio=2., upscale=1, img_range=1.,
            upsampler='', resi_connection='1conv')
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location='cpu')['params']
            print(self.body.load_state_dict(state, strict=True))
        self.body.conv_last.requires_grad_(False)     # bypassed by the ab head; DDP errors on params with no grad

        self.head = nn.Conv2d(174, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, L):
        H, W = L.shape[-2:]
        assert H % self.body.window_size == 0 and W % self.body.window_size == 0, 'crop must be a multiple of window_size'
        params = {'attn_mask': self.body.calculate_mask([H, W]).to(L.device),   # as in MambaIRv2.forward
                  'rpi_sa': self.body.relative_position_index_SA}

        x = L.repeat(1, 3, 1, 1)                      # grey -> 3 channels for pretrained conv
        feat = self.body.conv_first(x)
        feat = self.body.conv_after_body(self.forward_features(feat, params)) + feat
        return self.head(feat)                        # predicted ab

    def forward_features(self, x, params):
        """MambaIRv2.forward_features with gradient checkpointing per block (its use_checkpoint
        flag is accepted but never used). Each block keeps ~3.5 GB of attention/scan activations
        at batch 8 @ 128², so even one 6-block group doesn't fit in 22 GB without this."""
        body = self.body
        ckpt = self.training and torch.is_grad_enabled()
        x_size = (x.shape[2], x.shape[3])
        x = body.patch_embed(x)
        for group in body.layers:                     # ASSB.forward, unrolled per block
            res = x
            for blk in group.residual_group.layers:
                x = checkpoint(blk, x, x_size, params, use_reentrant=False) if ckpt else blk(x, x_size, params)
            x = group.patch_embed(group.conv(group.patch_unembed(x, x_size))) + res
        x = body.norm(x)
        return body.patch_unembed(x, x_size)
