import torch, torch.nn as nn
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
        feat = self.body.conv_after_body(self.body.forward_features(feat, params)) + feat
        return self.head(feat)                        # predicted ab
