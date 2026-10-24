import torch


def normalize(lab):
    """numpy Lab (H,W,3) -> L in [0,1], ab in [-1,1], both float32."""
    L = lab[:, :, 0:1] / 100.0
    ab = lab[:, :, 1:3] / 128.0
    return L.astype('float32'), ab.astype('float32')


def lab_to_rgb(L, ab):
    """torch L (B,1,H,W) in [0,1], ab (B,2,H,W) in [-1,1] -> sRGB (B,3,H,W) in [0,1].
    Differentiable, so it can sit inside the LPIPS loss."""
    L = L * 100
    a = ab[:, 0:1] * 128
    b = ab[:, 1:2] * 128

    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def f_inv(t):
        return torch.where(t > 6 / 29, t ** 3, 3 * (6 / 29) ** 2 * (t - 4 / 29))

    X = 0.95047 * f_inv(fx)
    Y = 1.00000 * f_inv(fy)
    Z = 1.08883 * f_inv(fz)

    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    rgb = torch.cat([r, g, b], dim=1).clamp(0, 1)

    # linear -> sRGB gamma
    rgb = torch.where(rgb <= 0.0031308,
                      12.92 * rgb,
                      1.055 * rgb.clamp(min=1e-6) ** (1 / 2.4) - 0.055)
    return rgb.clamp(0, 1)