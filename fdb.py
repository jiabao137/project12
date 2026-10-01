"""
Feature Deblurring Branch (FDB) — report Section III-B, Eqs. (1)-(5).

Five stages: projection -> multi-scale stroke extraction -> local spatial
attention -> residual prediction -> frequency-aware gating -> F_b.

forward() returns a dict (not just F_b) because the auxiliary losses in
losses.py need R_b (for L_res) and F_b/F (for L_cons) as well.
"""

import torch
import torch.nn as nn

import config


class MultiScaleStrokeExtraction(nn.Module):
    """
    Stage 2: parallel 3x3 / 5x5 / dilated-3x3 depthwise convs, summed then
    fused by a single 1x1 conv. (Summing before the 1x1 fuse — rather than
    concatenating — keeps the fuse step at C*C params instead of 3C*C,
    matching the ~25k "channel fusion" budget assumed in the report; the
    three depthwise convs still see F_p independently before being combined.)
    """

    def __init__(self, channels):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                              groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2,
                              groups=channels, bias=False)
        self.dw3_dilated = nn.Conv2d(channels, channels, kernel_size=3, padding=2,
                                      dilation=2, groups=channels, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, f_p):
        f_sum = self.dw3(f_p) + self.dw5(f_p) + self.dw3_dilated(f_p)
        return self.fuse(f_sum)  # F_m


class LocalSpatialAttention(nn.Module):
    """
    Stage 3: single-channel spatial attention map, broadcast across channels.
    This is the CBAM-style spatial branch cited in the report (Section III-B) —
    channel-wise scaling is deliberately left to Stage 5's gate instead, so
    this module does not also do channel attention.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, f_m):
        attn = torch.sigmoid(self.conv(f_m))   # (B, 1, H, W)
        return attn * f_m, attn                 # F_a, A


class FrequencyAwareGate(nn.Module):
    """Stage 5 gate: SE-style channel gate computed from F + R_b (Eq. 2)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, f_plus_rb):
        b, c, _, _ = f_plus_rb.shape
        g = self.gap(f_plus_rb).view(b, c)
        g = self.mlp(g).view(b, c, 1, 1)
        return g  # (B, C, 1, 1), broadcasts over H, W


class FeatureDeblurringBranch(nn.Module):
    """
    Full FDB:
        F_p = phi(W_p * F)                           Stage 1, Eq. -
        F_m = MultiScale(F_p)                          Stage 2
        F_a = A (dot) F_m,  A = sigmoid(Conv3x3(F_m))  Stage 3
        R_b = psi(F_a)                                  Stage 4
        G   = sigmoid(MLP(GAP(F + R_b)))               Stage 5 gate, Eq. (2)
        F_b = F + G (dot) R_b                           Stage 5 output, Eq. (3)
    """

    def __init__(self, channels=None):
        super().__init__()
        channels = channels or config.BACKBONE_OUT_CHANNELS

        # Stage 1: feature projection
        self.projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Stage 2
        self.multi_scale = MultiScaleStrokeExtraction(channels)
        # Stage 3
        self.local_attention = LocalSpatialAttention(channels)
        # Stage 4: residual correction head (depthwise 3x3 + pointwise 1x1,
        # i.e. one lightweight depthwise-separable block, consistent with the
        # rest of the backbone's design)
        self.residual_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        # Stage 5
        self.gate = FrequencyAwareGate(channels)

    def forward(self, f):
        f_p = self.projection(f)
        f_m = self.multi_scale(f_p)
        f_a, attn_map = self.local_attention(f_m)
        r_b = self.residual_head(f_a)

        g = self.gate(f + r_b)
        f_b = f + g * r_b

        return {
            "F_b": f_b,
            "R_b": r_b,
            "G": g,
            "attn_map": attn_map,
        }


if __name__ == "__main__":
    fdb = FeatureDeblurringBranch()
    f = torch.randn(2, config.BACKBONE_OUT_CHANNELS, 4, 13)  # matches Backbone's output shape
    out = fdb(f)
    for k, v in out.items():
        print(f"{k:10s}: {tuple(v.shape)}")
    n_params = sum(p.numel() for p in fdb.parameters())
    print(f"\nFDB parameters: {n_params:,}  (report design-phase estimate: ~70-80k)")
