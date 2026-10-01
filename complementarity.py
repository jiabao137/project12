"""
Complementarity term — report Section IV, Eqs. (14)-(15).

G (FDB's frequency-aware gate, Eq. 2) and M (FRB's spatial confidence map,
Section III-C Stage 6) are NOT directly comparable: G is a per-channel vector
(B, C, 1, 1) from global average pooling, M is a per-pixel map (B, 1, H, W).
Taking max(G, M) directly would broadcast into something without a clear
physical meaning.

SpatialActivityHead instead derives a spatial activity map G_s from the FDB
residual R_b — on the same spatial basis as M — by averaging |R_b| across
channels and passing the result through a single learnable 1x1 conv + sigmoid.
This is the only new parameter Section IV's fix introduces; everything else
reuses quantities already computed by the FDB and FRB branches.
"""

import torch
import torch.nn as nn


class SpatialActivityHead(nn.Module):
    """Eq. (14): G_s = sigmoid(Conv_1x1(mean_c(|R_b|)))."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, r_b):
        activity = r_b.abs().mean(dim=1, keepdim=True)   # (B, 1, H, W)
        return torch.sigmoid(self.conv(activity))           # G_s, same shape as M


def l_comp(g_s, m):
    """
    Eq. (15): L_comp = 1 - mean(max(G_s, M)).
    Rewards FDB and FRB for being active (high gate value) in different
    spatial regions rather than the same ones.
    """
    return 1 - torch.max(g_s, m).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    head = SpatialActivityHead()
    r_b = torch.randn(2, 160, 4, 13)
    m = torch.sigmoid(torch.randn(2, 1, 4, 13))

    g_s = head(r_b)
    print(f"G_s shape: {tuple(g_s.shape)} (should match M's shape: {tuple(m.shape)})")
    print(f"L_comp: {l_comp(g_s, m).item():.4f}")

    # sanity check: if G_s and M are identical everywhere, L_comp should equal
    # 1 - mean(G_s) exactly (max of two identical tensors is just the tensor)
    same = l_comp(g_s, g_s).item()
    expected = (1 - g_s.mean()).item()
    print(f"Identity check (should match): {same:.6f} vs {expected:.6f}")
