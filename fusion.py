"""
Confidence-aware spatial fusion — report Section III-D, Eqs. (11)-(12).

Implemented as a softmax over per-branch confidence scores at every spatial
location: this guarantees the confidence maps sum to 1 by construction
(Eq. 12), rather than needing a separate normalisation step. Written to
accept any number of branches, so the same module works for Configuration B
(2 branches: Original + FDB) now, and Configuration D (3 branches, once FRB
exists) later, without changes.
"""

import torch
import torch.nn as nn


class ConfidenceAwareFusion(nn.Module):
    def __init__(self, channels, num_branches):
        super().__init__()
        self.num_branches = num_branches
        self.confidence_head = nn.Conv2d(
            channels * num_branches, num_branches, kernel_size=3, padding=1
        )

    def forward(self, branch_features):
        """
        branch_features: list of (B, C, H, W) tensors, length == num_branches.
        Returns (fused, weights) where weights is (B, num_branches, H, W) and
        sums to 1 across the branch dimension at every (h, w).
        """
        assert len(branch_features) == self.num_branches, (
            f"Expected {self.num_branches} branches, got {len(branch_features)}"
        )
        concat = torch.cat(branch_features, dim=1)          # (B, N*C, H, W)
        scores = self.confidence_head(concat)                 # (B, N, H, W)
        weights = torch.softmax(scores, dim=1)                 # Eq. (12): sums to 1

        fused = sum(
            weights[:, i:i + 1, :, :] * branch_features[i]
            for i in range(self.num_branches)
        )
        return fused, weights


if __name__ == "__main__":
    fusion = ConfidenceAwareFusion(channels=160, num_branches=2)
    f_o = torch.randn(2, 160, 4, 13)
    f_b = torch.randn(2, 160, 4, 13)
    fused, weights = fusion([f_o, f_b])
    print(f"fused shape:   {tuple(fused.shape)}")
    print(f"weights shape: {tuple(weights.shape)}")
    print(f"weights sum across branches (should be ~1.0): "
          f"{weights.sum(dim=1).mean().item():.4f}")


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion — report Section III-D, Eq. (13):

        F_fused = F_o + softmax( Q_o [K_b ; K_g]^T / sqrt(d) ) [V_b ; V_g]

    Unlike ConfidenceAwareFusion, there is no sum-to-one constraint across
    branches: the FDB and FRB key/value tokens are concatenated and share a
    SINGLE softmax, so the two enhancement branches compete for attention
    mass jointly rather than being combined as two independently normalised
    outputs. This is what makes it a genuine alternative to the competitive,
    externally-normalised confidence maps of Eqs. (11)-(12), rather than just
    a more expensive way of computing the same thing.

    Only supports exactly one anchor branch (F_o) attending over exactly two
    enhancement branches (F_b, F_g), matching the report's formulation and
    Configuration E — not a generic N-branch module like ConfidenceAwareFusion.
    """

    def __init__(self, channels, proj_dim=None, num_heads=4):
        super().__init__()
        proj_dim = proj_dim or channels
        assert proj_dim % num_heads == 0, "proj_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        self.proj_dim = proj_dim

        self.q_proj = nn.Conv2d(channels, proj_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, proj_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, proj_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(proj_dim, channels, kernel_size=1)

    @staticmethod
    def _to_tokens(x):
        # (B, C, H, W) -> (B, N, C), N = H*W
        return x.flatten(2).permute(0, 2, 1)

    def _split_heads(self, t, b):
        # (B, T, D) -> (B, heads, T, head_dim)
        return t.view(b, t.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, f_o, f_b, f_g):
        """
        f_o, f_b, f_g: (B, C, H, W), same shape. Returns (fused, attn) where
        attn is (B, heads, N, 2N) — kept for inspection/debugging, not used
        downstream.
        """
        b, c, h, w = f_o.shape

        q = self._to_tokens(self.q_proj(f_o))                              # (B, N, D)
        k = torch.cat([self._to_tokens(self.k_proj(f_b)),
                       self._to_tokens(self.k_proj(f_g))], dim=1)            # (B, 2N, D)
        v = torch.cat([self._to_tokens(self.v_proj(f_b)),
                       self._to_tokens(self.v_proj(f_g))], dim=1)            # (B, 2N, D)

        qh, kh, vh = self._split_heads(q, b), self._split_heads(k, b), self._split_heads(v, b)

        scores = torch.matmul(qh, kh.transpose(-2, -1)) / (self.head_dim ** 0.5)   # (B, heads, N, 2N)
        attn = torch.softmax(scores, dim=-1)                                          # single softmax, Eq. (13)
        out = torch.matmul(attn, vh)                                                    # (B, heads, N, head_dim)

        out = out.permute(0, 2, 1, 3).reshape(b, h * w, self.proj_dim)                  # (B, N, D)
        out = out.permute(0, 2, 1).view(b, self.proj_dim, h, w)                          # (B, D, H, W)
        out = self.out_proj(out)                                                          # (B, C, H, W)

        fused = f_o + out
        return fused, attn


if __name__ == "__main__":
    print("\n--- CrossAttentionFusion ---")
    cross_fusion = CrossAttentionFusion(channels=160, num_heads=4)
    f_o = torch.randn(2, 160, 4, 13)
    f_b = torch.randn(2, 160, 4, 13)
    f_g = torch.randn(2, 160, 4, 13)
    fused, attn = cross_fusion(f_o, f_b, f_g)
    print(f"fused shape: {tuple(fused.shape)}")
    print(f"attn shape:  {tuple(attn.shape)}  (B, heads, N, 2N)")
    print(f"attn sums to 1 over last dim (should be ~1.0): {attn.sum(dim=-1).mean().item():.4f}")
