"""
Feature Rectification Branch (FRB) — report Section III-C, Eqs. (6)-(7).

Six stages: compression -> geometry context encoder -> TPS control-point
regressor -> differentiable feature warp -> geometry refinement -> spatial
confidence gate -> F_g.

The TPS warp operates on the feature map directly (Stage 4: "Feature-space
TPS Warping"), not on the input image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class GeometryContextEncoder(nn.Module):
    """
    Stage 2: residual bottleneck with increasing dilation (3x3, dilated 3x3,
    dilated 5x5) for a wide receptive field — rectification needs global
    context (plate orientation/skew), unlike the FDB's local receptive field.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                groups=channels, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2,
                                dilation=2, groups=channels, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=5, padding=4,
                                dilation=2, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, f_c):
        out = self.conv1(f_c)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act(self.bn(out))
        return f_c + out  # residual bottleneck


def build_canonical_points(k):
    """
    Fixed canonical control points: K/2 evenly spaced along the top border
    (y=-1) and K/2 along the bottom border (y=+1) of normalised [-1, 1]
    space. Shared by TPSControlPointRegressor (for identity initialisation)
    and TPSGridGenerator (for the TPS solution itself) so the two always
    agree on what "canonical" means.
    """
    half = k // 2
    xs = torch.linspace(-1, 1, half)
    top = torch.stack([xs, -torch.ones(half)], dim=1)
    bottom = torch.stack([xs, torch.ones(half)], dim=1)
    return torch.cat([top, bottom], dim=0)  # (K, 2)


class TPSControlPointRegressor(nn.Module):
    """
    Stage 3: predicts K (x, y) control points in normalised coordinates.

    Identity-initialised (Jaderberg et al., 2015): the final layer's weight
    is zeroed and its bias is set to the canonical control-point positions,
    so that at the START of training the predicted points already sit at
    their canonical (no-op) locations, and the network only has to learn how
    far to move them for each input.

    Without this, the layer's default (near-zero-output) initialisation
    predicts points clustered near the coordinate origin (0, 0) — nowhere
    near the canonical points, which span y=-1 and y=+1 — so the network
    starts from an essentially arbitrary control-point configuration. In
    practice this produced numerically unstable, visually meaningless TPS
    warps that training did not recover from even after many epochs
    (confirmed visually with visualize_frb_warp.py). This initialisation
    exists specifically to prevent that failure mode.
    """

    def __init__(self, channels, num_control_points):
        super().__init__()
        self.num_control_points = num_control_points
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.hidden = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Linear(channels // 2, num_control_points * 2)

        # Identity initialisation: weight=0 means the output equals the bias
        # regardless of input, so at step 0 the network predicts exactly the
        # canonical points (a no-op warp) and learns deviations from there.
        nn.init.zeros_(self.out.weight)
        self.out.bias.data.copy_(build_canonical_points(num_control_points).view(-1))

    def forward(self, f):
        b, c, _, _ = f.shape
        x = self.gap(f).view(b, c)
        x = self.hidden(x)
        points = self.out(x).view(b, self.num_control_points, 2)
        # Clamp (not Tanh) to guard against runaway predictions during
        # training without disturbing the identity initialisation above —
        # Tanh would squash the canonical +/-1 boundary values and undo it.
        points = torch.clamp(points, -1.5, 1.5)
        return points


class TPSGridGenerator(nn.Module):
    """
    Stage 4 (part 1): builds the dense backward-warping sampling grid.

    Fixed canonical control points sit evenly spaced along the top and bottom
    borders of normalised [-1, 1] space (K/2 per row). Given a batch of
    *predicted* source control points (where the network thinks those
    canonical points actually are in the distorted feature map), this module
    produces, for every position of an (out_h, out_w) canonical output grid,
    the coordinate to sample from in the input feature map — i.e. a grid
    directly usable with F.grid_sample.

    All of the fixed, input-independent parts of the classic TPS solution
    (the canonical kernel matrix and its inverse, and the kernel between the
    dense output grid and the canonical points) are precomputed once here and
    registered as buffers, since they never change across batches.
    """

    def __init__(self, num_control_points, out_h, out_w):
        super().__init__()
        assert num_control_points % 2 == 0, "control points split evenly between top/bottom rows"
        self.num_control_points = num_control_points
        self.out_h = out_h
        self.out_w = out_w

        canonical = build_canonical_points(num_control_points)          # (K, 2)
        target_grid = self._build_target_grid(out_h, out_w)              # (out_h*out_w, 2)

        delta_c = self._build_delta_c(canonical)                          # (K+3, K+3)
        target_kernel = self._build_kernel(target_grid, canonical)         # (out_h*out_w, K)
        p_hat = torch.cat([
            target_kernel,
            torch.ones(out_h * out_w, 1),
            target_grid,
        ], dim=1)                                                          # (out_h*out_w, K+3)

        self.register_buffer("canonical", canonical)
        self.register_buffer("delta_c_inv", torch.inverse(delta_c))
        self.register_buffer("p_hat", p_hat)

    @staticmethod
    def _build_target_grid(out_h, out_w):
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, out_h), torch.linspace(-1, 1, out_w), indexing="ij",
        )
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)  # (out_h*out_w, 2), (x, y) order

    @staticmethod
    def _u(r2):
        # classic TPS radial basis function r^2 * log(r^2); define U(0) = 0
        r2 = r2.clone()
        zero_mask = r2 == 0
        r2[zero_mask] = 1
        out = r2 * torch.log(r2)
        out[zero_mask] = 0
        return out

    def _build_kernel(self, points, control_points):
        diff = points.unsqueeze(1) - control_points.unsqueeze(0)   # (N, K, 2)
        r2 = (diff ** 2).sum(dim=2)
        return self._u(r2)                                           # (N, K)

    def _build_delta_c(self, control_points):
        k = control_points.shape[0]
        kernel = self._build_kernel(control_points, control_points)   # (K, K)
        ones = torch.ones(k, 1)
        top = torch.cat([kernel, ones, control_points], dim=1)          # (K, K+3)
        mid = torch.cat([ones.t(), torch.zeros(1, 1), torch.zeros(1, 2)], dim=1)      # (1, K+3)
        bot = torch.cat([control_points.t(), torch.zeros(2, 1), torch.zeros(2, 2)], dim=1)  # (2, K+3)
        return torch.cat([top, mid, bot], dim=0)                        # (K+3, K+3)

    def forward(self, source_points):
        """
        source_points: (B, K, 2) — predicted locations, in the distorted
        input, of the K canonical control points.
        Returns: (B, out_h, out_w, 2) sampling grid for F.grid_sample.
        """
        b = source_points.shape[0]
        dtype = source_points.dtype
        zeros = torch.zeros(b, 3, 2, device=source_points.device, dtype=dtype)
        y = torch.cat([source_points, zeros], dim=1)                      # (B, K+3, 2)

        delta_c_inv = self.delta_c_inv.unsqueeze(0).expand(b, -1, -1).to(dtype)
        mapping_matrix = torch.bmm(delta_c_inv, y)                          # (B, K+3, 2)

        p_hat = self.p_hat.unsqueeze(0).expand(b, -1, -1).to(dtype)
        grid = torch.bmm(p_hat, mapping_matrix)                            # (B, out_h*out_w, 2)
        return grid.view(b, self.out_h, self.out_w, 2)


class FeatureRectificationBranch(nn.Module):
    """
    Full FRB:
        F_c   = phi(W_c * F)                              Stage 1
        ctx   = GeometryContextEncoder(F_c)                 Stage 2
        P     = TPSControlPointRegressor(ctx)                Stage 3, K control points
        grid  = TPSGridGenerator(P)                           Stage 4a
        F_TPS = grid_sample(F, grid)                           Stage 4b (warp F itself, not F_c)
        F_r   = F_TPS + Refinement(F_TPS)                       Stage 5, Eq. (6)
        M     = sigmoid(Conv3x3(F_r))                            Stage 6 gate
        F_g   = F + M (dot) (F_r - F)                             Stage 6 output, Eq. (7)
    """

    def __init__(self, channels=None, num_control_points=None, out_h=None, out_w=None):
        super().__init__()
        channels = channels or config.BACKBONE_OUT_CHANNELS
        num_control_points = num_control_points or config.TPS_NUM_CONTROL_POINTS
        out_h = out_h or config.BACKBONE_FEAT_H
        out_w = out_w or config.BACKBONE_FEAT_W

        # Stage 1
        self.compression = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Stage 2
        self.geometry_encoder = GeometryContextEncoder(channels)
        # Stage 3
        self.tps_regressor = TPSControlPointRegressor(channels, num_control_points)
        # Stage 4
        self.grid_generator = TPSGridGenerator(num_control_points, out_h, out_w)
        # Stage 5: depthwise 3x3 + pointwise 1x1, same lightweight pattern as FDB
        self.refinement = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        # Stage 6
        self.confidence_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, f):
        f_c = self.compression(f)
        ctx = self.geometry_encoder(f_c)
        control_points = self.tps_regressor(ctx)                # (B, K, 2)

        grid = self.grid_generator(control_points)                # (B, H, W, 2)
        f_tps = F.grid_sample(f, grid, align_corners=True)          # warp F itself, Eq. Stage 4

        r = self.refinement(f_tps)
        f_r = f_tps + r                                              # Eq. (6)

        m = torch.sigmoid(self.confidence_conv(f_r))                  # (B, 1, H, W)
        f_g = f + m * (f_r - f)                                        # Eq. (7)

        return {
            "F_g": f_g,
            "F_r": f_r,
            "M": m,
            "control_points": control_points,
            "canonical_points": self.grid_generator.canonical,   # for L_smooth (Eq. 9)
        }


if __name__ == "__main__":
    frb = FeatureRectificationBranch()
    f = torch.randn(2, config.BACKBONE_OUT_CHANNELS, config.BACKBONE_FEAT_H, config.BACKBONE_FEAT_W)
    out = frb(f)
    for k, v in out.items():
        print(f"{k:18s}: {tuple(v.shape)}")
    n_params = sum(p.numel() for p in frb.parameters())
    print(f"\nFRB parameters: {n_params:,}  (report design-phase estimate: ~100k)")
