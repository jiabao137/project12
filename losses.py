"""
Auxiliary losses for the FDB (report Section III-B, Eqs. 4-5, and Section IV).

L_CTC^FDB is NOT defined here — it reuses the exact same CTCLoss function as
the primary decoder, just applied to the auxiliary head's logits (see
train_configB.py). Only L_res, L_cons and L_char are FDB-specific.
"""

import torch
import torch.nn.functional as F


def l_res(r_b):
    """Eq. (4): residual regularisation, L1 norm — penalises unnecessarily
    large corrections."""
    return r_b.abs().mean()


def l_cons(f_b, f):
    """Eq. (5): consistency loss, 1 - cosine similarity between F_b and F,
    keeping the enhanced feature close to the original feature's semantic
    content."""
    f_b_flat = f_b.flatten(1)
    f_flat = f.flatten(1)
    cos_sim = F.cosine_similarity(f_b_flat, f_flat, dim=1)
    return (1 - cos_sim).mean()


def l_char(seq_features, targets, target_lengths, plate_length=7, temperature=0.1):
    """
    Character-level contrastive loss (report Section III-B): pulls together
    feature vectors that correspond to the same ground-truth character across
    the batch, and pushes apart vectors for different characters, so the FDB
    learns blur-invariant per-character representations.

    IMPORTANT SIMPLIFICATION: CTC training does not provide an explicit
    per-timestep character alignment. This implementation uses an evenly
    spaced pseudo-alignment — character k of a plate is assumed to sit at
    time step round(k * T / 7) — rather than a true forced alignment. This is
    a reasonable first-pass approximation (CCPD characters are roughly evenly
    spaced across the plate), but if this loss term turns out to matter
    empirically, replacing the pseudo-alignment with a proper CTC forced
    alignment (e.g. Viterbi decoding against the trained model) would be the
    natural next refinement — flag this if it comes up with your supervisors.

    Args:
        seq_features: (B, T, C) — same sequence view fed to the BiLSTM.
        targets: flat concatenated label tensor (from ctc_collate_fn).
        target_lengths: (B,) tensor, all values should equal `plate_length` for CCPD.
    """
    B, T, C = seq_features.shape
    device = seq_features.device

    labels = torch.split(targets, target_lengths.tolist())  # list of (7,) tensors

    pseudo_positions = [
        min(int(round(k * T / plate_length)), T - 1) for k in range(plate_length)
    ]
    char_feats = seq_features[:, pseudo_positions, :]                     # (B, 7, C)
    char_feats = F.normalize(char_feats, dim=-1)

    char_ids = torch.stack([lbl[:plate_length] for lbl in labels]).to(device)  # (B, 7)

    flat_feats = char_feats.reshape(-1, C)      # (B*7, C)
    flat_ids = char_ids.reshape(-1)              # (B*7,)

    sim = flat_feats @ flat_feats.t() / temperature        # (N, N)
    same_char = (flat_ids.unsqueeze(0) == flat_ids.unsqueeze(1)).float()
    same_char.fill_diagonal_(0)                              # exclude self-pairs

    has_positive = same_char.sum(dim=1) > 0
    if has_positive.sum() == 0:
        # can happen with a tiny batch where every character is unique
        return torch.tensor(0.0, device=device)

    log_prob = torch.log_softmax(sim, dim=1)
    pos_log_prob = (log_prob * same_char).sum(dim=1) / same_char.sum(dim=1).clamp(min=1)
    loss = -pos_log_prob[has_positive].mean()
    return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C = 4, 13, 160
    seq = torch.randn(B, T, C)
    f_b = torch.randn(B, C, 4, 13)
    f = torch.randn(B, C, 4, 13)
    r_b = torch.randn(B, C, 4, 13) * 0.1

    labels = [torch.randint(1, 60, (7,)) for _ in range(B)]
    targets = torch.cat(labels)
    target_lengths = torch.tensor([7] * B)

    print("L_res :", l_res(r_b).item())
    print("L_cons:", l_cons(f_b, f).item())
    print("L_char:", l_char(seq, targets, target_lengths).item())


# ---------------------------------------------------------------------------
# FRB auxiliary losses (report Section III-C, Eqs. 8-10)
# ---------------------------------------------------------------------------

def l_geo(f_g, f_canonical):
    """
    Eq. (8): geometry consistency loss, ||F_g - F_canonical||_2^2.

    f_canonical is produced by running the SAME (currently-training) backbone
    on the canonical fronto-parallel reference crop, wrapped in torch.no_grad()
    by the caller — this is a stop-gradient "frozen" target in the sense that
    gradients from this loss do not flow back into computing F_canonical
    itself, not a separately pretrained network. See train_configC.py.
    """
    return ((f_g - f_canonical) ** 2).mean()


def l_smooth(control_points, canonical_points):
    """
    Eq. (9): transformation smoothness — penalises abrupt displacement
    differences between TPS-adjacent control points (neighbours along the
    same top/bottom row), discouraging unrealistic warps.

    control_points, canonical_points: (B, K, 2) and (K, 2) respectively
    (canonical_points broadcasts across the batch).
    """
    displacement = control_points - canonical_points.unsqueeze(0)   # (B, K, 2)
    k = displacement.shape[1]
    half = k // 2
    # neighbours within each row (top row: 0..half-1, bottom row: half..k-1)
    top = displacement[:, :half, :]
    bottom = displacement[:, half:, :]
    top_diff = (top[:, 1:, :] - top[:, :-1, :]) ** 2
    bottom_diff = (bottom[:, 1:, :] - bottom[:, :-1, :]) ** 2
    return top_diff.sum(dim=-1).mean() + bottom_diff.sum(dim=-1).mean()


def l_id(f_g, f):
    """Eq. (10): identity preservation, 1 - cosine similarity between F_g and F."""
    f_g_flat = f_g.flatten(1)
    f_flat = f.flatten(1)
    cos_sim = F.cosine_similarity(f_g_flat, f_flat, dim=1)
    return (1 - cos_sim).mean()


if __name__ == "__main__":
    print("\n--- FRB losses ---")
    K = 16
    canonical_pts = torch.stack([
        torch.linspace(-1, 1, K // 2), -torch.ones(K // 2)
    ], dim=1)
    canonical_pts = torch.cat([
        canonical_pts,
        torch.stack([torch.linspace(-1, 1, K // 2), torch.ones(K // 2)], dim=1),
    ], dim=0)
    pred_pts = canonical_pts.unsqueeze(0).repeat(4, 1, 1) + torch.randn(4, K, 2) * 0.05
    f_g = torch.randn(4, 160, 4, 13)
    f = torch.randn(4, 160, 4, 13)
    f_canonical = torch.randn(4, 160, 4, 13)

    print("L_geo   :", l_geo(f_g, f_canonical).item())
    print("L_smooth:", l_smooth(pred_pts, canonical_pts).item())
    print("L_id    :", l_id(f_g, f).item())
