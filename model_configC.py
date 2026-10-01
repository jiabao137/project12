"""
Configuration C (TABLE 7): Original Branch + FRB, confidence-aware fusion.

forward(x, return_aux=True) also returns everything train_configC.py needs
for the FRB auxiliary losses of Section III-C: control_points,
canonical_points (for L_smooth), F_g, F (for L_id), and the auxiliary CTC
head's logits. Unlike FDB's L_geo, computing L_geo needs a *canonical
reference image* run through the backbone — that happens in
train_configC.py, not inside this model, since it needs a second forward
pass through self.backbone on a different input.
"""

import torch
import torch.nn as nn

import config
from model import Backbone, OriginalBranch, SequenceDecoder
from frb import FeatureRectificationBranch
from fusion import ConfidenceAwareFusion


class ConfigCModel(nn.Module):
    def __init__(self, bilstm_layers=None):
        super().__init__()
        c = config.BACKBONE_OUT_CHANNELS

        self.backbone = Backbone()
        self.original_branch = OriginalBranch(c)
        self.frb = FeatureRectificationBranch(c)
        self.fusion = ConfidenceAwareFusion(c, num_branches=2)
        self.decoder = SequenceDecoder(in_channels=c, num_layers=bilstm_layers or config.BILSTM_LAYERS)

        # Auxiliary CTC head on F_g alone (Section III-C). Small on purpose —
        # discarded at inference.
        self.aux_decoder = SequenceDecoder(in_channels=c, num_layers=1)

    @staticmethod
    def _to_sequence(feat):
        feat = nn.functional.adaptive_avg_pool2d(feat, (1, feat.shape[-1]))
        return feat.squeeze(2).permute(0, 2, 1)

    def encode_canonical(self, canonical_img):
        """
        Runs the canonical reference image through the SAME backbone used for
        the main input, for use as the L_geo target. Callers should wrap this
        in torch.no_grad() (see train_configC.py) — it is a stop-gradient
        target, not a separately pretrained frozen network.
        """
        return self.backbone(canonical_img)

    def forward(self, x, return_aux=False):
        f = self.backbone(x)
        f_o = self.original_branch(f)
        frb_out = self.frb(f)
        f_g = frb_out["F_g"]

        fused, fusion_weights = self.fusion([f_o, f_g])
        logits = self.decoder(self._to_sequence(fused))

        if not return_aux:
            return logits

        aux_logits = self.aux_decoder(self._to_sequence(f_g))
        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "F_g": f_g,
            "F": f,
            "M": frb_out["M"],
            "control_points": frb_out["control_points"],
            "canonical_points": frb_out["canonical_points"],
            "fusion_weights": fusion_weights,
        }


if __name__ == "__main__":
    model = ConfigCModel()
    x = torch.randn(2, 3, config.IMG_H, config.IMG_W)

    logits = model(x)
    print(f"Primary logits (return_aux=False): {tuple(logits.shape)}")

    out = model(x, return_aux=True)
    for k, v in out.items():
        print(f"{k:18s}: {tuple(v.shape)}")

    canonical = torch.randn(2, 3, config.IMG_H, config.IMG_W)
    with torch.no_grad():
        f_canonical = model.encode_canonical(canonical)
    print(f"{'F_canonical':18s}: {tuple(f_canonical.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    n_params_no_aux = n_params - sum(p.numel() for p in model.aux_decoder.parameters())
    print(f"\nTotal parameters (incl. training-only aux head): {n_params:,}")
    print(f"Inference-time parameters (aux head discarded):   {n_params_no_aux:,}")
