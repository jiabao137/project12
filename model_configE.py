"""
Configuration E (TABLE 7): Original + FDB + FRB, CROSS-ATTENTION fusion
(Eq. 13) — the direct comparison point against Configuration D (Table III).

Identical to ConfigDModel except for the fusion module and how it's called
(CrossAttentionFusion takes f_o, f_b, f_g as three separate arguments and
returns attention weights, not a per-branch confidence map). Everything else
— both branches, both auxiliary CTC heads, the complementarity term inputs —
is unchanged, so the two configurations are comparable on everything except
the fusion mechanism itself.
"""

import torch
import torch.nn as nn

import config
from model import Backbone, OriginalBranch, SequenceDecoder
from fdb import FeatureDeblurringBranch
from frb import FeatureRectificationBranch
from fusion import CrossAttentionFusion
from complementarity import SpatialActivityHead


class ConfigEModel(nn.Module):
    def __init__(self, bilstm_layers=None, fusion_heads=4):
        super().__init__()
        c = config.BACKBONE_OUT_CHANNELS

        self.backbone = Backbone()
        self.original_branch = OriginalBranch(c)
        self.fdb = FeatureDeblurringBranch(c)
        self.frb = FeatureRectificationBranch(c)
        self.fusion = CrossAttentionFusion(c, num_heads=fusion_heads)
        self.decoder = SequenceDecoder(in_channels=c, num_layers=bilstm_layers or config.BILSTM_LAYERS)

        self.aux_decoder_fdb = SequenceDecoder(in_channels=c, num_layers=1)
        self.aux_decoder_frb = SequenceDecoder(in_channels=c, num_layers=1)
        self.spatial_activity_head = SpatialActivityHead()

    @staticmethod
    def _to_sequence(feat):
        feat = nn.functional.adaptive_avg_pool2d(feat, (1, feat.shape[-1]))
        return feat.squeeze(2).permute(0, 2, 1)

    def encode_canonical(self, canonical_img):
        return self.backbone(canonical_img)

    def forward(self, x, return_aux=False):
        f = self.backbone(x)
        f_o = self.original_branch(f)
        fdb_out = self.fdb(f)
        frb_out = self.frb(f)
        f_b = fdb_out["F_b"]
        f_g = frb_out["F_g"]

        fused, attn_weights = self.fusion(f_o, f_b, f_g)
        logits = self.decoder(self._to_sequence(fused))

        if not return_aux:
            return logits

        aux_logits_fdb = self.aux_decoder_fdb(self._to_sequence(f_b))
        aux_logits_frb = self.aux_decoder_frb(self._to_sequence(f_g))
        g_s = self.spatial_activity_head(fdb_out["R_b"])

        return {
            "logits": logits,
            "aux_logits_fdb": aux_logits_fdb,
            "aux_logits_frb": aux_logits_frb,
            "R_b": fdb_out["R_b"],
            "F_b": f_b,
            "F_g": f_g,
            "F": f,
            "G_s": g_s,
            "M": frb_out["M"],
            "control_points": frb_out["control_points"],
            "canonical_points": frb_out["canonical_points"],
            "attn_weights": attn_weights,
        }


if __name__ == "__main__":
    model = ConfigEModel()
    x = torch.randn(2, 3, config.IMG_H, config.IMG_W)

    logits = model(x)
    print(f"Primary logits (return_aux=False): {tuple(logits.shape)}")

    out = model(x, return_aux=True)
    for k, v in out.items():
        print(f"{k:18s}: {tuple(v.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    n_aux = (sum(p.numel() for p in model.aux_decoder_fdb.parameters())
             + sum(p.numel() for p in model.aux_decoder_frb.parameters())
             + sum(p.numel() for p in model.spatial_activity_head.parameters()))
    print(f"\nTotal parameters (incl. training-only heads): {n_params:,}")
    print(f"Inference-time parameters (aux heads discarded): {n_params - n_aux:,}")
