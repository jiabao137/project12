"""
Configuration B (TABLE 7): Original Branch + FDB, confidence-aware fusion.

forward(x, return_aux=True) also returns everything train_configB.py needs to
compute the FDB auxiliary losses of Section III-B: R_b, F_b, F, and the
auxiliary CTC head's logits. return_aux=False (the default) gives just the
primary logits, matching Configuration A's interface for evaluate()/metrics.py.
"""

import torch
import torch.nn as nn

import config
from model import Backbone, OriginalBranch, SequenceDecoder
from fdb import FeatureDeblurringBranch
from fusion import ConfidenceAwareFusion


class ConfigBModel(nn.Module):
    def __init__(self, bilstm_layers=None):
        super().__init__()
        c = config.BACKBONE_OUT_CHANNELS

        self.backbone = Backbone()
        self.original_branch = OriginalBranch(c)
        self.fdb = FeatureDeblurringBranch(c)
        self.fusion = ConfidenceAwareFusion(c, num_branches=2)
        self.decoder = SequenceDecoder(in_channels=c, num_layers=bilstm_layers or config.BILSTM_LAYERS)

        # Auxiliary CTC head on F_b alone (Section III-B). Deliberately small
        # (1 layer) since it is discarded at inference and only needs to be
        # good enough to provide a useful training signal to the FDB.
        self.aux_decoder = SequenceDecoder(in_channels=c, num_layers=1)

    @staticmethod
    def _to_sequence(feat):
        feat = nn.functional.adaptive_avg_pool2d(feat, (1, feat.shape[-1]))  # (B, C, 1, W)
        return feat.squeeze(2).permute(0, 2, 1)                                # (B, W, C)

    def forward(self, x, return_aux=False):
        f = self.backbone(x)
        f_o = self.original_branch(f)
        fdb_out = self.fdb(f)
        f_b = fdb_out["F_b"]

        fused, fusion_weights = self.fusion([f_o, f_b])
        logits = self.decoder(self._to_sequence(fused))

        if not return_aux:
            return logits

        aux_logits = self.aux_decoder(self._to_sequence(f_b))
        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "R_b": fdb_out["R_b"],
            "F_b": f_b,
            "F": f,
            "fusion_weights": fusion_weights,
        }


if __name__ == "__main__":
    model = ConfigBModel()
    x = torch.randn(2, 3, config.IMG_H, config.IMG_W)

    logits = model(x)
    print(f"Primary logits (return_aux=False): {tuple(logits.shape)}")

    out = model(x, return_aux=True)
    for k, v in out.items():
        print(f"{k:16s}: {tuple(v.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    n_params_no_aux = n_params - sum(p.numel() for p in model.aux_decoder.parameters())
    print(f"\nTotal parameters (incl. training-only aux head): {n_params:,}")
    print(f"Inference-time parameters (aux head discarded):   {n_params_no_aux:,}")
