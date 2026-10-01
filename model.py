"""
Configuration A baseline (report Table 7): Original Feature Branch only,
i.e. the shared backbone + identity-anchor branch + BiLSTM/CTC decoder,
with no FDB or FRB yet. This is the common foundation every other ablation
configuration (B-E) builds on, so it is implemented first.

FDB (Section III-B) and FRB (Section III-C) are NOT implemented here — they
are separate modules to be added later and fused in front of the same
BiLSTM/CTC head via the fusion module of Section III-D.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

import config
from vocab import NUM_CLASSES


def _make_divisible(v, divisor=8):
    return int((v + divisor / 2) // divisor) * divisor


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise conv + 1x1 pointwise conv, the basic MobileNet-style block."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                    padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class PretrainedBackbone(nn.Module):
    """
    ImageNet-pretrained MobileNetV3-Large, truncated to its first 7 feature
    blocks (torchvision indices 0-6). For a 3 x 32 x 100 input this produces
    a (40, 4, 13) feature map -- verified to match, exactly, the spatial
    resolution (4 x 13) the framework's earlier from-scratch LightweightBackbone
    always produced. Every downstream module (FDB, FRB's TPSGridGenerator,
    the fusion modules) was built against config.BACKBONE_FEAT_H/W = 4/13, so
    this is a drop-in replacement: no other file needs to change.

    This exists because the from-scratch LightweightBackbone (see below) has
    no pretrained weights and only ~3M parameters total across the whole
    network, which visibly bottlenecked recognition accuracy relative to
    published CCPD baselines. Swapping in real ImageNet-pretrained features
    is the first, lowest-effort lever for closing that gap; a distilled
    lightweight student network (e.g. from a YOLO-scale teacher) is a
    heavier follow-up direction, not attempted here.

    Requires internet access on first run to download the pretrained weights
    (~21MB) via torchvision; set pretrained=False to fall back to random
    initialisation of the same truncated architecture if offline.
    """

    def __init__(self, out_channels=None, pretrained=True):
        super().__init__()
        out_channels = out_channels or config.BACKBONE_OUT_CHANNELS
        weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        full = tv_models.mobilenet_v3_large(weights=weights)
        self.features = full.features[:7]   # -> (B, 40, 4, 13) for a 32x100 input
        self.project = nn.Sequential(
            nn.Conv2d(40, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.project(x)
        return x


class LightweightBackbone(nn.Module):
    """
    Original from-scratch MobileNetV3-Small-style backbone with explicit,
    controlled downsampling so the output feature map keeps a usable
    sequence length for CTC decoding. Kept for reference / rollback; the
    active `Backbone` alias below now points at PretrainedBackbone by default
    (config.USE_PRETRAINED_BACKBONE).

    For a 3 x 32 x 100 input:
        stem      stride 2  -> 16 x 16 x 50
        block 1   stride 2  -> 24 x  8 x 25
        block 2   stride 2  -> 48 x  4 x 13
        block 3   stride 1  -> 96 x  4 x 13
        project   1x1 conv  -> C  x  4 x 13   (C = config.BACKBONE_OUT_CHANNELS)

    Total stride 8 (not MobileNetV3's usual 32) is deliberate: full-stride
    downsampling collapses the width to ~3-4 positions, too short for 7-character
    CTC decoding.
    """

    def __init__(self, out_channels=None):
        super().__init__()
        out_channels = out_channels or config.BACKBONE_OUT_CHANNELS

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
        )
        self.block1 = DepthwiseSeparableConv(16, 24, stride=2)
        self.block2 = DepthwiseSeparableConv(24, 48, stride=2)
        self.block3 = DepthwiseSeparableConv(48, 96, stride=1)

        self.project = nn.Sequential(
            nn.Conv2d(96, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.project(x)
        return x  # (B, C, H, W)


# Active backbone used by BaselineLPRNet and all of Configurations B-E.
# Switch config.USE_PRETRAINED_BACKBONE to False to roll back to the
# from-scratch LightweightBackbone.
Backbone = PretrainedBackbone if getattr(config, "USE_PRETRAINED_BACKBONE", True) else LightweightBackbone


class ChannelAttention(nn.Module):
    """SE-style channel attention: GAP -> FC -> FC -> sigmoid -> per-channel scale."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class OriginalBranch(nn.Module):
    """
    Section III-A: identity-preserving anchor branch.
    A lightweight residual block followed by channel attention, producing F_o.
    """

    def __init__(self, channels):
        super().__init__()
        self.residual_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)
        self.channel_attention = ChannelAttention(channels)

    def forward(self, f):
        residual = self.residual_conv(f)
        f = self.act(f + residual)          # residual connection
        f_o = self.channel_attention(f)      # channel-reweighted output
        return f_o


class SequenceDecoder(nn.Module):
    """BiLSTM sequence encoder + CTC projection head (report Section V-C, TABLE 6)."""

    def __init__(self, in_channels, hidden_size=None, num_layers=None, num_classes=None):
        super().__init__()
        hidden_size = hidden_size or config.BILSTM_HIDDEN
        num_layers = num_layers or config.BILSTM_LAYERS
        num_classes = num_classes or NUM_CLASSES

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, feat_seq):
        # feat_seq: (B, T, C)
        out, _ = self.lstm(feat_seq)          # (B, T, 2*hidden)
        logits = self.fc(out)                  # (B, T, num_classes)
        return logits


class BaselineLPRNet(nn.Module):
    """
    Configuration A (TABLE 7): Backbone -> Original Branch -> BiLSTM -> CTC.

    Later configurations (B-E) add FDB / FRB modules that produce F_b / F_g in
    parallel with this branch's F_o, plus a fusion module (Section III-D)
    before the same SequenceDecoder. Keeping SequenceDecoder as its own module
    means it can be reused unchanged once those branches are added.
    """

    def __init__(self, bilstm_layers=None):
        super().__init__()
        self.backbone = Backbone()
        self.original_branch = OriginalBranch(config.BACKBONE_OUT_CHANNELS)
        self.decoder = SequenceDecoder(
            in_channels=config.BACKBONE_OUT_CHANNELS,
            num_layers=bilstm_layers or config.BILSTM_LAYERS,
        )

    def forward(self, x):
        f = self.backbone(x)                 # (B, C, H, W)
        f_o = self.original_branch(f)         # (B, C, H, W)

        # Squeeze height to 1 (adaptive avg pool over H only), keep width as
        # the CTC time axis, then reshape to (B, T, C) for the BiLSTM.
        f_o = nn.functional.adaptive_avg_pool2d(f_o, (1, f_o.shape[-1]))  # (B, C, 1, W)
        f_o = f_o.squeeze(2).permute(0, 2, 1)  # (B, W, C) = (B, T, C)

        logits = self.decoder(f_o)             # (B, T, num_classes)
        return logits


if __name__ == "__main__":
    model = BaselineLPRNet()
    x = torch.randn(2, 3, config.IMG_H, config.IMG_W)
    logits = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(logits.shape)}  (B, T, num_classes)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
