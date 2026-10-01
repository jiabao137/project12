"""
Central configuration for the Tri-RFN baseline (Configuration A: Original Branch only).

Matches the settings in the report, Section V-C (TABLE 6):
- MobileNetV3-Small-style backbone, ~C=160 channels
- Input resolution 3 x 32 x 100
- BiLSTM sequence encoder (depth is swept in Section V-B, see BILSTM_LAYERS below)
- CTC decoder
- Adam, lr=1e-4, batch size 32, up to 100 epochs, early stopping patience 10

Edit the paths below to match your machine before running.
"""

import os

# ---------------------------------------------------------------------------
# Paths (Windows / PyCharm, conda env lpr_env)
# CCPD is split across D:\ and E:\ per your existing setup — point these at the
# actual CCPD-Base / degradation-subset folders you use.
# ---------------------------------------------------------------------------
CCPD_BASE_DIR = r"D:\train_subset"          # training + validation split source
CCPD_TEST_DIRS = {
    "base":      r"D:\test_subset_base",
    "blur":      r"D:\CloudMusic\BaiduNetdiskDownload\CCPD2019\CCPD2019\ccpd_blur",
    "db":        r"D:\CloudMusic\BaiduNetdiskDownload\CCPD2019\CCPD2019\ccpd_db",
    "fn":        r"D:\CloudMusic\BaiduNetdiskDownload\CCPD2019\CCPD2019\ccpd_fn",
    "rotate":    r"D:\rotate_train_mix",  # 如果这个已经变成训练用的了，测试要指向剩余部分或者重新确认
    "tilt":      r"D:\CloudMusic\BaiduNetdiskDownload\CCPD2019\CCPD2019\ccpd_tilt",
    "challenge": r"D:\CloudMusic\BaiduNetdiskDownload\CCPD2019\CCPD2019\ccpd_challenge",
}
CHECKPOINT_DIR = r"D:\lpr_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
IMG_H = 32
IMG_W = 100
VAL_SPLIT = 0.2          # 80/20 train/val on CCPD-Base, per TABLE 4
RANDOM_SEED = 0          # override per run for the 3-seed protocol in Section V-D/V-B

# ---------------------------------------------------------------------------
# Backbone / feature dimension
# ---------------------------------------------------------------------------
BACKBONE_OUT_CHANNELS = 160   # matches the C ~ 160 assumption used throughout the report

# Use an ImageNet-pretrained MobileNetV3-Large, truncated to a 4x13 output
# resolution (model.py's PretrainedBackbone), instead of the from-scratch
# LightweightBackbone. Requires internet access on first run to download
# pretrained weights (~21MB). Set to False to roll back if offline or for
# comparison against the earlier from-scratch results.
USE_PRETRAINED_BACKBONE = True

# Backbone's output spatial size for the fixed IMG_H x IMG_W input above
# (stride-8 backbone, see model.py's Backbone docstring). FRB's TPS grid
# buffers are precomputed at this fixed resolution — if you change IMG_H/W
# or the backbone's stride, update these to match or FRB will break.
BACKBONE_FEAT_H = 4
BACKBONE_FEAT_W = 13

# Number of TPS control points for FRB (Section III-C, Stage 3). Half sit
# along the top border, half along the bottom, in normalised [-1, 1] space.
TPS_NUM_CONTROL_POINTS = 16

# ---------------------------------------------------------------------------
# Sequence encoder (BiLSTM)
# ---------------------------------------------------------------------------
BILSTM_LAYERS = 2          # default; sweep {1, 2, 3} for the Section V-B depth-selection study
BILSTM_HIDDEN = 256        # per direction -> 512-d output after concatenation

# ---------------------------------------------------------------------------
# Optimisation (TABLE 6)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10

# ---------------------------------------------------------------------------
# Loss weights (report Eq. 16, TABLE 6: "initialised at 0.1 and tuned on the
# validation split"). LAMBDA2/LAMBDA3 are unused until FRB exists.
# ---------------------------------------------------------------------------
LAMBDA1 = 0.1   # weight on the FDB auxiliary losses (L_CTC^FDB + L_char + L_res + L_cons)
LAMBDA2 = 0.1   # weight on the FRB auxiliary losses (not used yet)
LAMBDA3 = 0.1   # weight on L_comp (not used yet)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
import torch  # noqa: E402
DEVICE = "cuda" if (torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1") else "cpu"
