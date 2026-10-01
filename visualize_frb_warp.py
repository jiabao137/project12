"""
Visualizes what FRB's TPS warp actually does to a handful of real images —
lets you eyeball whether the predicted warp is doing anything sensible,
rather than only reading confidence-gate statistics (diagnose_frb.py).

For each sampled image, saves a side-by-side PNG with three panels:
    1. "input crop"                — the plain bbox crop the model sees
    2. "canonical (ground truth)"  — built from CCPD's annotated vertices,
                                       the same reference used for L_geo
    3. "FRB predicted warp"        — the input crop warped using FRB's
                                       ACTUAL predicted TPS control points
                                       for that image

Panel 3 uses the exact control points the trained model predicted (run
through the real forward pass), just applied to the input image at a larger
visualization resolution instead of the network's internal 4x13 feature
map — the underlying TPS math is identical, only the output resolution
differs, so this is a faithful visualization of what the network "decided"
to do, not an approximation.

Usage:
    D:\\Anaconda\\envs\\lpr_env\\python.exe visualize_frb_warp.py ^
        --checkpoint D:\\lpr_checkpoints\\configC_seed0.pt ^
        --distorted_subset tilt --n 6
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config
from dataset import parse_ccpd_filename, build_canonical_crop
from model_configC import ConfigCModel
from frb import TPSGridGenerator

VIS_H, VIS_W = 64, 200   # larger than the network's internal 4x13, for visibility


def tensor_from_bgr(img_bgr, h, w):
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (w, h)), cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)   # (1, 3, h, w)


def tensor_to_rgb_uint8(t):
    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5 + 0.5) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


@torch.no_grad()
def visualize_one(model, filepath, out_path, device):
    parsed = parse_ccpd_filename(filepath)
    if parsed is None:
        print(f"  skip (unparseable filename): {os.path.basename(filepath)}")
        return
    plate, bbox, vertices = parsed
    raw_img = cv2.imread(filepath)
    if raw_img is None:
        print(f"  skip (unreadable image): {os.path.basename(filepath)}")
        return

    x1, y1, x2, y2 = bbox
    h_img, w_img = raw_img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    input_crop_bgr = raw_img[y1:y2, x1:x2]
    if input_crop_bgr.size == 0:
        print(f"  skip (empty bbox crop): {os.path.basename(filepath)}")
        return

    # Run the real model to get FRB's actual predicted control points.
    input_tensor = tensor_from_bgr(input_crop_bgr, config.IMG_H, config.IMG_W).to(device)
    out = model(input_tensor, return_aux=True)
    control_points = out["control_points"]   # (1, K, 2), normalised [-1, 1]

    # Panel 2: ground-truth canonical crop (if vertices parsed OK).
    canonical_img = None
    if vertices is not None:
        canonical_bgr = build_canonical_crop(raw_img, vertices, VIS_W, VIS_H)
        canonical_img = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2RGB)

    # Panel 3: apply the model's ACTUAL predicted control points to the input
    # image at visualization resolution. Coordinates are resolution-independent,
    # so this is the same warp the network computed internally, just rendered
    # bigger so it's visible.
    vis_grid_gen = TPSGridGenerator(control_points.shape[1], VIS_H, VIS_W).to(device)
    grid = vis_grid_gen(control_points)
    input_vis_tensor = tensor_from_bgr(input_crop_bgr, VIS_H, VIS_W).to(device)
    warped_tensor = F.grid_sample(input_vis_tensor, grid, align_corners=True)
    warped_img = tensor_to_rgb_uint8(warped_tensor)

    # Assemble the side-by-side panel.
    input_vis = cv2.cvtColor(cv2.resize(input_crop_bgr, (VIS_W, VIS_H)), cv2.COLOR_BGR2RGB)
    panels = [input_vis]
    labels = ["input crop"]
    if canonical_img is not None:
        panels.append(canonical_img)
        labels.append("canonical (ground truth)")
    panels.append(warped_img)
    labels.append("FRB predicted warp")

    gap = 6
    canvas = np.full((VIS_H + 24, VIS_W * len(panels) + gap * (len(panels) - 1), 3), 255, dtype=np.uint8)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x_off = i * (VIS_W + gap)
        canvas[24:24 + VIS_H, x_off:x_off + VIS_W] = panel
        cv2.putText(canvas, label, (x_off + 2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"  saved {out_path}  (plate={plate})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clean_dir", default=None,
                         help="Defaults to config.CCPD_TEST_DIRS['base'] if not given.")
    parser.add_argument("--distorted_subset", default="tilt", choices=list(config.CCPD_TEST_DIRS.keys()))
    parser.add_argument("--n", type=int, default=6, help="Images to sample from each folder.")
    parser.add_argument("--bilstm_layers", type=int, default=config.BILSTM_LAYERS)
    parser.add_argument("--out_dir", default="frb_warp_previews")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    device = config.DEVICE
    model = ConfigCModel(bilstm_layers=args.bilstm_layers).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    os.makedirs(args.out_dir, exist_ok=True)

    clean_dir = args.clean_dir or config.CCPD_TEST_DIRS["base"]
    distorted_dir = config.CCPD_TEST_DIRS[args.distorted_subset]

    for name, folder in [("clean", clean_dir), (args.distorted_subset, distorted_dir)]:
        files = glob.glob(os.path.join(folder, "*.jpg"))
        random.shuffle(files)
        chosen = files[:args.n]
        print(f"\n{name}: sampling {len(chosen)} images from {folder}")
        for i, fp in enumerate(chosen):
            out_path = os.path.join(args.out_dir, f"{name}_{i:02d}.png")
            visualize_one(model, fp, out_path, device)

    print(f"\nAll previews saved to {args.out_dir}/ — open them and compare the three panels "
          f"(input vs. ground-truth canonical vs. what FRB actually predicted).")


if __name__ == "__main__":
    main()
