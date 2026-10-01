"""
CCPD dataset loader.

CCPD filenames encode all annotations in the filename itself, separated by '-':
    area-tilt-bbox-vertices-platenumber-brightness-blurriness.jpg

Example:
    025-95_113-154&383_386&473-386&473_154&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg

Field 3 (bbox)     : "x1&y1_x2&y2"                -> top-left / bottom-right corners
Field 4 (vertices) : "x1&y1_x2&y2_x3&y3_x4&y4"      -> 4 corner points, starting from
                                                        the bottom-right, going clockwise
Field 5 (plate)    : "p_a_d1_d2_d3_d4_d5"            -> indices into PROVINCES / ALPHABETS / ADS

This loader crops the plate region using the bbox field for the standard
(Configuration A/B) input, and also parses the four-vertex field so FRB's
L_geo (Section III-C, Eq. 8) can build a fronto-parallel "canonical" reference
crop via perspective warp — see build_canonical_crop() below.
"""

import os
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from vocab import PROVINCES, ALPHABETS, ADS, encode_plate


def parse_ccpd_filename(filename):
    """Return (plate_string, bbox, vertices) parsed from a CCPD filename, or None if malformed.
    vertices is a (4, 2) float32 array, or None if that field is missing/malformed."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    fields = stem.split("-")
    if len(fields) < 5:
        return None

    # --- bbox: "x1&y1_x2&y2" ---
    try:
        tl, br = fields[2].split("_")
        x1, y1 = map(int, tl.split("&"))
        x2, y2 = map(int, br.split("&"))
    except (ValueError, IndexError):
        return None
    bbox = (x1, y1, x2, y2)

    # --- four vertices: "x1&y1_x2&y2_x3&y3_x4&y4" ---
    vertices = None
    try:
        pts = [tuple(map(int, p.split("&"))) for p in fields[3].split("_")]
        if len(pts) == 4:
            vertices = np.array(pts, dtype=np.float32)
    except (ValueError, IndexError):
        vertices = None  # not fatal — only needed for FRB's L_geo

    # --- plate number: 7 underscore-separated indices ---
    try:
        idx = list(map(int, fields[4].split("_")))
        if len(idx) != 7:
            return None
        plate = (
            PROVINCES[idx[0]] + ALPHABETS[idx[1]]
            + "".join(ADS[i] for i in idx[2:])
        )
    except (ValueError, IndexError):
        return None

    return plate, bbox, vertices


def build_canonical_crop(img, vertices, out_w, out_h):
    """
    Perspective-warps the plate region defined by `vertices` into a flat,
    fronto-parallel out_w x out_h rectangle. Used to build the canonical
    reference image for FRB's L_geo (Section III-C, Eq. 8): this reference is
    passed through the (stop-gradient) backbone to get F_canonical.

    vertices: (4, 2) float32 array in CCPD's documented corner order —
    right-bottom, left-bottom, left-top, right-top. Worth double-checking
    against a few real samples (visually) if the canonical crops come out
    mirrored or rotated — vertex ordering has varied slightly across CCPD
    releases in the wild, and getting it wrong would silently teach FRB a
    consistently-flipped notion of "canonical" rather than crash.
    """
    dst = np.array([
        [out_w - 1, out_h - 1],   # right-bottom
        [0, out_h - 1],            # left-bottom
        [0, 0],                     # left-top
        [out_w - 1, 0],              # right-top
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(vertices, dst)
    warped = cv2.warpPerspective(img, matrix, (out_w, out_h))
    return warped


class CCPDDataset(Dataset):
    """
    Loads (image, label) pairs from one or more CCPD directories.

    Args:
        root_dirs: str or list[str] — directory (or directories) of .jpg files.
        img_h, img_w: target resize dimensions (default from config).
        augment: whether to apply light training-time augmentation
                 (brightness/contrast jitter only; geometric augmentation is
                 deliberately left out since FRB is meant to learn that).
        return_canonical: if True, __getitem__ also returns a fronto-parallel
                 canonical crop (for FRB's L_geo). Samples whose filename
                 lacks a parseable vertex field are skipped entirely when
                 this is True, since there is nothing to build a canonical
                 crop from. Use frb_collate_fn (not ctc_collate_fn) when this
                 is True — the batch tuple shape is different.
    """

    def __init__(self, root_dirs, img_h=None, img_w=None, augment=False, return_canonical=False):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.img_h = img_h or config.IMG_H
        self.img_w = img_w or config.IMG_W
        self.augment = augment
        self.return_canonical = return_canonical

        self.samples = []  # list of (filepath, plate_str, bbox, vertices)
        for d in root_dirs:
            for fp in glob.glob(os.path.join(d, "*.jpg")):
                parsed = parse_ccpd_filename(fp)
                if parsed is None:
                    continue
                plate, bbox, vertices = parsed
                if return_canonical and vertices is None:
                    continue  # nothing to build a canonical crop from
                self.samples.append((fp, plate, bbox, vertices))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid CCPD samples found in {root_dirs}. "
                f"Check that the path(s) exist, contain CCPD-format filenames, "
                f"and (if return_canonical=True) that the vertex field is present."
            )

    def __len__(self):
        return len(self.samples)

    def _crop_resize_rgb(self, img, bbox):
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def _augment(self, img):
        # Light photometric jitter only. Kept intentionally simple; extend here
        # if you want brightness/contrast augmentation for the low-quality-image
        # study, but avoid geometric jitter (that is FRB's job to learn to undo).
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return img

    def _to_tensor(self, img):
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # normalise to [-1, 1]
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()  # C,H,W

    def __getitem__(self, i):
        filepath, plate, bbox, vertices = self.samples[i]
        raw_img = cv2.imread(filepath)
        if raw_img is None:
            raise FileNotFoundError(filepath)

        img = self._crop_resize_rgb(raw_img, bbox)
        if self.augment:
            img = self._augment(img)
        img_t = self._to_tensor(img)

        label = torch.tensor(encode_plate(plate), dtype=torch.long)  # length 7

        if not self.return_canonical:
            return img_t, label, plate

        canonical = build_canonical_crop(raw_img, vertices, self.img_w, self.img_h)
        canonical = cv2.cvtColor(canonical, cv2.COLOR_BGR2RGB)
        canonical_t = self._to_tensor(canonical)
        return img_t, canonical_t, label, plate


def ctc_collate_fn(batch):
    """
    For return_canonical=False datasets (Configuration A/B).
    Stacks images; concatenates labels for CTCLoss along with per-sample lengths
    (CTCLoss expects a flat target tensor plus target_lengths, not padded labels).
    """
    imgs, labels, plates = zip(*batch)
    imgs = torch.stack(imgs, dim=0)                      # (B, 3, H, W)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels, dim=0)                   # flat, length = sum(target_lengths)
    return imgs, targets, target_lengths, plates


def frb_collate_fn(batch):
    """For return_canonical=True datasets (Configuration C and later)."""
    imgs, canonicals, labels, plates = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    canonicals = torch.stack(canonicals, dim=0)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels, dim=0)
    return imgs, canonicals, targets, target_lengths, plates


if __name__ == "__main__":
    # Quick smoke test — point this at a small local folder to sanity-check parsing.
    import sys
    test_dir = sys.argv[1] if len(sys.argv) > 1 else config.CCPD_BASE_DIR

    ds = CCPDDataset(test_dir)
    print(f"[return_canonical=False] Loaded {len(ds)} samples from {test_dir}")
    img, label, plate = ds[0]
    print(f"Image shape: {tuple(img.shape)}, plate: {plate}, label: {label.tolist()}")

    ds_canon = CCPDDataset(test_dir, return_canonical=True)
    print(f"\n[return_canonical=True] Loaded {len(ds_canon)} samples "
          f"(some may be skipped if the vertex field is missing)")
    if len(ds_canon) > 0:
        img, canonical, label, plate = ds_canon[0]
        print(f"Image shape: {tuple(img.shape)}, canonical shape: {tuple(canonical.shape)}, plate: {plate}")
