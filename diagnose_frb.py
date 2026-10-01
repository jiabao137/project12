"""
Diagnostic: is FRB's spatial confidence gate (M) responding to how distorted
the input actually is, or has it learned to be uniformly conservative
regardless of what it's shown?

Loads a Configuration C checkpoint and compares, on a CLEAN sample (e.g. your
test_subset_base) versus a DISTORTED sample (e.g. ccpd_tilt):
    - M's mean/std/min/max (Section III-C Stage 6 spatial confidence map)
    - the mean TPS control-point displacement from canonical (how far the
      predicted control points sit from where they'd be for an
      already-fronto-parallel plate)

Two possible diagnoses:
    - M / displacement stay similarly small on BOTH clean and distorted data
      -> FRB learned to barely correct anything regardless of input, most
      likely because the training subset didn't contain enough genuinely
      distorted examples for it to learn when to act. Fix: retrain with some
      distorted examples mixed into training, not an architecture change.
    - M / displacement are clearly larger on the distorted sample, but
      SRR/CAR still didn't improve on that subset -> FRB IS reacting to
      distortion, but the TPS warp itself is likely inaccurate (correcting
      in the wrong way, not failing to correct at all). Worth visually
      inspecting a few warped feature maps / canonical crops next, rather
      than changing the training data.

Usage:
    D:\\Anaconda\\envs\\lpr_env\\python.exe diagnose_frb.py ^
        --checkpoint D:\\lpr_checkpoints\\configC_seed0.pt --distorted_subset tilt
"""

import argparse

import torch
from torch.utils.data import DataLoader

import config
from dataset import CCPDDataset, ctc_collate_fn
from model_configC import ConfigCModel


@torch.no_grad()
def collect_frb_stats(model, loader, device, max_batches):
    m_values = []
    disp_values = []
    for i, (imgs, _targets, _target_lengths, _plates) in enumerate(loader):
        if i >= max_batches:
            break
        imgs = imgs.to(device)
        out = model(imgs, return_aux=True)
        m_values.append(out["M"].flatten().cpu())

        cp = out["control_points"]                          # (B, K, 2)
        canon = out["canonical_points"].unsqueeze(0)          # (1, K, 2)
        disp = (cp - canon).norm(dim=-1)                        # (B, K)
        disp_values.append(disp.flatten().cpu())

    m_values = torch.cat(m_values)
    disp_values = torch.cat(disp_values)
    return {
        "M_mean": m_values.mean().item(), "M_std": m_values.std().item(),
        "M_min": m_values.min().item(), "M_max": m_values.max().item(),
        "disp_mean": disp_values.mean().item(), "disp_std": disp_values.std().item(),
        "n_samples": m_values.shape[0],
    }


def print_stats(name, stats):
    print(f"\n[{name}]  (n={stats['n_samples']} spatial positions/control points scanned)")
    print(f"  M (spatial confidence gate): mean={stats['M_mean']:.4f}  std={stats['M_std']:.4f}  "
          f"min={stats['M_min']:.4f}  max={stats['M_max']:.4f}")
    print(f"  TPS control-point displacement from canonical: mean={stats['disp_mean']:.4f}  "
          f"std={stats['disp_std']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clean_dir", default=None,
                         help="Folder of clean images (e.g. your test_subset_base). "
                              "Defaults to config.CCPD_TEST_DIRS['base'] if not given.")
    parser.add_argument("--distorted_subset", default="tilt",
                         choices=list(config.CCPD_TEST_DIRS.keys()))
    parser.add_argument("--bilstm_layers", type=int, default=config.BILSTM_LAYERS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max_batches", type=int, default=50,
                         help="Caps batches scanned per folder, for speed — this is a "
                              "diagnostic, not a full evaluation.")
    args = parser.parse_args()

    device = config.DEVICE
    model = ConfigCModel(bilstm_layers=args.bilstm_layers).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    clean_dir = args.clean_dir or config.CCPD_TEST_DIRS["base"]
    distorted_dir = config.CCPD_TEST_DIRS[args.distorted_subset]

    clean_set = CCPDDataset(clean_dir, augment=False, return_canonical=False)
    distorted_set = CCPDDataset(distorted_dir, augment=False, return_canonical=False)
    clean_loader = DataLoader(clean_set, batch_size=args.batch_size, shuffle=False,
                               collate_fn=ctc_collate_fn, num_workers=2)
    distorted_loader = DataLoader(distorted_set, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=ctc_collate_fn, num_workers=2)

    print(f"Clean folder: {clean_dir} ({len(clean_set)} samples)")
    print(f"Distorted folder ({args.distorted_subset}): {distorted_dir} ({len(distorted_set)} samples)")
    print(f"Scanning up to {args.max_batches} batches from each...")

    clean_stats = collect_frb_stats(model, clean_loader, device, args.max_batches)
    distorted_stats = collect_frb_stats(model, distorted_loader, device, args.max_batches)

    print_stats("CLEAN", clean_stats)
    print_stats(f"DISTORTED / {args.distorted_subset}", distorted_stats)

    m_ratio = distorted_stats["M_mean"] / max(clean_stats["M_mean"], 1e-6)
    disp_ratio = distorted_stats["disp_mean"] / max(clean_stats["disp_mean"], 1e-6)

    print("\n--- Diagnosis ---")
    print(f"M mean ratio (distorted / clean):            {m_ratio:.2f}x")
    print(f"Displacement mean ratio (distorted / clean):  {disp_ratio:.2f}x")

    if m_ratio < 1.3 and disp_ratio < 1.3:
        print(
            "\nFRB does NOT appear to react meaningfully to distortion level -- M and "
            "control-point displacement are similar on clean and distorted data.\n"
            "Most likely explanation: the training subset did not contain enough genuinely "
            "distorted examples for FRB to learn when to act. Suggested fix: mix some "
            "distorted (tilt/rotate) samples into training rather than changing FRB's "
            "architecture."
        )
    else:
        print(
            "\nFRB DOES appear to react to distortion level (M / displacement are "
            "meaningfully larger on distorted data), yet SRR/CAR did not improve on that "
            "subset.\nThis points toward the TPS warp itself being inaccurate -- FRB is "
            "trying to correct, just not correctly. Worth visually inspecting a few warped "
            "feature maps or canonical crops next, rather than changing the training data."
        )


if __name__ == "__main__":
    main()
