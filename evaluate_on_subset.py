"""
Evaluates an already-trained checkpoint (any Configuration A-E) on a specific
CCPD test subset (TABLE 4) — reuses checkpoints already trained on
CCPD-Base, no retraining needed.

Use this to test the hypothesis from the A-E ablation results: does FRB
(Configurations C/D/E) help more on genuinely geometrically-distorted plates
(ccpd_tilt / ccpd_rotate) than it did on the CCPD-Base training distribution,
where its contribution looked weaker than FDB's?

Works on the FULL test subset (not just samples with a parseable vertex
field) since evaluation only needs the primary logits (return_aux=False path),
which never touches L_geo / the canonical crop.

Usage:
    D:\\Anaconda\\envs\\lpr_env\\python.exe evaluate_on_subset.py ^
        --config B --checkpoint D:\\lpr_checkpoints\\configB_seed0.pt --subset tilt

    D:\\Anaconda\\envs\\lpr_env\\python.exe evaluate_on_subset.py ^
        --config D --checkpoint D:\\lpr_checkpoints\\configD_seed0.pt --subset rotate
"""

import argparse

import torch
from torch.utils.data import DataLoader

import config
from dataset import CCPDDataset, ctc_collate_fn
from metrics import ctc_greedy_decode, compute_srr, compute_car

from model import BaselineLPRNet
from model_configB import ConfigBModel
from model_configC import ConfigCModel
from model_configD import ConfigDModel
from model_configE import ConfigEModel

MODEL_CLASSES = {
    "A": BaselineLPRNet,
    "B": ConfigBModel,
    "C": ConfigCModel,
    "D": ConfigDModel,
    "E": ConfigEModel,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subset", required=True, choices=list(config.CCPD_TEST_DIRS.keys()))
    parser.add_argument("--bilstm_layers", type=int, default=config.BILSTM_LAYERS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    device = config.DEVICE
    model_cls = MODEL_CLASSES[args.config]
    model = model_cls(bilstm_layers=args.bilstm_layers).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    trained_srr = checkpoint.get("val_srr")
    trained_srr_str = f"{trained_srr:.4f}" if trained_srr is not None else "unknown"
    print(f"Loaded Configuration {args.config} from {args.checkpoint} "
          f"(trained epoch {checkpoint.get('epoch', '?')}, "
          f"CCPD-Base val_SRR at save time: {trained_srr_str})")

    test_dir = config.CCPD_TEST_DIRS[args.subset]
    test_set = CCPDDataset(test_dir, augment=False, return_canonical=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              collate_fn=ctc_collate_fn, num_workers=2)
    print(f"Test subset '{args.subset}': {len(test_set)} samples from {test_dir}")

    all_preds, all_gts = [], []
    with torch.no_grad():
        for imgs, _targets, _target_lengths, plates in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)          # works uniformly across A-E: return_aux defaults to False
            log_probs = logits.log_softmax(dim=-1)
            preds = ctc_greedy_decode(log_probs)
            all_preds.extend(preds)
            all_gts.extend(plates)

    srr = compute_srr(all_preds, all_gts)
    car = compute_car(all_preds, all_gts)
    print(f"\n[Configuration {args.config} | subset={args.subset}] "
          f"SRR={srr:.4f} CAR={car:.4f} (n={len(all_gts)})")


if __name__ == "__main__":
    main()
