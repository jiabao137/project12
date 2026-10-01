"""
Training script for Configuration A (Original Branch only).

This same script covers two uses described in the report:
    1. Section V-B: sweep --bilstm_layers in {1, 2, 3} on CCPD-Base to select
       the sequence-encoder depth (run 3 seeds per depth).
    2. Section V-D, Configuration A row of TABLE 7: the baseline entry in the
       five-configuration ablation study, once the depth is fixed.

Usage (Windows / PyCharm, conda env lpr_env):
    D:\\Anaconda\\envs\\lpr_env\\python.exe train.py --bilstm_layers 2 --seed 0 --tag configA_L2_seed0

Add --subset {base,blur,db,fn,rotate,tilt,challenge} to evaluate on a specific
CCPD degradation subset (TABLE 4) instead of the clean CCPD-Base test split.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import config
from dataset import CCPDDataset, ctc_collate_fn
from model import BaselineLPRNet
from metrics import evaluate, ctc_greedy_decode, compute_srr, compute_car
from vocab import BLANK_IDX


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_backbone(model):
    """
    Sets requires_grad=False on every backbone parameter, so it receives no
    gradient updates during training and keeps exactly its pretrained
    (ImageNet) weights throughout. Only the newly-added modules (FDB, FRB,
    fusion, decoder, auxiliary heads) continue to train normally.

    Call this AFTER building the model and BEFORE constructing the
    optimizer -- the optimizer must be built from a parameter list that
    already reflects which parameters are frozen, e.g.:

        model = SomeConfigModel().to(device)
        if args.freeze_backbone:
            freeze_backbone(model)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE,
        )

    Prints how many parameters were frozen, as a sanity check that this
    actually did something (e.g. catches a case where model.backbone
    doesn't exist under that name).
    """
    n_frozen = 0
    for p in model.backbone.parameters():
        p.requires_grad = False
        n_frozen += p.numel()
    print(f"Backbone frozen: {n_frozen:,} parameters will not be updated.")


def build_dataloaders(batch_size, val_split, seed, subset=None, extra_dirs=None):
    train_dirs = [config.CCPD_BASE_DIR] + list(extra_dirs or [])
    full_train = CCPDDataset(train_dirs, augment=True)
    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)
    # val split should not use training-time augmentation; wrap with a plain copy
    val_set.dataset = CCPDDataset(train_dirs, augment=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               collate_fn=ctc_collate_fn, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             collate_fn=ctc_collate_fn, num_workers=2)

    test_loader = None
    if subset is not None:
        test_dir = config.CCPD_TEST_DIRS[subset]
        test_set = CCPDDataset(test_dir, augment=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                  collate_fn=ctc_collate_fn, num_workers=2)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, ctc_loss_fn, device):
    model.train()
    total_loss = 0.0
    for imgs, targets, target_lengths, _plates in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(imgs)                          # (B, T, num_classes)
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, B, num_classes) for CTCLoss
        T, B, _ = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * B
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bilstm_layers", type=int, default=config.BILSTM_LAYERS,
                         help="Sweep 1/2/3 for the Section V-B depth-selection study.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="configA",
                         help="Run name, used for the checkpoint filename.")
    parser.add_argument("--subset", type=str, default=None,
                         choices=list(config.CCPD_TEST_DIRS.keys()),
                         help="Optional CCPD degradation subset to report test SRR/CAR on.")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    parser.add_argument("--extra_train_dirs", type=str, default=None,
                         help="Comma-separated list of additional folders to mix into "
                              "training on top of CCPD_BASE_DIR, e.g. a tilt/rotate subset.")
    parser.add_argument("--freeze_backbone", action="store_true",
                         help="Freeze the pretrained backbone (no gradient updates); "
                              "only the decoder is trained.")
    args = parser.parse_args()

    extra_dirs = [d.strip() for d in args.extra_train_dirs.split(",")] if args.extra_train_dirs else None

    set_seed(args.seed)
    device = config.DEVICE
    print(f"Device: {device} | bilstm_layers={args.bilstm_layers} | seed={args.seed} | "
          f"freeze_backbone={args.freeze_backbone}")
    if extra_dirs:
        print(f"Mixing in extra training directories: {extra_dirs}")

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=config.BATCH_SIZE, val_split=config.VAL_SPLIT,
        seed=args.seed, subset=args.subset, extra_dirs=extra_dirs,
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = BaselineLPRNet(bilstm_layers=args.bilstm_layers).to(device)
    if args.freeze_backbone:
        freeze_backbone(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE
    )
    ctc_loss_fn = torch.nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    best_srr = -1.0
    epochs_without_improvement = 0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{args.tag}.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, ctc_loss_fn, device)
        val_srr, val_car = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
              f"val_SRR={val_srr:.4f} val_CAR={val_car:.4f} ({dt:.1f}s)")

        if val_srr > best_srr:
            best_srr = val_srr
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "bilstm_layers": args.bilstm_layers,
                "epoch": epoch,
                "val_srr": val_srr,
                "val_car": val_car,
            }, ckpt_path)
            print(f"  -> new best (SRR={best_srr:.4f}), checkpoint saved to {ckpt_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    if test_loader is not None:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_srr, test_car = evaluate(model, test_loader, device)
        print(f"[TEST | subset={args.subset}] SRR={test_srr:.4f} CAR={test_car:.4f}")


if __name__ == "__main__":
    main()
