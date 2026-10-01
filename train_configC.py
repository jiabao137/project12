"""
Training script for Configuration C (TABLE 7): Original Branch + FRB.

Combines the primary CTC loss with the FRB auxiliary losses of Section III-C,
i.e. Eq. (16) restricted to the FRB terms only (no FDB / L_comp yet):

    L_total = L_rec + lambda2 * (L_CTC^FRB + L_geo + L_smooth + L_id)

Uses frb_collate_fn / return_canonical=True since L_geo needs the
fronto-parallel canonical reference crop built from CCPD's vertex annotation
(dataset.py). Samples whose filename lacks a parseable vertex field are
silently skipped by the dataset — check the printed sample count against
Configuration A/B's if this matters for your comparison.

Usage:
    D:\\Anaconda\\envs\\lpr_env\\python.exe train_configC.py --seed 0 --tag configC_seed0
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, random_split

import config
from dataset import CCPDDataset, frb_collate_fn
from train import set_seed, freeze_backbone
from model_configC import ConfigCModel
from metrics import ctc_greedy_decode, compute_srr, compute_car
from vocab import BLANK_IDX
from losses import l_geo, l_smooth, l_id


def build_dataloaders_configC(batch_size, val_split, seed, extra_dirs=None):
    """
    extra_dirs: optional list of additional folders to mix into training
    (e.g. a tilt/rotate subset carved out for this purpose), on top of
    config.CCPD_BASE_DIR. CCPDDataset already accepts a list of root_dirs and
    merges all samples together, so this needed no new dataset code — just
    passing a longer directory list through. Validation is drawn from the
    SAME mixed pool (not CCPD-Base only), so the reported val_SRR/val_CAR
    during training reflects performance across the mixed distribution, not
    just the clean one.
    """
    train_dirs = [config.CCPD_BASE_DIR] + list(extra_dirs or [])
    full_train = CCPDDataset(train_dirs, augment=True, return_canonical=True)
    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)
    val_set.dataset = CCPDDataset(train_dirs, augment=False, return_canonical=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               collate_fn=frb_collate_fn, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             collate_fn=frb_collate_fn, num_workers=2)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_configC(model, loader, device):
    model.eval()
    all_preds, all_gts = [], []
    for imgs, _canonical, _targets, _target_lengths, plates in loader:
        imgs = imgs.to(device)
        logits = model(imgs, return_aux=False)
        log_probs = logits.log_softmax(dim=-1)
        preds = ctc_greedy_decode(log_probs)
        all_preds.extend(preds)
        all_gts.extend(plates)
    return compute_srr(all_preds, all_gts), compute_car(all_preds, all_gts)


def train_one_epoch(model, loader, optimizer, ctc_loss_fn, device, lambda2):
    model.train()
    totals = {"total": 0.0, "rec": 0.0, "aux_ctc": 0.0, "geo": 0.0, "smooth": 0.0, "id": 0.0}

    for imgs, canonical, targets, target_lengths, _plates in loader:
        imgs = imgs.to(device)
        canonical = canonical.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        B = imgs.shape[0]

        out = model(imgs, return_aux=True)
        with torch.no_grad():
            f_canonical = model.encode_canonical(canonical)   # stop-gradient target

        log_probs = out["logits"].log_softmax(dim=-1).permute(1, 0, 2)
        T = log_probs.shape[0]
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        loss_rec = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        aux_log_probs = out["aux_logits"].log_softmax(dim=-1).permute(1, 0, 2)
        loss_aux_ctc = ctc_loss_fn(aux_log_probs, targets, input_lengths, target_lengths)

        loss_geo = l_geo(out["F_g"], f_canonical)
        loss_smooth = l_smooth(out["control_points"], out["canonical_points"])
        loss_id = l_id(out["F_g"], out["F"])

        loss = loss_rec + lambda2 * (loss_aux_ctc + loss_geo + loss_smooth + loss_id)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        totals["total"] += loss.item() * B
        totals["rec"] += loss_rec.item() * B
        totals["aux_ctc"] += loss_aux_ctc.item() * B
        totals["geo"] += loss_geo.item() * B
        totals["smooth"] += loss_smooth.item() * B
        totals["id"] += loss_id.item() * B

    n = len(loader.dataset)
    return {k: v / n for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="configC")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    parser.add_argument("--bilstm_layers", type=int, default=config.BILSTM_LAYERS)
    parser.add_argument("--lambda2", type=float, default=config.LAMBDA2)
    parser.add_argument("--extra_train_dirs", type=str, default=None,
                         help="Comma-separated list of additional folders to mix into "
                              "training on top of CCPD_BASE_DIR, e.g. a tilt/rotate subset "
                              "carved out for this purpose. Example: "
                              "--extra_train_dirs D:\\tilt_train_mix,D:\\rotate_train_mix")
    parser.add_argument("--freeze_backbone", action="store_true",
                         help="Freeze the pretrained backbone (no gradient updates); "
                              "only FRB/fusion/decoder are trained.")
    args = parser.parse_args()

    extra_dirs = [d.strip() for d in args.extra_train_dirs.split(",")] if args.extra_train_dirs else None

    set_seed(args.seed)
    device = config.DEVICE
    print(f"Device: {device} | Configuration C (Original + FRB) | "
          f"bilstm_layers={args.bilstm_layers} | lambda2={args.lambda2} | seed={args.seed} | "
          f"freeze_backbone={args.freeze_backbone}")
    if extra_dirs:
        print(f"Mixing in extra training directories: {extra_dirs}")

    train_loader, val_loader = build_dataloaders_configC(
        batch_size=config.BATCH_SIZE, val_split=config.VAL_SPLIT, seed=args.seed,
        extra_dirs=extra_dirs,
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)} "
          f"(samples without a parseable vertex field are excluded)")

    model = ConfigCModel(bilstm_layers=args.bilstm_layers).to(device)
    if args.freeze_backbone:
        freeze_backbone(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters (incl. training-only aux head): {n_params:,} total, {n_trainable:,} trainable")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE
    )
    ctc_loss_fn = torch.nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    best_srr = -1.0
    epochs_without_improvement = 0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{args.tag}.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = train_one_epoch(model, train_loader, optimizer, ctc_loss_fn, device, args.lambda2)
        val_srr, val_car = evaluate_configC(model, val_loader, device)
        dt = time.time() - t0

        print(f"[epoch {epoch:03d}] loss={losses['total']:.4f} "
              f"(rec={losses['rec']:.3f} aux_ctc={losses['aux_ctc']:.3f} "
              f"geo={losses['geo']:.3f} smooth={losses['smooth']:.3f} id={losses['id']:.3f}) "
              f"val_SRR={val_srr:.4f} val_CAR={val_car:.4f} ({dt:.1f}s)")

        if val_srr > best_srr:
            best_srr = val_srr
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_srr": val_srr, "val_car": val_car,
            }, ckpt_path)
            print(f"  -> new best (SRR={best_srr:.4f}), checkpoint saved to {ckpt_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break


if __name__ == "__main__":
    main()
