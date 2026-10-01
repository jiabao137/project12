"""
Training script for Configuration B (TABLE 7): Original Branch + FDB.

Combines the primary CTC loss with the FDB auxiliary losses of Section III-B,
i.e. Eq. (16) restricted to the FDB terms only (no FRB / L_comp yet):

    L_total = L_rec + lambda1 * (L_CTC^FDB + L_char + L_res + L_cons)

Reuses build_dataloaders() and set_seed() from train.py so Configuration A
and B always split data / seed identically.

Usage:
    D:\\Anaconda\\envs\\lpr_env\\python.exe train_configB.py --seed 0 --tag configB_seed0
"""

import argparse
import os
import time

import torch

import config
from train import build_dataloaders, set_seed, freeze_backbone
from model_configB import ConfigBModel
from metrics import ctc_greedy_decode, compute_srr, compute_car
from vocab import BLANK_IDX
from losses import l_res, l_cons, l_char


@torch.no_grad()
def evaluate_configB(model, loader, device):
    """Evaluates using only the primary (fused) logits — the aux head is a
    training-only signal and is never used for reported metrics."""
    model.eval()
    all_preds, all_gts = [], []
    for imgs, _targets, _target_lengths, plates in loader:
        imgs = imgs.to(device)
        logits = model(imgs, return_aux=False)
        log_probs = logits.log_softmax(dim=-1)
        preds = ctc_greedy_decode(log_probs)
        all_preds.extend(preds)
        all_gts.extend(plates)
    return compute_srr(all_preds, all_gts), compute_car(all_preds, all_gts)


def train_one_epoch(model, loader, optimizer, ctc_loss_fn, device, lambda1):
    model.train()
    totals = {"total": 0.0, "rec": 0.0, "aux_ctc": 0.0, "char": 0.0, "res": 0.0, "cons": 0.0}

    for imgs, targets, target_lengths, _plates in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        B = imgs.shape[0]

        out = model(imgs, return_aux=True)

        log_probs = out["logits"].log_softmax(dim=-1).permute(1, 0, 2)     # (T, B, num_classes)
        T = log_probs.shape[0]
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        loss_rec = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        aux_log_probs = out["aux_logits"].log_softmax(dim=-1).permute(1, 0, 2)
        loss_aux_ctc = ctc_loss_fn(aux_log_probs, targets, input_lengths, target_lengths)

        loss_res = l_res(out["R_b"])
        loss_cons = l_cons(out["F_b"], out["F"])
        loss_char = l_char(model._to_sequence(out["F_b"]), targets, target_lengths)

        loss = loss_rec + lambda1 * (loss_aux_ctc + loss_char + loss_res + loss_cons)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        totals["total"] += loss.item() * B
        totals["rec"] += loss_rec.item() * B
        totals["aux_ctc"] += loss_aux_ctc.item() * B
        totals["char"] += loss_char.item() * B
        totals["res"] += loss_res.item() * B
        totals["cons"] += loss_cons.item() * B

    n = len(loader.dataset)
    return {k: v / n for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="configB")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    parser.add_argument("--bilstm_layers", type=int, default=config.BILSTM_LAYERS)
    parser.add_argument("--lambda1", type=float, default=config.LAMBDA1)
    parser.add_argument("--extra_train_dirs", type=str, default=None,
                         help="Comma-separated list of additional folders to mix into "
                              "training on top of CCPD_BASE_DIR, e.g. a tilt/rotate subset.")
    parser.add_argument("--freeze_backbone", action="store_true",
                         help="Freeze the pretrained backbone (no gradient updates); "
                              "only FDB/fusion/decoder are trained.")
    args = parser.parse_args()

    extra_dirs = [d.strip() for d in args.extra_train_dirs.split(",")] if args.extra_train_dirs else None

    set_seed(args.seed)
    device = config.DEVICE
    print(f"Device: {device} | Configuration B (Original + FDB) | "
          f"bilstm_layers={args.bilstm_layers} | lambda1={args.lambda1} | seed={args.seed} | "
          f"freeze_backbone={args.freeze_backbone}")
    if extra_dirs:
        print(f"Mixing in extra training directories: {extra_dirs}")

    train_loader, val_loader, _ = build_dataloaders(
        batch_size=config.BATCH_SIZE, val_split=config.VAL_SPLIT, seed=args.seed,
        extra_dirs=extra_dirs,
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = ConfigBModel(bilstm_layers=args.bilstm_layers).to(device)
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
        losses = train_one_epoch(model, train_loader, optimizer, ctc_loss_fn, device, args.lambda1)
        val_srr, val_car = evaluate_configB(model, val_loader, device)
        dt = time.time() - t0

        print(f"[epoch {epoch:03d}] loss={losses['total']:.4f} "
              f"(rec={losses['rec']:.3f} aux_ctc={losses['aux_ctc']:.3f} "
              f"char={losses['char']:.3f} res={losses['res']:.3f} cons={losses['cons']:.3f}) "
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
