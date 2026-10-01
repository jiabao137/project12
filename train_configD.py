"""
Training script for Configuration D (TABLE 7): the core ablation result —
all three branches, confidence-aware fusion, the FULL joint objective Eq. (16):

    L_total = L_rec
            + lambda1 * (L_CTC^FDB + L_char + L_res  + L_cons)
            + lambda2 * (L_CTC^FRB + L_geo  + L_smooth + L_id)
            + lambda3 * L_comp

Uses frb_collate_fn / return_canonical=True (needed for L_geo), same as
Configuration C.

Usage:
    D:\\Anaconda\\envs\\lpr_env\\python.exe train_configD.py --seed 0 --tag configD_seed0
"""

import argparse
import os
import time

import torch

import config
from dataset import CCPDDataset, frb_collate_fn
from train import set_seed, freeze_backbone
from train_configC import build_dataloaders_configC   # reuse: same return_canonical=True loader
from model_configD import ConfigDModel
from metrics import ctc_greedy_decode, compute_srr, compute_car
from vocab import BLANK_IDX
from losses import l_res, l_cons, l_char, l_geo, l_smooth, l_id
from complementarity import l_comp


@torch.no_grad()
def evaluate_configD(model, loader, device):
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


def train_one_epoch(model, loader, optimizer, ctc_loss_fn, device, lambda1, lambda2, lambda3):
    model.train()
    keys = ["total", "rec", "fdb_ctc", "char", "res", "cons",
            "frb_ctc", "geo", "smooth", "id", "comp"]
    totals = {k: 0.0 for k in keys}

    for imgs, canonical, targets, target_lengths, _plates in loader:
        imgs = imgs.to(device)
        canonical = canonical.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        B = imgs.shape[0]

        out = model(imgs, return_aux=True)
        with torch.no_grad():
            f_canonical = model.encode_canonical(canonical)

        log_probs = out["logits"].log_softmax(dim=-1).permute(1, 0, 2)
        T = log_probs.shape[0]
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        loss_rec = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        # FDB terms
        aux_fdb_log_probs = out["aux_logits_fdb"].log_softmax(dim=-1).permute(1, 0, 2)
        loss_fdb_ctc = ctc_loss_fn(aux_fdb_log_probs, targets, input_lengths, target_lengths)
        loss_res = l_res(out["R_b"])
        loss_cons = l_cons(out["F_b"], out["F"])
        loss_char = l_char(model._to_sequence(out["F_b"]), targets, target_lengths)

        # FRB terms
        aux_frb_log_probs = out["aux_logits_frb"].log_softmax(dim=-1).permute(1, 0, 2)
        loss_frb_ctc = ctc_loss_fn(aux_frb_log_probs, targets, input_lengths, target_lengths)
        loss_geo = l_geo(out["F_g"], f_canonical)
        loss_smooth = l_smooth(out["control_points"], out["canonical_points"])
        loss_id = l_id(out["F_g"], out["F"])

        # Section IV
        loss_comp = l_comp(out["G_s"], out["M"])

        loss = (loss_rec
                + lambda1 * (loss_fdb_ctc + loss_char + loss_res + loss_cons)
                + lambda2 * (loss_frb_ctc + loss_geo + loss_smooth + loss_id)
                + lambda3 * loss_comp)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        totals["total"] += loss.item() * B
        totals["rec"] += loss_rec.item() * B
        totals["fdb_ctc"] += loss_fdb_ctc.item() * B
        totals["char"] += loss_char.item() * B
        totals["res"] += loss_res.item() * B
        totals["cons"] += loss_cons.item() * B
        totals["frb_ctc"] += loss_frb_ctc.item() * B
        totals["geo"] += loss_geo.item() * B
        totals["smooth"] += loss_smooth.item() * B
        totals["id"] += loss_id.item() * B
        totals["comp"] += loss_comp.item() * B

    n = len(loader.dataset)
    return {k: v / n for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="configD")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    parser.add_argument("--bilstm_layers", type=int, default=config.BILSTM_LAYERS)
    parser.add_argument("--lambda1", type=float, default=config.LAMBDA1)
    parser.add_argument("--lambda2", type=float, default=config.LAMBDA2)
    parser.add_argument("--lambda3", type=float, default=config.LAMBDA3)
    parser.add_argument("--extra_train_dirs", type=str, default=None,
                         help="Comma-separated list of additional folders to mix into "
                              "training on top of CCPD_BASE_DIR, e.g. a tilt/rotate subset.")
    parser.add_argument("--freeze_backbone", action="store_true",
                         help="Freeze the pretrained backbone (no gradient updates); "
                              "only FDB/FRB/fusion/decoder are trained.")
    args = parser.parse_args()

    extra_dirs = [d.strip() for d in args.extra_train_dirs.split(",")] if args.extra_train_dirs else None

    set_seed(args.seed)
    device = config.DEVICE
    print(f"Device: {device} | Configuration D (Original + FDB + FRB, confidence-aware fusion) | "
          f"bilstm_layers={args.bilstm_layers} | "
          f"lambda1={args.lambda1} lambda2={args.lambda2} lambda3={args.lambda3} | seed={args.seed} | "
          f"freeze_backbone={args.freeze_backbone}")
    if extra_dirs:
        print(f"Mixing in extra training directories: {extra_dirs}")

    train_loader, val_loader = build_dataloaders_configC(
        batch_size=config.BATCH_SIZE, val_split=config.VAL_SPLIT, seed=args.seed,
        extra_dirs=extra_dirs,
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = ConfigDModel(bilstm_layers=args.bilstm_layers).to(device)
    if args.freeze_backbone:
        freeze_backbone(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters (incl. training-only aux heads): {n_params:,} total, {n_trainable:,} trainable")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE
    )
    ctc_loss_fn = torch.nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    best_srr = -1.0
    epochs_without_improvement = 0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{args.tag}.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = train_one_epoch(model, train_loader, optimizer, ctc_loss_fn, device,
                                  args.lambda1, args.lambda2, args.lambda3)
        val_srr, val_car = evaluate_configD(model, val_loader, device)
        dt = time.time() - t0

        print(f"[epoch {epoch:03d}] loss={losses['total']:.4f} "
              f"(rec={losses['rec']:.3f} | "
              f"FDB: ctc={losses['fdb_ctc']:.3f} char={losses['char']:.3f} "
              f"res={losses['res']:.3f} cons={losses['cons']:.3f} | "
              f"FRB: ctc={losses['frb_ctc']:.3f} geo={losses['geo']:.3f} "
              f"smooth={losses['smooth']:.3f} id={losses['id']:.3f} | "
              f"comp={losses['comp']:.3f}) "
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
