# Tri-RFN Baseline Skeleton — Configurations A-E (complete ablation suite, TABLE 7)

All five ablation configurations from the report are now implemented. Real-data
results so far (1.5万-image CCPD-Base subset, seed=0, 40 epochs, 2-layer BiLSTM):

| Config | Branches | Fusion | val_SRR | val_CAR |
|---|---|---|---|---|
| A | Original only | \u2014 | 0.4527 | 0.8729 |
| **B** | **Original + FDB** | \u2014 | **0.5663** | **0.9063** |
| C | Original + FRB | \u2014 | 0.5140 | 0.8860 |
| D | Original + FDB + FRB | Confidence-aware (Eqs. 11-12) | 0.5567 | 0.9004 |
| E | Original + FDB + FRB | Cross-attention (Eq. 13) | *not yet run on real data* | |

**Notable finding so far**: D does not beat B — three-branch confidence-aware
fusion underperforms FDB alone, despite clearly beating both A and C
individually, and despite the complementarity loss (`comp`) converging well
(0.42 -> 0.14 over 40 epochs, meaning FDB and FRB are correcting different
spatial regions, not fighting over the same ones). This is consistent with
confidence-aware fusion's competitive, sum-to-one weighting costing FDB some
of its influence to FRB without FRB fully making up the difference — exactly
the limitation Section III-D raised as the motivation for Configuration E.
Whether cross-attention fusion resolves this is the open question E answers.

All modules were smoke-tested end-to-end (forward pass, all 11 loss terms,
backward pass through both branches and both fusion mechanisms, checkpoint
save/load) with synthetic CCPD-format files before being handed over.

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths + hyperparameters, incl. `LAMBDA1/2/3`, `TPS_NUM_CONTROL_POINTS` |
| `vocab.py` | CCPD character vocabulary, CTC index mapping |
| `dataset.py` | CCPD filename parsing (bbox + 4 vertices), cropping, canonical-crop builder, both collate functions |
| `model.py` | Backbone + Original Branch (Section III-A) + BiLSTM/CTC — **Configuration A** |
| `fdb.py` | Feature Deblurring Branch, 5 stages (Section III-B, Eqs. 1-5) |
| `frb.py` | Feature Rectification Branch, 6 stages incl. differentiable TPS warp (Section III-C, Eqs. 6-7) |
| `fusion.py` | `ConfidenceAwareFusion` (N branches, Eqs. 11-12) + `CrossAttentionFusion` (Eq. 13) |
| `complementarity.py` | `SpatialActivityHead` (derives G_s from R_b) + `l_comp` (Section IV, Eqs. 14-15) |
| `losses.py` | FDB losses `l_res`, `l_cons`, `l_char` + FRB losses `l_geo`, `l_smooth`, `l_id` |
| `model_configB.py` | Original + FDB — **Configuration B** |
| `model_configC.py` | Original + FRB — **Configuration C** |
| `model_configD.py` | All 3 branches, confidence-aware fusion — **Configuration D** |
| `model_configE.py` | All 3 branches, cross-attention fusion — **Configuration E** |
| `metrics.py` | SRR, CAR, CTC greedy decoding |
| `train.py` | Training loop for Configuration A (also used for the Section V-B depth sweep) |
| `train_configB.py` | Training loop for Configuration B (Eq. 16, FDB terms only) |
| `train_configC.py` | Training loop for Configuration C (Eq. 16, FRB terms only); exports `build_dataloaders_configC`, reused by D and E |
| `train_configD.py` | Training loop for Configuration D — full Eq. 16 objective |
| `train_configE.py` | Training loop for Configuration E — full Eq. 16 objective, cross-attention fusion |
| `smoke_test.py` | Standalone real-data sanity check (run this first on new data) |

## Running

```
D:\Anaconda\envs\lpr_env\python.exe train.py         --tag configA_seed0 --seed 0 --epochs 40
D:\Anaconda\envs\lpr_env\python.exe train_configB.py --tag configB_seed0 --seed 0 --epochs 40
D:\Anaconda\envs\lpr_env\python.exe train_configC.py --tag configC_seed0 --seed 0 --epochs 40
D:\Anaconda\envs\lpr_env\python.exe train_configD.py --tag configD_seed0 --seed 0 --epochs 40
D:\Anaconda\envs\lpr_env\python.exe train_configE.py --tag configE_seed0 --seed 0 --epochs 40
```

`train_configE.py` is essentially `train_configD.py` with the fusion module
swapped — same 11-term loss, same per-epoch printout format, same runtime
cost per epoch (still two backbone passes for the L_geo canonical reference).
Diff the two files directly if you want to see exactly what changes between
D and E.

`--fusion_heads` (default 4) controls the number of attention heads in
`CrossAttentionFusion`; worth a quick sensitivity check (e.g. 1 vs 4 vs 8) if
E's results are close to D's, since head count is a free hyperparameter the
report doesn't pin down.

## What to check once Configuration E has run on real data

1. **val_SRR/val_CAR vs D** — this is Table III's comparison, made concrete.
   If E beats D (and ideally also beats B), that's a strong result: it would
   mean removing the sum-to-one constraint really does let both enhancement
   branches contribute without stealing from each other. If E doesn't beat D
   either, that raises a different (also reportable) finding — under this
   architecture, three-branch fusion may not be improving on FDB alone
   regardless of fusion mechanism, and B might be your strongest configuration
   headline result.
2. **`attn_weights` distribution** — inspect whether attention mass splits
   roughly evenly between the FDB and FRB halves of the concatenated K/V, or
   concentrates heavily on one branch (would suggest the other branch's
   contribution is being ignored despite being present).
3. **Training time** — cross-attention has more parameters and a full
   attention matrix per head; compare wall-clock time per epoch against D to
   know the practical cost of whichever fusion method you end up recommending.

## What's next

- All five TABLE 7 configurations exist. Natural next steps: 3-seed runs once
  single-seed numbers across A-E look sane (Section V-D's stated protocol),
  and filling in TABLE 6/7 of the report with these measured numbers.
- Revisit the report's estimated parameter counts (FDB ~70-80k vs ~91k
  measured; FRB ~100k vs ~77k measured) with the actual figures once you're
  citing measured rather than design-phase numbers.
- Section V-E (planned application study): cross-domain generalisation to
  non-CCPD plate formats — not started, deliberately out of scope until the
  CCPD ablation results (A-E) are settled.
