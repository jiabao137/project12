"""
Evaluation metrics used throughout the report (TABLE 4, TABLE 7):
    - SRR: Sequence Recognition Rate, perfect 7-character match only.
    - CAR: Character Accuracy Rate.

Both are computed from CTC logits via greedy decoding (argmax + collapse
repeats + remove blanks). Beam search can be swapped in later without
changing the metric functions themselves.
"""

import torch

from vocab import BLANK_IDX, IDX2CHAR


def ctc_greedy_decode(logits):
    """
    logits: (B, T, num_classes) raw or log-softmax scores.
    Returns: list[str], one decoded plate string per batch element.
    """
    pred_indices = logits.argmax(dim=-1)  # (B, T)
    results = []
    for seq in pred_indices.tolist():
        # collapse repeated consecutive symbols, then drop blanks
        collapsed = []
        prev = None
        for idx in seq:
            if idx != prev:
                collapsed.append(idx)
            prev = idx
        chars = [IDX2CHAR[i] for i in collapsed if i != BLANK_IDX]
        results.append("".join(chars))
    return results


def compute_srr(pred_strings, gt_strings):
    """Sequence Recognition Rate: fraction of samples with a perfect string match."""
    assert len(pred_strings) == len(gt_strings)
    if len(gt_strings) == 0:
        return 0.0
    correct = sum(p == g for p, g in zip(pred_strings, gt_strings))
    return correct / len(gt_strings)


def compute_car(pred_strings, gt_strings):
    """
    Character Accuracy Rate via Levenshtein (edit-distance) alignment, robust to
    predictions that are shorter/longer than the 7-character ground truth
    (which greedy CTC decoding can occasionally produce early in training).

    CAR = 1 - (edit_distance / len(ground_truth)), averaged over the batch,
    clipped to [0, 1] per sample so a badly garbled prediction cannot drag the
    average below 0.
    """
    assert len(pred_strings) == len(gt_strings)
    if len(gt_strings) == 0:
        return 0.0

    total = 0.0
    for pred, gt in zip(pred_strings, gt_strings):
        dist = _levenshtein(pred, gt)
        car = 1.0 - dist / max(len(gt), 1)
        total += max(0.0, car)
    return total / len(gt_strings)


def _levenshtein(a, b):
    """Standard O(len(a)*len(b)) edit distance."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[lb]


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Run the model over a dataloader and return (SRR, CAR) as floats in [0, 1]."""
    model.eval()
    all_preds, all_gts = [], []
    for imgs, _targets, _target_lengths, plates in dataloader:
        imgs = imgs.to(device)
        logits = model(imgs)                      # (B, T, num_classes)
        log_probs = logits.log_softmax(dim=-1)
        preds = ctc_greedy_decode(log_probs)
        all_preds.extend(preds)
        all_gts.extend(plates)
    srr = compute_srr(all_preds, all_gts)
    car = compute_car(all_preds, all_gts)
    return srr, car


if __name__ == "__main__":
    # Quick correctness check with known strings.
    preds = ["皖A12345", "沪B67890", "京C11111"]
    gts   = ["皖A12345", "沪B67891", "京C1111X"]
    print("SRR:", compute_srr(preds, gts))   # expect 1/3
    print("CAR:", compute_car(preds, gts))   # expect close to 1.0 (mostly 1 char off)
