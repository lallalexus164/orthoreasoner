import numpy as np
import torch
from orthoreasoner_rc.core.types import MetricReport
from torch import Tensor


def binary_counts(prediction: Tensor, target: Tensor) -> tuple[int, int, int, int]:
    pred = prediction.long().cpu()
    tgt = target.long().cpu()
    tp = int(((pred == 1) & (tgt == 1)).sum().item())
    tn = int(((pred == 0) & (tgt == 0)).sum().item())
    fp = int(((pred == 1) & (tgt == 0)).sum().item())
    fn = int(((pred == 0) & (tgt == 1)).sum().item())
    return tp, tn, fp, fn


def accuracy(prediction: Tensor, target: Tensor) -> float:
    return float((prediction.long() == target.long()).float().mean().item())


def sensitivity_specificity(prediction: Tensor, target: Tensor) -> tuple[float, float]:
    tp, tn, fp, fn = binary_counts(prediction, target)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return sensitivity, specificity


def cohen_kappa(prediction: Tensor, target: Tensor) -> float:
    tp, tn, fp, fn = binary_counts(prediction, target)
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    observed = (tp + tn) / total
    pred_pos = (tp + fp) / total
    pred_neg = (tn + fn) / total
    tgt_pos = (tp + fn) / total
    tgt_neg = (tn + fp) / total
    expected = pred_pos * tgt_pos + pred_neg * tgt_neg
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def auroc(score: Tensor, target: Tensor) -> float:
    y = target.detach().cpu().numpy().astype(np.int64)
    s = score.detach().cpu().numpy().astype(np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    total = float(len(pos) * len(neg))
    for value in pos:
        wins += float(np.sum(value > neg))
        wins += 0.5 * float(np.sum(value == neg))
    return wins / total


def brier(score: Tensor, target: Tensor) -> float:
    return float(torch.mean((score - target.float()).square()).item())


def report(score: Tensor, prediction: Tensor, target: Tensor) -> MetricReport:
    sens, spec = sensitivity_specificity(prediction, target)
    return MetricReport(
        auroc=auroc(score, target),
        accuracy=accuracy(prediction, target),
        sensitivity=sens,
        specificity=spec,
        kappa=cohen_kappa(prediction, target),
        brier=brier(score, target),
    )
