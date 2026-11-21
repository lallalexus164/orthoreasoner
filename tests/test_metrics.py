import torch
from orthoreasoner_rc.metrics.classification import auroc, cohen_kappa, sensitivity_specificity


def test_metrics_known_values() -> None:
    score = torch.tensor([0.1, 0.2, 0.8, 0.9])
    target = torch.tensor([0, 0, 1, 1])
    pred = (score > 0.5).long()
    assert auroc(score, target) == 1.0
    assert cohen_kappa(pred, target) == 1.0
    assert sensitivity_specificity(pred, target) == (1.0, 1.0)
