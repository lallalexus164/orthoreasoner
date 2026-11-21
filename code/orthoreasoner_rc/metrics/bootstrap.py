import torch
from orthoreasoner_rc.metrics.classification import auroc
from torch import Tensor


def bootstrap_auroc(
    score: Tensor, target: Tensor, samples: int, seed: int
) -> tuple[float, float, float]:
    generator = torch.Generator().manual_seed(seed)
    n = score.numel()
    values: list[float] = []
    for _ in range(samples):
        index = torch.randint(0, n, (n,), generator=generator)
        values.append(auroc(score[index], target[index]))
    if not values:
        center = auroc(score, target)
        return center, center, center
    ordered = sorted(values)
    low = ordered[int(0.025 * (len(ordered) - 1))]
    high = ordered[int(0.975 * (len(ordered) - 1))]
    return auroc(score, target), low, high
