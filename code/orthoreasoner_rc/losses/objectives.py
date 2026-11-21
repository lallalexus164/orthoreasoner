from typing import cast

import torch
from orthoreasoner_rc.arbitration.calibra import arbitrate_scores
from orthoreasoner_rc.constraints.compiler import compile_margins
from torch import Tensor, nn


class DecisionLoss(nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self, outputs: dict[str, Tensor], clinical: Tensor, decision: Tensor, favorable: Tensor
    ) -> Tensor:
        benefit = self.bce(outputs["benefit_logit"], favorable)
        diagnostic = self.bce(outputs["diagnostic_logit"], decision)
        margins = compile_margins(clinical)
        scores = arbitrate_scores(outputs["benefit"], margins, self.beta)
        arbitration = self.ce(scores, decision.long())
        return cast(Tensor, benefit + 0.5 * diagnostic + 0.5 * arbitration)


def brier_loss(probability: Tensor, target: Tensor) -> Tensor:
    return torch.mean((probability - target).square())


def calibration_penalty(probability: Tensor, target: Tensor, bins: int = 10) -> Tensor:
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probability.device)
    total = torch.zeros((), dtype=probability.dtype, device=probability.device)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probability >= lower) & (probability < upper)
        if bool(torch.any(mask)):
            total = (
                total
                + torch.abs(probability[mask].mean() - target[mask].mean()) * mask.float().mean()
            )
    return total
