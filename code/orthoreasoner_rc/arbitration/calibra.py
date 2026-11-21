import math

import torch
from orthoreasoner_rc.constraints.compiler import (
    feasible_from_margins,
    margin_support,
    named_margins,
)
from orthoreasoner_rc.core.types import ArbitrationTrace, DecisionName
from orthoreasoner_rc.ontology.typing import nodes_from_scores, typed_scores
from torch import Tensor


def posterior_log_odds(posterior: Tensor) -> Tensor:
    eps = torch.finfo(posterior.dtype).eps
    clipped = posterior.clamp(eps, 1.0 - eps)
    return torch.log(clipped / (1.0 - clipped))


def arbitrate_scores(posterior: Tensor, margins: Tensor, beta: float) -> Tensor:
    log_odds = posterior_log_odds(posterior)
    support = margin_support(margins)
    conservative = -log_odds + beta * support[:, 0]
    surgery = log_odds + beta * support[:, 1]
    return torch.stack([conservative, surgery], dim=1)


def choose_decisions(posterior: Tensor, margins: Tensor, beta: float) -> Tensor:
    scores = arbitrate_scores(posterior, margins, beta)
    feasible = feasible_from_margins(margins)
    adjusted = scores.clone()
    for i, item in enumerate(feasible):
        if not item.conservative:
            adjusted[i, 0] = -math.inf
        if not item.surgery:
            adjusted[i, 1] = -math.inf
    return torch.argmax(adjusted, dim=1)


def override_margin(scores: Tensor) -> Tensor:
    top = torch.topk(scores, k=2, dim=1).values
    return top[:, 0] - top[:, 1]


def trace_one(
    posterior: float, margins: Tensor, clinical: Tensor, beta: float, iterations: int
) -> ArbitrationTrace:
    p = torch.tensor([posterior], dtype=torch.float32)
    m = margins.unsqueeze(0)
    scores = arbitrate_scores(p, m, beta)[0]
    choice = int(torch.argmax(scores).item())
    decision: DecisionName = "surgery" if choice == 1 else "conservative"
    named = named_margins(margins)
    aligned = [item for item in named if item.decision == decision]
    binding = max(aligned, key=lambda item: item.margin * item.weight) if aligned else named[0]
    evidence = nodes_from_scores(typed_scores(clinical.unsqueeze(0))[0])
    sorted_scores = torch.sort(scores, descending=True).values
    margin = float((sorted_scores[0] - sorted_scores[1]).item())
    return ArbitrationTrace(decision, binding.name, margin, posterior, iterations, evidence)
