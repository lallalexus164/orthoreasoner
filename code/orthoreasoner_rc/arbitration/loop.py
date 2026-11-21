import torch
from orthoreasoner_rc.arbitration.calibra import arbitrate_scores, choose_decisions, override_margin
from orthoreasoner_rc.constraints.compiler import compile_margins
from orthoreasoner_rc.ontology.typing import typed_scores
from torch import Tensor


class SelfCorrectingLoop:
    def __init__(self, beta: float, threshold: float, max_iterations: int) -> None:
        self.beta = beta
        self.threshold = threshold
        self.max_iterations = max_iterations

    def refine(self, posterior: Tensor, clinical: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        current = clinical.clone()
        final_scores = torch.empty(current.shape[0], 2, dtype=current.dtype, device=current.device)
        final_margin = torch.empty(current.shape[0], dtype=current.dtype, device=current.device)
        iterations = torch.zeros(current.shape[0], dtype=torch.long, device=current.device)
        for k in range(self.max_iterations + 1):
            margins = compile_margins(current)
            scores = arbitrate_scores(posterior, margins, self.beta)
            margin = override_margin(scores)
            final_scores = scores
            final_margin = margin
            pending = margin < self.threshold
            iterations[pending] = k
            if not bool(torch.any(pending)):
                break
            typed = typed_scores(current)
            inflammation = typed[:, :3].mean(dim=1)
            current[pending, 11] = torch.clamp(
                current[pending, 11] + 0.08 * inflammation[pending], 0.0, 1.0
            )
            current[pending, 12] = torch.clamp(
                current[pending, 12] + 0.05 * inflammation[pending], 0.0, 1.0
            )
            current[pending, 13] = torch.clamp(
                current[pending, 13] + 0.05 * inflammation[pending], 0.0, 1.0
            )
        decision = choose_decisions(posterior, compile_margins(current), self.beta)
        return decision, final_scores, final_margin
