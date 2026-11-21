import torch
from orthoreasoner_rc.constraints.compiler import compile_margins, feasible_from_margins
from orthoreasoner_rc.ontology.typing import typed_scores


def test_typed_scores_shape() -> None:
    clinical = torch.rand(4, 14)
    scores = typed_scores(clinical)
    assert scores.shape == (4, 12)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


def test_constraint_feasible_set() -> None:
    clinical = torch.rand(3, 14)
    margins = compile_margins(clinical)
    feasible = feasible_from_margins(margins)
    assert len(feasible) == 3
    assert all(item.surgery or item.conservative for item in feasible)
