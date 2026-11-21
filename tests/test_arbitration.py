import torch
from orthoreasoner_rc.arbitration.calibra import choose_decisions, trace_one
from orthoreasoner_rc.constraints.compiler import compile_margins


def test_arbitration_returns_binary_decision() -> None:
    clinical = torch.rand(5, 14)
    posterior = torch.linspace(0.2, 0.8, 5)
    margins = compile_margins(clinical)
    decisions = choose_decisions(posterior, margins, 0.7)
    assert decisions.shape == (5,)
    assert set(decisions.tolist()).issubset({0, 1})


def test_trace_contains_binding_constraint() -> None:
    clinical = torch.rand(14)
    margins = compile_margins(clinical.unsqueeze(0))[0]
    trace = trace_one(0.7, margins, clinical, 0.7, 1)
    assert trace.decision in {"surgery", "conservative"}
    assert len(trace.binding_constraint) > 0
    assert len(trace.evidence) == 12
