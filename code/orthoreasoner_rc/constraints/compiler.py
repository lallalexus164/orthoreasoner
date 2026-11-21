import torch
from orthoreasoner_rc.core.types import ConstraintMargin, DecisionName, FeasibleSet
from torch import Tensor


def compile_margins(clinical: Tensor) -> Tensor:
    tear = clinical[:, 2]
    gout = clinical[:, 3]
    acuity = clinical[:, 6]
    dominant = clinical[:, 7]
    activity = clinical[:, 8]
    retraction = clinical[:, 9]
    bursal = clinical[:, 10]
    ases = clinical[:, 4]
    functional_deficit = torch.clamp(1.0 - ases, 0.0, 1.0)
    acute_surgery = 0.45 * (1.0 - acuity) + 0.35 * tear + 0.20 * functional_deficit
    dominant_demand = 0.25 * dominant + 0.35 * activity + 0.25 * tear + 0.15 * bursal
    retraction_stage = 0.5 * retraction + 0.3 * tear - 0.2 * gout
    conservative_fatty = 0.65 * gout + 0.20 * acuity + 0.15 * (1.0 - activity)
    conservative_low = 0.45 * (1.0 - tear) + 0.25 * acuity + 0.30 * (1.0 - functional_deficit)
    margins = torch.stack(
        [
            acute_surgery * 2.0 - 1.0,
            dominant_demand * 2.0 - 1.0,
            retraction_stage * 2.0 - 1.0,
            conservative_fatty * 2.0 - 1.0,
            conservative_low * 2.0 - 1.0,
        ],
        dim=1,
    )
    return margins


def feasible_from_margins(margins: Tensor) -> tuple[FeasibleSet, ...]:
    result: list[FeasibleSet] = []
    for row in margins:
        surgery = bool(torch.max(row[:3]) > -0.75)
        conservative = bool(torch.max(row[3:]) > -0.75)
        if not surgery and not conservative:
            conservative = True
        result.append(FeasibleSet(surgery=surgery, conservative=conservative))
    return tuple(result)


def named_margins(row: Tensor) -> tuple[ConstraintMargin, ...]:
    names: tuple[tuple[str, DecisionName, float], ...] = (
        ("acute_full_thickness_functional_deficit", "surgery", 1.0),
        ("dominant_high_demand_partial_or_full", "surgery", 0.8),
        ("retraction_structural_stage", "surgery", 0.7),
        ("advanced_fatty_infiltration", "conservative", 1.0),
        ("low_structural_burden", "conservative", 0.7),
    )
    return tuple(
        ConstraintMargin(name, decision, float(value), weight)
        for value, (name, decision, weight) in zip(row.tolist(), names, strict=True)
    )


def margin_support(margins: Tensor) -> Tensor:
    surgery = torch.max(margins[:, :3], dim=1).values
    conservative = torch.max(margins[:, 3:], dim=1).values
    return torch.stack([conservative, surgery], dim=1)
