import torch
from orthoreasoner_rc.core.types import EvidenceNode
from orthoreasoner_rc.ontology.vocabulary import TERMS
from torch import Tensor


def typed_scores(clinical: Tensor) -> Tensor:
    tear = clinical[:, 2]
    gout = clinical[:, 3]
    acuity = clinical[:, 6]
    dominant = clinical[:, 7]
    activity = clinical[:, 8]
    retraction = clinical[:, 9]
    bursal = clinical[:, 10]
    hyperemia = clinical[:, 11]
    effusion = clinical[:, 12]
    edema = clinical[:, 13]
    scores = [
        edema,
        effusion,
        hyperemia,
        torch.clamp(1.0 - (tear - 0.45).abs() * 2.0, 0.0, 1.0),
        torch.clamp((tear - 0.5) * 2.0, 0.0, 1.0),
        torch.clamp(1.0 - (gout - 0.55).abs() * 3.0, 0.0, 1.0),
        torch.clamp((gout - 0.7) * 3.0, 0.0, 1.0),
        torch.clamp(1.0 - acuity, 0.0, 1.0),
        torch.clamp(acuity, 0.0, 1.0),
        torch.clamp((dominant + activity) * 0.5, 0.0, 1.0),
        bursal,
        retraction,
    ]
    return torch.stack(scores, dim=1)


def nodes_from_scores(scores: Tensor) -> tuple[EvidenceNode, ...]:
    if scores.ndim != 1:
        raise ValueError("scores must be one dimensional")
    nodes: list[EvidenceNode] = []
    for value, term in zip(scores.tolist(), TERMS, strict=True):
        nodes.append(
            EvidenceNode(
                term.key, term.anatomy, term.phase, float(value), min(1.0, abs(float(value)))
            )
        )
    return tuple(nodes)


def likelihood_ratio(scores: Tensor, polarity: Tensor) -> Tensor:
    centered = scores * 2.0 - 1.0
    return centered * polarity


def ontology_polarity(device: torch.device | str) -> Tensor:
    return torch.tensor([term.polarity for term in TERMS], dtype=torch.float32, device=device)
