from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch
from torch import Tensor

DecisionName = Literal["surgery", "conservative"]


class Decision(Enum):
    surgery = 1
    conservative = 0


@dataclass(frozen=True)
class ClinicalRecord:
    record_id: str
    age: float
    sex: float
    tear_depth: float
    goutallier: float
    ases: float
    constant: float
    acuity_days: float
    dominant: float
    activity: float
    retraction: float
    bursal_side: float
    hyperemia: float
    effusion: float
    edema: float

    def tensor(self) -> Tensor:
        values = [
            self.age,
            self.sex,
            self.tear_depth,
            self.goutallier,
            self.ases,
            self.constant,
            self.acuity_days,
            self.dominant,
            self.activity,
            self.retraction,
            self.bursal_side,
            self.hyperemia,
            self.effusion,
            self.edema,
        ]
        return torch.tensor(values, dtype=torch.float32)


@dataclass(frozen=True)
class EvidenceNode:
    name: str
    anatomy: str
    phase: str
    value: float
    confidence: float


@dataclass(frozen=True)
class ConstraintMargin:
    name: str
    decision: DecisionName
    margin: float
    weight: float


@dataclass(frozen=True)
class FeasibleSet:
    surgery: bool
    conservative: bool

    def contains(self, decision: DecisionName) -> bool:
        if decision == "surgery":
            return self.surgery
        return self.conservative

    def decisions(self) -> tuple[DecisionName, ...]:
        items: list[DecisionName] = []
        if self.surgery:
            items.append("surgery")
        if self.conservative:
            items.append("conservative")
        return tuple(items)


@dataclass(frozen=True)
class ArbitrationTrace:
    decision: DecisionName
    binding_constraint: str
    margin: float
    posterior: float
    iterations: int
    evidence: tuple[EvidenceNode, ...]


@dataclass(frozen=True)
class BatchView:
    images: Tensor
    clinical: Tensor
    decision: Tensor
    favorable: Tensor


@dataclass(frozen=True)
class MetricReport:
    auroc: float
    accuracy: float
    sensitivity: float
    specificity: float
    kappa: float
    brier: float


@dataclass(frozen=True)
class TrainingResult:
    losses: tuple[float, ...]
    checkpoint: str
    metrics: MetricReport


@dataclass(frozen=True)
class CohortSplit:
    train: Sequence[int]
    valid: Sequence[int]
    test: Sequence[int]


@dataclass(frozen=True)
class DatasetCard:
    name: str
    access: str
    url: str
    license: str
    role: str


@dataclass(frozen=True)
class CalibrationState:
    temperature: float
    bias: float

    def apply(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(logits / self.temperature + self.bias)


def decision_to_index(decision: DecisionName) -> int:
    return 1 if decision == "surgery" else 0


def index_to_decision(index: int) -> DecisionName:
    return "surgery" if index == 1 else "conservative"


def stack_records(records: Iterable[ClinicalRecord]) -> Tensor:
    rows = [record.tensor() for record in records]
    if not rows:
        return torch.empty(0, 14, dtype=torch.float32)
    return torch.stack(rows, dim=0)
