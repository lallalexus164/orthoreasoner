from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class DataConfig:
    mode: str
    records: int
    image_size: int
    batch_size: int
    workers: int


@dataclass(frozen=True)
class ModelConfig:
    image_channels: int
    clinical_features: int
    hidden: int
    typed_nodes: int
    dropout: float


@dataclass(frozen=True)
class ArbitrationConfig:
    beta: float
    threshold: float
    max_iterations: int


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    checkpoint_dir: str


@dataclass(frozen=True)
class MetricConfig:
    bootstrap: int
    target_decision_auroc: float
    target_kappa: float


@dataclass(frozen=True)
class RunConfig:
    seed: int
    device: str
    precision: str
    data: DataConfig
    model: ModelConfig
    arbitration: ArbitrationConfig
    train: TrainConfig
    metrics: MetricConfig


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("configuration section must be a mapping")
    return value


def load_config(path: str | Path) -> RunConfig:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    cfg = _mapping(raw)
    data = _mapping(cfg["data"])
    model = _mapping(cfg["model"])
    arbitration = _mapping(cfg["arbitration"])
    train = _mapping(cfg["train"])
    metrics = _mapping(cfg["metrics"])
    return RunConfig(
        seed=int(cfg["seed"]),
        device=str(cfg["device"]),
        precision=str(cfg["precision"]),
        data=DataConfig(
            mode=str(data["mode"]),
            records=int(data["records"]),
            image_size=int(data["image_size"]),
            batch_size=int(data["batch_size"]),
            workers=int(data["workers"]),
        ),
        model=ModelConfig(
            image_channels=int(model["image_channels"]),
            clinical_features=int(model["clinical_features"]),
            hidden=int(model["hidden"]),
            typed_nodes=int(model["typed_nodes"]),
            dropout=float(model["dropout"]),
        ),
        arbitration=ArbitrationConfig(
            beta=float(arbitration["beta"]),
            threshold=float(arbitration["threshold"]),
            max_iterations=int(arbitration["max_iterations"]),
        ),
        train=TrainConfig(
            epochs=int(train["epochs"]),
            lr=float(train["lr"]),
            weight_decay=float(train["weight_decay"]),
            grad_clip=float(train["grad_clip"]),
            checkpoint_dir=str(train["checkpoint_dir"]),
        ),
        metrics=MetricConfig(
            bootstrap=int(metrics["bootstrap"]),
            target_decision_auroc=float(metrics["target_decision_auroc"]),
            target_kappa=float(metrics["target_kappa"]),
        ),
    )
