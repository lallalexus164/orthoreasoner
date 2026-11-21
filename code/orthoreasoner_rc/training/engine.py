from pathlib import Path

import torch
from orthoreasoner_rc.arbitration.loop import SelfCorrectingLoop
from orthoreasoner_rc.core.config import RunConfig
from orthoreasoner_rc.core.randomness import set_seed
from orthoreasoner_rc.core.types import BatchView, TrainingResult
from orthoreasoner_rc.data.factory import make_loaders
from orthoreasoner_rc.losses.objectives import DecisionLoss
from orthoreasoner_rc.metrics.classification import report
from orthoreasoner_rc.models.system import OrthoReasonerNet
from orthoreasoner_rc.training.checkpoint import save_checkpoint
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader


def move_batch(batch: BatchView, device: torch.device) -> BatchView:
    return BatchView(
        images=batch.images.to(device),
        clinical=batch.clinical.to(device),
        decision=batch.decision.to(device),
        favorable=batch.favorable.to(device),
    )


def train_epoch(
    model: OrthoReasonerNet,
    loader: DataLoader[object],
    loss_fn: DecisionLoss,
    optimizer: AdamW,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for raw in loader:
        batch = move_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch.images, batch.clinical)
        loss = loss_fn(outputs, batch.clinical, batch.decision, batch.favorable)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item()) * batch.images.shape[0]
        count += batch.images.shape[0]
    return total / max(1, count)


def collect_predictions(
    model: OrthoReasonerNet,
    loader: DataLoader[object],
    loop: SelfCorrectingLoop,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    model.eval()
    scores: list[Tensor] = []
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            outputs = model(batch.images, batch.clinical)
            decision, _, _ = loop.refine(outputs["benefit"], batch.clinical)
            scores.append(outputs["benefit"].detach().cpu())
            predictions.append(decision.detach().cpu())
            targets.append(batch.decision.detach().cpu())
    return torch.cat(scores), torch.cat(predictions), torch.cat(targets)


def run_training(config: RunConfig) -> TrainingResult:
    set_seed(config.seed)
    device = torch.device(config.device)
    train_loader, valid_loader, _ = make_loaders(config.data, config.seed)
    model = OrthoReasonerNet(
        config.model.image_channels,
        config.model.clinical_features,
        config.model.hidden,
        config.model.dropout,
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    loss_fn = DecisionLoss(config.arbitration.beta)
    losses: list[float] = []
    for _ in range(config.train.epochs):
        losses.append(
            train_epoch(model, train_loader, loss_fn, optimizer, device, config.train.grad_clip)
        )
    loop = SelfCorrectingLoop(
        config.arbitration.beta, config.arbitration.threshold, config.arbitration.max_iterations
    )
    score, prediction, target = collect_predictions(model, valid_loader, loop, device)
    metric = report(score, prediction, target)
    checkpoint = save_checkpoint(
        Path(config.train.checkpoint_dir) / "best.pt",
        model,
        optimizer,
        config.seed,
        config.train.epochs,
    )
    return TrainingResult(tuple(losses), checkpoint, metric)
