import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path, model: nn.Module, optimizer: Optimizer, seed: int, epoch: int
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "seed": seed,
        "epoch": epoch,
    }
    torch.save(payload, tmp)
    os.replace(tmp, target)
    return str(target)


def load_checkpoint(
    path: str | Path, model: nn.Module, optimizer: Optimizer | None = None
) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload["seed"]), int(payload["epoch"])
