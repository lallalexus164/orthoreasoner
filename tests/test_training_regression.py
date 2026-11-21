from pathlib import Path

from orthoreasoner_rc.core.config import load_config
from orthoreasoner_rc.training.engine import run_training


def test_training_regression(tmp_path: Path) -> None:
    config = load_config("configs/experiments/smoke.yaml")
    patched = config.__class__(
        seed=config.seed,
        device=config.device,
        precision=config.precision,
        data=config.data,
        model=config.model,
        arbitration=config.arbitration,
        train=config.train.__class__(
            epochs=2,
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
            grad_clip=config.train.grad_clip,
            checkpoint_dir=str(tmp_path),
        ),
        metrics=config.metrics,
    )
    result = run_training(patched)
    assert len(result.losses) == 2
    assert result.losses[-1] <= result.losses[0] + 0.2
    assert Path(result.checkpoint).exists()
