import argparse
import json
import sys

from orthoreasoner_rc.core.config import load_config
from orthoreasoner_rc.training.engine import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/smoke.yaml")
    args = parser.parse_args()
    result = run_training(load_config(args.config))
    payload = {
        "losses": result.losses,
        "checkpoint": result.checkpoint,
        "auroc": result.metrics.auroc,
        "kappa": result.metrics.kappa,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
