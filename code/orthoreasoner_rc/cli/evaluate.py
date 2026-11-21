import argparse
import json
import sys

from orthoreasoner_rc.core.config import load_config
from orthoreasoner_rc.evaluation.protocol import evaluate_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/smoke.yaml")
    args = parser.parse_args()
    sys.stdout.write(json.dumps(evaluate_config(load_config(args.config)), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
