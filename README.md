# OrthoReasoner: A Knowledge-Graph-Guided Agentic AI Synergizing Ontologies and Clinical Standards for Rotator Cuff Injury

This package contains the rotator cuff decision pipeline used for CALIBRA arbitration, inflammation-typed evidence routing, AAOS-style constraint margins, confidence-gated requery, training, and evaluation on private cohort manifests or synthetic fixtures.

## Installation

```bash
pip install -e .
```

```bash
conda env create -f environment.yml
conda activate orthoreasoner-rc
pip install -e .
```

```bash
docker build -t orthoreasoner-rc .
```

## Data

Dataset links are listed in `datasets.txt`. The private multicenter cohort is not included. Expected manifest columns are `record_id`, `mri_path`, `us_path`, `age`, `sex`, `tear_depth`, `goutallier`, `ases`, `constant`, `acuity_days`, `dominant`, `activity`, `retraction`, `bursal_side`, `hyperemia`, `effusion`, `edema`, `decision`, and `favorable`.

## Training

```bash
python -m orthoreasoner_rc.cli.train --config configs/experiments/main.yaml
```

The main configuration targets mixed MRI, power Doppler ultrasound, and clinical covariates. The manuscript reports cohort sizes and validation metrics but does not report optimizer, batch size, epoch count, or GPU hardware. Defaults in this package are declared in config files and can be overridden from the command line.

## Evaluation

```bash
python -m orthoreasoner_rc.cli.evaluate --config configs/experiments/main.yaml
```

Expected primary-arm values from the manuscript are decision AUROC 0.834 with 95 percent interval 0.807 to 0.860, decision kappa 0.84 with interval 0.80 to 0.88, diagnostic AUROC 0.985 with interval 0.978 to 0.991, and partial-thickness AUROC 0.933 with interval 0.908 to 0.955.

## Checkpoints

Trained weights are not included. Private cohort checkpoints can be exported with SHA-256 manifests using `python -m orthoreasoner_rc.cli.export --checkpoint checkpoints/best.pt --out exports`.

## Validation Commands

```bash
pytest -q
ruff check .
mypy --strict code/orthoreasoner_rc
```
