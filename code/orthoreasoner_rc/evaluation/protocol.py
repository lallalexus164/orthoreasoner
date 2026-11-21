import torch
from orthoreasoner_rc.arbitration.loop import SelfCorrectingLoop
from orthoreasoner_rc.core.config import RunConfig
from orthoreasoner_rc.core.randomness import set_seed
from orthoreasoner_rc.data.factory import make_loaders
from orthoreasoner_rc.metrics.bootstrap import bootstrap_auroc
from orthoreasoner_rc.metrics.classification import report
from orthoreasoner_rc.models.system import OrthoReasonerNet
from orthoreasoner_rc.training.engine import collect_predictions


def evaluate_config(config: RunConfig) -> dict[str, float]:
    set_seed(config.seed)
    device = torch.device(config.device)
    _, _, test_loader = make_loaders(config.data, config.seed)
    model = OrthoReasonerNet(
        config.model.image_channels,
        config.model.clinical_features,
        config.model.hidden,
        config.model.dropout,
    ).to(device)
    loop = SelfCorrectingLoop(
        config.arbitration.beta, config.arbitration.threshold, config.arbitration.max_iterations
    )
    score, prediction, target = collect_predictions(model, test_loader, loop, device)
    metric = report(score, prediction, target)
    center, low, high = bootstrap_auroc(score, target, config.metrics.bootstrap, config.seed)
    return {
        "auroc": metric.auroc,
        "accuracy": metric.accuracy,
        "sensitivity": metric.sensitivity,
        "specificity": metric.specificity,
        "kappa": metric.kappa,
        "brier": metric.brier,
        "bootstrap_auroc": center,
        "bootstrap_low": low,
        "bootstrap_high": high,
    }
