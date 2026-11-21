import torch

from orthoreasoner_rc.arbitration.loop import SelfCorrectingLoop
from orthoreasoner_rc.constraints.compiler import compile_margins
from orthoreasoner_rc.core.config import RunConfig
from orthoreasoner_rc.models.system import OrthoReasonerNet


def build_system(config: RunConfig) -> tuple[OrthoReasonerNet, SelfCorrectingLoop]:
    model = OrthoReasonerNet(
        config.model.image_channels,
        config.model.clinical_features,
        config.model.hidden,
        config.model.dropout,
    )
    loop = SelfCorrectingLoop(
        config.arbitration.beta, config.arbitration.threshold, config.arbitration.max_iterations
    )
    return model, loop


def run_record(
    model: OrthoReasonerNet, loop: SelfCorrectingLoop, image: torch.Tensor, clinical: torch.Tensor
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        outputs = model(image.unsqueeze(0), clinical.unsqueeze(0))
        decision, scores, margin = loop.refine(outputs["benefit"], clinical.unsqueeze(0))
        margins = compile_margins(clinical.unsqueeze(0))
        return {"decision": decision, "scores": scores, "margin": margin, "constraints": margins}
