from typing import cast

import torch
from orthoreasoner_rc.models.blocks import ClinicalEncoder, ConvStem, FusionGate, TemperatureHead
from orthoreasoner_rc.ontology.typing import ontology_polarity, typed_scores
from torch import Tensor, nn


class OrthoReasonerNet(nn.Module):
    def __init__(
        self, image_channels: int, clinical_features: int, hidden: int, dropout: float
    ) -> None:
        super().__init__()
        self.vision = ConvStem(image_channels, hidden)
        self.clinical = ClinicalEncoder(clinical_features, hidden, dropout)
        self.fusion = FusionGate(hidden)
        self.posterior = TemperatureHead(hidden)
        self.diagnostic = nn.Linear(hidden, 1)
        self.typed_projection = nn.Linear(12, hidden)

    def encode(self, image: Tensor, clinical: Tensor) -> Tensor:
        vision = self.vision(image)
        clin = self.clinical(clinical)
        typed = typed_scores(clinical)
        polarity = ontology_polarity(clinical.device)
        typed_feature = cast(Tensor, self.typed_projection(typed * polarity))
        return cast(Tensor, self.fusion(vision, clin + typed_feature))

    def forward(self, image: Tensor, clinical: Tensor) -> dict[str, Tensor]:
        features = self.encode(image, clinical)
        benefit_logit = self.posterior(features)
        diagnostic_logit = self.diagnostic(features).squeeze(1)
        return {
            "benefit_logit": benefit_logit,
            "benefit": torch.sigmoid(benefit_logit),
            "diagnostic_logit": diagnostic_logit,
            "diagnostic": torch.sigmoid(diagnostic_logit),
            "features": features,
        }
