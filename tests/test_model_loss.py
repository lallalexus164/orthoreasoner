import torch
from orthoreasoner_rc.losses.objectives import DecisionLoss
from orthoreasoner_rc.models.system import OrthoReasonerNet


def test_model_and_loss_backward() -> None:
    model = OrthoReasonerNet(2, 14, 24, 0.0)
    image = torch.rand(6, 2, 16, 16)
    clinical = torch.rand(6, 14)
    decision = torch.randint(0, 2, (6,), dtype=torch.float32)
    favorable = torch.randint(0, 2, (6,), dtype=torch.float32)
    outputs = model(image, clinical)
    loss = DecisionLoss(0.7)(outputs, clinical, decision, favorable)
    loss.backward()
    assert float(loss.item()) > 0.0
