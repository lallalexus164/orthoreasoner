from dataclasses import dataclass

import torch
from orthoreasoner_rc.core.randomness import seeded_generator
from orthoreasoner_rc.core.types import BatchView
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticSpec:
    records: int
    image_size: int
    seed: int


class SyntheticRotatorCuffDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, spec: SyntheticSpec) -> None:
        self.spec = spec
        generator = seeded_generator(spec.seed)
        n = spec.records
        h = spec.image_size
        self.clinical = torch.rand(n, 14, generator=generator)
        age = self.clinical[:, 0]
        tear = self.clinical[:, 2]
        gout = self.clinical[:, 3]
        acuity = self.clinical[:, 6]
        activity = self.clinical[:, 8]
        edema = self.clinical[:, 13]
        hyperemia = self.clinical[:, 11]
        effusion = self.clinical[:, 12]
        raw = (
            1.8 * tear
            + 1.2 * activity
            + 0.9 * edema
            + 0.7 * hyperemia
            - 1.1 * gout
            - 0.6 * age
            - 0.3 * acuity
        )
        prob = torch.sigmoid(raw)
        self.decision = (prob > 0.52).float()
        favorable_raw = 1.4 * prob + 0.5 * effusion - 0.4 * gout + 0.25 * activity
        self.favorable = (torch.sigmoid(favorable_raw) > 0.58).float()
        base = torch.randn(n, 2, h, h, generator=generator) * 0.04
        grid = torch.linspace(-1.0, 1.0, h)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        blob = torch.exp(-(xx.square() + yy.square()) * 3.0)
        self.images = base
        self.images[:, 0] += tear[:, None, None] * blob
        self.images[:, 1] += (edema + hyperemia)[:, None, None] * blob * 0.5

    def __len__(self) -> int:
        return self.spec.records

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.images[index],
            self.clinical[index],
            self.decision[index],
            self.favorable[index],
        )


def collate_batch(items: list[tuple[Tensor, Tensor, Tensor, Tensor]]) -> BatchView:
    images = torch.stack([item[0] for item in items], dim=0)
    clinical = torch.stack([item[1] for item in items], dim=0)
    decision = torch.stack([item[2] for item in items], dim=0)
    favorable = torch.stack([item[3] for item in items], dim=0)
    return BatchView(images=images, clinical=clinical, decision=decision, favorable=favorable)
