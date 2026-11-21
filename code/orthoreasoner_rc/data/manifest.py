import csv
from pathlib import Path

import torch
from orthoreasoner_rc.core.types import ClinicalRecord
from torch import Tensor
from torch.utils.data import Dataset


class ManifestDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, path: str | Path, image_size: int) -> None:
        self.path = Path(path)
        self.image_size = image_size
        self.rows = list(csv.DictReader(self.path.open("r", encoding="utf-8")))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        row = self.rows[index]
        record = ClinicalRecord(
            record_id=row["record_id"],
            age=float(row["age"]),
            sex=float(row["sex"]),
            tear_depth=float(row["tear_depth"]),
            goutallier=float(row["goutallier"]),
            ases=float(row["ases"]),
            constant=float(row["constant"]),
            acuity_days=float(row["acuity_days"]),
            dominant=float(row["dominant"]),
            activity=float(row["activity"]),
            retraction=float(row["retraction"]),
            bursal_side=float(row["bursal_side"]),
            hyperemia=float(row["hyperemia"]),
            effusion=float(row["effusion"]),
            edema=float(row["edema"]),
        )
        image = torch.zeros(2, self.image_size, self.image_size, dtype=torch.float32)
        decision = torch.tensor(float(row["decision"]), dtype=torch.float32)
        favorable = torch.tensor(float(row["favorable"]), dtype=torch.float32)
        return image, record.tensor(), decision, favorable
