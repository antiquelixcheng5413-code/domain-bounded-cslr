from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cslr.contracts import SampleRecord
from cslr.data.gloss import encode_gloss_tokens


class LandmarkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, records: list[SampleRecord], feature_root: Path, label_to_index: dict[str, int]
    ) -> None:
        self.records = records
        self.feature_root = feature_root
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        path = self.feature_root / f"{record.sample_id}.npy"
        features = np.load(path).astype(np.float32)
        label = self.label_to_index[record.label]
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)


class GlossTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, records: list[SampleRecord], feature_root: Path, token_to_index: dict[str, int]
    ) -> None:
        self.records = records
        self.feature_root = feature_root
        self.token_to_index = token_to_index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        path = self.feature_root / f"{record.sample_id}.npy"
        features = np.load(path).astype(np.float32)
        target = encode_gloss_tokens(record.label, self.token_to_index)
        return torch.from_numpy(features), torch.tensor(target, dtype=torch.float32)
