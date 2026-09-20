from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .io import YearFileStore


class OceanForecastDataset(Dataset):
    def __init__(self, cfg: dict, split: str):
        artifacts = Path(cfg["project"]["artifacts_dir"])
        index = pd.read_csv(artifacts / "sample_index.csv", parse_dates=["input_end", "target"])
        self.rows = index[index["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"样本索引中没有 split={split}")
        norm = np.load(artifacts / "normalization.npz")
        self.mean = norm["mean"].astype(np.float32)[:, :, None, None]
        self.std = norm["std"].astype(np.float32)[:, :, None, None]
        self.mask = np.load(artifacts / "wet_mask.npy").astype(bool)
        self.history = int(cfg["data"]["history_days"])
        self.max_lead = max(int(x) for x in cfg["data"]["lead_days"])
        self.store = YearFileStore(cfg)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[idx]
        input_end = pd.Timestamp(row.input_end)
        dates = pd.date_range(input_end - pd.Timedelta(days=self.history - 1), input_end, freq="D")
        history_phys = self.store.read_state(dates)
        target_phys = self.store.read_state([pd.Timestamp(row.target)])[0]

        history_norm = (history_phys - self.mean[None]) / self.std[None]
        target_norm = (target_phys - self.mean) / self.std
        history_norm = np.nan_to_num(history_norm, nan=0.0, posinf=0.0, neginf=0.0)
        target_norm = np.nan_to_num(target_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # 历史时间和变量合并为通道；三维卷积仍保留 depth/lat/lon。
        channels = history_norm.reshape(self.history * 2, *history_norm.shape[2:])
        d, h, w = target_norm.shape[1:]
        lead = np.full((1, d, h, w), float(row.lead) / self.max_lead, dtype=np.float32)
        month_angle = 2.0 * np.pi * (pd.Timestamp(row.target).month - 1) / 12.0
        month_sin = np.full((1, d, h, w), np.sin(month_angle), dtype=np.float32)
        month_cos = np.full((1, d, h, w), np.cos(month_angle), dtype=np.float32)
        mask_channel = self.mask.astype(np.float32)[None]
        model_input = np.concatenate([channels, lead, month_sin, month_cos, mask_channel], axis=0)

        return {
            "x": torch.from_numpy(model_input),
            "y": torch.from_numpy(target_norm),
            "last_phys": torch.from_numpy(np.nan_to_num(history_phys[-1], nan=0.0)),
            "target_phys": torch.from_numpy(np.nan_to_num(target_phys, nan=0.0)),
            "lead": torch.tensor(int(row.lead), dtype=torch.long),
            "month": torch.tensor(pd.Timestamp(row.target).month, dtype=torch.long),
            "target_ordinal": torch.tensor(pd.Timestamp(row.target).toordinal(), dtype=torch.long),
        }

