from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from diffts.config import load_config, set_seed  # noqa: E402
from diffts.dataset import OceanForecastDataset  # noqa: E402
from diffts.model import UNet3D  # noqa: E402


def stratified_indices(dataset: OceanForecastDataset, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or maximum >= len(dataset):
        return np.arange(len(dataset))
    table = dataset.rows.copy()
    table["month"] = table.target.dt.month
    rng = np.random.default_rng(seed)
    chosen = []
    groups = list(table.groupby(["lead", "month"]).groups.values())
    quota = max(1, maximum // len(groups))
    for group in groups:
        values = np.asarray(list(group), dtype=np.int64)
        chosen.extend(rng.choice(values, size=min(quota, len(values)), replace=False).tolist())
    remaining = np.setdiff1d(np.arange(len(table)), np.asarray(chosen), assume_unique=False)
    if len(chosen) < maximum:
        chosen.extend(rng.choice(remaining, size=min(maximum - len(chosen), len(remaining)), replace=False).tolist())
    return np.asarray(chosen[:maximum], dtype=np.int64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT / "config.yaml"))
    p.add_argument("--max-samples", type=int, default=512, help="0表示使用全部训练样本")
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    cfg = load_config(args.config); set_seed(int(cfg["project"]["seed"]))
    sc = cfg["stage3"]
    checkpoint = Path(args.checkpoint or sc["deterministic_checkpoint"])
    output = Path(sc["residual_stats"]); output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    mc = cfg["model"]
    model = UNet3D(ckpt["in_channels"], 2, mc["base_channels"], mc["levels"], mc["dropout"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    dataset = OceanForecastDataset(cfg, "train")
    indices = stratified_indices(dataset, args.max_samples, int(cfg["project"]["seed"]))
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=1, shuffle=False, num_workers=0)
    mask = dataset.mask
    coords = np.load(Path(cfg["project"]["artifacts_dir"]) / "coordinates.npz")
    area = np.cos(np.deg2rad(coords["lat"])).clip(1e-6, None)[:, None]
    sums = np.zeros((2, mask.shape[0]), dtype=np.float64)
    sums2 = np.zeros_like(sums); weights = np.zeros_like(sums)
    with torch.inference_mode():
        for batch in tqdm(loader, desc="residual statistics"):
            pred = model(batch["x"].to(device)).cpu().numpy()[0]
            residual = batch["y"].numpy()[0] - pred
            for var in range(2):
                for depth in range(mask.shape[0]):
                    valid = mask[depth] & np.isfinite(residual[var, depth])
                    w = np.broadcast_to(area, valid.shape)[valid]
                    values = residual[var, depth][valid]
                    sums[var, depth] += np.sum(w * values)
                    sums2[var, depth] += np.sum(w * values**2)
                    weights[var, depth] += w.sum()
    mean = sums / weights
    std = np.sqrt(np.maximum(sums2 / weights - mean**2, 1e-8))
    np.savez_compressed(output, mean=mean.astype(np.float32), std=std.astype(np.float32),
                        samples=np.asarray([len(indices)]), checkpoint=np.asarray([str(checkpoint)]))
    metadata = {"samples": int(len(indices)), "full_training_samples": len(dataset),
                "checkpoint": str(checkpoint), "device": str(device), "output": str(output),
                "note": "按预报时效和月份分层抽样；正式论文训练前建议在GPU节点以--max-samples 0重算。"}
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
