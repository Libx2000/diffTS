from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from diffts.config import ensure_dirs, load_config  # noqa: E402
from diffts.dataset import OceanForecastDataset  # noqa: E402
from diffts.metrics import MetricAccumulator  # noqa: E402
from diffts.model import UNet3D  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "config.yaml"))
    parser.add_argument("--split", choices=("val", "test", "ood"), default="test")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    artifacts, runs = ensure_dirs(cfg)
    run_dir = runs / "deterministic"
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"未找到模型：{checkpoint}")
    clim_path = artifacts / "monthly_climatology.nc"
    if not clim_path.exists():
        raise FileNotFoundError("缺少 monthly_climatology.nc；请以 qc_mode=full 执行阶段一")

    ds = OceanForecastDataset(cfg, args.split)
    ec = cfg["evaluation"]
    loader = DataLoader(ds, batch_size=ec["batch_size"], shuffle=False, num_workers=ec["num_workers"], pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    mc = cfg["model"]
    model = UNet3D(ckpt["in_channels"], 2, mc["base_channels"], mc["levels"], mc["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    norm = np.load(artifacts / "normalization.npz")
    mean = norm["mean"].astype(np.float32)[:, :, None, None]
    std = norm["std"].astype(np.float32)[:, :, None, None]
    coords = np.load(artifacts / "coordinates.npz")
    mask = np.load(artifacts / "wet_mask.npy")
    clim_ds = xr.open_dataset(clim_path)
    climatology = np.stack([clim_ds["thetao"].values, clim_ds["so"].values], axis=1)  # month,var,D,H,W
    acc = MetricAccumulator(coords["lat"], mask)

    with torch.no_grad():
        for batch in tqdm(loader):
            prediction_norm = model(batch["x"].to(device)).cpu().numpy()
            prediction = prediction_norm * std[None] + mean[None]
            obs = batch["target_phys"].numpy()
            persistence = batch["last_phys"].numpy()
            leads = batch["lead"].numpy()
            months = batch["month"].numpy()
            for i in range(prediction.shape[0]):
                clim = climatology[int(months[i]) - 1]
                acc.update("persistence", int(leads[i]), persistence[i], obs[i], clim)
                acc.update("climatology", int(leads[i]), clim, obs[i], clim)
                acc.update("unet3d", int(leads[i]), prediction[i], obs[i], clim)

    clim_ds.close()
    result = acc.dataframe(coords["depth"])
    output = run_dir / f"metrics_{args.split}.csv"
    result.to_csv(output, index=False)
    summary = result.groupby(["model", "lead_day", "variable"])[["bias", "mae", "rmse", "acc"]].mean()
    print(summary.to_string())
    print(f"详细分层指标：{output}")


if __name__ == "__main__":
    main()

