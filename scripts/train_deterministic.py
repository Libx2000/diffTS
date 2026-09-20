from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from diffts.config import ensure_dirs, load_config, set_seed  # noqa: E402
from diffts.dataset import OceanForecastDataset  # noqa: E402
from diffts.losses import horizontal_gradient_loss, make_spatial_weight, weighted_mse  # noqa: E402
from diffts.model import UNet3D  # noqa: E402


def run_epoch(model, loader, optimizer, scaler, device, weight, grad_weight, amp, train):
    model.train(train)
    total = 0.0
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                pred = model(x)
                loss = weighted_mse(pred, y, weight)
                if grad_weight > 0:
                    loss = loss + grad_weight * horizontal_gradient_loss(pred, y, weight)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            total += float(loss.detach()) * x.shape[0]
            count += x.shape[0]
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    artifacts, runs = ensure_dirs(cfg)
    set_seed(int(cfg["project"]["seed"]))
    run_dir = runs / "deterministic"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = OceanForecastDataset(cfg, "train")
    val_ds = OceanForecastDataset(cfg, "val")
    tc = cfg["training"]
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, num_workers=tc["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False, num_workers=tc["num_workers"], pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 2 * int(cfg["data"]["history_days"]) + 4
    mc = cfg["model"]
    model = UNet3D(in_channels, 2, mc["base_channels"], mc["levels"], mc["dropout"]).to(device)
    optimizer = AdamW(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=tc["amp"] and device.type == "cuda")
    coords = np.load(artifacts / "coordinates.npz")
    mask = torch.from_numpy(np.load(artifacts / "wet_mask.npy").astype(np.float32))
    weight = make_spatial_weight(torch.from_numpy(coords["lat"].astype(np.float32)), mask, device)

    best = math.inf
    patience = 0
    history = []
    for epoch in range(1, int(tc["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, weight, tc["gradient_loss_weight"], tc["amp"], True)
        val_loss = run_epoch(model, val_loader, optimizer, scaler, device, weight, tc["gradient_loss_weight"], tc["amp"], False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
            "in_channels": in_channels,
        }
        torch.save(state, run_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            patience = 0
            torch.save(state, run_dir / "best.pt")
        else:
            patience += 1
        print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} best={best:.6f}")
        if patience >= int(tc["early_stopping_patience"]):
            print("触发早停。")
            break
    print(f"训练完成。最佳模型：{run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

