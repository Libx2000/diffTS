from __future__ import annotations

import argparse
import copy
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
from diffts.diffusion import DiffusionSchedule  # noqa: E402
from diffts.diffusion_model import ConditionalResidualUNet3D  # noqa: E402
from diffts.model import UNet3D  # noqa: E402
from diffts.stage3_utils import spatial_coordinate_channels  # noqa: E402


def load_deterministic(cfg, device):
    path = Path(cfg["stage3"]["deterministic_checkpoint"])
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mc = cfg["model"]
    model = UNet3D(ckpt["in_channels"], 2, mc["base_channels"], mc["levels"], mc["dropout"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval().requires_grad_(False)
    return model


def update_ema(ema, model, decay):
    with torch.no_grad():
        for e, p in zip(ema.parameters(), model.parameters()): e.mul_(decay).add_(p, alpha=1-decay)
        for e, p in zip(ema.buffers(), model.buffers()): e.copy_(p)


def epoch_pass(model, deterministic, schedule, loader, optimizer, scaler, device, residual_mean,
               residual_std, spatial_weight, coordinates, cfg, train, ema=None):
    model.train(train); total = 0.0; count = 0
    accum = int(cfg["gradient_accumulation"]); clip = float(cfg["grad_clip_norm"])
    if train: optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, batch in enumerate(tqdm(loader, leave=False), 1):
            x = batch["x"].to(device, non_blocking=True); y = batch["y"].to(device, non_blocking=True)
            with torch.no_grad(): det = deterministic(x)
            clean = (y - det - residual_mean) / residual_std
            t = torch.randint(0, schedule.steps, (x.shape[0],), device=device)
            noise = torch.randn_like(clean); noisy = schedule.q_sample(clean, t, noise)
            condition = torch.cat([x, det, coordinates.expand(x.shape[0], -1, -1, -1, -1)], dim=1)
            with torch.autocast(device_type=device.type, enabled=bool(cfg["amp"]) and device.type == "cuda"):
                predicted = model(noisy, condition, t)
                loss = ((predicted-noise)**2 * spatial_weight).sum() / (spatial_weight.sum()*x.shape[0]*2)
                scaled_loss = loss / accum
            if train:
                scaler.scale(scaled_loss).backward()
                if step % accum == 0 or step == len(loader):
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    if ema is not None: update_ema(ema, model, float(cfg["ema_decay"]))
            total += float(loss.detach()) * x.shape[0]; count += x.shape[0]
    return total / max(count, 1)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default=str(PROJECT/"config.yaml")); p.add_argument("--resume",default=None); args=p.parse_args()
    cfg=load_config(args.config); set_seed(int(cfg["project"]["seed"])); artifacts,runs=ensure_dirs(cfg); sc=cfg["stage3"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("阶段三全量训练需要CUDA GPU，请在超算GPU节点运行。")
    train_ds=OceanForecastDataset(cfg,"train"); val_ds=OceanForecastDataset(cfg,"val")
    train_loader=DataLoader(train_ds,batch_size=int(sc["batch_size"]),shuffle=True,num_workers=int(sc["num_workers"]),pin_memory=True,persistent_workers=int(sc["num_workers"])>0)
    val_loader=DataLoader(val_ds,batch_size=int(sc["batch_size"]),shuffle=False,num_workers=int(sc["num_workers"]),pin_memory=True,persistent_workers=int(sc["num_workers"])>0)
    deterministic=load_deterministic(cfg,device); condition_channels=deterministic.encoders[0].net[0].conv.in_channels+2+4
    model=ConditionalResidualUNet3D(condition_channels,2,int(sc["base_channels"]),int(sc["levels"]),float(sc["dropout"]),int(sc["time_embedding_dim"])).to(device)
    ema=copy.deepcopy(model).eval().requires_grad_(False); schedule=DiffusionSchedule(int(sc["diffusion_steps"]),sc["beta_schedule"]).to(device)
    optimizer=AdamW(model.parameters(),lr=float(sc["learning_rate"]),weight_decay=float(sc["weight_decay"])); scaler=torch.amp.GradScaler("cuda",enabled=bool(sc["amp"]))
    stats=np.load(sc["residual_stats"]); residual_mean=torch.from_numpy(stats["mean"])[None,:,:,None,None].to(device); residual_std=torch.from_numpy(stats["std"])[None,:,:,None,None].to(device)
    coords=np.load(artifacts/"coordinates.npz"); mask=torch.from_numpy(np.load(artifacts/"wet_mask.npy").astype(np.float32)).to(device)
    area=torch.cos(torch.deg2rad(torch.from_numpy(coords["lat"].astype(np.float32)))).clamp_min(1e-6).to(device)
    spatial_weight=(mask*area[None,:,None])[None,None]
    coordinate_channels=spatial_coordinate_channels(artifacts,device)
    run_dir=runs/sc["run_name"]; run_dir.mkdir(parents=True,exist_ok=True); start=1; best=math.inf; patience=0; history=[]
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"]); optimizer.load_state_dict(state["optimizer"]); start=int(state["epoch"])+1; best=float(state.get("best_val",math.inf)); history_path=run_dir/"history.csv"; history=pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    for epoch in range(start,int(sc["epochs"])+1):
        tr=epoch_pass(model,deterministic,schedule,train_loader,optimizer,scaler,device,residual_mean,residual_std,spatial_weight,coordinate_channels,sc,True,ema)
        va=epoch_pass(ema,deterministic,schedule,val_loader,optimizer,scaler,device,residual_mean,residual_std,spatial_weight,coordinate_channels,sc,False)
        history.append({"epoch":epoch,"train_noise_mse":tr,"val_noise_mse":va}); pd.DataFrame(history).to_csv(run_dir/"history.csv",index=False)
        state={"epoch":epoch,"model":model.state_dict(),"ema":ema.state_dict(),"optimizer":optimizer.state_dict(),"val_loss":va,"best_val":min(best,va),"config":cfg,"condition_channels":condition_channels}
        torch.save(state,run_dir/"last.pt")
        if va<best: best=va; patience=0; state["best_val"]=best; torch.save(state,run_dir/"best.pt")
        else: patience+=1
        print(f"epoch={epoch:03d} train={tr:.6f} val={va:.6f} best={best:.6f}")
        if patience>=int(sc["early_stopping_patience"]): print("触发早停。"); break


if __name__=="__main__": main()
