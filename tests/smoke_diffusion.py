from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT/"src"))
from diffts.diffusion import DiffusionSchedule  # noqa:E402
from diffts.diffusion_model import ConditionalResidualUNet3D  # noqa:E402


def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ConditionalResidualUNet3D(condition_channels=14,state_channels=2,base=4,levels=2,time_dim=32).to(device)
    condition=torch.randn(1,14,3,16,32,device=device); clean=torch.randn(1,2,3,16,32,device=device)
    schedule=DiffusionSchedule(20,"cosine").to(device); t=torch.tensor([10],device=device); noise=torch.randn_like(clean)
    noisy=schedule.q_sample(clean,t,noise); output=model(noisy,condition,t)
    assert output.shape==clean.shape and torch.isfinite(output).all()
    sample=schedule.ddim_sample(model,condition,tuple(clean.shape),sampling_steps=4,eta=0.0)
    assert sample.shape==clean.shape and torch.isfinite(sample).all()
    print(f"stage3 smoke test passed on {device}: {tuple(sample.shape)}")


if __name__=="__main__": main()
