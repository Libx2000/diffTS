from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn

from .model import UNet3D
from .physics import density_gradient, density_unesco_torch


class BoundedPhysicsCorrector(nn.Module):
    """Small correction head. Output is bounded separately for T and S."""
    def __init__(self, in_channels:int, base:int=8, levels:int=2, dropout:float=.05):
        super().__init__(); self.net=UNet3D(in_channels,2,base,levels,dropout)
    def forward(self,condition:torch.Tensor,bound_norm:torch.Tensor)->torch.Tensor:
        return torch.tanh(self.net(condition))*bound_norm


def weighted_mean(values:torch.Tensor,weight:torch.Tensor)->torch.Tensor:
    return (values*weight).sum()/(weight.sum()*values.shape[0]*values.shape[1]).clamp_min(1e-12)


class PhysicsV2Calibration:
    """The exact fixed Stage-3 physics-v2 calibration used by the gate baseline."""
    def __init__(self,path: str | Path,device:torch.device|None=None):
        with np.load(path) as d:
            self.leads=d["lead_days"].astype(int); self.bias_np=d["bias"].astype(np.float32); self.blend_np=d["blend"].astype(np.float32); self.scale_np=d["scale"].astype(np.float32)
        self.index={int(x):i for i,x in enumerate(self.leads)}
        if device is not None:
            self.bias=torch.from_numpy(self.bias_np).to(device);self.blend=torch.from_numpy(self.blend_np).to(device)
    def expected_center_norm(self,raw_mean_norm,det_norm,leads,nmean,nstd):
        ids=torch.tensor([self.index[int(x)] for x in leads.detach().cpu().tolist()],device=raw_mean_norm.device)
        bias=self.bias[ids,:,None,None,None];blend=self.blend[ids,:,None,None,None]
        raw_phys=raw_mean_norm*nstd+nmean;det_phys=det_norm*nstd+nmean
        return (det_phys+blend*(raw_phys-det_phys)+bias-nmean)/nstd
    def apply_numpy(self,ensemble_phys,det_phys,lead:int):
        i=self.index[int(lead)];center=ensemble_phys.mean(0);anomaly=ensemble_phys-center[None];bias=self.bias_np[i,:,None,None,None];blend=self.blend_np[i,:,None,None,None];scale=self.scale_np[i,:,None,None,None]
        calibrated_center=det_phys[None]+blend*(center[None]-det_phys[None])+bias
        return calibrated_center+scale*anomaly


def correction_losses(corrected,baseline,truth_norm,truth_phys,nmean,nstd,depth,mask,weight,cfg):
    pred_phys=corrected*nstd+nmean
    mse=weighted_mean((corrected-truth_norm).square(),weight)
    base_mse=weighted_mean((baseline-truth_norm).square(),weight)
    point_delta=(corrected-truth_norm).square()-(baseline-truth_norm).square()+float(cfg["degradation_margin"])
    degradation=weighted_mean(torch.relu(point_delta),weight)
    anchor=weighted_mean((corrected-baseline).square(),weight)
    base_phys=baseline*nstd+nmean
    pred_rho=density_unesco_torch(pred_phys[:,0],pred_phys[:,1]);base_rho=density_unesco_torch(base_phys[:,0],base_phys[:,1]);true_rho=density_unesco_torch(truth_phys[:,0],truth_phys[:,1])
    density=weighted_mean(((pred_rho-true_rho)/float(cfg["density_scale"])).square()[:,None],weight[:,:1])
    density_degradation=weighted_mean(torch.relu((pred_rho-true_rho).square()-(base_rho-true_rho).square())[:,None]/float(cfg["density_scale"])**2,weight[:,:1])
    pg=density_gradient(pred_phys,depth);bg=density_gradient(base_phys,depth);tg=density_gradient(truth_phys,depth); adjacent=(mask[1:]&mask[:-1])[None]
    valid=adjacent.expand_as(pg); gradient=torch.abs((pg-tg)/float(cfg["gradient_scale"]))[valid].mean()
    gradient_degradation=torch.relu(torch.abs(pg-tg)-torch.abs(bg-tg))[valid].mean()/float(cfg["gradient_scale"])
    instability=torch.abs(torch.relu(-pg/float(cfg["gradient_scale"]))-torch.relu(-tg/float(cfg["gradient_scale"])))[valid].mean()
    total=(float(cfg["mse_weight"])*mse+float(cfg["degradation_weight"])*degradation+float(cfg["anchor_weight"])*anchor+float(cfg["density_weight"])*density+float(cfg["density_degradation_weight"])*density_degradation+float(cfg["gradient_weight"])*gradient+float(cfg["gradient_degradation_weight"])*gradient_degradation+float(cfg["instability_weight"])*instability)
    return total,(mse,base_mse,degradation,anchor,density,density_degradation,gradient,gradient_degradation,instability)
