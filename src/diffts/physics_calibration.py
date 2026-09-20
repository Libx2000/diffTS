from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np


class PhysicsAwareValidationCalibrator:
    """Lead/variable calibration that preserves vertical and spatial model structure."""
    def __init__(self, lat: np.ndarray, wet_mask: np.ndarray):
        self.area=np.cos(np.deg2rad(lat)).clip(1e-6,None)[:,None]
        self.mask=wet_mask.astype(bool)
        # w, delta, error, delta2, error2, delta*error, raw_spread2
        self.stats=defaultdict(lambda:np.zeros(7,dtype=np.float64))

    def update(self,lead:int,ensemble:np.ndarray,deterministic:np.ndarray,truth:np.ndarray)->None:
        raw_mean=ensemble.mean(0); raw_spread=ensemble.std(0,ddof=1)
        for var in range(truth.shape[0]):
            valid=self.mask & np.isfinite(truth[var]) & np.isfinite(raw_mean[var]) & np.isfinite(deterministic[var])
            if not valid.any(): continue
            w=np.broadcast_to(self.area,self.mask.shape)[valid]
            delta=(raw_mean[var]-deterministic[var])[valid]
            error=(truth[var]-deterministic[var])[valid]
            spread=raw_spread[var][valid]
            self.stats[(int(lead),var)]+=np.asarray([w.sum(),np.sum(w*delta),np.sum(w*error),np.sum(w*delta**2),np.sum(w*error**2),np.sum(w*delta*error),np.sum(w*spread**2)])

    def save(self,path:Path,min_scale:float=.25,max_scale:float=4.,min_blend:float=0.,max_blend:float=1.)->dict:
        leads=sorted({k[0] for k in self.stats}); bias=np.zeros((len(leads),2),np.float32); blend=np.zeros_like(bias); scale=np.ones_like(bias)
        for li,lead in enumerate(leads):
            for var in range(2):
                w,sd,se,sd2,se2,sde,ss2=self.stats[(lead,var)]
                md,me=sd/w,se/w; covariance=sde/w-md*me; variance=max(sd2/w-md**2,1e-12)
                b=np.clip(covariance/variance,min_blend,max_blend); a=me-b*md
                mse=max((se2+b*b*sd2+a*a*w-2*b*sde-2*a*se+2*a*b*sd)/w,0.)
                bias[li,var]=a;blend[li,var]=b;scale[li,var]=np.clip(np.sqrt(mse/max(ss2/w,1e-12)),min_scale,max_scale)
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,lead_days=np.asarray(leads),bias=bias,blend=blend,scale=scale)
        return {"blend_mean":float(blend.mean()),"blend_min":float(blend.min()),"blend_max":float(blend.max()),"scale_mean":float(scale.mean())}


def load_physics_calibration(path:Path):
    d=np.load(path);return {int(lead):(d["bias"][i],d["blend"][i],d["scale"][i]) for i,lead in enumerate(d["lead_days"])}


def apply_physics_calibration(ensemble:np.ndarray,deterministic:np.ndarray,bias:np.ndarray,blend:np.ndarray,scale:np.ndarray)->np.ndarray:
    raw_mean=ensemble.mean(0);anomalies=ensemble-raw_mean[None]
    center=deterministic+blend[:,None,None,None]*(raw_mean-deterministic)+bias[:,None,None,None]
    return center[None]+scale[None,:,None,None,None]*anomalies
