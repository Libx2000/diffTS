from __future__ import annotations
import sys,tempfile
from pathlib import Path
import numpy as np,torch
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT/"src"))
from diffts.stage4 import BoundedPhysicsCorrector,PhysicsV2Calibration,correction_losses
def main():
 b,d,h,w=1,3,8,8;model=BoundedPhysicsCorrector(8,4,2,0);x=torch.randn(b,8,d,h,w);bound=torch.tensor([.1,.02])[None,:,None,None,None];cor=model(x,bound);assert cor.shape==(b,2,d,h,w) and torch.all(cor.abs()<=bound+1e-6)
 mask=torch.ones(d,h,w,dtype=torch.bool);weight=mask.float()[None,None];baseline=torch.randn(b,2,d,h,w)*.01;truth=torch.zeros_like(baseline);nmean=torch.tensor([10.,35.])[:,None,None,None];nstd=torch.ones(2,d,1,1);truthp=truth*nstd+nmean;cfg={"degradation_margin":0.,"density_scale":.1,"gradient_scale":.001,"mse_weight":1.,"degradation_weight":2.,"anchor_weight":.05,"density_weight":.002,"density_degradation_weight":.01,"gradient_weight":.002,"gradient_degradation_weight":.01,"instability_weight":.001};loss,_=correction_losses(baseline+cor,baseline,truth,truthp,nmean,nstd,torch.tensor([0.,10.,30.]),mask,weight,cfg);loss.backward();assert torch.isfinite(loss)
 ens=np.random.default_rng(1).normal(size=(10,2,d,h,w)).astype(np.float32);det=np.zeros((2,d,h,w),np.float32)
 with tempfile.TemporaryDirectory() as td:
  path=Path(td)/"p.npz";np.savez(path,lead_days=np.array([1]),bias=np.array([[.1,.2]],np.float32),blend=np.array([[.5,.25]],np.float32),scale=np.array([[.8,1.2]],np.float32));cal=PhysicsV2Calibration(path,torch.device("cpu"));out=cal.apply_numpy(ens,det,1);expected=det+np.array([.1,.2])[:,None,None,None]+np.array([.5,.25])[:,None,None,None]*(ens.mean(0)-det);assert np.allclose(out.mean(0),expected,atol=1e-6);center=cal.expected_center_norm(torch.zeros(1,2,d,h,w),torch.zeros(1,2,d,h,w),torch.tensor([1]),torch.zeros(2,d,1,1),torch.ones(2,d,1,1));assert center.shape==(1,2,d,h,w)
 print("Stage 4 smoke test passed: bounds, strong-baseline calibration, and degradation guards verified")
if __name__=="__main__":main()
