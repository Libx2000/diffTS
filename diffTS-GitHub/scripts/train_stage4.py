from __future__ import annotations
import argparse,math,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.optim import AdamW
from torch.utils.data import ConcatDataset,DataLoader,Subset
from tqdm import tqdm
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT/"src"))
from diffts.config import ensure_dirs,load_config,set_seed
from diffts.dataset import OceanForecastDataset
from diffts.model import UNet3D
from diffts.stage3_utils import spatial_coordinate_channels
from diffts.stage4 import BoundedPhysicsCorrector,PhysicsV2Calibration,correction_losses

def years_subset(ds,first,last):
    years=pd.to_datetime(ds.rows["target"]).dt.year.to_numpy(); return Subset(ds,np.flatnonzero((years>=first)&(years<=last)).tolist())

def epoch(model,deterministic,loader,optimizer,scaler,device,t,cfg,train):
    model.train(train); total=np.zeros(10); n=0
    rmean,nmean,nstd,bound,coords,mask,depth,weight,calibration=t
    context=torch.enable_grad() if train else torch.no_grad()
    with context:
      for batch in tqdm(loader,leave=False):
        x,y=batch["x"].to(device),batch["y"].to(device); truth=batch["target_phys"].to(device);leads=batch["lead"].to(device)
        with torch.no_grad():
            det=deterministic(x);raw_mean=det+rmean;baseline=calibration.expected_center_norm(raw_mean,det,leads,nmean,nstd)
        condition=torch.cat([x,baseline,coords.expand(x.shape[0],-1,-1,-1,-1)],1)
        with torch.autocast(device_type=device.type,enabled=bool(cfg["amp"])):
            corrected=baseline+model(condition,bound); loss,parts=correction_losses(corrected,baseline,y,truth,nmean,nstd,depth,mask,weight,cfg)
        if train:
            optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["grad_clip_norm"]));scaler.step(optimizer);scaler.update()
        vals=(loss,*parts);total+=np.asarray([float(v.detach()) for v in vals])*x.shape[0];n+=x.shape[0]
    return total/max(n,1)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(PROJECT/"config.yaml"));p.add_argument("--resume");args=p.parse_args()
    cfg=load_config(args.config);set_seed(int(cfg["project"]["seed"]));artifacts,runs=ensure_dirs(cfg);sc=cfg["stage4"];device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda":raise RuntimeError("Stage 4 training requires CUDA.")
    original_train=OceanForecastDataset(cfg,"train");original_val=OceanForecastDataset(cfg,"val")
    train_ds=years_subset(original_train,*sc["train_years"]);val_ds=ConcatDataset([years_subset(original_train,sc["validation_years"][0],sc["validation_years"][1]),years_subset(original_val,sc["validation_years"][0],sc["validation_years"][1])])
    loaders=[DataLoader(ds,batch_size=int(sc["batch_size"]),shuffle=i==0,num_workers=int(sc["num_workers"]),pin_memory=True,persistent_workers=int(sc["num_workers"])>0) for i,ds in enumerate((train_ds,val_ds))]
    ck=torch.load(sc["deterministic_checkpoint"],map_location=device,weights_only=False);mc=cfg["model"];det=UNet3D(ck["in_channels"],2,mc["base_channels"],mc["levels"],mc["dropout"]).to(device);det.load_state_dict(ck["model"]);det.eval().requires_grad_(False)
    coords=spatial_coordinate_channels(artifacts,device);channels=ck["in_channels"]+2+coords.shape[1];model=BoundedPhysicsCorrector(channels,int(sc["base_channels"]),int(sc["levels"]),float(sc["dropout"])).to(device)
    opt=AdamW(model.parameters(),lr=float(sc["learning_rate"]),weight_decay=float(sc["weight_decay"]));scaler=torch.amp.GradScaler("cuda",enabled=bool(sc["amp"]));rs=np.load(sc["residual_stats"]);norm=np.load(artifacts/"normalization.npz");c=np.load(artifacts/"coordinates.npz");mask=torch.from_numpy(np.load(artifacts/"wet_mask.npy").astype(bool)).to(device);nstd=torch.from_numpy(norm["std"].astype(np.float32))[:,:,None,None].to(device);nmean=torch.from_numpy(norm["mean"].astype(np.float32))[:,:,None,None].to(device);bounds=torch.tensor([float(sc["max_temperature_correction_c"]),float(sc["max_salinity_correction_psu"])],device=device)[:,None,None,None]/nstd;area=torch.cos(torch.deg2rad(torch.from_numpy(c["lat"].astype(np.float32)))).clamp_min(1e-6).to(device);weight=(mask.float()*area[None,:,None])[None,None]
    calibration=PhysicsV2Calibration(sc["baseline_calibration"],device)
    tensors=(torch.from_numpy(rs["mean"].astype(np.float32))[None,:,:,None,None].to(device),nmean,nstd,bounds[None],coords,mask,torch.from_numpy(np.abs(c["depth"]).astype(np.float32)).to(device),weight,calibration)
    rd=runs/sc["run_name"];rd.mkdir(parents=True,exist_ok=True);best=math.inf;start=1;patience=0;history=[]
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False);model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"]);start=state["epoch"]+1;best=state["best_score"]
    names=("total","mse","baseline_mse","degradation","anchor","density","density_degradation","gradient","gradient_degradation","instability")
    for ep in range(start,int(sc["epochs"])+1):
        tr=epoch(model,det,loaders[0],opt,scaler,device,tensors,sc,True);va=epoch(model,det,loaders[1],opt,scaler,device,tensors,sc,False);ratio=float(va[1]/max(va[2],1e-12));score=ratio+float(sc["selection_density_weight"])*va[5]+float(sc["selection_gradient_weight"])*va[7]+float(sc["selection_instability_weight"])*va[9]
        row={"epoch":ep,"validation_mse_ratio":ratio,"selection_score":score,**{f"train_{k}":v for k,v in zip(names,tr)},**{f"val_{k}":v for k,v in zip(names,va)}};history.append(row);pd.DataFrame(history).to_csv(rd/"history.csv",index=False)
        state={"epoch":ep,"model":model.state_dict(),"optimizer":opt.state_dict(),"best_score":min(best,score),"condition_channels":int(channels),"config":cfg};torch.save(state,rd/"last.pt")
        eligible=ratio<=1+float(sc["max_validation_mse_degradation"])
        if eligible and score<best:best=score;patience=0;state["best_score"]=best;torch.save(state,rd/"best.pt")
        else:patience+=1
        print(f"epoch={ep:03d} score={score:.6f} mse_ratio={ratio:.6f} eligible={eligible} physics={va[5:].tolist()}")
        if patience>=int(sc["early_stopping_patience"]):break
    if not (rd/"best.pt").exists():raise RuntimeError("No checkpoint passed the baseline-protection validation gate.")
if __name__=="__main__":main()
