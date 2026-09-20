from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,Subset
from tqdm import tqdm
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT/"src"))
from diffts.config import ensure_dirs,load_config,set_seed
from diffts.dataset import OceanForecastDataset
from diffts.diffusion import DiffusionSchedule
from diffts.diffusion_model import ConditionalResidualUNet3D
from diffts.model import UNet3D
from diffts.stage3_utils import spatial_coordinate_channels
from diffts.stage4 import BoundedPhysicsCorrector,PhysicsV2Calibration

def scope_masks(depth,lat,wet):
 d=np.abs(depth);la=np.abs(lat);shape=wet.shape
 def broad(x):return np.broadcast_to(x,shape)&wet
 return {"global":wet,"surface_0_50m":broad((d<=50)[:,None,None]),"mid_50_300m":broad(((d>50)&(d<=300))[:,None,None]),"deep_over_300m":broad((d>300)[:,None,None]),"tropics":broad((la<23.5)[None,:,None]),"midlatitudes":broad(((la>=23.5)&(la<60))[None,:,None]),"polar":broad((la>=60)[None,:,None])}

def metrics(ensemble,truth,lead,ordinal,split,alpha,scopes,area):
 center=ensemble.mean(0);lo=np.quantile(ensemble,.1,axis=0);hi=np.quantile(ensemble,.9,axis=0);rows=[]
 for var,name in enumerate(("thetao","so")):
  first=np.abs(ensemble[:,var]-truth[var][None]).mean(0);pair=np.abs(ensemble[:,None,var]-ensemble[None,:,var]).mean((0,1));crps=first-.5*pair
  for scope,mask in scopes.items():
   valid=mask&np.isfinite(truth[var])&np.isfinite(center[var]);w=np.broadcast_to(area,mask.shape)[valid];den=w.sum()
   if den<=0:continue
   err=center[var][valid]-truth[var][valid]
   rows.append({"split":split,"target_ordinal":ordinal,"lead_day":lead,"variable":name,"scope":scope,"alpha":alpha,"rmse":float(np.sqrt(np.sum(w*err**2)/den)),"mae":float(np.sum(w*np.abs(err))/den),"crps":float(np.sum(w*crps[valid])/den),"coverage80":float(np.sum(w*((truth[var][valid]>=lo[var][valid])&(truth[var][valid]<=hi[var][valid])))/den),"weight_sum":float(den)})
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default=str(PROJECT/"config.yaml"));p.add_argument("--split",choices=("test","ood"),required=True);p.add_argument("--max-samples",type=int);args=p.parse_args();cfg=load_config(args.config);set_seed(int(cfg["project"]["seed"]));artifacts,runs=ensure_dirs(cfg);sc=cfg["stage5"];device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 if device.type!="cuda":raise RuntimeError("Stage 5 paired ensemble evaluation requires CUDA.")
 dc=torch.load(sc["deterministic_checkpoint"],map_location=device,weights_only=False);mc=cfg["model"];det=UNet3D(dc["in_channels"],2,mc["base_channels"],mc["levels"],mc["dropout"]).to(device);det.load_state_dict(dc["model"]);det.eval()
 bc=torch.load(sc["stage3_checkpoint"],map_location=device,weights_only=False);diff=ConditionalResidualUNet3D(int(bc["condition_channels"]),2,int(sc["stage3_base_channels"]),int(sc["stage3_levels"]),float(sc["stage3_dropout"]),int(sc["stage3_time_embedding_dim"])).to(device);diff.load_state_dict(bc["ema"]);diff.eval();schedule=DiffusionSchedule(int(sc["diffusion_steps"]),sc["beta_schedule"]).to(device)
 cc=torch.load(sc["stage4_checkpoint"],map_location=device,weights_only=False);corr=BoundedPhysicsCorrector(int(cc["condition_channels"]),int(sc["correction_base_channels"]),int(sc["correction_levels"]),float(sc["correction_dropout"])).to(device);corr.load_state_dict(cc["model"]);corr.eval()
 rs=np.load(sc["residual_stats"]);rmean=torch.from_numpy(rs["mean"].astype(np.float32))[None,:,:,None,None].to(device);rstd=torch.from_numpy(rs["std"].astype(np.float32))[None,:,:,None,None].to(device);norm=np.load(artifacts/"normalization.npz");mean=norm["mean"].astype(np.float32)[:,:,None,None];std=norm["std"].astype(np.float32)[:,:,None,None];nmean=torch.from_numpy(mean).to(device);nstd=torch.from_numpy(std).to(device);bound=torch.tensor(sc["correction_bounds"],device=device)[None,:,None,None,None]/nstd[None]
 co=np.load(artifacts/"coordinates.npz");wet=np.load(artifacts/"wet_mask.npy").astype(bool);scopes=scope_masks(co["depth"],co["lat"],wet);area=np.cos(np.deg2rad(co["lat"])).clip(1e-6,None)[None,:,None];coords=spatial_coordinate_channels(artifacts,device);cal=PhysicsV2Calibration(sc["baseline_calibration"],device);ds=OceanForecastDataset(cfg,args.split);maximum=args.max_samples if args.max_samples is not None else int(sc["evaluation_max_samples"])
 if maximum>0 and maximum<len(ds):ds=Subset(ds,np.linspace(0,len(ds)-1,maximum,dtype=int).tolist())
 loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=int(sc["num_workers"]),pin_memory=True);members=int(sc["ensemble_members"]);alphas=[float(x) for x in sc["ablation_alphas"]];rows=[];cases={};started=time.time()
 with torch.inference_mode():
  for bi,batch in enumerate(tqdm(loader,desc=f"stage5 {args.split}")):
   x=batch["x"].to(device);lead=int(batch["lead"].item());ordinal=int(batch["target_ordinal"].item());d=det(x);expected=cal.expected_center_norm(d+rmean,d,batch["lead"].to(device),nmean,nstd);correction=corr(torch.cat([x,expected,coords.expand(1,-1,-1,-1,-1)],1),bound).cpu().numpy()[0]*std;condition=torch.cat([x,d,coords.expand(1,-1,-1,-1,-1)],1);raw=[]
   for member in range(members):
    gen=torch.Generator(device=device).manual_seed(int(cfg["project"]["seed"])+bi*members+member);raw.append(d+schedule.ddim_sample(diff,condition,tuple(d.shape),int(sc["ddim_steps"]),float(sc["ddim_eta"]),gen)*rstd+rmean)
   raw_phys=torch.stack(raw)[:,0].cpu().numpy()*std[None]+mean[None];base=cal.apply_numpy(raw_phys,d.cpu().numpy()[0]*std+mean,lead);truth=batch["target_phys"].numpy()[0]
   for alpha in alphas:rows.extend(metrics(base+alpha*correction[None],truth,lead,ordinal,args.split,alpha,scopes,area))
   if lead not in cases:cases[lead]={"baseline":base.astype(np.float32),"correction":correction.astype(np.float32),"truth":truth.astype(np.float32),"target_ordinal":np.array([ordinal])}
 out=runs/sc["run_name"];out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/f"paired_metrics_{args.split}.csv",index=False)
 for lead,data in cases.items():np.savez_compressed(out/f"case_{args.split}_lead{lead}.npz",**data)
 (out/f"evaluation_{args.split}.json").write_text(json.dumps({"split":args.split,"samples":len(ds),"members":members,"alphas":alphas,"scopes":list(scopes),"elapsed_seconds":time.time()-started,"paired_by":"target_ordinal"},indent=2),encoding="utf-8")
if __name__=="__main__":main()
