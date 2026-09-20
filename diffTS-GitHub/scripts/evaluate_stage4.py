from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,Subset
from tqdm import tqdm
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT/"src"))
from diffts.config import ensure_dirs,load_config,set_seed
from diffts.dataset import OceanForecastDataset
from diffts.diffusion import DiffusionSchedule
from diffts.diffusion_model import ConditionalResidualUNet3D
from diffts.model import UNet3D
from diffts.physics import PhysicalAccumulator
from diffts.prob_metrics import ProbabilisticAccumulator
from diffts.stage3_utils import spatial_coordinate_channels
from diffts.stage4 import BoundedPhysicsCorrector,PhysicsV2Calibration

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default=str(PROJECT/"config.yaml"));p.add_argument("--split",choices=("val","test","ood"),default="test");p.add_argument("--members",type=int);p.add_argument("--max-samples",type=int);args=p.parse_args()
 cfg=load_config(args.config);set_seed(int(cfg["project"]["seed"]));artifacts,runs=ensure_dirs(cfg);sc=cfg["stage4"];device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 if device.type!="cuda":raise RuntimeError("Stage 4 evaluation requires CUDA.")
 dc=torch.load(sc["deterministic_checkpoint"],map_location=device,weights_only=False);mc=cfg["model"];det=UNet3D(dc["in_channels"],2,mc["base_channels"],mc["levels"],mc["dropout"]).to(device);det.load_state_dict(dc["model"]);det.eval()
 bc=torch.load(sc["stage3_checkpoint"],map_location=device,weights_only=False);diff=ConditionalResidualUNet3D(int(bc["condition_channels"]),2,int(sc["stage3_base_channels"]),int(sc["stage3_levels"]),float(sc["stage3_dropout"]),int(sc["stage3_time_embedding_dim"])).to(device);diff.load_state_dict(bc["ema"]);diff.eval();schedule=DiffusionSchedule(int(sc["diffusion_steps"]),sc["beta_schedule"]).to(device)
 rd=runs/sc["run_name"];cc=torch.load(rd/"best.pt",map_location=device,weights_only=False);corr=BoundedPhysicsCorrector(int(cc["condition_channels"]),int(sc["base_channels"]),int(sc["levels"]),float(sc["dropout"])).to(device);corr.load_state_dict(cc["model"]);corr.eval()
 rs=np.load(sc["residual_stats"]);rmean=torch.from_numpy(rs["mean"].astype(np.float32))[None,:,:,None,None].to(device);rstd=torch.from_numpy(rs["std"].astype(np.float32))[None,:,:,None,None].to(device);norm=np.load(artifacts/"normalization.npz");mean=norm["mean"].astype(np.float32)[:,:,None,None];std=norm["std"].astype(np.float32)[:,:,None,None];nmean=torch.from_numpy(mean).to(device);nstd=torch.from_numpy(std).to(device);bound=torch.tensor([float(sc["max_temperature_correction_c"]),float(sc["max_salinity_correction_psu"])],device=device)[None,:,None,None,None]/nstd[None]
 co=np.load(artifacts/"coordinates.npz");mask=np.load(artifacts/"wet_mask.npy");coords=spatial_coordinate_channels(artifacts,device);calibration=PhysicsV2Calibration(sc["baseline_calibration"],device);members=int(args.members or sc["ensemble_members"]);ds=OceanForecastDataset(cfg,args.split);maximum=args.max_samples if args.max_samples is not None else int(sc["evaluation_max_samples"])
 if maximum>0 and maximum<len(ds):ds=Subset(ds,np.linspace(0,len(ds)-1,maximum,dtype=int).tolist())
 loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=int(sc["num_workers"]),pin_memory=True);prob=ProbabilisticAccumulator(co["lat"],mask,members);physical=PhysicalAccumulator(co["depth"],co["lat"],co["lon"],mask)
 with torch.inference_mode():
  for bi,batch in enumerate(tqdm(loader,desc=f"stage4 {args.split}")):
   x=batch["x"].to(device);lead=int(batch["lead"].item());d=det(x);raw_expected=d+rmean;baseline_expected=calibration.expected_center_norm(raw_expected,d,batch["lead"].to(device),nmean,nstd);base_condition=torch.cat([x,baseline_expected,coords.expand(1,-1,-1,-1,-1)],1);correction_norm=corr(base_condition,bound);condition=torch.cat([x,d,coords.expand(1,-1,-1,-1,-1)],1);raw=[]
   for member in range(members):
    gen=torch.Generator(device=device).manual_seed(int(cfg["project"]["seed"])+bi*members+member);sample=schedule.ddim_sample(diff,condition,tuple(d.shape),int(sc["ddim_steps"]),float(sc["ddim_eta"]),gen);raw.append(d+sample*rstd+rmean)
   raw_phys=torch.stack(raw)[:,0].cpu().numpy()*std[None]+mean[None];det_phys=d.cpu().numpy()[0]*std+mean;baseline_ensemble=calibration.apply_numpy(raw_phys,det_phys,lead);correction_phys=correction_norm.cpu().numpy()[0]*std;used=baseline_ensemble+correction_phys[None];truth=batch["target_phys"].numpy()[0]
   prob.update(lead,used,truth);physical.update(lead,used.mean(0),truth)
 rd.mkdir(parents=True,exist_ok=True);tag=f"{args.split}_strong_baseline";metrics,ranks=prob.dataframes(co["depth"]);metrics.to_csv(rd/f"probabilistic_metrics_{tag}.csv",index=False);ranks.to_csv(rd/f"rank_histogram_{tag}.csv",index=False);rows=physical.rows()
 with (rd/f"physical_metrics_{tag}.csv").open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=metrics.groupby(["lead_day","variable"])[["rmse_ensemble_mean","rms_spread","spread_skill_ratio","crps","coverage80","coverage90"]].mean();(rd/f"probabilistic_summary_{tag}.txt").write_text(summary.to_string(),encoding="utf-8");(rd/f"evaluation_{tag}.json").write_text(json.dumps({"split":args.split,"samples":len(ds),"members":members,"baseline":"fixed residual_diffusion_physics_v2 calibration","extra_stage4_calibration":False},indent=2),encoding="utf-8");print(summary.to_string())
if __name__=="__main__":main()
