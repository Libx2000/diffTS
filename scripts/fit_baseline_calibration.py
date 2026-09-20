from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT/"src"))
from diffts.physics_calibration import PhysicsAwareValidationCalibrator,apply_physics_calibration,load_physics_calibration  # noqa:E402
from diffts.config import ensure_dirs,load_config,set_seed  # noqa:E402
from diffts.dataset import OceanForecastDataset  # noqa:E402
from diffts.diffusion import DiffusionSchedule  # noqa:E402
from diffts.diffusion_model import ConditionalResidualUNet3D  # noqa:E402
from diffts.model import UNet3D  # noqa:E402
from diffts.physics import PhysicalAccumulator  # noqa:E402
from diffts.prob_metrics import ProbabilisticAccumulator  # noqa:E402
from diffts.stage3_utils import spatial_coordinate_channels  # noqa:E402


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default=str(PROJECT/"config.yaml")); p.add_argument("--mode",choices=("fit","evaluate"),required=True); p.add_argument("--split",choices=("val","test","ood"),default=None); p.add_argument("--checkpoint",default=None); p.add_argument("--calibration",default=None); p.add_argument("--output-run-name",default="residual_diffusion_physics_v2"); p.add_argument("--members",type=int,default=None); p.add_argument("--max-samples",type=int,default=None); args=p.parse_args()
    split=args.split or ("val" if args.mode=="fit" else "test")
    if args.mode=="fit" and split!="val": raise ValueError("Fit calibration on validation data only.")
    cfg=load_config(args.config); set_seed(int(cfg["project"]["seed"])); artifacts,runs=ensure_dirs(cfg); sc=cfg["stage3"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("Stage-4 ensemble evaluation requires CUDA.")
    det_ck=torch.load(sc["deterministic_checkpoint"],map_location=device,weights_only=False); mc=cfg["model"]
    deterministic=UNet3D(det_ck["in_channels"],2,mc["base_channels"],mc["levels"],mc["dropout"]).to(device); deterministic.load_state_dict(det_ck["model"]); deterministic.eval()
    model_run_dir=runs/sc["run_name"]; run_dir=runs/(args.output_run_name or sc["run_name"]); checkpoint=Path(args.checkpoint) if args.checkpoint else model_run_dir/"best.pt"; ck=torch.load(checkpoint,map_location=device,weights_only=False)
    model=ConditionalResidualUNet3D(int(ck["condition_channels"]),2,int(sc["base_channels"]),int(sc["levels"]),float(sc["dropout"]),int(sc["time_embedding_dim"])).to(device); model.load_state_dict(ck["ema"]); model.eval()
    schedule=DiffusionSchedule(int(sc["diffusion_steps"]),sc["beta_schedule"]).to(device); rs=np.load(sc["residual_stats"]); rmean=torch.from_numpy(rs["mean"])[None,:,:,None,None].to(device); rstd=torch.from_numpy(rs["std"])[None,:,:,None,None].to(device)
    norm=np.load(artifacts/"normalization.npz"); mean=norm["mean"].astype(np.float32)[:,:,None,None]; std=norm["std"].astype(np.float32)[:,:,None,None]; coords=np.load(artifacts/"coordinates.npz"); mask=np.load(artifacts/"wet_mask.npy"); coord_channels=spatial_coordinate_channels(artifacts,device)
    members=int(args.members or sc["ensemble_members"]); ds=OceanForecastDataset(cfg,split); maximum=args.max_samples if args.max_samples is not None else int(sc["evaluation_max_samples"])
    if maximum>0 and maximum<len(ds): ds=Subset(ds,np.linspace(0,len(ds)-1,maximum,dtype=int).tolist())
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=int(sc["num_workers"]),pin_memory=True)
    cal_path=Path(args.calibration) if args.calibration else run_dir/"validation_physics_calibration.npz"; fitter=PhysicsAwareValidationCalibrator(coords["lat"],mask) if args.mode=="fit" else None; calibration=load_physics_calibration(cal_path) if args.mode=="evaluate" else None
    prob=ProbabilisticAccumulator(coords["lat"],mask,members); physical=PhysicalAccumulator(coords["depth"],coords["lat"],coords["lon"],mask)
    with torch.inference_mode():
        for bi,batch in enumerate(tqdm(loader,desc=f"stage4 {args.mode} {split}")):
            x=batch["x"].to(device); det=deterministic(x); condition=torch.cat([x,det,coord_channels.expand(1,-1,-1,-1,-1)],dim=1); generated=[]
            for member in range(members):
                generator=torch.Generator(device=device).manual_seed(int(cfg["project"]["seed"])+bi*members+member)
                sample=schedule.ddim_sample(model,condition,tuple(det.shape),int(sc["ddim_steps"]),float(sc["ddim_eta"]),generator); generated.append(det+sample*rstd+rmean)
            ens=torch.stack(generated)[:,0].cpu().numpy()*std[None]+mean[None]; det_phys=det.cpu().numpy()[0]*std+mean; truth=batch["target_phys"].numpy()[0]; lead=int(batch["lead"].item())
            if fitter is not None: fitter.update(lead,ens,det_phys,truth); used=ens
            else:
                bias,blend,scale=calibration[lead]; used=apply_physics_calibration(ens,det_phys,bias,blend,scale)
            prob.update(lead,used,truth); physical.update(lead,used.mean(0),truth)
    run_dir.mkdir(parents=True,exist_ok=True)
    calibration_summary={}
    if fitter is not None:
        calibration_summary=fitter.save(cal_path,float(sc["calibration_min_scale"]),float(sc["calibration_max_scale"]),float(sc.get("calibration_min_blend",0.0)),float(sc.get("calibration_max_blend",1.0))); tag="val_raw_for_physics_calibration"
    else: tag=f"{split}_physics_calibrated"
    metrics,ranks=prob.dataframes(coords["depth"]); metrics.to_csv(run_dir/f"probabilistic_metrics_{tag}.csv",index=False); ranks.to_csv(run_dir/f"rank_histogram_{tag}.csv",index=False)
    physical_rows=physical.rows()
    with (run_dir/f"physical_metrics_{tag}.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(physical_rows[0])); writer.writeheader(); writer.writerows(physical_rows)
    summary=metrics.groupby(["lead_day","variable"])[["rmse_ensemble_mean","rms_spread","spread_skill_ratio","crps","coverage80","coverage90"]].mean(); (run_dir/f"probabilistic_summary_{tag}.txt").write_text(summary.to_string(),encoding="utf-8")
    (run_dir/f"evaluation_{tag}.json").write_text(json.dumps({"mode":args.mode,"split":split,"samples":len(ds),"members":members,"ddim_steps":int(sc["ddim_steps"]),"checkpoint":str(checkpoint),"calibration":str(cal_path),"calibration_method":"lead-variable structure-preserving affine blend","density_equation":"TEOS-10 in-situ density and sigma0 stability",**calibration_summary},indent=2),encoding="utf-8")
    print(summary.to_string()); print(physical_rows)


if __name__=="__main__": main()
