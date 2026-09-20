from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT/"src"))
from diffts.config import ensure_dirs,load_config,set_seed  # noqa:E402
from diffts.dataset import OceanForecastDataset  # noqa:E402
from diffts.diffusion import DiffusionSchedule  # noqa:E402
from diffts.diffusion_model import ConditionalResidualUNet3D  # noqa:E402
from diffts.model import UNet3D  # noqa:E402
from diffts.prob_metrics import ProbabilisticAccumulator  # noqa:E402
from diffts.stage3_utils import spatial_coordinate_channels  # noqa:E402


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default=str(PROJECT/"config.yaml")); p.add_argument("--split",choices=("val","test","ood"),default="test"); p.add_argument("--checkpoint",default=None); p.add_argument("--members",type=int,default=None); p.add_argument("--max-samples",type=int,default=None); args=p.parse_args()
    cfg=load_config(args.config); set_seed(int(cfg["project"]["seed"])); artifacts,runs=ensure_dirs(cfg); sc=cfg["stage3"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("集合扩散评价计算量很大，请在超算GPU节点运行。")
    det_ck=torch.load(sc["deterministic_checkpoint"],map_location=device,weights_only=False); mc=cfg["model"]
    deterministic=UNet3D(det_ck["in_channels"],2,mc["base_channels"],mc["levels"],mc["dropout"]).to(device); deterministic.load_state_dict(det_ck["model"]); deterministic.eval()
    path=Path(args.checkpoint) if args.checkpoint else runs/sc["run_name"]/"best.pt"; ck=torch.load(path,map_location=device,weights_only=False)
    model=ConditionalResidualUNet3D(int(ck["condition_channels"]),2,int(sc["base_channels"]),int(sc["levels"]),float(sc["dropout"]),int(sc["time_embedding_dim"])).to(device); model.load_state_dict(ck["ema"]); model.eval()
    schedule=DiffusionSchedule(int(sc["diffusion_steps"]),sc["beta_schedule"]).to(device)
    stats=np.load(sc["residual_stats"]); rmean=torch.from_numpy(stats["mean"])[None,:,:,None,None].to(device); rstd=torch.from_numpy(stats["std"])[None,:,:,None,None].to(device)
    norm=np.load(artifacts/"normalization.npz"); mean=norm["mean"].astype(np.float32)[:,:,None,None]; std=norm["std"].astype(np.float32)[:,:,None,None]
    coords=np.load(artifacts/"coordinates.npz"); mask=np.load(artifacts/"wet_mask.npy"); members=int(args.members or sc["ensemble_members"])
    coordinate_channels=spatial_coordinate_channels(artifacts,device)
    dataset=OceanForecastDataset(cfg,args.split); maximum=args.max_samples if args.max_samples is not None else int(sc["evaluation_max_samples"])
    if maximum>0 and maximum<len(dataset): dataset=Subset(dataset,np.linspace(0,len(dataset)-1,maximum,dtype=int).tolist())
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=int(sc["num_workers"]),pin_memory=True)
    acc=ProbabilisticAccumulator(coords["lat"],mask,members); example=None
    with torch.inference_mode():
        for batch_index,batch in enumerate(tqdm(loader,desc=f"ensemble {args.split}")):
            x=batch["x"].to(device); det=deterministic(x); condition=torch.cat([x,det,coordinate_channels.expand(x.shape[0],-1,-1,-1,-1)],dim=1); generated=[]
            for member in range(members):
                generator=torch.Generator(device=device).manual_seed(int(cfg["project"]["seed"])+batch_index*members+member)
                sample=schedule.ddim_sample(model,condition,tuple(det.shape),int(sc["ddim_steps"]),float(sc["ddim_eta"]),generator)
                generated.append(det+sample*rstd+rmean)
            ensemble_norm=torch.stack(generated,dim=0)[:,0].cpu().numpy(); ensemble_phys=ensemble_norm*std[None]+mean[None]
            truth=batch["target_phys"].numpy()[0]; lead=int(batch["lead"].item()); acc.update(lead,ensemble_phys,truth)
            if example is None or lead==10:
                example={"ensemble":ensemble_phys.astype(np.float32),"truth":truth.astype(np.float32),"deterministic":(det.cpu().numpy()[0]*std+mean).astype(np.float32),"lead":np.asarray([lead]),"target_ordinal":batch["target_ordinal"].numpy()}
    metrics,ranks=acc.dataframes(coords["depth"]); run_dir=runs/sc["run_name"]; run_dir.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(run_dir/f"probabilistic_metrics_{args.split}.csv",index=False); ranks.to_csv(run_dir/f"rank_histogram_{args.split}.csv",index=False)
    if example is not None: np.savez_compressed(run_dir/f"example_ensemble_{args.split}.npz",**example)
    summary=metrics.groupby(["lead_day","variable"])[["rmse_ensemble_mean","rms_spread","spread_skill_ratio","crps","coverage80","coverage90"]].mean()
    (run_dir/f"probabilistic_summary_{args.split}.txt").write_text(summary.to_string(),encoding="utf-8")
    metadata={"split":args.split,"samples":len(dataset),"members":members,"ddim_steps":int(sc["ddim_steps"]),"checkpoint":str(path)}
    (run_dir/f"evaluation_{args.split}.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(summary.to_string())


if __name__=="__main__": main()
