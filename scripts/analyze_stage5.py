from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd

def paired_bootstrap(frame,metric,nboot,seed,margin):
 pivot=frame.pivot_table(index="target_ordinal",columns="alpha",values=metric,aggfunc="mean").dropna();base=pivot.iloc[:,0].to_numpy();full=pivot.iloc[:,-1].to_numpy();observed=100*(full.mean()/base.mean()-1);rng=np.random.default_rng(seed);n=len(base);draw=np.empty(nboot)
 for i in range(nboot):
  ids=rng.integers(0,n,n);draw[i]=100*(full[ids].mean()/base[ids].mean()-1)
 lo,hi=np.quantile(draw,[.025,.975]);return observed,float(lo),float(hi),n,bool(hi<=margin)

def main():
 p=argparse.ArgumentParser();p.add_argument("--runs-dir",required=True);p.add_argument("--bootstrap",type=int,default=2000);p.add_argument("--seed",type=int,default=42);p.add_argument("--noninferiority-margin-percent",type=float,default=.5);a=p.parse_args();runs=Path(a.runs_dir);src=runs/"stage5_paper_evaluation";out=src/"analysis";out.mkdir(parents=True,exist_ok=True);data=pd.concat([pd.read_csv(src/f"paired_metrics_{s}.csv") for s in ("test","ood")],ignore_index=True);rows=[]
 for keys,g in data.groupby(["split","lead_day","variable","scope"]):
  for metric in ("crps","rmse","mae","coverage80"):
   obs,lo,hi,n,noninf=paired_bootstrap(g,metric,a.bootstrap,a.seed,a.noninferiority_margin_percent if metric=="crps" else np.inf);rows.append(dict(zip(("split","lead_day","variable","scope"),keys),metric=metric,change_percent=obs,ci95_low=lo,ci95_high=hi,n_dates=n,crps_noninferior=noninf if metric=="crps" else True))
 stats=pd.DataFrame(rows);stats.to_csv(out/"paired_bootstrap.csv",index=False)
 global_crps=stats[(stats.scope=="global")&(stats.metric=="crps")];summary={"bootstrap_replicates":a.bootstrap,"crps_noninferiority_margin_percent":a.noninferiority_margin_percent,"all_global_crps_noninferior":bool(global_crps.crps_noninferior.all()),"mean_global_crps_change_percent":float(global_crps.change_percent.mean()),"global_crps_ci_upper_max_percent":float(global_crps.ci95_high.max())}
 physical=json.loads((runs/"stage4_comparison"/"stage4_gate.json").read_text(encoding="utf-8"));summary["stage4_gate"]=physical;(out/"stage5_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 fig,axes=plt.subplots(1,2,figsize=(9,3.8))
 for split,marker in (("test","o"),("ood","s")):
  for var,color in (("thetao","#0072B2"),("so","#009E73")):
   z=global_crps[(global_crps.split==split)&(global_crps.variable==var)];axes[0].plot(z.lead_day,z.change_percent,marker=marker,color=color,ls="-" if split=="test" else "--",label=f"{split}-{var}");axes[0].fill_between(z.lead_day,z.ci95_low,z.ci95_high,color=color,alpha=.08)
 axes[0].axhline(0,color="black",ls=":");axes[0].axhline(a.noninferiority_margin_percent,color="#D55E00",ls="--");axes[0].set(xlabel="Lead time (days)",ylabel="Paired CRPS change (%)");axes[0].grid(alpha=.25);axes[0].legend(fontsize=7,ncol=2)
 z=stats[(stats.metric=="crps")&(stats.lead_day==1)].pivot_table(index="scope",columns="split",values="change_percent");z.plot.bar(ax=axes[1],color=["#56B4E9","#E69F00"]);axes[1].axhline(0,color="black",ls=":");axes[1].set(ylabel="Day-1 CRPS change (%)",xlabel="Region/depth scope");axes[1].tick_params(axis="x",rotation=55);axes[1].grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(out/"stage5_crps_bootstrap.png",dpi=300,bbox_inches="tight");fig.savefig(out/"stage5_crps_bootstrap.svg",bbox_inches="tight");plt.close(fig)
 abl=data[data.scope=="global"].groupby(["split","lead_day","variable","alpha"],as_index=False)[["crps","rmse","coverage80"]].mean();abl.to_csv(out/"correction_strength_ablation.csv",index=False);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
