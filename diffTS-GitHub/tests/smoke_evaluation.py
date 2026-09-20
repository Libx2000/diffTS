from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,pandas as pd
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT/"scripts"))
from evaluate_stage5 import metrics,scope_masks
from analyze_stage5 import paired_bootstrap
def main():
 rng=np.random.default_rng(2);d,h,w=4,8,10;depth=np.array([0,25,100,500]);lat=np.linspace(-80,80,h);wet=np.ones((d,h,w),bool);scopes=scope_masks(depth,lat,wet);truth=rng.normal(size=(2,d,h,w));base=truth[None]+rng.normal(0,.2,size=(10,2,d,h,w));rows=[]
 for ordinal in range(20):
  for alpha,ens in ((0.,base),(1.,truth[None]+.9*(base-truth[None]))):rows.extend(metrics(ens,truth,1,ordinal,"test",alpha,scopes,np.cos(np.deg2rad(lat))[None,:,None]))
 df=pd.DataFrame(rows);assert set(scopes)==set(df.scope.unique());g=df[(df.scope=="global")&(df.variable=="thetao")];change,lo,hi,n,noninf=paired_bootstrap(g,"crps",200,1,.5);assert n==20 and change<0 and np.isfinite([lo,hi]).all()
 print("Stage 5 smoke test passed: scopes, ensemble CRPS, ablation, and paired bootstrap verified")
if __name__=="__main__":main()
