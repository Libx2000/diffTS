from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


class MetricAccumulator:
    def __init__(self, lat: np.ndarray, mask: np.ndarray):
        self.area = np.cos(np.deg2rad(lat)).clip(1e-6, None)[:, None]
        self.mask = mask.astype(bool)
        self.stats = defaultdict(lambda: np.zeros(9, dtype=np.float64))

    def update(self, model: str, lead: int, pred: np.ndarray, obs: np.ndarray, clim: np.ndarray) -> None:
        # 输入均为 [variable, depth, lat, lon]
        for var in range(pred.shape[0]):
            for depth in range(pred.shape[1]):
                valid = self.mask[depth] & np.isfinite(pred[var, depth]) & np.isfinite(obs[var, depth])
                if not valid.any():
                    continue
                w = np.broadcast_to(self.area, valid.shape)[valid]
                e = pred[var, depth][valid] - obs[var, depth][valid]
                pa = pred[var, depth][valid] - clim[var, depth][valid]
                oa = obs[var, depth][valid] - clim[var, depth][valid]
                s = self.stats[(model, int(lead), var, depth)]
                s[0] += w.sum()
                s[1] += np.sum(w * e)
                s[2] += np.sum(w * np.abs(e))
                s[3] += np.sum(w * e * e)
                s[4] += np.sum(w * pa)
                s[5] += np.sum(w * oa)
                s[6] += np.sum(w * pa * pa)
                s[7] += np.sum(w * oa * oa)
                s[8] += np.sum(w * pa * oa)

    def dataframe(self, depths: np.ndarray) -> pd.DataFrame:
        rows = []
        names = ["thetao", "so"]
        for (model, lead, var, depth), s in sorted(self.stats.items()):
            w, sum_e, sum_ae, sum_se, sum_p, sum_o, sum_pp, sum_oo, sum_po = s
            cov = sum_po - sum_p * sum_o / w
            vp = sum_pp - sum_p * sum_p / w
            vo = sum_oo - sum_o * sum_o / w
            acc = cov / np.sqrt(max(vp * vo, 1e-30))
            rows.append({
                "model": model,
                "lead_day": lead,
                "variable": names[var],
                "depth_index": depth,
                "depth": float(depths[depth]),
                "bias": sum_e / w,
                "mae": sum_ae / w,
                "rmse": np.sqrt(sum_se / w),
                "acc": acc,
                "weight_sum": w,
            })
        return pd.DataFrame(rows)
