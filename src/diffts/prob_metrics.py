from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


class ProbabilisticAccumulator:
    def __init__(self, lat: np.ndarray, mask: np.ndarray, members: int):
        self.area = np.cos(np.deg2rad(lat)).clip(1e-6, None)[:, None]
        self.mask = mask.astype(bool)
        self.members = members
        self.stats = defaultdict(lambda: np.zeros(8, dtype=np.float64))
        self.ranks = defaultdict(lambda: np.zeros(members + 1, dtype=np.int64))

    def update(self, lead: int, ensemble: np.ndarray, truth: np.ndarray) -> None:
        # ensemble [M,V,D,H,W], truth [V,D,H,W]
        mean = ensemble.mean(axis=0)
        spread = ensemble.std(axis=0, ddof=1)
        sorted_ens = np.sort(ensemble, axis=0)
        coeff = (2 * np.arange(self.members) - self.members + 1).reshape(self.members, 1, 1, 1, 1)
        crps = np.mean(np.abs(ensemble - truth[None]), axis=0) - np.sum(coeff * sorted_ens, axis=0) / self.members**2
        lower80, upper80 = np.quantile(ensemble, [0.1, 0.9], axis=0)
        lower90, upper90 = np.quantile(ensemble, [0.05, 0.95], axis=0)
        for var in range(truth.shape[0]):
            for depth in range(truth.shape[1]):
                valid = self.mask[depth] & np.isfinite(truth[var, depth])
                if not valid.any():
                    continue
                w = np.broadcast_to(self.area, valid.shape)[valid]
                error = mean[var, depth][valid] - truth[var, depth][valid]
                values = np.array([
                    w.sum(), np.sum(w * np.abs(error)), np.sum(w * error**2),
                    np.sum(w * spread[var, depth][valid]), np.sum(w * spread[var, depth][valid]**2),
                    np.sum(w * crps[var, depth][valid]),
                    np.sum(w * ((truth[var, depth][valid] >= lower80[var, depth][valid]) & (truth[var, depth][valid] <= upper80[var, depth][valid]))),
                    np.sum(w * ((truth[var, depth][valid] >= lower90[var, depth][valid]) & (truth[var, depth][valid] <= upper90[var, depth][valid]))),
                ])
                self.stats[(int(lead), var, depth)] += values
                ranks = np.sum(ensemble[:, var, depth][:, valid] < truth[var, depth][valid][None], axis=0)
                self.ranks[(int(lead), var)] += np.bincount(ranks, minlength=self.members + 1)

    def dataframes(self, depths: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        names = ["thetao", "so"]
        rows = []
        for (lead, var, depth), s in sorted(self.stats.items()):
            w = s[0]
            rmse = np.sqrt(s[2] / w)
            spread_rms = np.sqrt(s[4] / w)
            rows.append({"lead_day": lead, "variable": names[var], "depth_index": depth,
                         "depth": float(depths[depth]), "mae_ensemble_mean": s[1] / w,
                         "rmse_ensemble_mean": rmse, "mean_spread": s[3] / w,
                         "rms_spread": spread_rms, "spread_skill_ratio": spread_rms / max(rmse, 1e-12),
                         "crps": s[5] / w, "coverage80": s[6] / w, "coverage90": s[7] / w,
                         "weight_sum": w})
        rank_rows = []
        for (lead, var), counts in sorted(self.ranks.items()):
            for rank, count in enumerate(counts):
                rank_rows.append({"lead_day": lead, "variable": names[var], "rank": rank, "count": int(count)})
        return pd.DataFrame(rows), pd.DataFrame(rank_rows)
