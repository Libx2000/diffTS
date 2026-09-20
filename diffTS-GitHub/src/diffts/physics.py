from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def density_unesco_torch(temperature: torch.Tensor, salinity: torch.Tensor) -> torch.Tensor:
    """Differentiable UNESCO-1983 density at atmospheric pressure (training surrogate)."""
    t = temperature.clamp(-4.0, 40.0)
    s = salinity.clamp(0.0, 45.0)
    rho_w = (999.842594 + 6.793952e-2*t - 9.095290e-3*t**2 +
             1.001685e-4*t**3 - 1.120083e-6*t**4 + 6.536332e-9*t**5)
    a = 0.824493 - 4.0899e-3*t + 7.6438e-5*t**2 - 8.2467e-7*t**3 + 5.3875e-9*t**4
    b = -5.72466e-3 + 1.0227e-4*t - 1.6546e-6*t**2
    return rho_w + a*s + b*s.pow(1.5) + 4.8314e-4*s**2


class PhysicsPrior:
    def __init__(self, path: str | Path, device: torch.device):
        data = np.load(path)
        self.mean = torch.from_numpy(data["mean"].astype(np.float32)).to(device)
        self.inv_cov = torch.from_numpy(data["inv_cov"].astype(np.float32)).to(device)
        self.lat_band_index = torch.from_numpy(data["lat_band_index"].astype(np.int64)).to(device)


def ts_prior_loss(state: torch.Tensor, months: torch.Tensor, prior: PhysicsPrior,
                  wet_mask: torch.Tensor, threshold: float = 9.21) -> torch.Tensor:
    losses = []
    for bi in range(state.shape[0]):
        mi = int(months[bi].item()) - 1
        mu = prior.mean[mi, :, prior.lat_band_index, :]  # D,H,2
        inv = prior.inv_cov[mi, :, prior.lat_band_index, :, :]  # D,H,2,2
        values = torch.stack([state[bi, 0], state[bi, 1]], dim=-1)  # D,H,W,2
        diff = values - mu[:, :, None, :]
        maha = torch.einsum("dhwi,dhij,dhwj->dhw", diff, inv, diff)
        valid = wet_mask & torch.isfinite(maha)
        excess = torch.relu(maha - threshold)
        losses.append((excess[valid].square()).mean() if valid.any() else state.new_zeros(()))
    return torch.stack(losses).mean()


def density_gradient(state: torch.Tensor, positive_depth: torch.Tensor) -> torch.Tensor:
    rho = density_unesco_torch(state[:, 0], state[:, 1])
    dz = positive_depth[1:] - positive_depth[:-1]
    return (rho[:, 1:] - rho[:, :-1]) / dz[None, :, None, None].clamp(min=-1e6, max=1e6)


def vertical_physics_losses(state: torch.Tensor, truth: torch.Tensor, positive_depth: torch.Tensor,
                            wet_mask: torch.Tensor, gradient_scale: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    pred_grad = density_gradient(state, positive_depth)
    true_grad = density_gradient(truth, positive_depth)
    adjacent = wet_mask[1:] & wet_mask[:-1]
    valid = adjacent[None].expand_as(pred_grad)
    normalized = pred_grad / gradient_scale
    stability = torch.relu(-normalized[valid]).mean() if valid.any() else state.new_zeros(())
    match = torch.abs((pred_grad - true_grad) / gradient_scale)
    gradient_match = match[valid].mean() if valid.any() else state.new_zeros(())
    return stability, gradient_match


def teos10_density_numpy(temperature: np.ndarray, salinity: np.ndarray, depth: np.ndarray,
                         lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return in-situ density and potential-density anomaly sigma0 using TEOS-10."""
    try:
        import gsw
    except ImportError as exc:
        raise ImportError("Install gsw (`conda install -c conda-forge gsw`) for Stage-4 evaluation.") from exc
    z = np.asarray(depth, dtype=np.float64)
    if np.nanmean(z) > 0:
        z = -np.abs(z)
    p = gsw.p_from_z(z[:, None], np.asarray(lat)[None, :])[:, :, None]
    lon3 = np.broadcast_to(np.asarray(lon)[None, None, :], temperature.shape)
    lat3 = np.broadcast_to(np.asarray(lat)[None, :, None], temperature.shape)
    p3 = np.broadcast_to(p, temperature.shape)
    sa = gsw.SA_from_SP(salinity, p3, lon3, lat3)
    ct = gsw.CT_from_pt(sa, temperature)
    return gsw.rho(sa, ct, p3), gsw.sigma0(sa, ct)


class PhysicalAccumulator:
    def __init__(self, depth: np.ndarray, lat: np.ndarray, lon: np.ndarray, mask: np.ndarray):
        self.depth, self.lat, self.lon = depth, lat, lon
        self.mask = mask.astype(bool)
        self.area = np.cos(np.deg2rad(lat)).clip(1e-6, None)[:, None]
        self.stats: dict[int, np.ndarray] = {}

    def update(self, lead: int, state: np.ndarray, truth: np.ndarray) -> None:
        rho, sigma0 = teos10_density_numpy(state[0], state[1], self.depth, self.lat, self.lon)
        rho_true, sigma0_true = teos10_density_numpy(truth[0], truth[1], self.depth, self.lat, self.lon)
        depth = np.abs(self.depth)
        dz = depth[1:] - depth[:-1]
        grad = (sigma0[1:] - sigma0[:-1]) / dz[:, None, None]
        grad_true = (sigma0_true[1:] - sigma0_true[:-1]) / dz[:, None, None]
        adjacent = self.mask[1:] & self.mask[:-1]
        w3 = np.broadcast_to(self.area, self.mask.shape)
        wa = np.broadcast_to(self.area, adjacent.shape)
        valid = self.mask & np.isfinite(rho_true) & np.isfinite(rho)
        valid_a = adjacent & np.isfinite(grad_true) & np.isfinite(grad)
        values = np.asarray([
            w3[valid].sum(), np.sum(w3[valid] * (rho[valid] - rho_true[valid])**2),
            wa[valid_a].sum(), np.sum(wa[valid_a] * np.abs(grad[valid_a] - grad_true[valid_a])),
            np.sum(wa[valid_a] * (grad[valid_a] < 0)),
            np.sum(wa[valid_a] * (grad_true[valid_a] < 0)),
        ])
        self.stats[int(lead)] = self.stats.get(int(lead), np.zeros(6)) + values

    def rows(self) -> list[dict]:
        rows = []
        for lead, s in sorted(self.stats.items()):
            rows.append({"lead_day": lead, "density_rmse": np.sqrt(s[1]/s[0]),
                         "density_gradient_mae": s[3]/s[2],
                         "unstable_fraction_forecast": s[4]/s[2],
                         "unstable_fraction_truth": s[5]/s[2]})
        return rows
