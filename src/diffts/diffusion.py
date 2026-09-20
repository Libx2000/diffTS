from __future__ import annotations

import math

import torch
from torch import nn


def _cosine_betas(steps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((x / steps + s) / (1 + s)) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


def _extract(values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    return values.gather(0, t).reshape(t.shape[0], *((1,) * (ndim - 1)))


class DiffusionSchedule(nn.Module):
    def __init__(self, steps: int = 1000, schedule: str = "cosine"):
        super().__init__()
        if schedule == "cosine":
            betas = _cosine_betas(steps)
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, steps)
        else:
            raise ValueError(f"未知噪声日程: {schedule}")
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.steps = int(steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    def q_sample(self, clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_alpha_bars, t, clean.ndim) * clean + _extract(
            self.sqrt_one_minus_alpha_bars, t, clean.ndim
        ) * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        shape: tuple[int, ...],
        sampling_steps: int = 50,
        eta: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = condition.device
        x = torch.randn(shape, device=device, generator=generator)
        times = torch.linspace(self.steps - 1, 0, sampling_steps, device=device).long().unique_consecutive()
        for index, time in enumerate(times):
            t = torch.full((shape[0],), int(time), device=device, dtype=torch.long)
            eps = model(x, condition, t)
            ab_t = self.alpha_bars[time]
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0 = x0.clamp(-8.0, 8.0)
            if index == len(times) - 1:
                x = x0
                continue
            prev = times[index + 1]
            ab_prev = self.alpha_bars[prev]
            sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).clamp_min(0).sqrt()
            direction = (1 - ab_prev - sigma**2).clamp_min(0).sqrt() * eps
            noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)
            x = ab_prev.sqrt() * x0 + direction + sigma * noise
        return x
