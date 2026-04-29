from __future__ import annotations

import torch
import torch.nn.functional as F

from .flow_matching import build_flow_mlp, build_flow_unet


class StochasticInterpolant:
    def __init__(self, sigma_coef: float = 1.0, beta_fn: str = "t^2"):
        self.sigma_coef = float(sigma_coef)
        if beta_fn not in ("t", "t^2"):
            raise ValueError("beta_fn must be 't' or 't^2'")
        self.beta_fn = beta_fn

    def wide(self, t: torch.Tensor, spatial: bool) -> torch.Tensor:
        if spatial:
            return t.view(t.shape[0], 1, 1, 1)
        return t.view(t.shape[0], 1)

    def alpha(self, t: torch.Tensor, spatial: bool) -> torch.Tensor:
        return self.wide(1.0 - t, spatial)

    def alpha_dot(self, t: torch.Tensor, spatial: bool) -> torch.Tensor:
        return self.wide(-torch.ones_like(t), spatial)

    def beta(self, t: torch.Tensor, spatial: bool) -> torch.Tensor:
        if self.beta_fn == "t^2":
            return self.wide(t * t, spatial)
        return self.wide(t, spatial)

    def beta_dot(self, t: torch.Tensor, spatial: bool) -> torch.Tensor:
        if self.beta_fn == "t^2":
            return self.wide(2.0 * t, spatial)
        return self.wide(torch.ones_like(t), spatial)

    def sigma(self, t: torch.Tensor, spatial: bool) -> torch.Tensor:
        return self.sigma_coef * self.wide(1.0 - t, spatial)

    def sigma_dot(self, t: torch.Tensor, spatial: bool) -> torch.Tensor:
        return self.sigma_coef * self.wide(-torch.ones_like(t), spatial)

    def gamma(self, t: torch.Tensor, spatial: bool) -> torch.Tensor:
        return self.wide(torch.sqrt(t.clamp(min=0.0)), spatial) * self.sigma(t, spatial)

    def build_state(
        self,
        t: torch.Tensor,
        z0: torch.Tensor,
        z1: torch.Tensor,
        noise: torch.Tensor,
        spatial: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        at = self.alpha(t, spatial)
        bt = self.beta(t, spatial)
        adot = self.alpha_dot(t, spatial)
        bdot = self.beta_dot(t, spatial)
        root_t = self.wide(torch.sqrt(t.clamp(min=0.0)), spatial)
        gamma_t = self.gamma(t, spatial)
        sdot = self.sigma_dot(t, spatial)
        zt = at * z0 + bt * z1 + gamma_t * noise
        drift_target = adot * z0 + bdot * z1 + (sdot * root_t) * noise
        return zt, drift_target


class StochasticInterpolantUtils:
    def __init__(
        self,
        sigma_coef: float = 1.0,
        beta_fn: str = "t^2",
    ):
        self.interpolant = StochasticInterpolant(sigma_coef=sigma_coef, beta_fn=beta_fn)
        self.sigma_coef = sigma_coef
        self.beta_fn = beta_fn

    def compute_loss(self, model, x1, condition, t=None):
        b = x1.shape[0]
        device = x1.device
        spatial = x1.dim() == 4

        if x1.dim() == 2:
            if condition.shape != x1.shape:
                raise ValueError(
                    f"condition and x1 must match for vector latents; got {condition.shape} vs {x1.shape}"
                )
        elif x1.dim() == 4:
            if condition.shape[2:] != x1.shape[2:]:
                condition = F.interpolate(
                    condition,
                    size=x1.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            raise ValueError(f"x1 must be 2D or 4D, got shape {x1.shape}")

        z0 = torch.randn_like(x1)
        noise = torch.randn_like(x1)
        if t is None:
            t = torch.rand(b, device=device)

        z_t, drift_target = self.interpolant.build_state(t, z0, x1, noise, spatial=spatial)
        t_scaled = t * 1000.0
        b_pred = model(z_t, condition, t_scaled)

        if spatial:
            if b_pred.shape[2:] != drift_target.shape[2:]:
                b_pred = F.interpolate(
                    b_pred,
                    size=drift_target.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if b_pred.shape[1] != drift_target.shape[1]:
                if b_pred.shape[1] > drift_target.shape[1]:
                    b_pred = b_pred[:, : drift_target.shape[1], :, :]
                else:
                    pad = torch.zeros(
                        b_pred.shape[0],
                        drift_target.shape[1] - b_pred.shape[1],
                        b_pred.shape[2],
                        b_pred.shape[3],
                        device=b_pred.device,
                        dtype=b_pred.dtype,
                    )
                    b_pred = torch.cat([b_pred, pad], dim=1)

        return F.mse_loss(b_pred, drift_target)

    @torch.no_grad()
    def sample(self, model, condition, steps=25, z0=None, t_max=0.999):
        spatial = condition.dim() == 4
        b = condition.shape[0]
        device = condition.device
        if z0 is None:
            if spatial:
                z0 = torch.randn(
                    b,
                    condition.shape[1],
                    condition.shape[2],
                    condition.shape[3],
                    device=device,
                    dtype=condition.dtype,
                )
            else:
                z0 = torch.randn(
                    b, condition.shape[1], device=device, dtype=condition.dtype
                )
        xt = z0
        times = torch.linspace(0, t_max, steps + 1, device=device)
        dt = t_max / steps
        I = self.interpolant
        for i in range(steps):
            t_curr = times[i]
            t_batch = torch.full((b,), t_curr, device=device)
            t_scaled = t_batch * 1000.0
            b_pred = model(xt, condition, t_scaled)
            if spatial:
                if b_pred.shape[2:] != xt.shape[2:]:
                    b_pred = F.interpolate(
                        b_pred,
                        size=xt.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if b_pred.shape[1] != xt.shape[1]:
                    if b_pred.shape[1] > xt.shape[1]:
                        b_pred = b_pred[:, : xt.shape[1], :, :]
                    else:
                        pad = torch.zeros(
                            b_pred.shape[0],
                            xt.shape[1] - b_pred.shape[1],
                            b_pred.shape[2],
                            b_pred.shape[3],
                            device=b_pred.device,
                            dtype=b_pred.dtype,
                        )
                        b_pred = torch.cat([b_pred, pad], dim=1)
            g = I.sigma(t_batch, spatial)
            eps = torch.randn_like(xt)
            xt = xt + b_pred * dt + g * eps * (dt**0.5)
        return xt


@torch.no_grad()
def sample_latent_stochastic_interpolant(
    drift_model,
    vae,
    condition_image,
    si_utils,
    num_inference_steps=25,
    device="cpu",
):
    vae.eval()
    drift_model.eval()
    condition_z = vae.encode_to_latent(condition_image)
    predicted_z = si_utils.sample(
        drift_model, condition_z, steps=num_inference_steps
    )
    predicted_image = vae.decode_from_latent(
        predicted_z, target_size=condition_image.shape
    )
    return torch.clamp(predicted_image, 0.0, 1.0)


@torch.no_grad()
def sample_pixel_stochastic_interpolant(
    drift_model,
    condition_image,
    si_utils,
    num_inference_steps=25,
    device="cpu",
):
    drift_model.eval()
    condition_image = condition_image.to(device)
    predicted = si_utils.sample(
        drift_model, condition_image, steps=num_inference_steps
    )
    return torch.clamp(predicted, 0.0, 1.0)


__all__ = [
    "StochasticInterpolant",
    "StochasticInterpolantUtils",
    "build_flow_mlp",
    "build_flow_unet",
    "sample_latent_stochastic_interpolant",
    "sample_pixel_stochastic_interpolant",
]
