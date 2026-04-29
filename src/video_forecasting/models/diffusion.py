from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionScheduler:
    """
    Diffusion noise scheduler for DDPM.

    Implements linear and cosine noise schedules.
    """

    def __init__(
        self,
        num_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type="linear",
    ):
        """
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
            schedule_type: 'linear' or 'cosine'
        """
        self.num_timesteps = num_timesteps

        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            # Cosine schedule as in Improved DDPM
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For sampling
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_start, t, noise=None):
        """
        Add noise to x_start according to timestep t.

        Args:
            x_start: Clean latents [B, latent_dim] (1D vectors)
            t: Timesteps [B] (on same device as x_start)
            noise: Optional noise tensor (if None, sample new noise)

        Returns:
            Noisy latents x_t [B, latent_dim]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Move scheduler tensors to same device as x_start
        device = x_start.device
        t_cpu = t.cpu() if t.device != torch.device("cpu") else t
        view_shape = (x_start.shape[0],) + (1,) * (x_start.dim() - 1)
        sqrt_alphas_cumprod_t = (
            self.sqrt_alphas_cumprod[t_cpu].to(device).reshape(view_shape)
        )
        sqrt_one_minus_alphas_cumprod_t = (
            self.sqrt_one_minus_alphas_cumprod[t_cpu].to(device).reshape(view_shape)
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalLatentMLP(nn.Module):
    """
    Conditional MLP for latent diffusion with 1D latents.

    Predicts noise ε in latent space given:
    - Noisy future frame latent z_t (1D vector)
    - Current frame latent (condition, 1D vector)
    - Timestep t
    """

    def __init__(
        self,
        latent_dim=64,  # Dimension of 1D latent vector
        condition_dim=None,
        time_emb_dim=64,
        hidden_dims=[256, 512, 256],  # Hidden layer dimensions
        dropout=0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_dim = latent_dim if condition_dim is None else condition_dim
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Input: concatenate noisy latent, condition latent, and time embedding
        input_dim = latent_dim + self.condition_dim + time_emb_dim

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, z, condition_z, timestep):
        """
        Args:
            z: Noisy future frame latent [B, latent_dim] (1D vector)
            condition_z: Current frame latent [B, latent_dim] (1D vector)
            timestep: Diffusion timestep [B] (scaled to [0, 1000])

        Returns:
            Predicted noise in latent space [B, latent_dim]
        """
        # Time embedding
        time_emb = self.time_embed(timestep)  # [B, time_emb_dim]

        # Concatenate noisy latent, condition latent, and time embedding
        x = torch.cat(
            [z, condition_z, time_emb], dim=1
        )  # [B, latent_dim + latent_dim + time_emb_dim]

        # Pass through MLP
        noise_pred = self.mlp(x)  # [B, latent_dim]

        return noise_pred


@torch.no_grad()
def sample_latent_diffusion(
    diffusion_model,
    vae,
    condition_image,
    scheduler,
    num_inference_steps=50,
    device="cpu",
):
    """
    Complete pipeline: encode condition, sample in latent space using DDPM, decode to image.
    Args:
        diffusion_model: Trained noise prediction model
        vae: Trained VAE encoder/decoder
        condition_image: Current frame [1, C, H, W] in [0, 1] range
        scheduler: Diffusion scheduler
        num_inference_steps: Number of denoising steps
        device: Device to run on
    Returns:
        Generated future frame [1, C, H, W] in [0, 1] range
    """
    vae.eval()
    diffusion_model.eval()
    if condition_image.dim() == 5:
        bsz, context_frames, channels, height, width = condition_image.shape
        flat = condition_image.reshape(bsz * context_frames, channels, height, width)
        condition_z = vae.encode_to_latent(flat)
        target_size = condition_image[:, -1].shape
        sample_shape = (bsz, condition_z.shape[1])
        condition_z = condition_z.reshape(bsz, context_frames * condition_z.shape[1])
    else:
        condition_z = vae.encode_to_latent(condition_image)
        target_size = condition_image.shape
        sample_shape = condition_z.shape

    # Start from pure noise
    z = torch.randn(*sample_shape, device=condition_z.device, dtype=condition_z.dtype)

    # Create timestep schedule for inference (can use fewer steps than training)
    timesteps = torch.linspace(
        scheduler.num_timesteps - 1, 0, num_inference_steps, device=device
    ).long()

    # Denoising loop
    for i, t in enumerate(timesteps):
        # Scale timestep for time embedding
        t_scaled = t.float().unsqueeze(0) * (1000.0 / scheduler.num_timesteps)

        # Predict noise
        noise_pred = diffusion_model(z, condition_z, t_scaled)

        # Compute coefficients for denoising step
        alpha_t = scheduler.alphas[t].to(device)
        alpha_cumprod_t = scheduler.alphas_cumprod[t].to(device)
        beta_t = scheduler.betas[t].to(device)

        if i < len(timesteps) - 1:
            alpha_cumprod_t_prev = scheduler.alphas_cumprod[timesteps[i + 1]].to(device)
            # Compute predicted x0
            pred_x0 = (z - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(
                alpha_cumprod_t
            )
            # Compute direction pointing to x_t
            pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev) * noise_pred
            # Compute random noise for stochastic sampling
            random_noise = torch.randn_like(z)
            # Compute variance
            variance = scheduler.posterior_variance[t].to(device)
            # Sample next z
            z = (
                torch.sqrt(alpha_cumprod_t_prev) * pred_x0
                + pred_dir
                + torch.sqrt(variance) * random_noise
            )
        else:
            # Last step: deterministic
            z = (
                z - beta_t * noise_pred / torch.sqrt(1 - alpha_cumprod_t)
            ) / torch.sqrt(alpha_t)

    # Decode to image space
    predicted_image = vae.decode_from_latent(z, target_size=target_size)
    # Clamp to [0, 1]
    predicted_image = torch.clamp(predicted_image, 0.0, 1.0)
    return predicted_image


@torch.no_grad()
def sample_pixel_diffusion(
    diffusion_model,
    condition_image,
    scheduler,
    num_inference_steps=50,
    device="cpu",
):
    """
    Sample a future frame directly in pixel space using conditional diffusion.

    Args:
        diffusion_model: Trained noise prediction model
        condition_image: Current frame [B, C, H, W] in [0, 1]
        scheduler: DiffusionScheduler
        num_inference_steps: Number of denoising steps
        device: Device to run on
    Returns:
        Predicted future frame [B, C, H, W] in [0, 1]
    """
    diffusion_model.eval()
    condition_image = condition_image.to(device)
    if condition_image.dim() == 5:
        bsz, context_frames, channels, height, width = condition_image.shape
        sample_shape = (bsz, channels, height, width)
        condition_image = condition_image.reshape(
            bsz, context_frames * channels, height, width
        )
    else:
        sample_shape = condition_image.shape
    x = torch.randn(*sample_shape, device=device, dtype=condition_image.dtype)
    timesteps = torch.linspace(
        scheduler.num_timesteps - 1, 0, num_inference_steps, device=device
    ).long()

    for i, t in enumerate(timesteps):
        t_scaled = torch.full(
            (condition_image.shape[0],),
            float(t.item()) * (1000.0 / scheduler.num_timesteps),
            device=device,
        )
        noise_pred = diffusion_model(x, condition_image, t_scaled)

        alpha_t = scheduler.alphas[t].to(device)
        alpha_cumprod_t = scheduler.alphas_cumprod[t].to(device)
        beta_t = scheduler.betas[t].to(device)

        if i < len(timesteps) - 1:
            alpha_cumprod_t_prev = scheduler.alphas_cumprod[timesteps[i + 1]].to(device)
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(
                alpha_cumprod_t
            )
            pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev) * noise_pred
            variance = scheduler.posterior_variance[t].to(device)
            x = (
                torch.sqrt(alpha_cumprod_t_prev) * pred_x0
                + pred_dir
                + torch.sqrt(variance) * torch.randn_like(x)
            )
        else:
            x = (
                x - beta_t * noise_pred / torch.sqrt(1 - alpha_cumprod_t)
            ) / torch.sqrt(alpha_t)

    return torch.clamp(x, 0.0, 1.0)


def build_diffusion_mlp(**kwargs):
    return ConditionalLatentMLP(**kwargs)
