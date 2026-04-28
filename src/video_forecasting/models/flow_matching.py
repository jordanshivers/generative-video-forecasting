from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingUtils:
    """
    Utilities for Conditional Flow Matching (CFM).
    Implements:
    1. Optimal Transport path interpolation
    2. Velocity field target calculation
    3. ODE sampling (Euler method)
    """

    def __init__(self, sigma_min=1e-5):
        """
        Args:
            sigma_min: Small noise level for numerical stability
        """
        self.sigma_min = sigma_min

    def compute_loss(self, model, x1, condition, t=None):
        """
        Compute CFM loss.
        Args:
            model: The velocity prediction model v_theta(x, t, condition)
            x1: Target data (Frame t+m latent) [B, C, H, W]
            condition: Conditioning data (Frame t latent) [B, C, H, W]
            t: Timesteps [B] (optional, if None, sample uniformly)
        Returns:
            loss: Mean squared error between predicted and target velocity
        """
        b = x1.shape[0]
        device = x1.device
        # Ensure condition has same spatial dimensions as x1
        if condition.shape[2:] != x1.shape[2:]:
            condition = F.interpolate(
                condition, size=x1.shape[2:], mode="bilinear", align_corners=False
            )
        # 1. Sample x0 from Normal(0, I) - match x1's shape
        x0 = torch.randn_like(x1)
        # 2. Sample timesteps t ~ U[0, 1] if not provided
        if t is None:
            t = torch.rand(b, device=device)
        # Reshape t for broadcasting [B, 1, 1, 1]
        t_expand = t.view(b, 1, 1, 1)
        # 3. Compute x_t (interpolation)
        # Optimal Transport path: x_t = (1 - (1-sigma_min)t) * x0 + t * x1
        t_flow = 1 - (1 - self.sigma_min) * t_expand
        x_t = t_flow * x0 + t_expand * x1
        # 4. Compute target velocity u_t
        # u_t = dx_t/dt = x1 - (1-sigma_min)x0
        u_t = x1 - (1 - self.sigma_min) * x0
        # 5. Predict velocity
        # Scaling time by 1000 for UNet embedding which typically expects larger values
        t_scaled = t * 1000.0
        v_pred = model(x_t, condition, t_scaled)
        # 6. Ensure v_pred matches u_t spatial dimensions (UNet may change them due to upsampling)
        # Check both spatial dimensions and channel dimensions
        if v_pred.shape[2] != u_t.shape[2] or v_pred.shape[3] != u_t.shape[3]:
            v_pred = F.interpolate(
                v_pred,
                size=(u_t.shape[2], u_t.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
        # Also ensure channel dimensions match (should be same, but be safe)
        if v_pred.shape[1] != u_t.shape[1]:
            # This shouldn't happen, but if it does, use a 1x1 conv or take first channels
            if v_pred.shape[1] > u_t.shape[1]:
                v_pred = v_pred[:, : u_t.shape[1], :, :]
            else:
                # Pad with zeros if needed
                padding = torch.zeros(
                    v_pred.shape[0],
                    u_t.shape[1] - v_pred.shape[1],
                    v_pred.shape[2],
                    v_pred.shape[3],
                    device=v_pred.device,
                    dtype=v_pred.dtype,
                )
                v_pred = torch.cat([v_pred, padding], dim=1)
        # 7. MSE Loss
        loss = F.mse_loss(v_pred, u_t)
        return loss

    @torch.no_grad()
    def sample(self, model, condition, steps=25, x0=None):
        """
        Sample from the model using Euler ODE solver.
        Args:
            model: The velocity prediction model
            condition: Conditioning data (Frame t latent)
            steps: Number of integration steps
            x0: Initial noise (optional)
        Returns:
            Generated latents (Frame t+m)
        """
        b = condition.shape[0]
        device = condition.device
        # Start from Gaussian noise
        if x0 is None:
            x = torch.randn_like(condition)
        else:
            x = x0
        # Time steps 0 -> 1
        times = torch.linspace(0, 1, steps + 1, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t_curr = times[i]
            # Broadcast t for batch
            t_batch = torch.ones(b, device=device) * t_curr
            # Predict velocity (scale t by 1000)
            t_scaled = t_batch * 1000.0
            v_pred = model(x, condition, t_scaled)
            # Ensure v_pred matches x spatial dimensions (UNet may change them)
            if v_pred.shape[2] != x.shape[2] or v_pred.shape[3] != x.shape[3]:
                v_pred = F.interpolate(
                    v_pred,
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            # Ensure channel dimensions match
            if v_pred.shape[1] != x.shape[1]:
                if v_pred.shape[1] > x.shape[1]:
                    v_pred = v_pred[:, : x.shape[1], :, :]
                else:
                    padding = torch.zeros(
                        v_pred.shape[0],
                        x.shape[1] - v_pred.shape[1],
                        v_pred.shape[2],
                        v_pred.shape[3],
                        device=v_pred.device,
                        dtype=v_pred.dtype,
                    )
                    v_pred = torch.cat([v_pred, padding], dim=1)
            # Euler step: x_{t+dt} = x_t + v_t * dt
            x = x + v_pred * dt
        return x


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


class ResidualBlock(nn.Module):
    """Residual block with group normalization."""

    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        h = self.block1(x)
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for low-resolution feature maps."""

    def __init__(self, channels, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        h_in = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)
        attn = torch.einsum("bci,bcj->bij", q, k) / math.sqrt(c)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bij,bcj->bci", attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return h_in + out


class ConditionalLatentUNet(nn.Module):
    """
    Conditional U-Net for latent flow matching.
    Predicts velocity fields v_t in latent space given:
    - Interpolated latent x_t along the flow path
    - Current frame latent (condition)
    - Timestep t
    """

    def __init__(
        self,
        latent_channels=4,  # Channels in latent space
        condition_channels=4,  # Channels in condition latent
        out_channels=4,  # Output noise channels (same as latent_channels)
        time_emb_dim=64,
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        groups=8,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        # Initial projection
        self.input_conv = nn.Conv2d(
            latent_channels + condition_channels, base_channels, 3, padding=1
        )
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            # Add residual blocks for this level
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResidualBlock(ch, out_ch, time_emb_dim, groups)
                )
                ch = out_ch
            # Add downsample (except for last level)
            if i < len(channel_multipliers) - 1:
                self.encoder_downsamples.append(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                )
            else:
                self.encoder_downsamples.append(nn.Identity())
        # Bottleneck
        self.bottleneck = ResidualBlock(ch, ch, time_emb_dim, groups)
        self.attn_bottleneck = AttentionBlock(ch, groups=groups)
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        reversed_mults = list(reversed(self.channel_multipliers))
        encoder_channels = [base_channels * mult for mult in self.channel_multipliers]
        ch = encoder_channels[-1]
        for i, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            # Conv to handle skip connection
            if i < len(reversed_mults) - 1:
                encoder_level_idx = len(reversed_mults) - 1 - i
                skip_ch = encoder_channels[encoder_level_idx]
                self.decoder_convs.append(nn.Conv2d(ch + skip_ch, out_ch, 1))
            else:
                self.decoder_convs.append(nn.Conv2d(ch, out_ch, 1))
            # Residual blocks for this level
            for _ in range(self.num_res_blocks + 1):
                self.decoder_blocks.append(
                    ResidualBlock(out_ch, out_ch, time_emb_dim, groups)
                )
            # Upsample (except last level)
            if i < len(reversed_mults) - 1:
                self.decoder_upsamples.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    )
                )
                ch = out_ch
            else:
                self.decoder_upsamples.append(nn.Identity())
                ch = out_ch
        # Output
        self.output_norm = nn.GroupNorm(groups, ch)
        self.output_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, z, condition_z, timestep):
        """
        Args:
            z: Noisy future frame latent [B, latent_channels, H, W]
            condition_z: Current frame latent [B, condition_channels, H, W]
            timestep: Flow matching timestep [B] (scaled to [0, 1000])
        Returns:
            Predicted noise in latent space [B, out_channels, H, W]
        """
        # Time embedding
        time_emb = self.time_embed(timestep)
        # Concatenate noisy latent and condition
        x = torch.cat([z, condition_z], dim=1)
        x = self.input_conv(x)
        # Encoder with skip connections
        skip_connections = []
        block_idx = 0
        for level_idx in range(len(self.channel_multipliers)):
            num_blocks_per_level = self.num_res_blocks
            for _ in range(num_blocks_per_level):
                x = self.encoder_blocks[block_idx](x, time_emb)
                block_idx += 1
            skip_connections.append(x)
            x = self.encoder_downsamples[level_idx](x)
        # Bottleneck
        x = self.bottleneck(x, time_emb)
        x = self.attn_bottleneck(x)
        # Decoder with skip connections
        skip_idx = len(skip_connections) - 1
        block_idx = 0
        for level_idx in range(len(self.decoder_upsamples)):
            if skip_idx >= 0 and level_idx < len(self.decoder_upsamples) - 1:
                skip = skip_connections[skip_idx]
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(
                        x, size=skip.shape[2:], mode="bilinear", align_corners=False
                    )
                x = torch.cat([x, skip], dim=1)
                skip_idx -= 1
            x = self.decoder_convs[level_idx](x)
            num_blocks = len(self.decoder_blocks) // len(self.decoder_upsamples)
            for _ in range(num_blocks):
                x = self.decoder_blocks[block_idx](x, time_emb)
                block_idx += 1
            x = self.decoder_upsamples[level_idx](x)
        # Output
        x = self.output_norm(x)
        x = F.silu(x)
        x = self.output_conv(x)
        return x


class LatentSinusoidalPositionEmbeddings(nn.Module):
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


class ConditionalFlowMLP(nn.Module):
    """
    Conditional MLP for latent flow matching with 1D latents.

    Predicts velocity fields v_t in latent space given:
    - Interpolated latent x_t along the flow path (1D vector)
    - Current frame latent (condition, 1D vector)
    - Timestep t
    """

    def __init__(
        self,
        latent_dim=64,  # Dimension of 1D latent vector
        time_emb_dim=64,
        hidden_dims=[256, 512, 256],  # Hidden layer dimensions
        dropout=0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            LatentSinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Input: concatenate noisy latent, condition latent, and time embedding
        input_dim = latent_dim + latent_dim + time_emb_dim  # z + condition_z + time_emb

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
            timestep: Flow matching timestep [B] (scaled to [0, 1000])

        Returns:
            Predicted velocity in latent space [B, latent_dim]
        """
        # Time embedding
        time_emb = self.time_embed(timestep)  # [B, time_emb_dim]

        # Concatenate noisy latent, condition latent, and time embedding
        x = torch.cat(
            [z, condition_z, time_emb], dim=1
        )  # [B, latent_dim + latent_dim + time_emb_dim]

        # Pass through MLP
        velocity = self.mlp(x)  # [B, latent_dim]

        return velocity


@torch.no_grad()
def sample_latent_flow_matching(
    flow_matching_model,
    vae,
    condition_image,
    flow_utils,
    num_inference_steps=25,
    device="cpu",
):
    """
    Complete pipeline: encode condition, sample in latent space using Flow Matching, decode to image.
    Args:
        flow_matching_model: Trained velocity prediction model
        vae: Trained VAE encoder/decoder
        condition_image: Current frame [1, C, H, W] in [0, 1] range
        flow_utils: Flow Matching Utils
        num_inference_steps: Number of Euler integration steps
        device: Device to run on
    Returns:
        Generated future frame [1, C, H, W] in [0, 1] range
    """
    vae.eval()
    flow_matching_model.eval()
    # Encode condition to latent
    condition_z = vae.encode_to_latent(condition_image)
    # Sample in latent space using ODE solver
    predicted_z = flow_utils.sample(
        flow_matching_model, condition_z, steps=num_inference_steps
    )
    # Decode to image space
    predicted_image = vae.decode_from_latent(
        predicted_z, target_size=condition_image.shape
    )
    # Clamp to [0, 1]
    predicted_image = torch.clamp(predicted_image, 0.0, 1.0)
    return predicted_image


def build_flow_unet(**kwargs):
    return ConditionalLatentUNet(**kwargs)


def build_flow_mlp(**kwargs):
    return ConditionalFlowMLP(**kwargs)
