from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Spatial latent VAE used by latent flow matching.
class SpatialVAEResidualBlock(nn.Module):
    """Residual block for SpatialVAE with GroupNorm (more stable than BatchNorm for generative models)."""

    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()
        # Ensure groups doesn't exceed channel count
        groups_out = min(groups, out_channels)
        groups_in = min(groups, in_channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups_out, out_channels),
            nn.SiLU(),  # SiLU (Swish) - smoother than ReLU, better for generative models
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups_out, out_channels),
        )
        # Residual connection: 1x1 conv if channels or stride changes
        self.res_conv = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(groups_out, out_channels),
            )
            if (in_channels != out_channels or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        out = self.block(x)
        out = out + self.res_conv(x)
        # SiLU activation after residual connection (matching diffusion model pattern)
        out = F.silu(out)
        return out


class SpatialVAEEncoder(nn.Module):
    """
    SpatialVAE Encoder - maps images to latent distribution parameters.
    Uses spatial latents (maintains spatial structure).
    Now with residual connections for better gradient flow.
    """

    def __init__(
        self, in_channels=1, latent_channels=4, hidden_dims=[32, 64, 128, 256]
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims.copy()
        # Encoder layers with residual blocks (using GroupNorm)
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                # First layer: input -> first hidden dim (stride=2 for downsampling)
                self.encoder_blocks.append(
                    SpatialVAEResidualBlock(
                        in_channels, hidden_dims[i], stride=2, groups=8
                    )
                )
            else:
                # Subsequent layers: hidden_dims[i-1] -> hidden_dims[i] (stride=2 for downsampling)
                self.encoder_blocks.append(
                    SpatialVAEResidualBlock(
                        hidden_dims[i - 1], hidden_dims[i], stride=2, groups=8
                    )
                )
        # Final layers to produce mu and logvar (spatial latents)
        self.conv_mu = nn.Conv2d(hidden_dims[-1], latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(hidden_dims[-1], latent_channels, kernel_size=1)

    def forward(self, x):
        """
        Encode input to latent distribution parameters.
        Args:
            x: Input images [B, C, H, W] in [0, 1] range
        Returns:
            mu: Mean of latent distribution [B, latent_channels, H', W']
            logvar: Log variance of latent distribution [B, latent_channels, H', W']
        """
        h = x
        for block in self.encoder_blocks:
            h = block(h)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar


class SpatialVAEDecoder(nn.Module):
    """
    SpatialVAE Decoder with residual connections for better gradient flow and reconstruction quality.
    Adapted for spatial latents.
    """

    def __init__(
        self, latent_channels=4, out_channels=1, hidden_dims=[256, 128, 64, 32]
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims.copy()
        # Project latent to encoder output shape (using GroupNorm and SiLU)
        groups_proj = min(8, hidden_dims[0])
        self.input_proj = nn.Sequential(
            nn.Conv2d(
                latent_channels, hidden_dims[0], kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(groups_proj, hidden_dims[0]),
            nn.SiLU(),  # SiLU (Swish) - smoother activation, better for generative models
        )
        # Decoder blocks with residual connections (using GroupNorm)
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Each block: upsample + residual block
            self.decoder_blocks.append(
                nn.ModuleDict(
                    {
                        "upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                        "residual": SpatialVAEResidualBlock(
                            hidden_dims[i], hidden_dims[i + 1], stride=1, groups=8
                        ),
                    }
                )
            )
        # Final layer to output channels
        self.final_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dims[-1], out_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),  # Output directly in [0, 1]
        )

    def forward(self, z, target_size=None):
        """
        Decode latent to image.
        Args:
            z: Latent codes [B, latent_channels, H', W']
            target_size: Optional target size (B, C, H, W)
        Returns:
            Reconstructed images [B, C, H, W] in [0, 1] range
        """
        # Project latent
        x = self.input_proj(z)
        # Decode with residual blocks
        for block in self.decoder_blocks:
            x = block["upsample"](x)
            x = block["residual"](x)
        # Final upsampling and output
        x = self.final_upsample(x)
        x_recon = self.final_conv(x)
        # Crop or pad to match target size if provided
        if target_size is not None:
            batch_size, channels, target_h, target_w = target_size
            _, _, recon_h, recon_w = x_recon.shape
            # Handle height mismatch
            if recon_h != target_h:
                if recon_h > target_h:
                    # Crop from center
                    crop_h = (recon_h - target_h) // 2
                    x_recon = x_recon[:, :, crop_h : crop_h + target_h, :]
                else:
                    # Pad with zeros to keep values in [0, 1]
                    pad_h_top = (target_h - recon_h) // 2
                    pad_h_bottom = target_h - recon_h - pad_h_top
                    x_recon = F.pad(
                        x_recon,
                        (0, 0, pad_h_top, pad_h_bottom),
                        mode="constant",
                        value=0.0,
                    )
            # Handle width mismatch
            if recon_w != target_w:
                if recon_w > target_w:
                    # Crop from center
                    crop_w = (recon_w - target_w) // 2
                    x_recon = x_recon[:, :, :, crop_w : crop_w + target_w]
                else:
                    # Pad with zeros to keep values in [0, 1]
                    pad_w_left = (target_w - recon_w) // 2
                    pad_w_right = target_w - recon_w - pad_w_left
                    x_recon = F.pad(
                        x_recon,
                        (pad_w_left, pad_w_right, 0, 0),
                        mode="constant",
                        value=0.0,
                    )
        return x_recon


class SpatialVAE(nn.Module):
    """
    Variational Autoencoder - combines encoder and decoder.
    Uses spatial latents (maintains spatial structure).
    """

    def __init__(
        self, in_channels=1, latent_channels=4, hidden_dims=[32, 64, 128, 256]
    ):
        super().__init__()
        self.encoder = SpatialVAEEncoder(in_channels, latent_channels, hidden_dims)
        self.decoder = SpatialVAEDecoder(
            latent_channels, in_channels, list(reversed(hidden_dims))
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample from latent distribution.
        Args:
            mu: Mean [B, latent_channels, H', W']
            logvar: Log variance [B, latent_channels, H', W']
        Returns:
            Sampled latent codes [B, latent_channels, H', W']
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, target_size=None):
        """
        Forward pass: encode, reparameterize, decode.
        Args:
            x: Input images [B, C, H, W] in [0, 1] range
            target_size: Optional target size for decoder output
        Returns:
            x_recon: Reconstructed images [B, C, H, W] in [0, 1] range
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, target_size=target_size)
        return x_recon, mu, logvar

    def encode_to_latent(self, x):
        """
        Encode to deterministic latent (mean only, no sampling).
        Useful for inference.
        """
        mu, _ = self.encoder(x)
        return mu

    def decode_from_latent(self, z, target_size=None):
        """
        Decode from latent codes.
        """
        return self.decoder(z, target_size=target_size)


# 1D latent VAE used by latent flow matching and latent diffusion.
class VectorVAEResidualBlock(nn.Module):
    """Residual block for VectorVAE with GroupNorm (more stable than BatchNorm for generative models)."""

    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()
        # Ensure groups doesn't exceed channel count
        groups_out = min(groups, out_channels)
        groups_in = min(groups, in_channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups_out, out_channels),
            nn.SiLU(),  # SiLU (Swish) - smoother than ReLU, better for generative models
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups_out, out_channels),
        )
        # Residual connection: 1x1 conv if channels or stride changes
        self.res_conv = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(groups_out, out_channels),
            )
            if (in_channels != out_channels or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        out = self.block(x)
        out = out + self.res_conv(x)
        # SiLU activation after residual connection (matching diffusion model pattern)
        out = F.silu(out)
        return out


class VectorVAEEncoder(nn.Module):
    """
    VectorVAE Encoder - maps images to latent distribution parameters.
    Uses 1D latent bottleneck - outputs flattened vectors.
    Now with residual connections for better gradient flow.
    """

    def __init__(self, in_channels=1, latent_dim=64, hidden_dims=[32, 64, 128, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims.copy()
        # Encoder layers with residual blocks (using GroupNorm)
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                # First layer: input -> first hidden dim (stride=2 for downsampling)
                self.encoder_blocks.append(
                    VectorVAEResidualBlock(
                        in_channels, hidden_dims[i], stride=2, groups=8
                    )
                )
            else:
                # Subsequent layers: hidden_dims[i-1] -> hidden_dims[i] (stride=2 for downsampling)
                self.encoder_blocks.append(
                    VectorVAEResidualBlock(
                        hidden_dims[i - 1], hidden_dims[i], stride=2, groups=8
                    )
                )
        # Global average pooling to get [B, C] from [B, C, H, W]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Linear layers to produce mu and logvar (1D vectors)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        """
        Encode input to latent distribution parameters.
        Args:
            x: Input images [B, C, H, W] in [0, 1] range
        Returns:
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log variance of latent distribution [B, latent_dim]
        """
        h = x
        for block in self.encoder_blocks:
            h = block(h)
        # Global pooling: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        h = self.global_pool(h).squeeze(-1).squeeze(-1)
        mu = self.fc_mu(h)  # [B, latent_dim]
        logvar = self.fc_logvar(h)  # [B, latent_dim]
        return mu, logvar


class VectorVAEDecoder(nn.Module):
    """
    VectorVAE Decoder with residual connections for better gradient flow and reconstruction quality.
    Adapted for 1D latent input.
    """

    def __init__(
        self,
        latent_dim=64,
        out_channels=1,
        hidden_dims=[256, 128, 64, 32],
        max_initial_spatial_size=64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims.copy()
        # Support up to max_initial_spatial_size * 2^num_upsampling_steps output size
        # e.g., 64 * 2^4 = 1024x1024 max output
        self.max_initial_spatial_size = max_initial_spatial_size
        self.num_upsampling_steps = (
            len(hidden_dims) - 1 + 1
        )  # decoder blocks + final upsample

        # Linear layer to expand 1D latent to spatial feature map (output enough for max size)
        initial_channels = hidden_dims[0]
        self.fc_expand = nn.Linear(
            latent_dim,
            initial_channels * max_initial_spatial_size * max_initial_spatial_size,
        )

        # Reshape and project to spatial features (using GroupNorm and SiLU)
        groups_proj = min(8, initial_channels)
        self.input_proj = nn.Sequential(
            nn.Conv2d(
                initial_channels, initial_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(groups_proj, initial_channels),
            nn.SiLU(),  # SiLU (Swish) - smoother activation, better for generative models
        )

        # Decoder blocks with residual connections (using GroupNorm)
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Each block: upsample + residual block
            self.decoder_blocks.append(
                nn.ModuleDict(
                    {
                        "upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                        "residual": VectorVAEResidualBlock(
                            hidden_dims[i], hidden_dims[i + 1], stride=1, groups=8
                        ),
                    }
                )
            )

        # Final layer to output channels
        self.final_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dims[-1], out_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),  # Output directly in [0, 1]
        )

    def forward(self, z, target_size=None):
        """
        Decode latent to image.
        Args:
            z: Latent codes [B, latent_dim] (1D vector)
            target_size: Optional target size (B, C, H, W)
        Returns:
            Reconstructed images [B, C, H, W] in [0, 1] range
        """
        # Expand 1D latent: [B, latent_dim] -> [B, C*H*W]
        B = z.shape[0]
        h = self.fc_expand(z)  # [B, C*H*W]

        # Reshape to spatial: [B, C*H*W] -> [B, C, H, W]
        # Calculate required initial spatial size from target_size
        if target_size is not None:
            _, _, target_h, target_w = target_size
            # Calculate what initial size we need: target / 2^num_upsampling_steps
            # Use ceiling division to ensure we don't undershoot
            import math

            required_h = int(math.ceil(target_h / (2**self.num_upsampling_steps)))
            required_w = int(math.ceil(target_w / (2**self.num_upsampling_steps)))
            # Ensure we don't exceed max size
            required_h = min(required_h, self.max_initial_spatial_size)
            required_w = min(required_w, self.max_initial_spatial_size)
        else:
            # Use default (smallest) if no target size
            required_h = required_w = 16

        # Reshape to max spatial size first
        initial_channels = self.hidden_dims[0]
        h = h.view(
            B,
            initial_channels,
            self.max_initial_spatial_size,
            self.max_initial_spatial_size,
        )

        # Crop to required size (from center) if needed
        if (
            required_h != self.max_initial_spatial_size
            or required_w != self.max_initial_spatial_size
        ):
            crop_h = (self.max_initial_spatial_size - required_h) // 2
            crop_w = (self.max_initial_spatial_size - required_w) // 2
            h = h[:, :, crop_h : crop_h + required_h, crop_w : crop_w + required_w]

        # Project latent
        x = self.input_proj(h)

        # Decode with residual blocks
        for block in self.decoder_blocks:
            x = block["upsample"](x)
            x = block["residual"](x)

        # Final upsampling and output
        x = self.final_upsample(x)
        x_recon = self.final_conv(x)

        # Final adaptive interpolation to ensure exact target size match
        if target_size is not None:
            _, _, target_h, target_w = target_size
            _, _, recon_h, recon_w = x_recon.shape
            if recon_h != target_h or recon_w != target_w:
                x_recon = F.interpolate(
                    x_recon,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )

        return x_recon


class VectorVAE(nn.Module):
    """
    Variational Autoencoder - combines encoder and decoder.
    Uses 1D latent bottleneck.
    """

    def __init__(
        self,
        in_channels=1,
        latent_dim=64,
        hidden_dims=[32, 64, 128, 256],
        max_initial_spatial_size=64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VectorVAEEncoder(in_channels, latent_dim, hidden_dims)
        self.decoder = VectorVAEDecoder(
            latent_dim,
            in_channels,
            list(reversed(hidden_dims)),
            max_initial_spatial_size,
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample from latent distribution.
        Args:
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]
        Returns:
            Sampled latent codes [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, target_size=None):
        """
        Forward pass: encode, reparameterize, decode.
        Args:
            x: Input images [B, C, H, W] in [0, 1] range
            target_size: Optional target size for decoder output
        Returns:
            x_recon: Reconstructed images [B, C, H, W] in [0, 1] range
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, target_size=target_size)
        return x_recon, mu, logvar

    def encode_to_latent(self, x):
        """
        Encode to deterministic latent (mean only, no sampling).
        Useful for inference.
        """
        mu, _ = self.encoder(x)
        return mu

    def decode_from_latent(self, z, target_size=None):
        """
        Decode from latent codes.
        """
        return self.decoder(z, target_size=target_size)


# Spatial latent VAE used by the MDN-RNN notebook.
class SequenceSpatialVAEResidualBlock(nn.Module):
    """Residual block for SequenceSpatialVAE with GroupNorm (more stable than BatchNorm for generative models)."""

    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()
        # Ensure groups doesn't exceed channel count
        groups_out = min(groups, out_channels)
        groups_in = min(groups, in_channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups_out, out_channels),
            nn.SiLU(),  # SiLU (Swish) - smoother than ReLU, better for generative models
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups_out, out_channels),
        )
        # Residual connection: 1x1 conv if channels or stride changes
        self.res_conv = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(groups_out, out_channels),
            )
            if (in_channels != out_channels or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        out = self.block(x)
        out = out + self.res_conv(x)
        # SiLU activation after residual connection (matching diffusion model pattern)
        out = F.silu(out)
        return out


class SequenceSpatialVAEEncoder(nn.Module):
    """
    SequenceSpatialVAE Encoder - maps images to latent distribution parameters.
    Uses spatial latents (maintains spatial structure).
    Now with residual connections for better gradient flow.
    """

    def __init__(
        self, in_channels=1, latent_channels=4, hidden_dims=[32, 64, 128, 256]
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims.copy()
        # Encoder layers with residual blocks (using GroupNorm)
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                # First layer: input -> first hidden dim (stride=2 for downsampling)
                self.encoder_blocks.append(
                    SequenceSpatialVAEResidualBlock(
                        in_channels, hidden_dims[i], stride=2, groups=8
                    )
                )
            else:
                # Subsequent layers: hidden_dims[i-1] -> hidden_dims[i] (stride=2 for downsampling)
                self.encoder_blocks.append(
                    SequenceSpatialVAEResidualBlock(
                        hidden_dims[i - 1], hidden_dims[i], stride=2, groups=8
                    )
                )
        # Final layers to produce mu and logvar (spatial latents)
        self.conv_mu = nn.Conv2d(
            hidden_dims[-1], latent_channels, kernel_size=1, bias=False
        )
        self.conv_logvar = nn.Conv2d(
            hidden_dims[-1], latent_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        """
        Encode input to latent distribution parameters.
        Args:
            x: Input images [B, C, H, W] in [0, 1] range
        Returns:
            mu: Mean of latent distribution [B, latent_channels, H', W']
            logvar: Log variance of latent distribution [B, latent_channels, H', W']
        """
        h = x
        for block in self.encoder_blocks:
            h = block(h)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar


class SequenceSpatialVAEDecoder(nn.Module):
    """
    SequenceSpatialVAE Decoder with residual connections for better gradient flow and reconstruction quality.
    Adapted for spatial latents.
    """

    def __init__(
        self, latent_channels=4, out_channels=1, hidden_dims=[256, 128, 64, 32]
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims.copy()
        # Project latent to encoder output shape (using GroupNorm and SiLU)
        groups_proj = min(8, hidden_dims[0])
        self.input_proj = nn.Sequential(
            nn.Conv2d(
                latent_channels, hidden_dims[0], kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(groups_proj, hidden_dims[0]),
            nn.SiLU(),  # SiLU (Swish) - smoother activation, better for generative models
        )
        # Decoder blocks with residual connections (using GroupNorm)
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Each block: upsample + residual block
            self.decoder_blocks.append(
                nn.ModuleDict(
                    {
                        "upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                        "residual": SequenceSpatialVAEResidualBlock(
                            hidden_dims[i], hidden_dims[i + 1], stride=1, groups=8
                        ),
                    }
                )
            )
        # Final layer to output channels
        self.final_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dims[-1],
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Sigmoid(),  # Output directly in [0, 1]
        )

    def forward(self, z, target_size=None):
        """
        Decode latent to image.
        Args:
            z: Latent codes [B, latent_channels, H', W']
            target_size: Optional target size (B, C, H, W)
        Returns:
            Reconstructed images [B, C, H, W] in [0, 1] range
        """
        # Project latent
        x = self.input_proj(z)
        # Decode with residual blocks
        for block in self.decoder_blocks:
            x = block["upsample"](x)
            x = block["residual"](x)
        # Final upsampling and output
        x = self.final_upsample(x)
        x_recon = self.final_conv(x)
        # Crop or pad to match target size if provided
        if target_size is not None:
            batch_size, channels, target_h, target_w = target_size
            _, _, recon_h, recon_w = x_recon.shape
            # Handle height mismatch
            if recon_h != target_h:
                if recon_h > target_h:
                    # Crop from center
                    crop_h = (recon_h - target_h) // 2
                    x_recon = x_recon[:, :, crop_h : crop_h + target_h, :]
                else:
                    # Pad with zeros to keep values in [0, 1]
                    pad_h_top = (target_h - recon_h) // 2
                    pad_h_bottom = target_h - recon_h - pad_h_top
                    x_recon = F.pad(
                        x_recon,
                        (0, 0, pad_h_top, pad_h_bottom),
                        mode="constant",
                        value=0.0,
                    )
            # Handle width mismatch
            if recon_w != target_w:
                if recon_w > target_w:
                    # Crop from center
                    crop_w = (recon_w - target_w) // 2
                    x_recon = x_recon[:, :, :, crop_w : crop_w + target_w]
                else:
                    # Pad with zeros to keep values in [0, 1]
                    pad_w_left = (target_w - recon_w) // 2
                    pad_w_right = target_w - recon_w - pad_w_left
                    x_recon = F.pad(
                        x_recon,
                        (pad_w_left, pad_w_right, 0, 0),
                        mode="constant",
                        value=0.0,
                    )
        return x_recon


class SequenceSpatialVAE(nn.Module):
    """
    Variational Autoencoder - combines encoder and decoder.
    Uses spatial latents (maintains spatial structure).
    """

    def __init__(
        self, in_channels=1, latent_channels=4, hidden_dims=[32, 64, 128, 256]
    ):
        super().__init__()
        self.encoder = SequenceSpatialVAEEncoder(
            in_channels, latent_channels, hidden_dims
        )
        self.decoder = SequenceSpatialVAEDecoder(
            latent_channels, in_channels, list(reversed(hidden_dims))
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample from latent distribution.
        Args:
            mu: Mean [B, latent_channels, H', W']
            logvar: Log variance [B, latent_channels, H', W']
        Returns:
            Sampled latent codes [B, latent_channels, H', W']
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, target_size=None):
        """
        Forward pass: encode, reparameterize, decode.
        Args:
            x: Input images [B, C, H, W] in [0, 1] range
            target_size: Optional target size for decoder output
        Returns:
            x_recon: Reconstructed images [B, C, H, W] in [0, 1] range
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, target_size=target_size)
        return x_recon, mu, logvar

    def encode_to_latent(self, x):
        """
        Encode to deterministic latent (mean only, no sampling).
        Useful for inference.
        """
        mu, _ = self.encoder(x)
        return mu

    def encode_and_sample(self, x, variance_scale=1.0):
        """
        Encode and sample from latent distribution.
        Samples z ~ N(μ, (variance_scale * σ)²) for each training batch to handle stochasticity.
        Sampling during MDN-RNN training helps avoid overfitting and makes the sequence
        model robust to VAE stochasticity.

        Args:
            x: Input images [B, C, H, W]
            variance_scale: Scale factor for standard deviation (0.0 = deterministic, 1.0 = full variance)
                          Default: 1.0 (full variance). Use smaller values (e.g., 0.1-0.5) to reduce noise.

        Returns:
            z: Sampled latent codes [B, latent_channels, H', W']
        """
        mu, logvar = self.encoder(x)
        # Compute standard deviation from logvar
        std = torch.exp(0.5 * logvar)
        # Scale the standard deviation directly (not logvar, which can be negative!)
        # This correctly reduces variance: scaled_std = variance_scale * std
        scaled_std = variance_scale * std
        # Sample: z = mu + eps * scaled_std
        eps = torch.randn_like(std)
        z = mu + eps * scaled_std
        return z

    def decode_from_latent(self, z, target_size=None):
        """
        Decode from latent codes.
        """
        return self.decoder(z, target_size=target_size)


# 1D latent VAE used by the MDN-RNN notebook.
class SequenceVectorVAEResidualBlock(nn.Module):
    """Residual block for SequenceVectorVAE with GroupNorm (more stable than BatchNorm for generative models)."""

    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()
        # Ensure groups doesn't exceed channel count
        groups_out = min(groups, out_channels)
        groups_in = min(groups, in_channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups_out, out_channels),
            nn.SiLU(),  # SiLU (Swish) - smoother than ReLU, better for generative models
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups_out, out_channels),
        )
        # Residual connection: 1x1 conv if channels or stride changes
        self.res_conv = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(groups_out, out_channels),
            )
            if (in_channels != out_channels or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        out = self.block(x)
        out = out + self.res_conv(x)
        # SiLU activation after residual connection (matching diffusion model pattern)
        out = F.silu(out)
        return out


class SequenceVectorVAEEncoder(nn.Module):
    """
    SequenceVectorVAE Encoder - maps images to latent distribution parameters.
    Uses 1D latent bottleneck - outputs flattened vectors.
    Now with residual connections for better gradient flow.
    """

    def __init__(self, in_channels=1, latent_dim=512, hidden_dims=[32, 64, 128, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims.copy()
        # Encoder layers with residual blocks (using GroupNorm)
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                # First layer: input -> first hidden dim (stride=2 for downsampling)
                self.encoder_blocks.append(
                    SequenceVectorVAEResidualBlock(
                        in_channels, hidden_dims[i], stride=2, groups=8
                    )
                )
            else:
                # Subsequent layers: hidden_dims[i-1] -> hidden_dims[i] (stride=2 for downsampling)
                self.encoder_blocks.append(
                    SequenceVectorVAEResidualBlock(
                        hidden_dims[i - 1], hidden_dims[i], stride=2, groups=8
                    )
                )
        # Global average pooling to get [B, C] from [B, C, H, W]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Linear layers to produce mu and logvar (1D vectors)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        """
        Encode input to latent distribution parameters.
        Args:
            x: Input images [B, C, H, W] in [0, 1] range
        Returns:
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log variance of latent distribution [B, latent_dim]
        """
        h = x
        for block in self.encoder_blocks:
            h = block(h)
        # Global pooling: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        h = self.global_pool(h).squeeze(-1).squeeze(-1)
        mu = self.fc_mu(h)  # [B, latent_dim]
        logvar = self.fc_logvar(h)  # [B, latent_dim]
        return mu, logvar


class SequenceVectorVAEDecoder(nn.Module):
    """
    SequenceVectorVAE Decoder with residual connections for better gradient flow and reconstruction quality.
    Adapted for 1D latent input.
    """

    def __init__(
        self,
        latent_dim=512,
        out_channels=1,
        hidden_dims=[256, 128, 64, 32],
        initial_spatial_size=8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims.copy()
        self.initial_spatial_size = (
            initial_spatial_size  # e.g., 8x8 for 64x64 input with 3 downsampling layers
        )

        # Linear layer to expand 1D latent to spatial feature map
        initial_channels = hidden_dims[0]
        self.fc_expand = nn.Linear(
            latent_dim, initial_channels * initial_spatial_size * initial_spatial_size
        )

        # Reshape and project to spatial features (using GroupNorm and SiLU)
        groups_proj = min(8, initial_channels)
        self.input_proj = nn.Sequential(
            nn.Conv2d(
                initial_channels, initial_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(groups_proj, initial_channels),
            nn.SiLU(),  # SiLU (Swish) - smoother activation, better for generative models
        )
        # Decoder blocks with residual connections (using GroupNorm)
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Each block: upsample + residual block
            self.decoder_blocks.append(
                nn.ModuleDict(
                    {
                        "upsample": nn.Upsample(scale_factor=2, mode="nearest"),
                        "residual": SequenceVectorVAEResidualBlock(
                            hidden_dims[i], hidden_dims[i + 1], stride=1, groups=8
                        ),
                    }
                )
            )
        # Final layer to output channels
        self.final_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dims[-1],
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Sigmoid(),  # Output directly in [0, 1]
        )

    def forward(self, z, target_size=None):
        """
        Decode latent to image.
        Args:
            z: Latent codes [B, latent_dim] (1D vector)
            target_size: Optional target size (B, C, H, W)
        Returns:
            Reconstructed images [B, C, H, W] in [0, 1] range
        """
        # Expand 1D latent: [B, latent_dim] -> [B, C*H*W]
        B = z.shape[0]
        h = self.fc_expand(z)  # [B, C*H*W]
        # Reshape to spatial: [B, C*H*W] -> [B, C, H, W]
        h = h.view(
            B, self.hidden_dims[0], self.initial_spatial_size, self.initial_spatial_size
        )
        # Project latent
        x = self.input_proj(h)
        # Decode with residual blocks
        for block in self.decoder_blocks:
            x = block["upsample"](x)
            x = block["residual"](x)
        # Final upsampling and output
        x = self.final_upsample(x)
        x_recon = self.final_conv(x)
        # Crop or pad to match target size if provided
        if target_size is not None:
            batch_size, channels, target_h, target_w = target_size
            _, _, recon_h, recon_w = x_recon.shape
            # Handle height mismatch
            if recon_h != target_h:
                if recon_h > target_h:
                    # Crop from center
                    crop_h = (recon_h - target_h) // 2
                    x_recon = x_recon[:, :, crop_h : crop_h + target_h, :]
                else:
                    # Pad with zeros to keep values in [0, 1]
                    pad_h_top = (target_h - recon_h) // 2
                    pad_h_bottom = target_h - recon_h - pad_h_top
                    x_recon = F.pad(
                        x_recon,
                        (0, 0, pad_h_top, pad_h_bottom),
                        mode="constant",
                        value=0.0,
                    )
            # Handle width mismatch
            if recon_w != target_w:
                if recon_w > target_w:
                    # Crop from center
                    crop_w = (recon_w - target_w) // 2
                    x_recon = x_recon[:, :, :, crop_w : crop_w + target_w]
                else:
                    # Pad with zeros to keep values in [0, 1]
                    pad_w_left = (target_w - recon_w) // 2
                    pad_w_right = target_w - recon_w - pad_w_left
                    x_recon = F.pad(
                        x_recon,
                        (pad_w_left, pad_w_right, 0, 0),
                        mode="constant",
                        value=0.0,
                    )
        return x_recon


class SequenceVectorVAE(nn.Module):
    """
    Variational Autoencoder - combines encoder and decoder.
    Uses 1D latent bottleneck.
    """

    def __init__(
        self,
        in_channels=1,
        latent_dim=512,
        hidden_dims=[32, 64, 128, 256],
        initial_spatial_size=8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SequenceVectorVAEEncoder(in_channels, latent_dim, hidden_dims)
        self.decoder = SequenceVectorVAEDecoder(
            latent_dim, in_channels, list(reversed(hidden_dims)), initial_spatial_size
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample from latent distribution.
        Args:
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]
        Returns:
            Sampled latent codes [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, target_size=None):
        """
        Forward pass: encode, reparameterize, decode.
        Args:
            x: Input images [B, C, H, W] in [0, 1] range
            target_size: Optional target size for decoder output
        Returns:
            x_recon: Reconstructed images [B, C, H, W] in [0, 1] range
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, target_size=target_size)
        return x_recon, mu, logvar

    def encode_to_latent(self, x):
        """
        Encode to deterministic latent (mean only, no sampling).
        Useful for inference.
        """
        mu, _ = self.encoder(x)
        return mu

    def encode_and_sample(self, x, variance_scale=1.0):
        """
        Encode and sample from latent distribution.
        Samples z ~ N(μ, (variance_scale * σ)²) for each training batch to handle stochasticity.
        Sampling during MDN-RNN training helps avoid overfitting and makes the sequence
        model robust to VAE stochasticity.

        Args:
            x: Input images [B, C, H, W]
            variance_scale: Scale factor for standard deviation (0.0 = deterministic, 1.0 = full variance)
                          Default: 1.0 (full variance). Use smaller values (e.g., 0.1-0.5) to reduce noise.

        Returns:
            z: Sampled latent codes [B, latent_dim]
        """
        mu, logvar = self.encoder(x)
        # Compute standard deviation from logvar
        std = torch.exp(0.5 * logvar)
        # Scale the standard deviation directly (not logvar, which can be negative!)
        # This correctly reduces variance: scaled_std = variance_scale * std
        scaled_std = variance_scale * std
        # Sample: z = mu + eps * scaled_std
        eps = torch.randn_like(std)
        z = mu + eps * scaled_std
        return z

    def decode_from_latent(self, z, target_size=None):
        """
        Decode from latent codes.
        """
        return self.decoder(z, target_size=target_size)


def build_spatial_vae(**kwargs):
    return SpatialVAE(**kwargs)


def build_vector_vae(**kwargs):
    return VectorVAE(**kwargs)


def build_sequence_spatial_vae(**kwargs):
    return SequenceSpatialVAE(**kwargs)


def build_sequence_vector_vae(**kwargs):
    return SequenceVectorVAE(**kwargs)
