from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class LatentTransformerForecaster(nn.Module):
    """Causal transformer that predicts the next VAE latent at each timestep."""

    def __init__(
        self,
        latent_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(latent_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, latent_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_sequence: Latents with shape [B, T, latent_dim].

        Returns:
            Predicted next latents for each input timestep, shape [B, T, latent_dim].
        """
        if latent_sequence.dim() != 3:
            raise ValueError(
                f"Expected latent_sequence with shape [B, T, D], got {latent_sequence.shape}"
            )
        batch_size, seq_len, _ = latent_sequence.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        x = self.input_proj(latent_sequence) * math.sqrt(self.d_model)
        x = x + self.pos_embedding[:, :seq_len, :]
        mask = self._causal_mask(seq_len, latent_sequence.device)
        x = self.encoder(x, mask=mask)
        x = self.norm(x)
        return self.output_proj(x).view(batch_size, seq_len, self.latent_dim)


@torch.no_grad()
def predict_next_latent(model, latent_context: torch.Tensor) -> torch.Tensor:
    """Predict one next latent from a latent context sequence."""
    model.eval()
    predictions = model(latent_context)
    return predictions[:, -1, :]


@torch.no_grad()
def generate_transformer_rollout(
    model,
    vae,
    initial_frames,
    num_predictions: int,
    context_size: int = 5,
    device: str | torch.device = "cpu",
):
    """
    Generate an autoregressive rollout from initial frames.

    Args:
        model: LatentTransformerForecaster.
        vae: VAE with encode_to_latent/decode_from_latent methods.
        initial_frames: Tensor or ndarray with shape [T, C, H, W] or [C, H, W].
        num_predictions: Number of future frames to generate.
        context_size: Maximum number of recent latents used as context.
        device: Torch device.

    Returns:
        Numpy array with shape [T + num_predictions, C, H, W].
    """
    model.eval()
    vae.eval()

    if isinstance(initial_frames, np.ndarray):
        initial_frames = torch.from_numpy(initial_frames).float()
    if initial_frames.dim() == 3:
        initial_frames = initial_frames.unsqueeze(0)
    if initial_frames.dim() != 4:
        raise ValueError(f"Unexpected initial_frames shape: {initial_frames.shape}")

    device = torch.device(device)
    frames = initial_frames.to(device)
    all_frames = [frame.detach().cpu().numpy() for frame in frames]

    latents = vae.encode_to_latent(frames)
    latent_history = [latent.detach() for latent in latents]
    target_size = frames.shape[-2:]

    for _ in range(num_predictions):
        context = torch.stack(latent_history[-context_size:], dim=0).unsqueeze(0)
        next_latent = predict_next_latent(model, context).squeeze(0)
        next_frame = vae.decode_from_latent(
            next_latent.unsqueeze(0), target_size=(1, frames.shape[1], *target_size)
        ).squeeze(0)
        next_frame = torch.clamp(next_frame, 0, 1)
        all_frames.append(next_frame.detach().cpu().numpy())
        latent_history.append(next_latent.detach())

    return np.stack(all_frames, axis=0)
