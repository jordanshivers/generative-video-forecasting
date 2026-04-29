from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .models.mdn_rnn import mdn_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Helper dataset for VAE training (uses all frames, not just pairs)
class FrameOnlyDataset(Dataset):
    """Dataset that returns individual frames for VAE training in [0, 1] range."""

    def __init__(self, frame_prediction_dataset, target_height=None, target_width=None):
        self.base_dataset = frame_prediction_dataset
        self.target_height = target_height or frame_prediction_dataset.target_height
        self.target_width = target_width or frame_prediction_dataset.target_width
        # Collect all frames
        self.frames = []
        for seq_idx, seq in enumerate(frame_prediction_dataset.sequences):
            for frame_idx in range(len(seq)):
                self.frames.append({"seq_idx": seq_idx, "frame_idx": frame_idx})

    def _pad(self, img: np.ndarray) -> np.ndarray:
        """Pad image to target size using this dataset's target dimensions."""
        C, H, W = img.shape
        pad_h = max(0, self.target_height - H)
        pad_w = max(0, self.target_width - W)
        if pad_h > 0 or pad_w > 0:
            # Pad: (left, right, top, bottom) for channels
            img = np.pad(
                img,
                (
                    (0, 0),
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                ),
                mode="constant",
                constant_values=0,
            )
        return img

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_info = self.frames[idx]
        seq = self.base_dataset.sequences[frame_info["seq_idx"]]
        img = seq[frame_info["frame_idx"]].copy()
        # Normalize to [0, 1]
        if self.base_dataset.normalize:
            img = self.base_dataset._normalize(img)
        # Pad using this dataset's target dimensions
        img = self._pad(img)
        # Convert to tensor
        tensor = torch.from_numpy(img).float()
        return tensor


class ContextFramePredictionDataset(Dataset):
    """Frame-window dataset with K context frames and one future target."""

    def __init__(self, frame_prediction_dataset, context_frames: int = 1):
        self.base_dataset = frame_prediction_dataset
        self.context_frames = int(context_frames)
        if self.context_frames < 1:
            raise ValueError("context_frames must be positive.")
        self.frame_separation = frame_prediction_dataset.frame_separation
        self.sequences = frame_prediction_dataset.sequences
        self.normalize = getattr(frame_prediction_dataset, "normalize", True)
        self.target_height = getattr(frame_prediction_dataset, "target_height", None)
        self.target_width = getattr(frame_prediction_dataset, "target_width", None)
        self.windows: list[tuple[int, int]] = []
        for seq_idx, seq in enumerate(frame_prediction_dataset.sequences):
            max_start = len(seq) - self.context_frames * self.frame_separation - 1
            for start_idx in range(max_start + 1):
                self.windows.append((seq_idx, start_idx))
        if not self.windows:
            raise ValueError("No context-frame prediction windows available.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        if idx < 0 or idx >= len(self.windows):
            raise IndexError("ContextFramePredictionDataset index out of range")
        seq_idx, start_idx = self.windows[idx]
        seq = self.base_dataset.sequences[seq_idx]
        context_indices = [
            start_idx + i * self.frame_separation for i in range(self.context_frames)
        ]
        target_idx = start_idx + self.context_frames * self.frame_separation
        context_np = np.stack([seq[i].copy() for i in context_indices])
        image1_np = context_np[-1].copy()
        image2_np = seq[target_idx].copy()
        return {
            "context": torch.from_numpy(context_np).float(),
            "image1": torch.from_numpy(image1_np).float(),
            "image2": torch.from_numpy(image2_np).float(),
            "seq_idx": seq_idx,
            "start_idx": start_idx,
            "frame_idx1": context_indices[-1],
            "frame_idx2": target_idx,
        }

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        return self.base_dataset._normalize(img)

    def _pad(self, img: np.ndarray) -> np.ndarray:
        return self.base_dataset._pad(img)


def get_condition_frames(batch, device):
    if "context" in batch:
        return batch["context"].to(device)
    return batch["image1"].to(device).unsqueeze(1)


def stack_pixel_context(context):
    if context.dim() == 4:
        return context
    if context.dim() != 5:
        raise ValueError(f"context must be 4D or 5D, got {context.shape}")
    bsz, context_frames, channels, height, width = context.shape
    return context.reshape(bsz, context_frames * channels, height, width)


def _resize_context_to_target(context, target):
    if context.shape[-2:] == target.shape[-2:]:
        return context
    if context.dim() == 4:
        return F.interpolate(
            context, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
    bsz, context_frames, channels, _, _ = context.shape
    flat = context.reshape(bsz * context_frames, channels, *context.shape[-2:])
    flat = F.interpolate(
        flat, size=target.shape[-2:], mode="bilinear", align_corners=False
    )
    return flat.reshape(bsz, context_frames, channels, *target.shape[-2:])


def encode_stack_spatial_context(vae, context):
    if context.dim() == 4:
        return vae.encode_to_latent(context)
    bsz, context_frames, channels, height, width = context.shape
    flat = context.reshape(bsz * context_frames, channels, height, width)
    latents = vae.encode_to_latent(flat)
    if latents.dim() != 4:
        raise ValueError(f"Expected spatial latents, got shape {latents.shape}")
    _, latent_channels, latent_height, latent_width = latents.shape
    return latents.reshape(
        bsz, context_frames * latent_channels, latent_height, latent_width
    )


def encode_stack_vector_context(vae, context):
    if context.dim() == 4:
        return vae.encode_to_latent(context)
    bsz, context_frames, channels, height, width = context.shape
    flat = context.reshape(bsz * context_frames, channels, height, width)
    latents = vae.encode_to_latent(flat)
    if latents.dim() != 2:
        raise ValueError(f"Expected vector latents, got shape {latents.shape}")
    return latents.reshape(bsz, context_frames * latents.shape[1])


def encode_stack_context(vae, context, target_z):
    if target_z.dim() == 4:
        return encode_stack_spatial_context(vae, context)
    if target_z.dim() == 2:
        return encode_stack_vector_context(vae, context)
    raise ValueError(f"target_z must be 2D or 4D, got shape {target_z.shape}")


def _pixel_condition_and_target(batch, device):
    target = batch["image2"].to(device)
    context = get_condition_frames(batch, device)
    context = _resize_context_to_target(context, target)
    return stack_pixel_context(context), target


def _latent_condition_and_target(vae, batch, device):
    target = batch["image2"].to(device)
    context = get_condition_frames(batch, device)
    context = _resize_context_to_target(context, target)
    with torch.no_grad():
        target_z = vae.encode_to_latent(target)
        condition_z = encode_stack_context(vae, context, target_z)
    if condition_z.dim() == 4 and target_z.dim() == 4:
        if condition_z.shape[2:] != target_z.shape[2:]:
            condition_z = F.interpolate(
                condition_z,
                size=target_z.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
    return condition_z, target_z


def train_vae_epoch(model, dataloader, optimizer, device, beta=1.0):
    """Train a VAE for one epoch."""
    model.train()
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training VAE"):
        if isinstance(batch, dict):
            # FramePredictionDataset returns dict
            x = batch["image1"].to(device)
        else:
            # FrameOnlyDataset returns tensor
            x = batch.to(device)
        # Forward pass - pass target_size to ensure output matches input size
        x_recon, mu, logvar = model(x, target_size=x.shape)
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
        # Sum over latent dimensions and normalize by batch size.
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)
        # Total loss
        loss = recon_loss + beta * kl_loss
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_loss += loss.item()
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    return avg_recon_loss, avg_kl_loss, avg_loss


def evaluate_vae(model, dataloader, device, beta=1.0):
    """Evaluate a VAE."""
    model.eval()
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating VAE"):
            if isinstance(batch, dict):
                x = batch["image1"].to(device)
            else:
                x = batch.to(device)
            x_recon, mu, logvar = model(x, target_size=x.shape)
            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / x.size(0)
            loss = recon_loss + beta * kl_loss
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    return avg_recon_loss, avg_kl_loss, avg_loss


# Default VAE training parameters used by the notebooks.
vae_beta = 1.0
vae_learning_rate = 3e-4
vae_num_epochs = 100
vae_batch_size = 32


def train_flow_matching_epoch(model, vae, dataloader, flow_utils, optimizer, device):
    """Train flow matching model for one epoch."""
    model.train()
    vae.eval()  # Freeze VAE
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training Flow Matching"):
        condition_z, target_z = _latent_condition_and_target(vae, batch, device)
        loss = flow_utils.compute_loss(model, target_z, condition_z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_flow_matching(model, vae, dataloader, flow_utils, device):
    """Evaluate flow matching model."""
    model.eval()
    vae.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Flow Matching"):
            condition_z, target_z = _latent_condition_and_target(vae, batch, device)
            loss = flow_utils.compute_loss(model, target_z, condition_z)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_pixel_flow_matching_epoch(model, dataloader, flow_utils, optimizer, device):
    """Train a pixel-space flow matching model for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training Pixel Flow Matching"):
        condition, target = _pixel_condition_and_target(batch, device)
        loss = flow_utils.compute_loss(model, target, condition)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_pixel_flow_matching(model, dataloader, flow_utils, device):
    """Evaluate a pixel-space flow matching model."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Pixel Flow Matching"):
            condition, target = _pixel_condition_and_target(batch, device)
            total_loss += flow_utils.compute_loss(model, target, condition).item()
    return total_loss / len(dataloader)


def train_stochastic_interpolant_epoch(model, vae, dataloader, si_utils, optimizer, device):
    """Train stochastic interpolant drift model for one epoch (latent space)."""
    model.train()
    vae.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training Stochastic Interpolant"):
        condition_z, target_z = _latent_condition_and_target(vae, batch, device)
        loss = si_utils.compute_loss(model, target_z, condition_z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_stochastic_interpolant(model, vae, dataloader, si_utils, device):
    """Evaluate stochastic interpolant drift model (latent space)."""
    model.eval()
    vae.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Stochastic Interpolant"):
            condition_z, target_z = _latent_condition_and_target(vae, batch, device)
            loss = si_utils.compute_loss(model, target_z, condition_z)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_pixel_stochastic_interpolant_epoch(model, dataloader, si_utils, optimizer, device):
    """Train a pixel-space stochastic interpolant drift model for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training Pixel Stochastic Interpolant"):
        condition, target = _pixel_condition_and_target(batch, device)
        loss = si_utils.compute_loss(model, target, condition)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_pixel_stochastic_interpolant(model, dataloader, si_utils, device):
    """Evaluate a pixel-space stochastic interpolant drift model."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Pixel Stochastic Interpolant"):
            condition, target = _pixel_condition_and_target(batch, device)
            total_loss += si_utils.compute_loss(model, target, condition).item()
    return total_loss / len(dataloader)


def train_diffusion_epoch(model, vae, dataloader, scheduler, optimizer, device):
    """Train diffusion model for one epoch."""
    model.train()
    vae.eval()  # Freeze VAE
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training Diffusion"):
        condition_z, target_z = _latent_condition_and_target(vae, batch, device)
        t = scheduler.sample_timesteps(target_z.shape[0], device)
        noise = torch.randn_like(target_z)
        noisy_z = scheduler.add_noise(target_z, t, noise)
        t_scaled = t.float() * (1000.0 / scheduler.num_timesteps)
        noise_pred = model(noisy_z, condition_z, t_scaled)
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train_pixel_diffusion_epoch(model, dataloader, scheduler, optimizer, device):
    """Train a pixel-space diffusion model for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training Pixel Diffusion"):
        condition, target = _pixel_condition_and_target(batch, device)
        t = scheduler.sample_timesteps(target.shape[0], device)
        noise = torch.randn_like(target)
        noisy_image = scheduler.add_noise(target, t, noise)
        t_scaled = t.float() * (1000.0 / scheduler.num_timesteps)
        noise_pred = model(noisy_image, condition, t_scaled)
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_diffusion(model, vae, dataloader, scheduler, device):
    """Evaluate diffusion model."""
    model.eval()
    vae.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Diffusion"):
            condition_z, target_z = _latent_condition_and_target(vae, batch, device)
            t = scheduler.sample_timesteps(target_z.shape[0], device)
            noise = torch.randn_like(target_z)
            noisy_z = scheduler.add_noise(target_z, t, noise)
            t_scaled = t.float() * (1000.0 / scheduler.num_timesteps)
            noise_pred = model(noisy_z, condition_z, t_scaled)
            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_pixel_diffusion(model, dataloader, scheduler, device):
    """Evaluate a pixel-space diffusion model."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Pixel Diffusion"):
            condition, target = _pixel_condition_and_target(batch, device)
            t = scheduler.sample_timesteps(target.shape[0], device)
            noise = torch.randn_like(target)
            noisy_image = scheduler.add_noise(target, t, noise)
            t_scaled = t.float() * (1000.0 / scheduler.num_timesteps)
            noise_pred = model(noisy_image, condition, t_scaled)
            total_loss += F.mse_loss(noise_pred, noise).item()
    return total_loss / len(dataloader)


def train_mdn_rnn_epoch(model, vae, dataloader, optimizer, device, latent_shape):
    """Train an MDN-RNN for one epoch."""
    model.train()
    vae.eval()  # Freeze VAE
    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(dataloader, desc="Training MDN-RNN"):
        sequences = batch["sequence"].to(device)  # [B, T, C, H, W]
        B, T, C, H, W = sequences.shape

        # Encode entire sequence to latent space (sample with reduced variance for easier learning)
        with torch.no_grad():
            sequences_flat = sequences.view(B * T, C, H, W)
            latents = vae.encode_and_sample(
                sequences_flat, variance_scale=0.1
            )
            latents = latents.view(
                B, T, *latents.shape[1:]
            )  # [B, T, C_latent, H_latent, W_latent]
            latents_flat = latents.view(B, T, -1)  # [B, T, latent_dim]

        # Process each sequence with teacher forcing.
        # Input: frames 0 to T-2, Target: frames 1 to T-1 (teacher forcing)
        input_latents = latents_flat[
            :, :-1, :
        ]  # [B, T-1, latent_dim] - frames 0 to T-2
        target_latents = latents_flat[
            :, 1:, :
        ]  # [B, T-1, latent_dim] - frames 1 to T-1

        # Forward pass through MDN-RNN (processes entire sequence at once)
        pi, mu, sigma, _ = model(
            input_latents, hidden=None
        )  # [B, T-1, n_mixtures], [B, T-1, n_mixtures, latent_dim], ...

        # Compute loss for all timesteps
        batch_loss = mdn_loss(pi, mu, sigma, target_latents)

        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        total_steps += 1

    avg_loss = total_loss / total_steps
    return avg_loss


def evaluate_mdn_rnn(model, vae, dataloader, device, latent_shape):
    """Evaluate MDN-RNN using sequential processing."""
    model.eval()
    vae.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating MDN-RNN"):
            sequences = batch["sequence"].to(device)  # [B, T, C, H, W]
            B, T, C, H, W = sequences.shape

            # Encode entire sequence to latent space (deterministic for consistent evaluation)
            sequences_flat = sequences.view(B * T, C, H, W)
            latents = vae.encode_to_latent(
                sequences_flat
            )  # Deterministic for consistent evaluation
            latents = latents.view(B, T, *latents.shape[1:])
            latents_flat = latents.view(B, T, -1)  # [B, T, latent_dim]

            # Process entire sequence at once (matching training)
            input_latents = latents_flat[:, :-1, :]  # [B, T-1, latent_dim]
            target_latents = latents_flat[:, 1:, :]  # [B, T-1, latent_dim]

            # Forward pass
            pi, mu, sigma, _ = model(input_latents, hidden=None)

            # Compute loss for all timesteps
            batch_loss = mdn_loss(pi, mu, sigma, target_latents)
            total_loss += batch_loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps
    return avg_loss


def train_transformer_epoch(model, vae, dataloader, optimizer, device):
    """Train a causal latent transformer to predict the next latent in a sequence."""
    model.train()
    vae.eval()
    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(dataloader, desc="Training Latent Transformer"):
        sequences = batch["sequence"].to(device)  # [B, T, C, H, W]
        bsz, seq_len, channels, height, width = sequences.shape

        with torch.no_grad():
            flat = sequences.view(bsz * seq_len, channels, height, width)
            latents = vae.encode_to_latent(flat).view(bsz, seq_len, -1)

        inputs = latents[:, :-1, :]
        targets = latents[:, 1:, :]
        predictions = model(inputs)
        loss = F.mse_loss(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


def evaluate_transformer(model, vae, dataloader, device):
    """Evaluate next-latent MSE for a causal latent transformer."""
    model.eval()
    vae.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Latent Transformer"):
            sequences = batch["sequence"].to(device)
            bsz, seq_len, channels, height, width = sequences.shape
            flat = sequences.view(bsz * seq_len, channels, height, width)
            latents = vae.encode_to_latent(flat).view(bsz, seq_len, -1)
            predictions = model(latents[:, :-1, :])
            loss = F.mse_loss(predictions, latents[:, 1:, :])
            total_loss += loss.item()
            total_steps += 1

    return total_loss / max(total_steps, 1)


def train_simvp_epoch(model, dataloader, optimizer, device):
    """Train SimVP for one epoch with frame-wise MSE."""
    model.train()
    total_loss = 0.0
    total_steps = 0
    for batch in tqdm(dataloader, desc="Training SimVP"):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        predictions = model(inputs)
        loss = F.mse_loss(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_steps += 1
    return total_loss / max(total_steps, 1)


def evaluate_simvp(model, dataloader, device):
    """Evaluate SimVP frame-wise MSE."""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating SimVP"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            predictions = model(inputs)
            total_loss += F.mse_loss(predictions, targets).item()
            total_steps += 1
    return total_loss / max(total_steps, 1)
