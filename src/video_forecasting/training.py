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
        image1 = batch["image1"].to(device)  # Condition (current frame)
        image2 = batch["image2"].to(device)  # Target (future frame)
        # Ensure image1 and image2 have the same spatial dimensions
        if image1.shape[2:] != image2.shape[2:]:
            target_h, target_w = image2.shape[2], image2.shape[3]
            image1 = F.interpolate(
                image1, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        # Encode to latent space
        with torch.no_grad():
            condition_z = vae.encode_to_latent(image1)
            target_z = vae.encode_to_latent(image2)
        if condition_z.dim() == 4 and target_z.dim() == 4:
            if condition_z.shape[2:] != target_z.shape[2:]:
                target_h, target_w = target_z.shape[2], target_z.shape[3]
                condition_z = F.interpolate(
                    condition_z,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
        elif condition_z.dim() == 2 and target_z.dim() == 2:
            if condition_z.shape != target_z.shape:
                raise ValueError(
                    f"Vector latent shape mismatch: {condition_z.shape} vs {target_z.shape}"
                )
        # Compute CFM loss
        loss = flow_utils.compute_loss(model, target_z, condition_z)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate_flow_matching(model, vae, dataloader, flow_utils, device):
    """Evaluate flow matching model."""
    model.eval()
    vae.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Flow Matching"):
            image1 = batch["image1"].to(device)
            image2 = batch["image2"].to(device)
            if image1.shape[2:] != image2.shape[2:]:
                target_h, target_w = image2.shape[2], image2.shape[3]
                image1 = F.interpolate(
                    image1,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            condition_z = vae.encode_to_latent(image1)
            target_z = vae.encode_to_latent(image2)
            if condition_z.dim() == 4 and target_z.dim() == 4:
                if condition_z.shape[2:] != target_z.shape[2:]:
                    target_h, target_w = target_z.shape[2], target_z.shape[3]
                    condition_z = F.interpolate(
                        condition_z,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
            elif condition_z.dim() == 2 and target_z.dim() == 2:
                if condition_z.shape != target_z.shape:
                    raise ValueError(
                        f"Vector latent shape mismatch: {condition_z.shape} vs {target_z.shape}"
                    )
            loss = flow_utils.compute_loss(model, target_z, condition_z)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_diffusion_epoch(model, vae, dataloader, scheduler, optimizer, device):
    """Train diffusion model for one epoch."""
    model.train()
    vae.eval()  # Freeze VAE
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training Diffusion"):
        image1 = batch["image1"].to(device)  # Condition (current frame)
        image2 = batch["image2"].to(device)  # Target (future frame)
        # Ensure image1 and image2 have the same spatial dimensions
        if image1.shape[2:] != image2.shape[2:]:
            target_h, target_w = image2.shape[2], image2.shape[3]
            image1 = F.interpolate(
                image1, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        # Encode to latent space (1D vectors)
        with torch.no_grad():
            condition_z = vae.encode_to_latent(image1)  # [B, latent_dim]
            target_z = vae.encode_to_latent(image2)  # [B, latent_dim]

        # Sample timesteps
        t = scheduler.sample_timesteps(target_z.shape[0], device)

        # Sample noise
        noise = torch.randn_like(target_z)

        # Add noise to target latents
        noisy_z = scheduler.add_noise(target_z, t, noise)

        # Scale timesteps for time embedding (model expects [0, 1000])
        t_scaled = t.float() * (1000.0 / scheduler.num_timesteps)

        # Predict noise
        noise_pred = model(noisy_z, condition_z, t_scaled)

        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(noise_pred, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate_diffusion(model, vae, dataloader, scheduler, device):
    """Evaluate diffusion model."""
    model.eval()
    vae.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Diffusion"):
            image1 = batch["image1"].to(device)
            image2 = batch["image2"].to(device)
            if image1.shape[2:] != image2.shape[2:]:
                target_h, target_w = image2.shape[2], image2.shape[3]
                image1 = F.interpolate(
                    image1,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            condition_z = vae.encode_to_latent(image1)  # [B, latent_dim]
            target_z = vae.encode_to_latent(image2)  # [B, latent_dim]

            # Sample timesteps
            t = scheduler.sample_timesteps(target_z.shape[0], device)

            # Sample noise
            noise = torch.randn_like(target_z)

            # Add noise to target latents
            noisy_z = scheduler.add_noise(target_z, t, noise)

            # Scale timesteps for time embedding
            t_scaled = t.float() * (1000.0 / scheduler.num_timesteps)

            # Predict noise
            noise_pred = model(noisy_z, condition_z, t_scaled)

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


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
