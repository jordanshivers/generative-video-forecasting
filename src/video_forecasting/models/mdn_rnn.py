from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Dataset that provides full sequences for MDN-RNN training.
    Returns full sequences for sequential processing with RNN hidden state.
    """

    def __init__(self, base_dataset):
        """
        Args:
            base_dataset: MovingMNISTDataset instance
        """
        self.base_dataset = base_dataset
        # Use all sequences from base dataset
        self.sequences = list(range(len(base_dataset.sequences)))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_idx = self.sequences[idx]
        seq = self.base_dataset.sequences[seq_idx]

        # Get all frames in sequence
        frames = []
        for i in range(len(seq)):
            frame = seq[i].copy()
            if self.base_dataset.normalize:
                frame = self.base_dataset._normalize(frame)
            frames.append(torch.from_numpy(frame).float())

        # Stack all frames: [T, C, H, W]
        sequence = torch.stack(frames, dim=0)

        return {
            "sequence": sequence,  # [T, C, H, W] - full sequence
            "seq_idx": seq_idx,
        }


class MDNRNN(nn.Module):
    """
    Mixture Density Network + RNN for predicting next latent representation.

    Takes a sequence of m latent vectors and predicts a mixture of Gaussians
    for the next latent vector.
    """

    def __init__(
        self, latent_dim, hidden_dim=256, num_layers=1, n_mixtures=5, rnn_type="lstm"
    ):
        """
        Args:
            latent_dim: Dimension of flattened latent vector (C*H*W)
            hidden_dim: Hidden dimension of RNN
            num_layers: Number of RNN layers
            n_mixtures: Number of Gaussian mixture components
            rnn_type: 'lstm' or 'gru'
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_mixtures = n_mixtures

        # RNN backbone
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        # MDN output layers
        # Mixture weights (logits, will be softmaxed)
        self.pi_layer = nn.Linear(hidden_dim, n_mixtures)
        # Means for each mixture component
        self.mu_layer = nn.Linear(hidden_dim, n_mixtures * latent_dim)
        # Standard deviations (logits, will be exponentiated or softplus)
        self.sigma_layer = nn.Linear(hidden_dim, n_mixtures * latent_dim)

    def forward(self, latent_sequence, hidden=None):
        """
        Forward pass through MDN-RNN.

        Args:
            latent_sequence: Sequence of latent vectors [B, T, latent_dim] where T can be any length
            hidden: Hidden state from previous step (optional)

        Returns:
            pi: Mixture weights [B, n_mixtures]
            mu: Means [B, n_mixtures, latent_dim]
            sigma: Standard deviations [B, n_mixtures, latent_dim]
            hidden: Updated hidden state
        """
        # RNN forward pass - processes entire sequence at once
        rnn_out, hidden = self.rnn(
            latent_sequence, hidden
        )  # rnn_out: [B, T, hidden_dim]

        # Compute mixture parameters for ALL timesteps
        B, T, hidden_dim = rnn_out.shape
        rnn_out_flat = rnn_out.reshape(B * T, hidden_dim)  # [B*T, hidden_dim]

        # Mixture weights (logits -> softmax)
        pi_logits = self.pi_layer(rnn_out_flat)  # [B*T, n_mixtures]
        pi = F.softmax(pi_logits, dim=-1)  # [B*T, n_mixtures]
        pi = pi.view(B, T, self.n_mixtures)  # [B, T, n_mixtures]

        # Means
        mu = self.mu_layer(rnn_out_flat)  # [B*T, n_mixtures * latent_dim]
        mu = mu.view(
            B, T, self.n_mixtures, self.latent_dim
        )  # [B, T, n_mixtures, latent_dim]

        # Standard deviations use an exp(log_sigma) parameterization.
        sigma_logits = self.sigma_layer(rnn_out_flat)  # [B*T, n_mixtures * latent_dim]
        sigma = torch.exp(sigma_logits)  # [B*T, n_mixtures * latent_dim]
        sigma = torch.clamp(
            sigma, min=1e-4
        )  # Only clamp minimum for numerical stability
        sigma = sigma.view(
            B, T, self.n_mixtures, self.latent_dim
        )  # [B, T, n_mixtures, latent_dim]

        return pi, mu, sigma, hidden


def mdn_loss(pi, mu, sigma, target):
    """
    Compute negative log-likelihood loss for mixture density network.
    Handles both single timestep [B, ...] and sequence [B, T, ...] inputs.

    Args:
        pi: Mixture weights [B, n_mixtures] or [B, T, n_mixtures]
        mu: Means [B, n_mixtures, latent_dim] or [B, T, n_mixtures, latent_dim]
        sigma: Standard deviations [B, n_mixtures, latent_dim] or [B, T, n_mixtures, latent_dim]
        target: Target latent vector [B, latent_dim] or [B, T, latent_dim]

    Returns:
        loss: Negative log-likelihood loss (scalar)
    """
    # Handle both sequence and single timestep cases
    if target.dim() == 2:
        # Single timestep: [B, latent_dim]
        target = target.unsqueeze(1)  # [B, 1, latent_dim]
    elif target.dim() == 3:
        # Sequence: [B, T, latent_dim] -> [B, T, 1, latent_dim]
        target = target.unsqueeze(2)  # [B, T, 1, latent_dim]

    # Compute squared differences
    diff = target - mu  # [B, n_mixtures, latent_dim] or [B, T, n_mixtures, latent_dim]
    diff_sq = diff.pow(2)

    # Compute log probability for each component (diagonal Gaussian)
    log_sigma_sq = 2 * torch.log(sigma + 1e-8)  # log(sigma^2)
    log_prob_k = -0.5 * (
        diff_sq / (sigma.pow(2) + 1e-8) + log_sigma_sq + math.log(2 * math.pi)
    )
    log_prob_k = log_prob_k.sum(dim=-1)  # Sum over latent dimensions

    # Add log mixture weights
    log_pi = torch.log(pi + 1e-8)
    log_weighted_prob = log_pi + log_prob_k

    # Log-sum-exp trick
    max_log_prob = log_weighted_prob.max(dim=-1, keepdim=True)[0]
    log_sum_exp = max_log_prob + torch.log(
        torch.exp(log_weighted_prob - max_log_prob).sum(dim=-1, keepdim=True) + 1e-8
    )

    # Negative log-likelihood - average over batch and time (if sequence)
    nll = -log_sum_exp.mean()

    return nll


@torch.no_grad()
def sample_from_mdn(pi, mu, sigma, deterministic=False):
    """
    Sample from mixture density network or return weighted mean.

    Args:
        pi: Mixture weights [B, n_mixtures]
        mu: Means [B, n_mixtures, latent_dim]
        sigma: Standard deviations [B, n_mixtures, latent_dim]
        deterministic: If True, return weighted mean instead of sampling

    Returns:
        sampled: Sampled latent vectors [B, latent_dim] or weighted mean if deterministic
    """
    if deterministic:
        # Return weighted mean: sum_k pi_k * mu_k
        weighted_mean = torch.sum(pi.unsqueeze(-1) * mu, dim=1)  # [B, latent_dim]
        return weighted_mean

    B = pi.shape[0]
    device = pi.device

    # Sample mixture component for each batch item
    # pi is already softmaxed, so we can sample from categorical distribution
    mixture_idx = torch.multinomial(pi, num_samples=1)  # [B, 1]

    # Select corresponding mu and sigma for each batch item
    batch_indices = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
    selected_mu = mu[batch_indices, mixture_idx].squeeze(1)  # [B, latent_dim]
    selected_sigma = sigma[batch_indices, mixture_idx].squeeze(1)  # [B, latent_dim]

    # Sample from selected Gaussian
    eps = torch.randn_like(selected_mu)
    sampled = selected_mu + eps * selected_sigma

    return sampled


@torch.no_grad()
def predict_next_frame(mdn_rnn, vae, current_frame, latent_shape, hidden=None):
    """
    Predict next frame given current frame.
    Uses RNN hidden state to maintain history.

    Args:
        mdn_rnn: Trained MDN-RNN model
        vae: Trained VAE model
        current_frame: Current frame [C, H, W] or [B, C, H, W]
        latent_shape: Shape of latent representation (C, H, W)
        hidden: RNN hidden state from previous step (optional)

    Returns:
        predicted_frame: Predicted frame [C, H, W] or [B, C, H, W]
        hidden: Updated hidden state
    """
    mdn_rnn.eval()
    vae.eval()

    # Handle single frame vs batch
    if current_frame.dim() == 3:
        current_frame = current_frame.unsqueeze(0)  # [1, C, H, W]
        squeeze_output = True
    elif current_frame.dim() == 4:
        # Input is already [B, C, H, W]
        squeeze_output = False
    else:
        raise ValueError(
            f"Unexpected current_frame dimension: {current_frame.dim()}. Expected 3 ([C, H, W]) or 4 ([B, C, H, W])."
        )

    B, C, H, W = current_frame.shape
    device = current_frame.device

    # Encode current frame to latent space
    current_latent = vae.encode_to_latent(
        current_frame
    )  # [B, C_latent, H_latent, W_latent]
    current_latent_flat = current_latent.view(B, -1).unsqueeze(1)  # [B, 1, latent_dim]

    # Predict next latent using MDN-RNN (single timestep with hidden state)
    pi, mu, sigma, hidden = mdn_rnn(current_latent_flat, hidden)

    # Extract last timestep (model returns [B, T, ...], but we only have T=1)
    pi = pi[:, -1, :]  # [B, n_mixtures]
    mu = mu[:, -1, :, :]  # [B, n_mixtures, latent_dim]
    sigma = sigma[:, -1, :, :]  # [B, n_mixtures, latent_dim]

    # Sample from mixture (sampling produces sharper predictions than weighted mean)
    predicted_latent_flat = sample_from_mdn(
        pi, mu, sigma, deterministic=False
    )  # [B, latent_dim]

    # Reshape to spatial latent
    C_latent, H_latent, W_latent = latent_shape[1:]  # Skip batch dimension
    predicted_latent = predicted_latent_flat.view(B, C_latent, H_latent, W_latent)

    # Decode to image space
    predicted_frame = vae.decode_from_latent(predicted_latent, target_size=(B, C, H, W))

    # Clamp to [0, 1]
    predicted_frame = torch.clamp(predicted_frame, 0.0, 1.0)

    if squeeze_output:
        predicted_frame = predicted_frame.squeeze(0)

    return predicted_frame, hidden


@torch.no_grad()
def sample_from_mdn(pi, mu, sigma, deterministic=False):
    """
    Sample from mixture density network or return weighted mean.

    Args:
        pi: Mixture weights [B, n_mixtures]
        mu: Means [B, n_mixtures, latent_dim]
        sigma: Standard deviations [B, n_mixtures, latent_dim]
        deterministic: If True, return weighted mean instead of sampling

    Returns:
        sampled: Sampled latent vectors [B, latent_dim] or weighted mean if deterministic
    """
    if deterministic:
        # Return weighted mean: sum_k pi_k * mu_k
        weighted_mean = torch.sum(pi.unsqueeze(-1) * mu, dim=1)  # [B, latent_dim]
        return weighted_mean

    B = pi.shape[0]
    device = pi.device

    # Sample mixture component for each batch item
    # pi is already softmaxed, so we can sample from categorical distribution
    mixture_idx = torch.multinomial(pi, num_samples=1)  # [B, 1]

    # Select corresponding mu and sigma for each batch item
    batch_indices = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
    selected_mu = mu[batch_indices, mixture_idx].squeeze(1)  # [B, latent_dim]
    selected_sigma = sigma[batch_indices, mixture_idx].squeeze(1)  # [B, latent_dim]

    # Sample from selected Gaussian
    eps = torch.randn_like(selected_mu)
    sampled = selected_mu + eps * selected_sigma

    return sampled


@torch.no_grad()
def predict_next_frame_vector(mdn_rnn, vae, current_frame, latent_dim, hidden=None):
    """
    Predict next frame given current frame.
    Uses RNN hidden state to maintain history.

    Args:
        mdn_rnn: Trained MDN-RNN model
        vae: Trained VAE model
        current_frame: Current frame [C, H, W] or [B, C, H, W]
        latent_dim: Dimension of 1D latent representation (scalar)
        hidden: RNN hidden state from previous step (optional)

    Returns:
        predicted_frame: Predicted frame [C, H, W] or [B, C, H, W]
        hidden: Updated hidden state
    """
    mdn_rnn.eval()
    vae.eval()

    # Handle single frame vs batch
    if current_frame.dim() == 3:
        current_frame = current_frame.unsqueeze(0)  # [1, C, H, W]
        squeeze_output = True
    elif current_frame.dim() == 4:
        # Input is already [B, C, H, W]
        squeeze_output = False
    else:
        raise ValueError(
            f"Unexpected current_frame dimension: {current_frame.dim()}. Expected 3 ([C, H, W]) or 4 ([B, C, H, W])."
        )

    B, C, H, W = current_frame.shape
    device = current_frame.device

    # Encode current frame to latent space
    current_latent = vae.encode_to_latent(current_frame)  # [B, latent_dim] - Already 1D
    current_latent = current_latent.unsqueeze(
        1
    )  # [B, 1, latent_dim] - Add time dimension

    # Predict next latent using MDN-RNN (single timestep with hidden state)
    pi, mu, sigma, hidden = mdn_rnn(current_latent, hidden)

    # Extract last timestep (model returns [B, T, ...], but we only have T=1)
    pi = pi[:, -1, :]  # [B, n_mixtures]
    mu = mu[:, -1, :, :]  # [B, n_mixtures, latent_dim]
    sigma = sigma[:, -1, :, :]  # [B, n_mixtures, latent_dim]

    # Sample from mixture (sampling produces sharper predictions than weighted mean)
    predicted_latent = sample_from_mdn(
        pi, mu, sigma, deterministic=False
    )  # [B, latent_dim]

    # Decode to image space (latent is already 1D, no reshaping needed)
    predicted_frame = vae.decode_from_latent(predicted_latent, target_size=(B, C, H, W))

    # Clamp to [0, 1]
    predicted_frame = torch.clamp(predicted_frame, 0.0, 1.0)

    if squeeze_output:
        predicted_frame = predicted_frame.squeeze(0)

    return predicted_frame, hidden


def generate_rollout(
    mdn_rnn, vae, initial_frames, num_predictions, latent_shape, device="cpu"
):
    """
    Generate rollout by iteratively predicting future frames.
    Uses RNN hidden state to maintain history.

    Args:
        mdn_rnn: Trained MDN-RNN model
        vae: Trained VAE model
        initial_frames: Initial frames [T, C, H, W] or single frame [C, H, W]
        num_predictions: Number of future frames to predict
        latent_shape: Shape of latent representation
        device: Device to run on

    Returns:
        rollout: All frames including initial and predicted [T + num_predictions, C, H, W]
    """
    mdn_rnn.eval()
    vae.eval()

    # Handle different input formats
    if isinstance(initial_frames, np.ndarray):
        initial_frames = torch.from_numpy(initial_frames).float()

    if initial_frames.dim() == 3:
        # Single frame - start with it
        all_frames = [initial_frames.cpu().numpy()]
        current_frame = initial_frames.to(device)  # [C, H, W]
    elif initial_frames.dim() == 4:
        # Sequence of frames - use all as initial frames
        initial_frames = initial_frames.to(device)
        all_frames = [f.cpu().numpy() for f in initial_frames]
        current_frame = initial_frames[-1]  # Use last frame as starting point
    else:
        raise ValueError(f"Unexpected initial_frames shape: {initial_frames.shape}")

    # Initialize hidden state by processing initial frames sequentially
    hidden = None
    with torch.no_grad():
        # Process all initial frames to build up hidden state
        # This is critical: RNN needs context to understand dynamics
        if initial_frames.dim() == 4:
            # Process ALL initial frames to build hidden state
            # The last frame updates hidden state, then we use that updated state for predictions
            for frame in initial_frames:
                _, hidden = predict_next_frame(
                    mdn_rnn, vae, frame, latent_shape, hidden
                )
            # current_frame is already set to initial_frames[-1] above

        # Now generate predictions
        for step in range(num_predictions):
            # Predict next frame using current frame and hidden state
            predicted_frame, hidden = predict_next_frame(
                mdn_rnn, vae, current_frame, latent_shape, hidden
            )

            # Convert to numpy and add to rollout
            pred_np = predicted_frame.cpu().numpy()
            all_frames.append(pred_np)

            # Update current frame for next iteration
            current_frame = predicted_frame

    # Stack all frames
    rollout = np.stack(all_frames, axis=0)  # [T + num_predictions, C, H, W]
    return rollout


def generate_rollout_vector(
    mdn_rnn, vae, initial_frames, num_predictions, latent_dim, device="cpu"
):
    """
    Generate rollout by iteratively predicting future frames.
    Uses RNN hidden state to maintain history.

    Args:
        mdn_rnn: Trained MDN-RNN model
        vae: Trained VAE model
        initial_frames: Initial frames [T, C, H, W] or single frame [C, H, W]
        num_predictions: Number of future frames to predict
        latent_dim: Dimension of 1D latent representation (scalar)
        device: Device to run on

    Returns:
        rollout: All frames including initial and predicted [T + num_predictions, C, H, W]
    """
    mdn_rnn.eval()
    vae.eval()

    # Handle different input formats
    if isinstance(initial_frames, np.ndarray):
        initial_frames = torch.from_numpy(initial_frames).float()

    if initial_frames.dim() == 3:
        # Single frame - start with it
        all_frames = [initial_frames.cpu().numpy()]
        current_frame = initial_frames.to(device)  # [C, H, W]
    elif initial_frames.dim() == 4:
        # Sequence of frames - use all as initial frames
        initial_frames = initial_frames.to(device)
        all_frames = [f.cpu().numpy() for f in initial_frames]
        current_frame = initial_frames[-1]  # Use last frame as starting point
    else:
        raise ValueError(f"Unexpected initial_frames shape: {initial_frames.shape}")

    # Initialize hidden state by processing initial frames sequentially
    hidden = None
    with torch.no_grad():
        # Process all initial frames to build up hidden state
        # This is critical: RNN needs context to understand dynamics
        if initial_frames.dim() == 4:
            # Process ALL initial frames to build hidden state
            # The last frame updates hidden state, then we use that updated state for predictions
            for frame in initial_frames:
                _, hidden = predict_next_frame_vector(
                    mdn_rnn, vae, frame, latent_dim, hidden
                )
            # current_frame is already set to initial_frames[-1] above

        # Now generate predictions
        for step in range(num_predictions):
            # Predict next frame using current frame and hidden state
            predicted_frame, hidden = predict_next_frame_vector(
                mdn_rnn, vae, current_frame, latent_dim, hidden
            )

            # Convert to numpy and add to rollout
            pred_np = predicted_frame.cpu().numpy()
            all_frames.append(pred_np)

            # Update current frame for next iteration
            current_frame = predicted_frame

    # Stack all frames
    rollout = np.stack(all_frames, axis=0)  # [T + num_predictions, C, H, W]
    return rollout
