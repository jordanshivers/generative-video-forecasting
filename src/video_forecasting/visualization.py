from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.diffusion import sample_latent_diffusion
from .models.flow_matching import sample_latent_flow_matching
from .models.transformer import generate_transformer_rollout


OUTPUT_DIR = Path("outputs")


def set_output_dir(path):
    global OUTPUT_DIR
    OUTPUT_DIR = Path(path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# Helper function to save reconstruction frames during training (for GIF creation)
def save_reconstruction_frame(
    vae, dataset, sample_indices, epoch, save_dir, device="cpu"
):
    """Save a single reconstruction frame for GIF creation."""
    vae.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]
            # Handle both tensor and dict formats
            if isinstance(sample, torch.Tensor):
                original = sample.unsqueeze(0).to(device)
            else:
                original = sample["image1"].unsqueeze(0).to(device)
            # Encode and decode
            reconstructed = vae.decode_from_latent(
                vae.encode_to_latent(original), target_size=original.shape
            )
            # Convert to numpy
            orig_np = original.squeeze(0).cpu().numpy()
            recon_np = reconstructed.squeeze(0).cpu().numpy()
            # Compute MSE
            mse = np.mean((orig_np - recon_np) ** 2)
            # Display images as RGB
            if orig_np.shape[0] == 1:  # Grayscale (single channel)
                # Copy single channel to all RGB channels
                orig_rgb = np.stack([orig_np[0]] * 3, axis=2)
                recon_rgb = np.stack([recon_np[0]] * 3, axis=2)
            elif orig_np.shape[0] == 2:  # Two-channel (original biological data)
                # Map two channels to RGB (channel 0 -> R, channel 1 -> G, B=0)
                orig_rgb = np.zeros((orig_np.shape[1], orig_np.shape[2], 3))
                orig_rgb[:, :, 0] = orig_np[0]
                orig_rgb[:, :, 1] = orig_np[1]
                recon_rgb = np.zeros((recon_np.shape[1], recon_np.shape[2], 3))
                recon_rgb[:, :, 0] = recon_np[0]
                recon_rgb[:, :, 1] = recon_np[1]
            else:  # Multi-channel or RGB
                # Use first channel for all RGB channels
                orig_rgb = np.stack([orig_np[0]] * 3, axis=2)
                recon_rgb = np.stack([recon_np[0]] * 3, axis=2)
            # Plot original
            axes[i, 0].imshow(np.clip(orig_rgb, 0, 1))
            axes[i, 0].set_title(f"Original", fontsize=12)
            axes[i, 0].axis("off")
            # Plot reconstruction
            axes[i, 1].imshow(np.clip(recon_rgb, 0, 1))
            axes[i, 1].set_title(f"Reconstruction (MSE: {mse:.4f})", fontsize=12)
            axes[i, 1].axis("off")
    fig.suptitle(f"Epoch {epoch + 1} - VAE Reconstructions", fontsize=14, y=0.995)
    plt.tight_layout()
    # Save frame
    frame_path = save_dir / f"epoch_{epoch + 1:03d}.png"
    plt.savefig(frame_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # Close to free memory
    return frame_path


# Visualize VAE Reconstructions
# This cell shows how well the VAE can encode and decode images
def visualize_vae_reconstructions(
    vae, dataset, num_samples=8, device="cpu", title_prefix=""
):
    """
    Visualize VAE reconstructions to assess reconstruction quality.
    Args:
        vae: Trained VAE model
        dataset: Dataset to sample from (expects data in [0, 1] range)
        num_samples: Number of samples to visualize
        device: Device to run on
        title_prefix: Prefix for plot titles
    """
    vae.eval()
    # Get random samples
    indices = torch.randperm(len(dataset))[:num_samples]
    # Create figure with 2 columns: original and reconstruction
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    total_mse = 0.0
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            # Use image1 (or image2, doesn't matter for VAE)
            original = (
                sample["image1"].unsqueeze(0).to(device)
            )  # [1, C, H, W] in [0, 1]
            # Encode and decode
            reconstructed = vae.decode_from_latent(
                vae.encode_to_latent(original), target_size=original.shape
            )  # Output in [0, 1]
            # Convert to numpy (in [0, 1])
            orig_np = original.squeeze(0).cpu().numpy()
            recon_np = reconstructed.squeeze(0).cpu().numpy()
            # Compute MSE in [0, 1] space
            mse = np.mean((orig_np - recon_np) ** 2)
            total_mse += mse
            # Already in [0, 1], so plot directly
            # Display images as RGB
            if orig_np.shape[0] == 1:  # Grayscale (single channel)
                # Copy single channel to all RGB channels
                orig_rgb = np.stack([orig_np[0]] * 3, axis=2)
                recon_rgb = np.stack([recon_np[0]] * 3, axis=2)
            elif orig_np.shape[0] == 2:  # Two-channel (original biological data)
                # Map two channels to RGB (channel 0 -> R, channel 1 -> G, B=0)
                orig_rgb = np.zeros((orig_np.shape[1], orig_np.shape[2], 3))
                orig_rgb[:, :, 0] = orig_np[0]
                orig_rgb[:, :, 1] = orig_np[1]
                recon_rgb = np.zeros((recon_np.shape[1], recon_np.shape[2], 3))
                recon_rgb[:, :, 0] = recon_np[0]
                recon_rgb[:, :, 1] = recon_np[1]
            else:  # Multi-channel or RGB
                # Use first channel for all RGB channels
                orig_rgb = np.stack([orig_np[0]] * 3, axis=2)
                recon_rgb = np.stack([recon_np[0]] * 3, axis=2)
            # Plot original
            axes[i, 0].imshow(np.clip(orig_rgb, 0, 1))
            axes[i, 0].set_title(f"{title_prefix}Original", fontsize=12)
            axes[i, 0].axis("off")
            # Plot reconstruction
            axes[i, 1].imshow(np.clip(recon_rgb, 0, 1))
            axes[i, 1].set_title(
                f"{title_prefix}Reconstruction (MSE: {mse:.4f})", fontsize=12
            )
            axes[i, 1].axis("off")
    avg_mse = total_mse / num_samples
    fig.suptitle(
        f"{title_prefix}VAE Reconstructions (Average MSE: {avg_mse:.4f})",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR
        / f"{title_prefix.lower().replace(' ', '_')}_vae_reconstructions.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
    print(f"Average reconstruction MSE: {avg_mse:.6f}")
    return avg_mse


def visualize_flow_predictions(
    flow_matching_model,
    vae,
    dataset,
    flow_utils,
    num_samples=4,
    device="cpu",
    title_prefix="",
):
    """
    Visualize predictions vs ground truth.
    Args:
        flow_matching_model: Trained velocity prediction model
        vae: Trained VAE
        dataset: Test dataset
        flow_utils: Flow Matching Utils
        num_samples: Number of samples to visualize
        device: Device to run on
        title_prefix: Prefix for plot titles
    """
    flow_matching_model.eval()
    vae.eval()
    # Get random samples
    indices = torch.randperm(len(dataset))[:num_samples]
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            image1 = sample["image1"].unsqueeze(0).to(device)  # Condition
            image2 = sample["image2"].unsqueeze(0).to(device)  # Ground truth
            # Generate prediction using latent flow matching
            predicted = sample_latent_flow_matching(
                flow_matching_model,
                vae,
                image1,
                flow_utils,
                num_inference_steps=25,
                device=device,
            )
            # Convert to numpy
            img1_np = image1.squeeze(0).cpu().numpy()
            img2_np = image2.squeeze(0).cpu().numpy()
            pred_np = predicted.squeeze(0).cpu().numpy()
            # Display two-channel images as RGB (channel 0 = red, channel 1 = green)
            if img1_np.shape[0] == 2:
                img1_rgb = np.zeros((img1_np.shape[1], img1_np.shape[2], 3))
                img1_rgb[:, :, 0] = img1_np[0]  # Red channel
                img1_rgb[:, :, 1] = img1_np[1]  # Green channel
                img2_rgb = np.zeros((img2_np.shape[1], img2_np.shape[2], 3))
                img2_rgb[:, :, 0] = img2_np[0]
                img2_rgb[:, :, 1] = img2_np[1]
                pred_rgb = np.zeros((pred_np.shape[1], pred_np.shape[2], 3))
                pred_rgb[:, :, 0] = pred_np[0]
                pred_rgb[:, :, 1] = pred_np[1]
            else:
                img1_rgb = np.stack([img1_np[0]] * 3, axis=2)
                img2_rgb = np.stack([img2_np[0]] * 3, axis=2)
                pred_rgb = np.stack([pred_np[0]] * 3, axis=2)
            # Plot
            axes[i, 0].imshow(np.clip(img1_rgb, 0, 1))
            axes[i, 0].set_title(f"{title_prefix}Input Frame", fontsize=12)
            axes[i, 0].axis("off")
            axes[i, 1].imshow(np.clip(pred_rgb, 0, 1))
            axes[i, 1].set_title(f"{title_prefix}Predicted Frame", fontsize=12)
            axes[i, 1].axis("off")
            axes[i, 2].imshow(np.clip(img2_rgb, 0, 1))
            axes[i, 2].set_title(f"{title_prefix}Ground Truth", fontsize=12)
            axes[i, 2].axis("off")
            # Compute MSE
            mse = np.mean((pred_np - img2_np) ** 2)
            axes[i, 1].text(
                0.02,
                0.98,
                f"MSE: {mse:.4f}",
                transform=axes[i, 1].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR
        / f"{title_prefix.lower().replace(' ', '_')}_latent_flow_matching_predictions.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def visualize_diffusion_predictions(
    diffusion_model,
    vae,
    dataset,
    scheduler,
    num_samples=4,
    device="cpu",
    title_prefix="",
    num_inference_steps=50,
):
    """
    Visualize predictions vs ground truth.
    Args:
        diffusion_model: Trained noise prediction model
        vae: Trained VAE
        dataset: Test dataset
        scheduler: Diffusion scheduler
        num_samples: Number of samples to visualize
        device: Device to run on
        title_prefix: Prefix for plot titles
        num_inference_steps: Number of denoising steps
    """
    diffusion_model.eval()
    vae.eval()
    # Get random samples
    indices = torch.randperm(len(dataset))[:num_samples]
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            image1 = sample["image1"].unsqueeze(0).to(device)  # Condition
            image2 = sample["image2"].unsqueeze(0).to(device)  # Ground truth
            # Generate prediction using latent diffusion
            predicted = sample_latent_diffusion(
                diffusion_model,
                vae,
                image1,
                scheduler,
                num_inference_steps=num_inference_steps,
                device=device,
            )
            # Convert to numpy
            img1_np = image1.squeeze(0).cpu().numpy()
            img2_np = image2.squeeze(0).cpu().numpy()
            pred_np = predicted.squeeze(0).cpu().numpy()
            # Display two-channel images as RGB (channel 0 = red, channel 1 = green)
            if img1_np.shape[0] == 2:
                img1_rgb = np.zeros((img1_np.shape[1], img1_np.shape[2], 3))
                img1_rgb[:, :, 0] = img1_np[0]  # Red channel
                img1_rgb[:, :, 1] = img1_np[1]  # Green channel
                img2_rgb = np.zeros((img2_np.shape[1], img2_np.shape[2], 3))
                img2_rgb[:, :, 0] = img2_np[0]
                img2_rgb[:, :, 1] = img2_np[1]
                pred_rgb = np.zeros((pred_np.shape[1], pred_np.shape[2], 3))
                pred_rgb[:, :, 0] = pred_np[0]
                pred_rgb[:, :, 1] = pred_np[1]
            else:
                img1_rgb = np.stack([img1_np[0]] * 3, axis=2)
                img2_rgb = np.stack([img2_np[0]] * 3, axis=2)
                pred_rgb = np.stack([pred_np[0]] * 3, axis=2)
            # Plot
            axes[i, 0].imshow(np.clip(img1_rgb, 0, 1))
            axes[i, 0].set_title(f"{title_prefix}Input Frame", fontsize=12)
            axes[i, 0].axis("off")
            axes[i, 1].imshow(np.clip(pred_rgb, 0, 1))
            axes[i, 1].set_title(f"{title_prefix}Predicted Frame", fontsize=12)
            axes[i, 1].axis("off")
            axes[i, 2].imshow(np.clip(img2_rgb, 0, 1))
            axes[i, 2].set_title(f"{title_prefix}Ground Truth", fontsize=12)
            axes[i, 2].axis("off")
            # Compute MSE
            mse = np.mean((pred_np - img2_np) ** 2)
            axes[i, 1].text(
                0.02,
                0.98,
                f"MSE: {mse:.4f}",
                transform=axes[i, 1].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR
        / f"{title_prefix.lower().replace(' ', '_')}_latent_diffusion_predictions.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def visualize_mdn_predictions(
    mdn_rnn,
    vae,
    dataset,
    num_samples=4,
    num_context_frames=3,
    device="cpu",
    latent_shape=None,
):
    """
    Visualize predictions vs ground truth (World Models style).

    Args:
        mdn_rnn: Trained MDN-RNN model
        vae: Trained VAE model
        dataset: SequenceDataset instance
        num_samples: Number of samples to visualize
        num_context_frames: Number of context frames to DISPLAY (visualization only).
                          NOTE: This does NOT affect prediction - the model uses ALL
                          frames in the sequence to build hidden state.
        device: Device to run on
        latent_shape: Shape of latent representation
    """
    mdn_rnn.eval()
    vae.eval()

    # Get random samples
    indices = torch.randperm(len(dataset))[:num_samples]

    fig, axes = plt.subplots(
        num_samples,
        num_context_frames + 2,
        figsize=((num_context_frames + 2) * 3, num_samples * 3),
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            sequence = sample["sequence"].to(device)  # [T, C, H, W]
            T = sequence.shape[0]

            # Show the last num_context_frames frames before the prediction
            # We're predicting frame T-1, so show frames T-4, T-3, T-2 (or available frames)
            # The model uses ALL frames in the sequence (0 to T-2) to build hidden state,
            # then predicts frame T-1 from frame T-2.
            start_idx = max(0, T - num_context_frames - 1)
            context_frames = sequence[
                start_idx : T - 1
            ]  # Last num_context_frames before prediction

            if T > 1:
                # Process frames sequentially: frame t predicts frame t+1
                # After processing frames 0 to T-2, the last prediction IS frame T-1
                hidden = None
                predicted_frame = None

                for t in range(T - 1):  # Process frames 0 to T-2
                    # Each call processes frame t and predicts frame t+1
                    pred, hidden = predict_next_frame(
                        mdn_rnn, vae, sequence[t], latent_shape, hidden
                    )
                    # Keep the last prediction (which is frame T-1 when t = T-2)
                    if t == T - 2:
                        predicted_frame = pred
                target_frame = sequence[T - 1]  # Last frame is the target
            else:
                # Sequence too short, just use what we have
                predicted_frame = sequence[-1]
                target_frame = sequence[-1]

            # Convert to numpy
            context_np = context_frames.cpu().numpy()
            target_np = target_frame.cpu().numpy()
            pred_np = predicted_frame.cpu().numpy()

            # Display context frames
            for j in range(num_context_frames):
                if j < len(context_np):
                    cond_frame = context_np[j]
                    if cond_frame.shape[0] == 1:  # Grayscale
                        axes[i, j].imshow(cond_frame[0], cmap="gray", vmin=0, vmax=1)
                    else:
                        axes[i, j].imshow(
                            np.transpose(cond_frame, (1, 2, 0)), vmin=0, vmax=1
                        )
                    # Calculate the actual frame index relative to the prediction (T-1)
                    frame_idx = start_idx + j
                    relative_idx = (
                        T - 1
                    ) - frame_idx  # How many frames before the prediction
                    axes[i, j].set_title(f"Frame t-{relative_idx}", fontsize=10)
                else:
                    axes[i, j].axis("off")
                axes[i, j].axis("off")

            # Display predicted frame
            if pred_np.shape[0] == 1:
                axes[i, num_context_frames].imshow(
                    pred_np[0], cmap="gray", vmin=0, vmax=1
                )
            else:
                axes[i, num_context_frames].imshow(
                    np.transpose(pred_np, (1, 2, 0)), vmin=0, vmax=1
                )
            axes[i, num_context_frames].set_title("Predicted", fontsize=10)
            axes[i, num_context_frames].axis("off")

            # Display ground truth
            if target_np.shape[0] == 1:
                axes[i, num_context_frames + 1].imshow(
                    target_np[0], cmap="gray", vmin=0, vmax=1
                )
            else:
                axes[i, num_context_frames + 1].imshow(
                    np.transpose(target_np, (1, 2, 0)), vmin=0, vmax=1
                )
            axes[i, num_context_frames + 1].set_title("Ground Truth", fontsize=10)
            axes[i, num_context_frames + 1].axis("off")

            # Compute MSE
            mse = np.mean((pred_np - target_np) ** 2)
            axes[i, num_context_frames].text(
                0.02,
                0.98,
                f"MSE: {mse:.4f}",
                transform=axes[i, num_context_frames].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mdn_rnn_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()


def generate_flow_rollout_movie(
    flow_matching_model,
    vae,
    test_dataset,
    sequence=None,  # Sequence array [T, C, H, W] for Moving MNIST
    tif_path=None,  # Path to TIF file (for backward compatibility)
    dataset_type="moving_mnist",
    frame_separation=5,
    start_frame=0,
    num_predictions=20,
    device="cpu",
    fps=10,
    output_dir=str(OUTPUT_DIR / "output_mp4s"),
    num_inference_steps=25,
    use_ddim=True,  # Ignored for flow matching but kept for compatibility
):
    """
    Generate rollout movie showing iterative predictions using Flow Matching.
    Args:
        flow_matching_model: Trained velocity prediction model
        vae: Trained VAE
        test_dataset: Test dataset (for normalization parameters)
        sequence: Sequence array [T, C, H, W] for Moving MNIST (if None, uses tif_path)
        tif_path: Path to TIF file for rollout (if None, uses sequence)
        dataset_type: 'moving_mnist' or 'pulsation' (for normalization)
        frame_separation: Frame separation m used during training
        start_frame: Starting frame index (default: 0)
        num_predictions: Number of prediction steps to generate
        device: Device to run on
        fps: Frames per second for output video
        output_dir: Directory to save output videos
        num_inference_steps: Number of Euler integration steps
        use_ddim: Ignored
    Returns:
        Path to saved video file
    """
    flow_matching_model.eval()
    vae.eval()
    # Load sequence (from array or TIF file)
    if sequence is not None:
        # Use provided sequence array
        data_array = sequence  # Already in [T, C, H, W] format
        print(f"Using provided sequence with shape {data_array.shape}...")
    elif tif_path is not None:
        # Load from TIF file
        print(f"Loading {tif_path.name}...")
        image_sequence = tifffile.imread(str(tif_path))
        # Handle different TIF formats
        if image_sequence.ndim == 4:
            if image_sequence.shape[1] == 2:
                data_array = image_sequence
            elif image_sequence.shape[-1] == 2:
                data_array = np.transpose(image_sequence, (0, 3, 1, 2))
            else:
                raise ValueError(f"Unexpected 4D shape: {image_sequence.shape}")
        elif image_sequence.ndim == 3:
            data_array = image_sequence[:, None, :, :]
        else:
            raise ValueError(f"Unexpected number of dimensions: {image_sequence.ndim}")
    else:
        raise ValueError("Either sequence or tif_path must be provided")
    T, C, H, W = data_array.shape
    print(f"  Sequence shape: {T} frames, {C} channels, {H}x{W}")
    # Get normalization parameters
    # Get normalization parameters (Moving MNIST is already in [0, 1])
    if hasattr(test_dataset, "normalization_params"):
        img_min, img_max = test_dataset.normalization_params
    else:
        # Moving MNIST is already normalized to [0, 1]
        img_min, img_max = 0.0, 1.0

    # Normalize function
    def normalize_frame(frame):
        """Normalize frame to [0, 1] using precomputed min/max."""
        normalized = (frame.astype(np.float32) - img_min) / (img_max - img_min)
        return np.clip(normalized, 0, 1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    # Start with real frame
    current_frame = data_array[start_frame].copy()  # [C, H, W]
    current_frame_norm = normalize_frame(current_frame)
    # Store predictions and ground truth
    predicted_frames = []
    ground_truth_frames = []
    error_maps = []
    # First frame is real
    predicted_frames.append(current_frame_norm.copy())
    if start_frame < T:
        ground_truth_frames.append(current_frame_norm.copy())
    else:
        ground_truth_frames.append(None)
    print(f"Generating {num_predictions} prediction steps...")
    print(f"  Starting from frame {start_frame}")
    with torch.no_grad():
        for step in tqdm(range(num_predictions), desc="Generating predictions"):
            # Convert current frame to tensor
            current_tensor = (
                torch.from_numpy(current_frame_norm).float().unsqueeze(0).to(device)
            )  # [1, C, H, W]
            # Predict next frame using latent flow matching
            predicted_tensor = sample_latent_flow_matching(
                flow_matching_model,
                vae,
                current_tensor,
                flow_utils,
                num_inference_steps=num_inference_steps,
                device=device,
            )
            predicted_norm = predicted_tensor.squeeze(0).cpu().numpy()  # [C, H, W]
            # Clip to valid range
            predicted_norm = np.clip(predicted_norm, 0, 1)
            # Store prediction
            predicted_frames.append(predicted_norm.copy())
            # Get ground truth if available
            gt_frame_idx = start_frame + (step + 1) * frame_separation
            if gt_frame_idx < T:
                gt_frame = normalize_frame(data_array[gt_frame_idx].copy())
                ground_truth_frames.append(gt_frame.copy())
                # Compute error map
                error = np.abs(predicted_norm - gt_frame)
                error_maps.append(error.copy())
            else:
                ground_truth_frames.append(None)
                error_maps.append(None)
            # Update current frame for next iteration
            current_frame_norm = predicted_norm.copy()
    # Create video frames
    print("Creating video frames...")
    video_frames = []
    for i in range(len(predicted_frames)):
        pred_frame = predicted_frames[i]
        gt_frame = ground_truth_frames[i] if i < len(ground_truth_frames) else None
        error_map = error_maps[i - 1] if i > 0 and i - 1 < len(error_maps) else None
        # Convert to RGB
        if pred_frame.shape[0] == 2:
            # Two channels: channel 0 = red, channel 1 = green
            pred_rgb = np.zeros((H, W, 3), dtype=np.float32)
            pred_rgb[:, :, 0] = pred_frame[0]  # Red
            pred_rgb[:, :, 1] = pred_frame[1]  # Green
            if gt_frame is not None:
                gt_rgb = np.zeros((H, W, 3), dtype=np.float32)
                gt_rgb[:, :, 0] = gt_frame[0]
                gt_rgb[:, :, 1] = gt_frame[1]
            else:
                gt_rgb = np.zeros((H, W, 3), dtype=np.float32)
            # Error map RGB (if available)
            if error_map is not None:
                error_rgb = np.zeros((H, W, 3), dtype=np.float32)
                error_sum = error_map.sum(axis=0)  # Sum across channels
                error_norm = (error_sum - error_sum.min()) / (
                    error_sum.max() - error_sum.min() + 1e-8
                )
                error_rgb[:, :, 0] = error_norm  # Red
                error_rgb[:, :, 1] = error_norm * 0.5  # Green (darker)
                error_rgb[:, :, 2] = error_norm * 0.1  # Blue (very dark)
            else:
                error_rgb = np.zeros((H, W, 3), dtype=np.float32)
        else:
            # Single channel: use grayscale
            pred_gray = pred_frame[0]
            pred_rgb = np.stack([pred_gray, pred_gray, pred_gray], axis=2)
            if gt_frame is not None:
                gt_gray = gt_frame[0]
                gt_rgb = np.stack([gt_gray, gt_gray, gt_gray], axis=2)
            else:
                gt_rgb = np.zeros((H, W, 3), dtype=np.float32)
            error_rgb = np.zeros((H, W, 3), dtype=np.float32)
        # Create composite frame: [predicted | ground truth | error]
        frame_width = W * 3
        composite_frame = np.zeros((H, frame_width, 3), dtype=np.float32)
        composite_frame[:, :W, :] = pred_rgb
        composite_frame[:, W : 2 * W, :] = gt_rgb
        composite_frame[:, 2 * W : 3 * W, :] = error_rgb
        # Convert to uint8
        composite_frame_uint8 = (np.clip(composite_frame, 0, 1) * 255).astype(np.uint8)
        video_frames.append(composite_frame_uint8)
    # Save video
    # Save video
    # Generate output filename based on input source
    if tif_path is not None:
        base_name = tif_path.stem
    elif sequence is not None:
        base_name = f"moving_mnist_sequence_{start_frame}"
    else:
        base_name = "rollout"
    output_filename = output_path / f"{base_name}_latent_flow_matching_rollout.mp4"
    print(f"Saving video to {output_filename}...")
    imageio.mimwrite(
        str(output_filename), video_frames, fps=fps, codec="libx264", quality=8
    )
    print(f"Video saved successfully!")
    return output_filename


def generate_diffusion_rollout_movie(
    diffusion_model,
    vae,
    scheduler,
    test_dataset,
    sequence=None,  # Sequence array [T, C, H, W] for Moving MNIST
    tif_path=None,  # Path to TIF file (for backward compatibility)
    dataset_type="moving_mnist",
    frame_separation=5,
    start_frame=0,
    num_predictions=20,
    device="cpu",
    fps=10,
    output_dir=str(OUTPUT_DIR / "output_mp4s"),
    num_inference_steps=50,
    use_ddim=False,  # Use DDIM for faster sampling if True
):
    """
    Generate rollout movie showing iterative predictions using Diffusion.
    Args:
        diffusion_model: Trained noise prediction model
        vae: Trained VAE
        scheduler: Diffusion scheduler
        test_dataset: Test dataset (for normalization parameters)
        sequence: Sequence array [T, C, H, W] for Moving MNIST (if None, uses tif_path)
        tif_path: Path to TIF file for rollout (if None, uses sequence)
        dataset_type: 'moving_mnist' or 'pulsation' (for normalization)
        frame_separation: Frame separation m used during training
        start_frame: Starting frame index (default: 0)
        num_predictions: Number of prediction steps to generate
        device: Device to run on
        fps: Frames per second for output video
        output_dir: Directory to save output videos
        num_inference_steps: Number of denoising steps
        use_ddim: Use DDIM for deterministic sampling (faster)
    Returns:
        Path to saved video file
    """
    diffusion_model.eval()
    vae.eval()
    # Load sequence (from array or TIF file)
    if sequence is not None:
        # Use provided sequence array
        data_array = sequence  # Already in [T, C, H, W] format
        print(f"Using provided sequence with shape {data_array.shape}...")
    elif tif_path is not None:
        # Load from TIF file
        print(f"Loading {tif_path.name}...")
        image_sequence = tifffile.imread(str(tif_path))
        # Handle different TIF formats
        if image_sequence.ndim == 4:
            if image_sequence.shape[1] == 2:
                data_array = image_sequence
            elif image_sequence.shape[-1] == 2:
                data_array = np.transpose(image_sequence, (0, 3, 1, 2))
            else:
                raise ValueError(f"Unexpected 4D shape: {image_sequence.shape}")
        elif image_sequence.ndim == 3:
            data_array = image_sequence[:, None, :, :]
        else:
            raise ValueError(f"Unexpected number of dimensions: {image_sequence.ndim}")
    else:
        raise ValueError("Either sequence or tif_path must be provided")
    T, C, H, W = data_array.shape
    print(f"  Sequence shape: {T} frames, {C} channels, {H}x{W}")
    # Get normalization parameters
    # Get normalization parameters (Moving MNIST is already in [0, 1])
    if hasattr(test_dataset, "normalization_params"):
        img_min, img_max = test_dataset.normalization_params
    else:
        # Moving MNIST is already normalized to [0, 1]
        img_min, img_max = 0.0, 1.0

    # Normalize function
    def normalize_frame(frame):
        """Normalize frame to [0, 1] using precomputed min/max."""
        normalized = (frame.astype(np.float32) - img_min) / (img_max - img_min)
        return np.clip(normalized, 0, 1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    # Start with real frame
    current_frame = data_array[start_frame].copy()  # [C, H, W]
    current_frame_norm = normalize_frame(current_frame)
    # Store predictions and ground truth
    predicted_frames = []
    ground_truth_frames = []
    error_maps = []
    # First frame is real
    predicted_frames.append(current_frame_norm.copy())
    if start_frame < T:
        ground_truth_frames.append(current_frame_norm.copy())
    else:
        ground_truth_frames.append(None)
    print(f"Generating {num_predictions} prediction steps...")
    print(f"  Starting from frame {start_frame}")
    with torch.no_grad():
        for step in tqdm(range(num_predictions), desc="Generating predictions"):
            # Convert current frame to tensor
            current_tensor = (
                torch.from_numpy(current_frame_norm).float().unsqueeze(0).to(device)
            )  # [1, C, H, W]
            # Predict next frame using latent diffusion
            predicted_tensor = sample_latent_diffusion(
                diffusion_model,
                vae,
                current_tensor,
                scheduler,
                num_inference_steps=num_inference_steps,
                device=device,
            )
            predicted_norm = predicted_tensor.squeeze(0).cpu().numpy()  # [C, H, W]
            # Clip to valid range
            predicted_norm = np.clip(predicted_norm, 0, 1)
            # Store prediction
            predicted_frames.append(predicted_norm.copy())
            # Get ground truth if available
            gt_frame_idx = start_frame + (step + 1) * frame_separation
            if gt_frame_idx < T:
                gt_frame = normalize_frame(data_array[gt_frame_idx].copy())
                ground_truth_frames.append(gt_frame.copy())
                # Compute error map
                error = np.abs(predicted_norm - gt_frame)
                error_maps.append(error.copy())
            else:
                ground_truth_frames.append(None)
                error_maps.append(None)
            # Update current frame for next iteration
            current_frame_norm = predicted_norm.copy()
    # Create video frames
    print("Creating video frames...")
    video_frames = []
    for i in range(len(predicted_frames)):
        pred_frame = predicted_frames[i]
        gt_frame = ground_truth_frames[i] if i < len(ground_truth_frames) else None
        error_map = error_maps[i - 1] if i > 0 and i - 1 < len(error_maps) else None
        # Convert to RGB
        if pred_frame.shape[0] == 2:
            # Two channels: channel 0 = red, channel 1 = green
            pred_rgb = np.zeros((H, W, 3), dtype=np.float32)
            pred_rgb[:, :, 0] = pred_frame[0]  # Red
            pred_rgb[:, :, 1] = pred_frame[1]  # Green
            if gt_frame is not None:
                gt_rgb = np.zeros((H, W, 3), dtype=np.float32)
                gt_rgb[:, :, 0] = gt_frame[0]
                gt_rgb[:, :, 1] = gt_frame[1]
            else:
                gt_rgb = np.zeros((H, W, 3), dtype=np.float32)
            # Error map RGB (if available)
            if error_map is not None:
                error_rgb = np.zeros((H, W, 3), dtype=np.float32)
                error_sum = error_map.sum(axis=0)  # Sum across channels
                error_norm = (error_sum - error_sum.min()) / (
                    error_sum.max() - error_sum.min() + 1e-8
                )
                error_rgb[:, :, 0] = error_norm  # Red
                error_rgb[:, :, 1] = error_norm * 0.5  # Green (darker)
                error_rgb[:, :, 2] = error_norm * 0.1  # Blue (very dark)
            else:
                error_rgb = np.zeros((H, W, 3), dtype=np.float32)
        else:
            # Single channel: use grayscale
            pred_gray = pred_frame[0]
            pred_rgb = np.stack([pred_gray, pred_gray, pred_gray], axis=2)
            if gt_frame is not None:
                gt_gray = gt_frame[0]
                gt_rgb = np.stack([gt_gray, gt_gray, gt_gray], axis=2)
            else:
                gt_rgb = np.zeros((H, W, 3), dtype=np.float32)
            error_rgb = np.zeros((H, W, 3), dtype=np.float32)
        # Create composite frame: [predicted | ground truth | error]
        frame_width = W * 3
        composite_frame = np.zeros((H, frame_width, 3), dtype=np.float32)
        composite_frame[:, :W, :] = pred_rgb
        composite_frame[:, W : 2 * W, :] = gt_rgb
        composite_frame[:, 2 * W : 3 * W, :] = error_rgb
        # Convert to uint8
        composite_frame_uint8 = (np.clip(composite_frame, 0, 1) * 255).astype(np.uint8)
        video_frames.append(composite_frame_uint8)
    # Save video
    # Save video
    # Generate output filename based on input source
    if tif_path is not None:
        base_name = tif_path.stem
    elif sequence is not None:
        base_name = f"moving_mnist_sequence_{start_frame}"
    else:
        base_name = "rollout"
    output_filename = output_path / f"{base_name}_latent_diffusion_rollout.mp4"
    print(f"Saving video to {output_filename}...")
    imageio.mimwrite(
        str(output_filename), video_frames, fps=fps, codec="libx264", quality=8
    )
    print(f"Video saved successfully!")
    return output_filename


def visualize_transformer_predictions(
    transformer_model,
    vae,
    dataset,
    num_samples=4,
    num_context_frames=5,
    device="cpu",
    title_prefix="",
):
    """Visualize latent transformer predictions against the next ground-truth frame."""
    transformer_model.eval()
    vae.eval()
    indices = torch.randperm(len(dataset))[:num_samples]
    fig, axes = plt.subplots(
        num_samples,
        num_context_frames + 2,
        figsize=((num_context_frames + 2) * 3, num_samples * 3),
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            sequence = sample["sequence"].to(device)
            context = sequence[:num_context_frames]
            target = sequence[num_context_frames]
            rollout = generate_transformer_rollout(
                transformer_model,
                vae,
                context,
                num_predictions=1,
                context_size=num_context_frames,
                device=device,
            )
            predicted = rollout[-1]

            for col in range(num_context_frames):
                frame = context[col].detach().cpu().numpy()
                if frame.shape[0] == 1:
                    axes[row, col].imshow(frame[0], cmap="gray", vmin=0, vmax=1)
                else:
                    axes[row, col].imshow(np.transpose(frame, (1, 2, 0)), vmin=0, vmax=1)
                axes[row, col].set_title(f"Context {col + 1}", fontsize=10)
                axes[row, col].axis("off")

            target_np = target.detach().cpu().numpy()
            pred_np = np.clip(predicted, 0, 1)
            for col, frame, title in [
                (num_context_frames, pred_np, "Predicted"),
                (num_context_frames + 1, target_np, "Ground Truth"),
            ]:
                if frame.shape[0] == 1:
                    axes[row, col].imshow(frame[0], cmap="gray", vmin=0, vmax=1)
                else:
                    axes[row, col].imshow(np.transpose(frame, (1, 2, 0)), vmin=0, vmax=1)
                axes[row, col].set_title(title, fontsize=10)
                axes[row, col].axis("off")

            mse = np.mean((pred_np - target_np) ** 2)
            axes[row, num_context_frames].text(
                0.02,
                0.98,
                f"MSE: {mse:.4f}",
                transform=axes[row, num_context_frames].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    plt.tight_layout()
    output_path = (
        OUTPUT_DIR
        / f"{title_prefix.lower().replace(' ', '_')}_latent_transformer_predictions.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return output_path


def generate_transformer_rollout_movie(
    transformer_model,
    vae,
    sequence,
    context_size=5,
    num_predictions=20,
    start_frame=0,
    device="cpu",
    fps=10,
    output_dir=str(OUTPUT_DIR / "output_mp4s"),
):
    """Generate a predicted/ground-truth/error MP4 for latent transformer rollouts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_array = np.asarray(sequence, dtype=np.float32)
    context = data_array[start_frame : start_frame + context_size]
    rollout = generate_transformer_rollout(
        transformer_model,
        vae,
        context,
        num_predictions=num_predictions,
        context_size=context_size,
        device=device,
    )

    _, channels, height, width = rollout.shape
    video_frames = []
    for idx, pred_frame in enumerate(rollout):
        gt_idx = start_frame + idx
        gt_frame = data_array[gt_idx] if gt_idx < len(data_array) else None
        pred_frame = np.clip(pred_frame, 0, 1)

        if channels == 1:
            pred_rgb = np.stack([pred_frame[0]] * 3, axis=2)
            if gt_frame is None:
                gt_rgb = np.zeros_like(pred_rgb)
                error_rgb = np.zeros_like(pred_rgb)
            else:
                gt_rgb = np.stack([gt_frame[0]] * 3, axis=2)
                error = np.abs(pred_frame[0] - gt_frame[0])
                error_rgb = np.zeros_like(pred_rgb)
                error_rgb[:, :, 0] = error
        else:
            pred_rgb = np.transpose(pred_frame[:3], (1, 2, 0))
            if pred_rgb.shape[2] < 3:
                pred_rgb = np.pad(pred_rgb, ((0, 0), (0, 0), (0, 3 - pred_rgb.shape[2])))
            if gt_frame is None:
                gt_rgb = np.zeros_like(pred_rgb)
                error_rgb = np.zeros_like(pred_rgb)
            else:
                gt_rgb = np.transpose(gt_frame[:3], (1, 2, 0))
                if gt_rgb.shape[2] < 3:
                    gt_rgb = np.pad(gt_rgb, ((0, 0), (0, 0), (0, 3 - gt_rgb.shape[2])))
                error = np.abs(pred_frame - gt_frame).sum(axis=0)
                error = (error - error.min()) / (error.max() - error.min() + 1e-8)
                error_rgb = np.zeros_like(pred_rgb)
                error_rgb[:, :, 0] = error

        composite = np.zeros((height, width * 3, 3), dtype=np.float32)
        composite[:, :width, :] = pred_rgb
        composite[:, width : 2 * width, :] = gt_rgb
        composite[:, 2 * width : 3 * width, :] = error_rgb
        video_frames.append((np.clip(composite, 0, 1) * 255).astype(np.uint8))

    output_file = output_path / f"moving_mnist_sequence_{start_frame}_latent_transformer_rollout.mp4"
    imageio.mimwrite(str(output_file), video_frames, fps=fps, codec="libx264", quality=8)
    return output_file


def plot_training_curves(
    train_losses,
    val_losses=None,
    output_path=None,
    title="Training Curves",
    ylabel="Loss",
):
    output_path = (
        Path(output_path)
        if output_path is not None
        else OUTPUT_DIR / "training_curves.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def write_comparison_mp4(video_frames, output_path, fps=10):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        str(output_path), video_frames, fps=fps, codec="libx264", quality=8
    )
    return output_path
