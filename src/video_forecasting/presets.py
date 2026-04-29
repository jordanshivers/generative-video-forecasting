from __future__ import annotations

from copy import deepcopy
from typing import Any


BASELINE_PRESETS: dict[str, dict[str, Any]] = {
    "moving_mnist": {
        "max_sequences": 5000,
        "sequence_length": 20,
        "frame_separation": 5,
    },
    "elastic_disks": {
        "num_sequences": 2500,
        "max_sequences": 2500,
        "sequence_length": 32,
        "frame_separation": 5,
        "num_particles": 6,
        "image_size": 64,
        "render_mode": "hard",
    },
    "vae_spatial": {
        "num_epochs": 50,
        "hidden_dims": [32, 64, 128],
        "learning_rate": 3e-4,
        "batch_size": {"cuda": 128, "mps": 64, "cpu": 32},
    },
    "vae_vector": {
        "latent_dim": 64,
        "num_epochs": 75,
        "hidden_dims": [32, 64, 128],
        "learning_rate": 3e-4,
        "batch_size": {"cuda": 128, "mps": 64, "cpu": 32},
    },
    "latent_flow_matching": {
        "context_frames": 1,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "time_emb_dim": 128,
        "base_channels": 32,
        "batch_size": {"cuda": 128, "mps": 64, "cpu": 32},
        "num_inference_steps": 50,
    },
    "latent_flow_matching_vector": {
        "context_frames": 1,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "time_emb_dim": 128,
        "hidden_dims": [512, 512, 512],
        "batch_size": {"cuda": 128, "mps": 64, "cpu": 32},
        "num_inference_steps": 50,
    },
    "latent_diffusion_vector": {
        "context_frames": 1,
        "num_epochs": 100,
        "num_timesteps": 200,
        "learning_rate": 1e-4,
        "time_emb_dim": 128,
        "hidden_dims": [512, 512, 512],
        "batch_size": {"cuda": 128, "mps": 64, "cpu": 32},
        "num_inference_steps": 100,
    },
    "pixel_flow_matching": {
        "context_frames": 1,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "time_emb_dim": 128,
        "base_channels": 32,
        "batch_size": {"cuda": 64, "mps": 32, "cpu": 16},
        "num_inference_steps": 50,
    },
    "latent_stochastic_interpolant": {
        "context_frames": 1,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "time_emb_dim": 128,
        "base_channels": 32,
        "batch_size": {"cuda": 128, "mps": 64, "cpu": 32},
        "num_inference_steps": 50,
        "sigma_coef": 1.0,
        "beta_fn": "t^2",
    },
    "pixel_stochastic_interpolant": {
        "context_frames": 1,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "time_emb_dim": 128,
        "base_channels": 32,
        "batch_size": {"cuda": 64, "mps": 32, "cpu": 16},
        "num_inference_steps": 50,
        "sigma_coef": 1.0,
        "beta_fn": "t^2",
    },
    "pixel_diffusion": {
        "context_frames": 1,
        "num_epochs": 50,
        "num_timesteps": 200,
        "learning_rate": 1e-4,
        "time_emb_dim": 128,
        "base_channels": 32,
        "batch_size": {"cuda": 64, "mps": 32, "cpu": 16},
        "num_inference_steps": 100,
    },
    "simvp": {
        "context_frames": 10,
        "pred_frames": 10,
        "hid_s": 32,
        "hid_t": 128,
        "num_spatial_layers": 4,
        "num_temporal_layers": 4,
        "num_epochs": 100,
        "batch_size": {"cuda": 128, "mps": 16, "cpu": 8},
    },
    "latent_transformer": {
        "context_size": 10,
        "latent_dim": 64,
        "d_model": 256,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "num_epochs": 100,
        "batch_size": {"cuda": 128, "mps": 64, "cpu": 32},
        "sequence_batch_size": {"cuda": 64, "mps": 32, "cpu": 16},
    },
    "mdn_rnn": {
        "hidden_dim": 256,
        "num_layers": 2,
        "n_mixtures": 5,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": {"cuda": 128, "mps": 64, "cpu": 32},
    },
}


def get_preset(name: str) -> dict[str, Any]:
    return deepcopy(BASELINE_PRESETS[name])


def batch_size_for_device(device: Any, sizes: dict[str, int]) -> int:
    device_type = getattr(device, "type", str(device))
    return sizes.get(device_type, sizes["cpu"])
