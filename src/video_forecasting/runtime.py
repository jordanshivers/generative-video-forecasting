"""Runtime helpers shared by local and Colab notebooks."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def get_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root from repo root, notebooks/, or a Colab clone path."""
    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "requirements.txt").exists() and (
            candidate / "src" / "video_forecasting"
        ).is_dir():
            return candidate
    if current.name == "notebooks" and (current.parent / "requirements.txt").exists():
        return current.parent
    raise FileNotFoundError(
        f"Could not find generative-video-forecasting repo root from {current}. "
        "Run notebooks from the repo root or notebooks/ directory."
    )


def get_data_dir(repo_root: str | Path | None = None) -> Path:
    root = get_repo_root(repo_root) if repo_root is not None else get_repo_root()
    path = root / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir(notebook_name: str, repo_root: str | Path | None = None) -> Path:
    root = get_repo_root(repo_root) if repo_root is not None else get_repo_root()
    path = root / "outputs" / notebook_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(prefer_mps: bool = True) -> torch.device:
    """Select CUDA, MPS, or CPU, with MOVING_MNIST_DEVICE override."""
    override = os.environ.get("MOVING_MNIST_DEVICE", "").strip().lower()
    if override:
        if override not in {"cuda", "mps", "cpu"}:
            raise ValueError("MOVING_MNIST_DEVICE must be one of: cuda, mps, cpu")
        if override == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "MOVING_MNIST_DEVICE=cuda was requested but CUDA is unavailable"
            )
        if override == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MOVING_MNIST_DEVICE=mps was requested but MPS is unavailable"
            )
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        prefer_mps
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
