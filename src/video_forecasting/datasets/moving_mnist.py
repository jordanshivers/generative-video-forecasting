"""Moving MNIST dataset utilities for frame-forecasting notebooks.

The dataset returns pairs of frames separated by ``frame_separation``:
``image1`` is frame ``t`` and ``image2`` is frame ``t + frame_separation``.
Both tensors are returned as ``float32`` with shape ``(1, 64, 64)`` and
values in ``[0, 1]``.
"""

from __future__ import annotations

import bisect
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MovingMNISTDataset(Dataset):
    """Dataset for future-frame prediction on Moving MNIST sequences.

    Args:
        root: Directory where Moving MNIST data will be stored.
        train: If True, use the first 80% of sequences; otherwise use the last 20%.
        sequence_length: Expected number of frames per sequence.
        normalize: If True, return frames in ``[0, 1]``.
        frame_separation: Gap between input and target frames.
        download: If True, download the dataset if needed.
        max_sequences: Optional cap before train/test splitting for quick runs.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        train: bool = True,
        sequence_length: int = 20,
        normalize: bool = True,
        frame_separation: int = 5,
        download: bool = True,
        max_sequences: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.frame_separation = frame_separation
        self.max_sequences = max_sequences

        if frame_separation < 1:
            raise ValueError(f"frame_separation must be >= 1, got {frame_separation}")
        if frame_separation >= sequence_length:
            raise ValueError(
                f"frame_separation ({frame_separation}) must be < sequence_length ({sequence_length})"
            )
        if max_sequences is not None and max_sequences < 2:
            raise ValueError(
                "max_sequences must be at least 2 so train/test splits are non-empty"
            )

        self.target_height = 64
        self.target_width = 64
        self.img_min = 0.0 if normalize else None
        self.img_max = 1.0 if normalize else None

        self.sequences = self._load_sequences(download=download)
        if not self.sequences:
            split_name = "train" if train else "test"
            raise ValueError(
                f"No {split_name} sequences available. Increase max_sequences."
            )

        self.pair_cum_counts = []
        running_total = 0
        for seq in self.sequences:
            running_total += max(0, len(seq) - self.frame_separation)
            self.pair_cum_counts.append(running_total)
        self.total_pairs = running_total

        if self.total_pairs == 0:
            raise ValueError(
                "No frame pairs available. Check sequence_length and frame_separation."
            )

        print(
            "\nDataset initialized:\n"
            f"  Split: {'train' if train else 'test'}\n"
            f"  Total sequences: {len(self.sequences)}\n"
            f"  Frame separation: {self.frame_separation}\n"
            f"  Total pairs: {self.total_pairs}\n"
            f"  Image size: {self.target_height}x{self.target_width}\n"
            "  Channels: 1 (grayscale)"
        )

    @property
    def normalization_params(self) -> Tuple[Optional[float], Optional[float]]:
        """Return the value range used by normalized frames."""
        return self.img_min, self.img_max

    def _load_sequences(self, download: bool) -> list[np.ndarray]:
        """Load Moving MNIST and return sequence-major arrays shaped ``(T, C, H, W)``."""
        print(f"Loading Moving MNIST dataset (train={self.train})...")
        try:
            return self._load_with_torchvision(download=download)
        except (ImportError, TypeError, AttributeError, RuntimeError) as exc:
            print(f"Using manual Moving MNIST loader ({type(exc).__name__}: {exc})...")
            return self._load_manual(download=download)

    def _load_with_torchvision(self, download: bool) -> list[np.ndarray]:
        from torchvision.datasets import MovingMNIST as TorchvisionMovingMNIST

        moving_mnist = TorchvisionMovingMNIST(
            root=str(self.root),
            split=None,
            download=download,
        )
        sequences = []
        for idx in range(len(moving_mnist)):
            video = moving_mnist[idx]
            if isinstance(video, torch.Tensor):
                video_np = video.detach().cpu().numpy()
            else:
                video_np = np.asarray(video)

            video_np = self._ensure_sequence_shape(video_np)
            sequences.append(self._to_float_frames(video_np))

        return self._split_sequences(sequences)

    def _load_manual(self, download: bool) -> list[np.ndarray]:
        url = "https://github.com/edenton/svg/raw/master/data/moving_mnist/mnist_test_seq.npy"
        data_dir = self.root / "MovingMNIST"
        data_dir.mkdir(parents=True, exist_ok=True)
        npy_path = data_dir / "mnist_test_seq.npy"

        if not npy_path.exists():
            if not download:
                raise FileNotFoundError(f"Moving MNIST not found at {npy_path}")
            print(f"Downloading Moving MNIST from {url}...")
            urllib.request.urlretrieve(url, npy_path)
            print("Download complete.")

        data = np.load(npy_path)
        data = self._ensure_sequence_major(data)

        sequences = []
        for seq in data:
            seq = self._ensure_sequence_shape(seq)
            sequences.append(self._to_float_frames(seq))

        return self._split_sequences(sequences)

    def _split_sequences(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        if self.max_sequences is not None:
            sequences = sequences[: self.max_sequences]
            print(
                f"  Limited total to {len(sequences)} sequences (max_sequences={self.max_sequences})"
            )

        split_idx = int(0.8 * len(sequences))
        if split_idx == 0 or split_idx == len(sequences):
            raise ValueError(
                f"Cannot create non-empty train/test split from {len(sequences)} sequence(s)."
            )
        return sequences[:split_idx] if self.train else sequences[split_idx:]

    def _ensure_sequence_major(self, data: np.ndarray) -> np.ndarray:
        """Return data shaped ``(N, T, H, W)`` for the canonical Moving MNIST file."""
        if data.ndim != 4:
            raise ValueError(
                f"Expected Moving MNIST data with 4 dimensions, got {data.shape}"
            )

        if data.shape[1] == self.sequence_length:
            return data
        if data.shape[0] == self.sequence_length:
            return np.transpose(data, (1, 0, 2, 3))

        raise ValueError(
            "Could not infer Moving MNIST sequence axis from shape "
            f"{data.shape}; expected one axis to equal sequence_length={self.sequence_length}."
        )

    def _ensure_sequence_shape(self, seq: np.ndarray) -> np.ndarray:
        """Return one sequence shaped ``(T, C, H, W)``."""
        if seq.ndim == 3:
            if seq.shape[0] != self.sequence_length:
                raise ValueError(
                    f"Expected sequence length {self.sequence_length}, got {seq.shape}"
                )
            return seq[:, None, :, :]

        if seq.ndim == 4:
            if seq.shape[0] == self.sequence_length and seq.shape[1] in (1, 3):
                return seq
            if seq.shape[-1] in (1, 3) and seq.shape[0] == self.sequence_length:
                return np.moveaxis(seq, -1, 1)

        raise ValueError(f"Unexpected Moving MNIST sequence shape: {seq.shape}")

    def _to_float_frames(self, frames: np.ndarray) -> np.ndarray:
        frames = frames.astype(np.float32, copy=False)
        if self.normalize:
            max_value = float(np.nanmax(frames)) if frames.size else 0.0
            if max_value > 1.0:
                frames = frames / 255.0
            frames = np.clip(frames, 0.0, 1.0)
        return frames

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Compatibility helper used by the notebook training datasets."""
        return self._to_float_frames(img)

    def _pad(self, img: np.ndarray) -> np.ndarray:
        """Compatibility no-op; Moving MNIST frames are already 64x64."""
        return img

    def __len__(self) -> int:
        return self.total_pairs

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        if idx < 0 or idx >= self.total_pairs:
            raise IndexError("MovingMNISTDataset index out of range")

        seq_idx = bisect.bisect_right(self.pair_cum_counts, idx)
        prev_count = 0 if seq_idx == 0 else self.pair_cum_counts[seq_idx - 1]
        frame_idx1 = idx - prev_count
        frame_idx2 = frame_idx1 + self.frame_separation

        seq = self.sequences[seq_idx]
        img1 = torch.from_numpy(seq[frame_idx1].copy()).float()
        img2 = torch.from_numpy(seq[frame_idx2].copy()).float()

        return {
            "image1": img1,
            "image2": img2,
            "seq_idx": seq_idx,
            "frame_idx1": frame_idx1,
            "frame_idx2": frame_idx2,
        }
