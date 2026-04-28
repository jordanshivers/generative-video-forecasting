"""Synthetic elastic-disk image dataset.

The dataset generates grayscale movies of equal-mass elastic disks moving in a
2D box. The rendered images use Gaussian blobs, but the dynamics are simple
elastic collisions rather than force-integrated particle dynamics. Frames are
returned as ``float32`` tensors with shape ``(1, H, W)`` and values in
``[0, 1]``. Generated arrays are cached under ``data/elastic_disks/`` so
Colab and local runs are deterministic.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ElasticDisksDataset(Dataset):
    """Frame-pair dataset for synthetic 2D elastic-disk trajectories."""

    def __init__(
        self,
        root: str | Path = "./data",
        train: bool = True,
        num_sequences: int = 200,
        sequence_length: int = 20,
        image_size: int = 64,
        num_particles: int = 6,
        radius: float = 0.06,
        speed_range: tuple[float, float] = (0.015, 0.035),
        boundary: str = "reflecting",
        normalize: bool = True,
        frame_separation: int = 5,
        seed: int = 42,
        max_sequences: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.num_particles = num_particles
        self.radius = radius
        self.speed_range = speed_range
        self.boundary = boundary
        self.normalize = normalize
        self.frame_separation = frame_separation
        self.seed = seed
        self.max_sequences = max_sequences
        self.target_height = image_size
        self.target_width = image_size
        self.img_min = 0.0 if normalize else None
        self.img_max = 1.0 if normalize else None

        self._validate()
        self.sequences = self._load_or_generate()
        if not self.sequences:
            split_name = "train" if train else "test"
            raise ValueError(f"No {split_name} sequences available.")

        self.pair_cum_counts = []
        running_total = 0
        for seq in self.sequences:
            running_total += max(0, len(seq) - self.frame_separation)
            self.pair_cum_counts.append(running_total)
        self.total_pairs = running_total
        if self.total_pairs == 0:
            raise ValueError("No frame pairs available for this frame_separation.")

        print(
            "\nDataset initialized:\n"
            f"  Dataset: elastic_disks\n"
            f"  Split: {'train' if train else 'test'}\n"
            f"  Total sequences: {len(self.sequences)}\n"
            f"  Boundary: {self.boundary}\n"
            f"  Particles: {self.num_particles}\n"
            f"  Frame separation: {self.frame_separation}\n"
            f"  Total pairs: {self.total_pairs}\n"
            f"  Image size: {self.target_height}x{self.target_width}\n"
            "  Channels: 1 (grayscale)"
        )

    @property
    def normalization_params(self) -> Tuple[Optional[float], Optional[float]]:
        return self.img_min, self.img_max

    def _validate(self) -> None:
        if self.boundary not in {"reflecting", "periodic"}:
            raise ValueError("boundary must be 'reflecting' or 'periodic'")
        if self.num_sequences < 2:
            raise ValueError("num_sequences must be at least 2")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.frame_separation < 1 or self.frame_separation >= self.sequence_length:
            raise ValueError("frame_separation must be in [1, sequence_length)")
        if self.num_particles < 1:
            raise ValueError("num_particles must be positive")
        if not (0.0 < self.radius < 0.5):
            raise ValueError("radius must be in (0, 0.5)")
        if self.speed_range[0] <= 0 or self.speed_range[1] < self.speed_range[0]:
            raise ValueError("speed_range must be positive and ordered")
        if self.max_sequences is not None and self.max_sequences < 2:
            raise ValueError("max_sequences must be at least 2")

    def _cache_path(self) -> Path:
        params = {
            "num_sequences": self.num_sequences,
            "sequence_length": self.sequence_length,
            "image_size": self.image_size,
            "num_particles": self.num_particles,
            "radius": self.radius,
            "speed_range": self.speed_range,
            "boundary": self.boundary,
            "seed": self.seed,
        }
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
        cache_dir = self.root / "elastic_disks"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"elastic_disks_{key}.npz"

    def _load_or_generate(self) -> list[np.ndarray]:
        cache_path = self._cache_path()
        if cache_path.exists():
            data = np.load(cache_path)["sequences"]
        else:
            data = self._generate_sequences()
            np.savez_compressed(cache_path, sequences=data)
        sequences = [seq for seq in data]
        return self._split_sequences(sequences)

    def _split_sequences(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        if self.max_sequences is not None:
            sequences = sequences[: self.max_sequences]
        split_idx = int(0.8 * len(sequences))
        if split_idx == 0 or split_idx == len(sequences):
            raise ValueError(
                f"Cannot create non-empty train/test split from {len(sequences)} sequence(s)."
            )
        return sequences[:split_idx] if self.train else sequences[split_idx:]

    def _generate_sequences(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        sequences = np.empty(
            (self.num_sequences, self.sequence_length, 1, self.image_size, self.image_size),
            dtype=np.float32,
        )
        for seq_idx in range(self.num_sequences):
            positions = self._initial_positions(rng)
            velocities = self._initial_velocities(rng)
            for frame_idx in range(self.sequence_length):
                sequences[seq_idx, frame_idx, 0] = self._render(positions)
                if frame_idx < self.sequence_length - 1:
                    positions, velocities = self._step(positions, velocities)
        return sequences

    def _initial_positions(self, rng: np.random.Generator) -> np.ndarray:
        positions: list[np.ndarray] = []
        attempts = 0
        min_dist = 2.2 * self.radius
        while len(positions) < self.num_particles and attempts < 5000:
            attempts += 1
            candidate = rng.uniform(self.radius, 1.0 - self.radius, size=2)
            if all(np.linalg.norm(candidate - pos) >= min_dist for pos in positions):
                positions.append(candidate)
        if len(positions) != self.num_particles:
            raise RuntimeError("Could not place non-overlapping particles; reduce radius or count.")
        return np.asarray(positions, dtype=np.float32)

    def _initial_velocities(self, rng: np.random.Generator) -> np.ndarray:
        angles = rng.uniform(0.0, 2.0 * np.pi, size=self.num_particles)
        speeds = rng.uniform(self.speed_range[0], self.speed_range[1], size=self.num_particles)
        velocities = np.stack([np.cos(angles), np.sin(angles)], axis=1) * speeds[:, None]
        return velocities.astype(np.float32)

    def _step(self, positions: np.ndarray, velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = positions + velocities
        if self.boundary == "periodic":
            positions = positions % 1.0
        else:
            low = positions < self.radius
            high = positions > (1.0 - self.radius)
            velocities[low | high] *= -1.0
            positions = np.clip(positions, self.radius, 1.0 - self.radius)

        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                delta = positions[j] - positions[i]
                if self.boundary == "periodic":
                    delta = delta - np.round(delta)
                dist = float(np.linalg.norm(delta))
                min_dist = 2.0 * self.radius
                if dist <= 1e-8 or dist >= min_dist:
                    continue
                normal = delta / dist
                rel_vel = velocities[j] - velocities[i]
                rel_normal = float(np.dot(rel_vel, normal))
                if rel_normal < 0.0:
                    velocities[i] += rel_normal * normal
                    velocities[j] -= rel_normal * normal
                overlap = min_dist - dist
                positions[i] -= 0.5 * overlap * normal
                positions[j] += 0.5 * overlap * normal
                if self.boundary == "periodic":
                    positions = positions % 1.0
                else:
                    positions = np.clip(positions, self.radius, 1.0 - self.radius)
        return positions.astype(np.float32), velocities.astype(np.float32)

    def _render(self, positions: np.ndarray) -> np.ndarray:
        coords = (np.arange(self.image_size, dtype=np.float32) + 0.5) / self.image_size
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        frame = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        sigma = self.radius / 2.0
        for x_pos, y_pos in positions:
            dx = xx - x_pos
            dy = yy - y_pos
            if self.boundary == "periodic":
                dx = dx - np.round(dx)
                dy = dy - np.round(dy)
            dist2 = dx * dx + dy * dy
            frame += np.exp(-0.5 * dist2 / (sigma * sigma))
        return np.clip(frame, 0.0, 1.0)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32, copy=False)
        return np.clip(img, 0.0, 1.0) if self.normalize else img

    def _pad(self, img: np.ndarray) -> np.ndarray:
        return img

    def __len__(self) -> int:
        return self.total_pairs

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        if idx < 0 or idx >= self.total_pairs:
            raise IndexError("ElasticDisksDataset index out of range")
        seq_idx = bisect.bisect_right(self.pair_cum_counts, idx)
        prev_count = 0 if seq_idx == 0 else self.pair_cum_counts[seq_idx - 1]
        frame_idx1 = idx - prev_count
        frame_idx2 = frame_idx1 + self.frame_separation
        seq = self.sequences[seq_idx]
        return {
            "image1": torch.from_numpy(seq[frame_idx1].copy()).float(),
            "image2": torch.from_numpy(seq[frame_idx2].copy()).float(),
            "seq_idx": seq_idx,
            "frame_idx1": frame_idx1,
            "frame_idx2": frame_idx2,
        }
