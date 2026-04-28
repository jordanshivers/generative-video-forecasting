from __future__ import annotations

import bisect

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def _stride_pattern(num_layers: int, reverse: bool = False) -> list[int]:
    strides = ([1, 2] * ((num_layers + 1) // 2))[:num_layers]
    return list(reversed(strides)) if reverse else strides


class BasicConv2d(nn.Module):
    """Convolution block used by SimVP encoder/decoder modules."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        transpose: bool = False,
        act_norm: bool = False,
        groups: int = 2,
    ) -> None:
        super().__init__()
        self.act_norm = act_norm
        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride // 2,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        groups = min(groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = self.activation(self.norm(y))
        return y


class ConvSC(nn.Module):
    """Spatial convolution block from SimVP."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, transpose: bool = False
    ) -> None:
        super().__init__()
        transpose = transpose and stride != 1
        self.conv = BasicConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            transpose=transpose,
            act_norm=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GroupConv2d(nn.Module):
    """Grouped convolution branch used inside the temporal translator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        groups: int,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0 or out_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class InceptionBlock(nn.Module):
    """Multi-kernel convolution block used by the SimVP translator."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernels: tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.branches = nn.ModuleList(
            [
                GroupConv2d(
                    hidden_channels,
                    out_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=groups,
                )
                for kernel in kernels
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(x)
        return sum(branch(projected) for branch in self.branches)


class SimVPEncoder(nn.Module):
    """Frame-wise spatial encoder."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int) -> None:
        super().__init__()
        strides = _stride_pattern(num_layers)
        layers = [ConvSC(in_channels, hidden_channels, stride=strides[0])]
        layers.extend(
            ConvSC(hidden_channels, hidden_channels, stride=stride)
            for stride in strides[1:]
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.layers[0](x)
        latent = skip
        for layer in self.layers[1:]:
            latent = layer(latent)
        return latent, skip


class SimVPDecoder(nn.Module):
    """Frame-wise spatial decoder."""

    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int) -> None:
        super().__init__()
        strides = _stride_pattern(num_layers, reverse=True)
        layers = [
            ConvSC(hidden_channels, hidden_channels, stride=stride, transpose=True)
            for stride in strides[:-1]
        ]
        layers.append(
            ConvSC(
                hidden_channels * 2,
                hidden_channels,
                stride=strides[-1],
                transpose=True,
            )
        )
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](torch.cat([x, skip], dim=1))
        return self.readout(x)


class SimVPTranslator(nn.Module):
    """Inception U-Net that mixes encoded features over time."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        kernels: tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        mid_channels = max(1, hidden_channels // 2)
        enc_layers = [
            InceptionBlock(in_channels, mid_channels, hidden_channels, kernels, groups)
        ]
        enc_layers.extend(
            InceptionBlock(hidden_channels, mid_channels, hidden_channels, kernels, groups)
            for _ in range(1, num_layers)
        )
        dec_layers = [
            InceptionBlock(hidden_channels, mid_channels, hidden_channels, kernels, groups)
        ]
        dec_layers.extend(
            InceptionBlock(
                hidden_channels * 2,
                mid_channels,
                hidden_channels if i < num_layers - 1 else in_channels,
                kernels,
                groups,
            )
            for i in range(1, num_layers)
        )
        self.encoder = nn.ModuleList(enc_layers)
        self.decoder = nn.ModuleList(dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps, channels, height, width = x.shape
        z = x.reshape(batch, timesteps * channels, height, width)
        skips = []
        for idx, layer in enumerate(self.encoder):
            z = layer(z)
            if idx < self.num_layers - 1:
                skips.append(z)
        z = self.decoder[0](z)
        for idx in range(1, self.num_layers):
            z = self.decoder[idx](torch.cat([z, skips[-idx]], dim=1))
        return z.reshape(batch, timesteps, channels, height, width)


class SimVP(nn.Module):
    """
    CNN-only SimVP forecaster.

    The model maps ``context_frames`` input frames to the same number of future
    frames. It follows the encoder-translator-decoder structure from SimVP.
    """

    def __init__(
        self,
        shape_in: tuple[int, int, int, int],
        hid_s: int = 16,
        hid_t: int = 128,
        num_spatial_layers: int = 4,
        num_temporal_layers: int = 4,
        kernels: tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
    ) -> None:
        super().__init__()
        timesteps, channels, _, _ = shape_in
        self.context_frames = timesteps
        self.encoder = SimVPEncoder(channels, hid_s, num_spatial_layers)
        self.translator = SimVPTranslator(
            timesteps * hid_s,
            hid_t,
            num_temporal_layers,
            kernels=kernels,
            groups=groups,
        )
        self.decoder = SimVPDecoder(hid_s, channels, num_spatial_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps, channels, height, width = x.shape
        if timesteps != self.context_frames:
            raise ValueError(
                f"Expected {self.context_frames} context frames, got {timesteps}."
            )
        flat = x.reshape(batch * timesteps, channels, height, width)
        encoded, skip = self.encoder(flat)
        _, hidden_channels, hidden_height, hidden_width = encoded.shape
        encoded = encoded.reshape(
            batch, timesteps, hidden_channels, hidden_height, hidden_width
        )
        translated = self.translator(encoded).reshape(
            batch * timesteps, hidden_channels, hidden_height, hidden_width
        )
        decoded = self.decoder(translated, skip)
        return decoded.reshape(batch, timesteps, channels, height, width)


class SimVPSequenceDataset(Dataset):
    """Sequence-window dataset for SimVP context-to-future training."""

    def __init__(
        self,
        base_dataset,
        context_frames: int = 5,
        pred_frames: int | None = None,
        stride: int = 1,
    ) -> None:
        self.base_dataset = base_dataset
        self.context_frames = context_frames
        self.pred_frames = pred_frames or context_frames
        self.stride = stride
        if self.context_frames != self.pred_frames:
            raise ValueError("SimVP expects context_frames == pred_frames.")
        if self.context_frames < 1:
            raise ValueError("context_frames must be positive.")
        if self.stride < 1:
            raise ValueError("stride must be positive.")
        self.windows: list[tuple[int, int]] = []
        for seq_idx, seq in enumerate(base_dataset.sequences):
            max_start = len(seq) - self.context_frames - self.pred_frames
            for start_idx in range(0, max_start + 1, self.stride):
                self.windows.append((seq_idx, start_idx))
        if not self.windows:
            raise ValueError("No SimVP sequence windows available.")
        self.cum_counts = []
        running = 0
        for _ in self.windows:
            running += 1
            self.cum_counts.append(running)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        if idx < 0 or idx >= len(self.windows):
            raise IndexError("SimVPSequenceDataset index out of range")
        window_idx = bisect.bisect_right(self.cum_counts, idx)
        seq_idx, start_idx = self.windows[window_idx]
        seq = self.base_dataset.sequences[seq_idx]
        input_np = seq[start_idx : start_idx + self.context_frames].copy()
        target_np = seq[
            start_idx
            + self.context_frames : start_idx
            + self.context_frames
            + self.pred_frames
        ].copy()
        if self.base_dataset.normalize:
            input_np = np.stack([self.base_dataset._normalize(frame) for frame in input_np])
            target_np = np.stack([self.base_dataset._normalize(frame) for frame in target_np])
        return {
            "input": torch.from_numpy(input_np).float(),
            "target": torch.from_numpy(target_np).float(),
            "seq_idx": seq_idx,
            "start_idx": start_idx,
        }


def build_simvp(**kwargs) -> SimVP:
    return SimVP(**kwargs)
