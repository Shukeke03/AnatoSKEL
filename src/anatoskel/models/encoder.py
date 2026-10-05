"""Native PyTorch volumetric encoders with bounded attention memory."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from anatoskel.config import ModelConfig


def bounded_pool_shape(shape: tuple[int, ...], limit: int) -> tuple[int, ...]:
    """Retain the volume aspect ratio while enforcing a strict token budget."""
    if limit < 1:
        raise ValueError("Token limit must be positive")
    result = list(shape)
    while math.prod(result) > limit:
        axis = max(range(3), key=lambda i: result[i])
        result[axis] -= 1
    return tuple(result)


def physical_grid(
    shape: tuple[int, ...], affine: Tensor, stride: int = 1,
    source_shape: tuple[int, ...] | None = None,
) -> Tensor:
    """XYZ millimetres for tensor DHW positions, including pooled cell centres.

    The affine maps input XYZ voxel indices to RAS millimetres. Strided padded
    convolutions have sample centres at j * stride, not at (j + .5) * stride.
    """
    source_shape = source_shape or shape
    axes = []
    for count, original in zip(shape, source_shape, strict=True):
        if count == original:
            axis = torch.arange(count, device=affine.device, dtype=affine.dtype)
        else:
            # Match adaptive-average-pooling intervals even for odd dimensions.
            starts = torch.floor(torch.arange(count, device=affine.device) * original / count)
            ends = torch.ceil((torch.arange(count, device=affine.device) + 1) * original / count)
            axis = (starts + ends - 1).to(affine.dtype) / 2
        axes.append(axis * stride)
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    xyz = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    return xyz.unsqueeze(0) @ affine[:, :3, :3].transpose(-1, -2) + affine[:, None, :3, 3]


def physical_center(shape: tuple[int, ...], affine: Tensor) -> Tensor:
    xyz = affine.new_tensor(tuple((size - 1) / 2 for size in reversed(shape)))
    return torch.einsum("bij,j->bi", affine[:, :3, :3], xyz) + affine[:, :3, 3]


class PhysicalPosition(nn.Module):
    """Fixed-frequency encoding of physical positions followed by projection."""

    def __init__(self, channels: int, coordinate_scale_mm: float):
        super().__init__()
        self.coordinate_scale_mm = coordinate_scale_mm
        self.register_buffer("frequencies", 2.0 ** torch.arange(4), persistent=False)
        self.projection = nn.Linear(24, channels)

    def forward(self, xyz_mm: Tensor) -> Tensor:
        angles = xyz_mm.unsqueeze(-1) / self.coordinate_scale_mm * self.frequencies * math.pi
        encoded = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return self.projection(encoded)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(0.01)
        self.skip = (nn.Identity() if in_channels == out_channels and stride == 1
                     else nn.Conv3d(in_channels, out_channels, 1, stride, bias=False))

    @staticmethod
    def normalize(norm: nn.InstanceNorm3d, value: Tensor) -> Tensor:
        # A small test crop can collapse to one spatial sample. Instance variance
        # is then undefined; the affine-only limit keeps inference well-defined.
        if math.prod(value.shape[-3:]) == 1:
            return value * norm.weight[None, :, None, None, None] + norm.bias[None, :, None, None, None]
        return norm(value)

    def forward(self, value: Tensor) -> Tensor:
        residual = self.skip(value)
        value = self.activation(self.normalize(self.norm1, self.conv1(value)))
        value = self.normalize(self.norm2, self.conv2(value))
        return self.activation(value + residual)


class AttentionBlock(nn.Module):
    """Pre-LN self attention and a GELU 4x-width feed-forward network."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 4), nn.GELU(),
                                 nn.Linear(channels * 4, channels))

    def forward(self, tokens: Tensor, position: Tensor, valid: Tensor | None = None) -> Tensor:
        value = self.norm1(tokens)
        attended = self.attention(value + position, value + position, value,
                                  key_padding_mask=None if valid is None else ~valid,
                                  need_weights=False)[0]
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens if valid is None else tokens.masked_fill(~valid[..., None], 0)


def partition_windows(value: Tensor, window_size: int, shift: int = 0) -> tuple[Tensor, Tensor, tuple]:
    """Offset windows by padding, with no cyclic connection across opposite edges."""
    if not 0 <= shift < window_size:
        raise ValueError("Window offset must lie in [0, window_size)")
    batch, channels, depth, height, width = value.shape
    padded = tuple(math.ceil((size + shift) / window_size) * window_size
                   for size in (depth, height, width))
    padding = (shift, padded[2] - width - shift, shift, padded[1] - height - shift,
               shift, padded[0] - depth - shift)
    value = F.pad(value, padding)
    valid = F.pad(torch.ones((batch, 1, depth, height, width), device=value.device,
                            dtype=torch.bool), padding)
    zd, yh, xw = (size // window_size for size in padded)

    def partition(tensor: Tensor) -> Tensor:
        return tensor.reshape(batch, -1, zd, window_size, yh, window_size, xw, window_size).permute(
            0, 2, 4, 6, 3, 5, 7, 1).reshape(batch * zd * yh * xw, window_size ** 3, -1)

    metadata = (batch, channels, (depth, height, width), padded, window_size, shift)
    return partition(value), partition(valid).squeeze(-1), metadata


def unpartition_windows(tokens: Tensor, metadata: tuple) -> Tensor:
    batch, channels, shape, padded, window_size, shift = metadata
    zd, yh, xw = (size // window_size for size in padded)
    value = tokens.reshape(batch, zd, yh, xw, window_size, window_size, window_size, channels).permute(
        0, 7, 1, 4, 2, 5, 3, 6).reshape(batch, channels, *padded)
    return value[:, :, shift:shift + shape[0], shift:shift + shape[1], shift:shift + shape[2]]


class WindowAttention(nn.Module):
    def __init__(self, channels: int, config: ModelConfig):
        super().__init__()
        self.window_size = config.window_size
        self.position = PhysicalPosition(channels, config.coordinate_scale_mm)
        self.blocks = nn.ModuleList(AttentionBlock(channels, config.heads)
                                    for _ in range(config.attention_blocks))

    def forward(self, feature: Tensor, affine: Tensor, stride: int, position_origin_mm: Tensor) -> Tensor:
        position = self.position(physical_grid(feature.shape[-3:], affine, stride) - position_origin_mm[:, None])
        position = position.transpose(1, 2).reshape(feature.shape)
        for index, block in enumerate(self.blocks):
            shift = 0 if index % 2 == 0 else self.window_size // 2
            tokens, valid, metadata = partition_windows(feature, self.window_size, shift)
            position_tokens, _, _ = partition_windows(position, self.window_size, shift)
            feature = unpartition_windows(block(tokens, position_tokens, valid), metadata)
        return feature


class GlobalAttention(nn.Module):
    def __init__(self, channels: int, config: ModelConfig):
        super().__init__()
        self.limit = config.max_global_tokens
        self.position = PhysicalPosition(channels, config.coordinate_scale_mm)
        self.blocks = nn.ModuleList(AttentionBlock(channels, config.heads)
                                    for _ in range(config.attention_blocks))

    def forward(self, feature: Tensor, affine: Tensor, stride: int, position_origin_mm: Tensor) -> Tensor:
        source_shape = tuple(feature.shape[-3:])
        shape = bounded_pool_shape(source_shape, self.limit)
        pooled = feature if shape == source_shape else F.adaptive_avg_pool3d(feature, shape)
        tokens = pooled.flatten(2).transpose(1, 2)
        position = self.position(physical_grid(shape, affine, stride, source_shape) - position_origin_mm[:, None])
        original = tokens
        for block in self.blocks:
            tokens = block(tokens, position)
        delta = (tokens - original).transpose(1, 2).reshape(pooled.shape)
        if shape != source_shape:
            delta = F.interpolate(delta, size=source_shape, mode="trilinear", align_corners=False)
        return feature + delta


class ResUNet(nn.Module):
    def __init__(self, config: ModelConfig, fine: bool = False):
        super().__init__()
        config.validate()
        channels = config.fine_channels if fine else config.coarse_channels
        counts = config.fine_blocks if fine else config.coarse_blocks
        self.fine = fine
        self.gradient_checkpointing = config.gradient_checkpointing
        self.stem = nn.Conv3d(1, channels[0], 3, padding=1, bias=False)
        self.stages = nn.ModuleList()
        for index, (width, count) in enumerate(zip(channels, counts, strict=True)):
            previous = channels[index - 1] if index else channels[0]
            blocks = [ResidualBlock(previous, width, 1 if index == 0 else 2)]
            blocks.extend(ResidualBlock(width, width) for _ in range(count - 1))
            self.stages.append(nn.Sequential(*blocks))
        self.window = WindowAttention(channels[3], config) if fine else None
        self.global_attention = GlobalAttention(channels[4], config)
        self.decoder = nn.ModuleList(ResidualBlock(channels[index + 1] + channels[index], channels[index])
                                     for index in range(3, -1, -1))

    def forward(self, volume: Tensor, affine: Tensor) -> dict:
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError("Volume must be [B, 1, D, H, W]")
        if affine.shape != (volume.shape[0], 4, 4):
            raise ValueError("Affine must be [B, 4, 4] and map XYZ indices to RAS mm")
        affine = affine.to(device=volume.device, dtype=volume.dtype)
        if not torch.isfinite(volume).all() or not torch.isfinite(affine).all():
            raise ValueError("Volume and affine must be finite")
        # Image origins vary across NIfTI files. Attention needs physical spacing
        # and orientation, while a common origin should not change appearance.
        position_origin_mm = physical_center(volume.shape[-3:], affine)
        feature = self.stem(volume)
        features = []
        for index, stage in enumerate(self.stages):
            feature = self.apply_block(stage, feature)
            if index == 3 and self.window is not None:
                feature = self.apply_block(self.window, feature, affine, 8, position_origin_mm)
            if index == 4:
                feature = self.apply_block(self.global_attention, feature, affine, 16, position_origin_mm)
            features.append(feature)
        decoded = features[-1]
        for block, skip in zip(self.decoder, reversed(features[:-1]), strict=True):
            decoded = F.interpolate(decoded, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            decoded = self.apply_block(block, torch.cat((decoded, skip), dim=1))
        return {"features": features, "decoded": decoded, "strides": [1, 2, 4, 8, 16],
                "volume": volume, "affine": affine, "position_origin_mm": position_origin_mm}

    def apply_block(self, module: nn.Module, *arguments) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *arguments, use_reentrant=False, preserve_rng_state=False)
        return module(*arguments)
