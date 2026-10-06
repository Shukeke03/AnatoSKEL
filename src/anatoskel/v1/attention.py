"""Physical-coordinate attention primitives for AnatoSKEL v1.

All world coordinates are RAS millimetres; volume axes are D,H,W. Geometry
sampling stays float32 under AMP and explicitly models the Swin patch centres.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.input = nn.Linear(width, hidden * 2)
        self.output = nn.Linear(hidden, width)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.input(x).chunk(2, dim=-1)
        return self.output(value * F.silu(gate))


class PhysicalPosition(nn.Module):
    def __init__(self, width: int, scale_mm: float = 64.):
        super().__init__()
        self.scale_mm = scale_mm
        self.register_buffer("frequency", 2. ** torch.arange(4), persistent=False)
        self.projection = nn.Linear(24, width)

    def forward(self, world_relative_mm: Tensor) -> Tensor:
        phase = world_relative_mm[..., None] / self.scale_mm * self.frequency * math.pi
        return self.projection(torch.cat((phase.sin(), phase.cos()), -1).flatten(-2))


def safe_padding_mask(valid: Tensor) -> tuple[Tensor, Tensor]:
    """MHA requires one key even for entirely missing CT; zero it afterwards."""
    empty = ~valid.any(-1)
    safe = valid.clone()
    safe[..., 0] |= empty
    return ~safe, empty


class SelfBlock(nn.Module):
    def __init__(self, width: int, heads: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = SwiGLU(width, hidden)

    def forward(self, x: Tensor, valid: Tensor | None = None) -> Tensor:
        value = self.norm(x)
        padding = None if valid is None else safe_padding_mask(valid)[0]
        x = x + self.attn(value, value, value, key_padding_mask=padding,
                          need_weights=False)[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x if valid is None else x.masked_fill(~valid[..., None], 0.)


class CrossBlock(nn.Module):
    def __init__(self, width: int, heads: int, hidden: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.self_block = SelfBlock(width, heads, hidden)

    def forward(self, query: Tensor, memory: Tensor,
                valid: Tensor | None = None) -> Tensor:
        memory_value = self.memory_norm(memory)
        if valid is not None:
            memory_value = memory_value.masked_fill(~valid[..., None], 0.)
            padding, empty = safe_padding_mask(valid)
        else:
            padding = None
            empty = query.new_zeros(query.shape[0], dtype=torch.bool)
        cross = self.cross(self.query_norm(query), memory_value, memory_value,
                           key_padding_mask=padding, need_weights=False)[0]
        cross = cross.masked_fill(empty[:, None, None], 0.)
        return self.self_block(query + cross)


def feature_world_grid(shape: tuple[int, int, int], affine: Tensor,
                       stride: float, offset_vox: float,
                       source_shape: tuple[int, int, int] | None = None) -> Tensor:
    """Feature/pooled-cell centre coordinates including patch-merge offsets."""
    source_shape = source_shape or shape
    axes = []
    for count, original in zip(shape, source_shape):
        index = torch.arange(count, device=affine.device, dtype=torch.float32)
        starts = torch.floor(index * original / count)
        ends = torch.ceil((index + 1) * original / count)
        axes.append(((starts + ends - 1) / 2) * stride + offset_vox)
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    xyz = torch.stack((xx, yy, zz), -1).reshape(-1, 3)
    with torch.autocast(device_type=affine.device.type, enabled=False):
        return xyz[None] @ affine.float()[:, :3, :3].transpose(-1, -2) + affine.float()[:, None, :3, 3]


def physical_sample(feature: Tensor, world_points: Tensor, affine: Tensor,
                    stride: float = 1., offset_vox: float = 0.) -> Tensor:
    """[B,N,P,3] world points -> [B,N,P,C], with correct Swin lattice."""
    if world_points.ndim != 4 or world_points.shape[-1] != 3:
        raise ValueError("World points must be [B,N,P,3]")
    with torch.autocast(device_type=feature.device.type, enabled=False):
        inverse = torch.linalg.inv(affine.float())
        vox = torch.einsum("bij,bnpj->bnpi", inverse[:, :3, :3], world_points.float())
        vox = vox + inverse[:, None, None, :3, 3]
        index = (vox - float(offset_vox)) / float(stride)
        shape_xyz = index.new_tensor(tuple(reversed(feature.shape[-3:])))
        grid = (2 * (index + .5) / shape_xyz - 1).unsqueeze(1)
        sampled = F.grid_sample(feature.float(), grid, align_corners=False,
                                padding_mode="zeros", mode="bilinear")
    return sampled[:, :, 0].permute(0, 2, 3, 1)
