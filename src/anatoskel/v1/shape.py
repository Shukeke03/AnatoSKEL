"""Deterministic indexed shape tokens and a differentiable millimetre SDF.

Coordinates are instance coordinates ``u = (x_RAS - anchor_RAS) / 64 mm``.
Only complete, reliable surfaces may be encoded as full shape targets.  The
caller controls that provenance; partial geometry belongs in observation losses.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import V1Config


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.input = nn.Linear(width, 2 * hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, value: Tensor) -> Tensor:
        gate, content = self.input(value).chunk(2, dim=-1)
        return self.output(torch.nn.functional.silu(gate) * content)


class CoordinateEncoding(nn.Module):
    """Continuous coordinates retain metric extent, without per-axis rescaling."""

    def __init__(self, bands: int = 6):
        super().__init__()
        self.register_buffer("frequencies", math.pi * 2.0 ** torch.arange(bands))
        self.output_dim = 3 + 6 * bands

    def forward(self, points_u: Tensor) -> Tensor:
        phase = points_u[..., None] * self.frequencies.to(points_u)
        return torch.cat((points_u, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)


class ShapeBlock(nn.Module):
    def __init__(self, width: int, heads: int, hidden: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = SwiGLU(width, hidden)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(normalized, normalized, normalized,
                                         need_weights=False)[0]
        return tokens + self.ffn(self.ffn_norm(tokens))


def _valid_mask(valid: Tensor | None, batch: int, length: int, device) -> Tensor | None:
    if valid is None:
        return None
    if valid.shape != (batch, length):
        raise ValueError(f"Expected valid mask {(batch, length)}, got {tuple(valid.shape)}")
    valid = valid.to(device=device, dtype=torch.bool)
    if not valid.any(dim=1).all():
        raise ValueError("Each shape must contain at least one reliable surface point")
    return valid


class ShapeAutoencoder(nn.Module):
    """One cross-attention encoder, indexed latent self-attention, and SDF decoder.

    ``freeze`` freezes parameters only. Decoding a predicted latent continues to
    propagate geometry gradients into that latent and into its predictor.
    """

    def __init__(self, config: V1Config):
        super().__init__()
        self.config = config
        width = config.shape_width
        self.coordinates = CoordinateEncoding()
        self.surface_projection = nn.Linear(self.coordinates.output_dim + 3, width)
        self.encoder_queries = nn.Parameter(torch.randn(config.latent_tokens, width) * 0.02)
        self.encoder_query_norm = nn.LayerNorm(width)
        self.encoder_surface_norm = nn.LayerNorm(width)
        self.encoder_cross = nn.MultiheadAttention(width, config.shape_heads, batch_first=True)
        self.encoder_blocks = nn.ModuleList([
            ShapeBlock(width, config.shape_heads, config.shape_ffn_hidden)
            for _ in range(config.shape_layers)
        ])
        self.to_latent = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, config.latent_dim))
        self.from_latent = nn.Linear(config.latent_dim, width)
        self.decoder_slots = nn.Parameter(torch.randn(config.latent_tokens, width) * 0.02)
        self.decoder_blocks = nn.ModuleList([
            ShapeBlock(width, config.shape_heads, config.shape_ffn_hidden)
            for _ in range(config.shape_layers)
        ])
        self.point_projection = nn.Linear(self.coordinates.output_dim, width)
        self.point_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.point_cross = nn.MultiheadAttention(width, config.shape_heads, batch_first=True)
        self.field_head = nn.Sequential(
            nn.Linear(width + self.coordinates.output_dim, config.field_hidden),
            nn.Softplus(beta=20),
            nn.Linear(config.field_hidden, config.field_hidden),
            nn.Softplus(beta=20),
            nn.Linear(config.field_hidden, 1),
        )

    def _check_latent(self, z: Tensor):
        expected = (self.config.latent_tokens, self.config.latent_dim)
        if z.ndim != 3 or tuple(z.shape[1:]) != expected:
            raise ValueError(f"Expected latent [B,{expected[0]},{expected[1]}]")

    def _checkpoint(self, function, *values):
        # Non-reentrant checkpointing supports trainable parameters even when
        # inputs do not require gradients, and frozen modules with live latents.
        live_input = any(isinstance(value, Tensor) and value.requires_grad for value in values)
        if (self.config.gradient_checkpointing and torch.is_grad_enabled()
                and (self.training or live_input)):
            return checkpoint(function, *values, use_reentrant=False)
        return function(*values)

    def _encode_cross(self, queries: Tensor, surface: Tensor, valid: Tensor | None) -> Tensor:
        surface = self.encoder_surface_norm(surface)
        return queries + self.encoder_cross(
            self.encoder_query_norm(queries), surface, surface,
            key_padding_mask=None if valid is None else ~valid, need_weights=False,
        )[0]

    def _decode_points(self, points: Tensor, memory: Tensor) -> Tensor:
        coordinates = self.coordinates(points)
        query = self.point_projection(coordinates)
        query = query + self.point_cross(self.point_norm(query), memory, memory,
                                         need_weights=False)[0]
        return self.field_head(torch.cat((query, coordinates), -1)).squeeze(-1)

    def encode(self, surface_u: Tensor, normals: Tensor, valid: Tensor | None = None) -> Tensor:
        if surface_u.ndim != 3 or surface_u.shape[-1] != 3 or normals.shape != surface_u.shape:
            raise ValueError("Surface coordinates and normals must both have shape [B,P,3]")
        if surface_u.shape[1] == 0:
            raise ValueError("Cannot encode an empty complete surface")
        valid = _valid_mask(valid, *surface_u.shape[:2], surface_u.device)
        if valid is not None:
            # Padded entries may be non-finite. Never feed them into projections.
            surface_u = torch.where(valid[..., None], surface_u, torch.zeros_like(surface_u))
            normals = torch.where(valid[..., None], normals, torch.zeros_like(normals))
        surface = self.surface_projection(torch.cat((self.coordinates(surface_u), normals), -1))
        queries = self.encoder_queries[None].expand(surface_u.shape[0], -1, -1)
        queries = self._checkpoint(self._encode_cross, queries, surface, valid)
        for block in self.encoder_blocks:
            queries = self._checkpoint(block, queries)
        return self.to_latent(queries)

    def prepare_memory(self, z: Tensor) -> Tensor:
        """Prepare once per field evaluation, retaining its latent autograd graph.

        Do not persist this tensor across optimizer steps or detach it while
        training a latent predictor. Chunked point evaluation can reuse it.
        """
        self._check_latent(z)
        memory = self.from_latent(z) + self.decoder_slots[None]
        for block in self.decoder_blocks:
            memory = self._checkpoint(block, memory)
        return self.memory_norm(memory)

    def decode(self, z: Tensor, points_u: Tensor, memory: Tensor | None = None,
               chunk_size: int = 2048) -> Tensor:
        self._check_latent(z)
        if points_u.ndim != 3 or points_u.shape[0] != z.shape[0] or points_u.shape[-1] != 3:
            raise ValueError("SDF points must have shape [B,P,3] and match the latent batch")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if memory is None:
            memory = self.prepare_memory(z)
        if memory.shape != (z.shape[0], self.config.latent_tokens, self.config.shape_width):
            raise ValueError("Prepared memory shape does not match the shape decoder")
        outputs = []
        for points in points_u.split(chunk_size, dim=1):
            outputs.append(self._checkpoint(self._decode_points, points, memory))
        if not outputs:
            return z.new_empty(z.shape[0], 0)
        # The linear output is in millimetres. There is no tanh or TSDF clamp.
        return torch.cat(outputs, dim=1)

    def forward(self, surface_u: Tensor, normals: Tensor, points_u: Tensor,
                valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
        z = self.encode(surface_u, normals, valid)
        return z, self.decode(z, points_u)

    def freeze(self) -> "ShapeAutoencoder":
        self.requires_grad_(False)
        self.eval()
        return self
