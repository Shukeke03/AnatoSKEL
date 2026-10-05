"""Set-based whole-scan vertebra detection with actual voxel instance masks."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from anatoskel.config import ModelConfig
from anatoskel.models.encoder import PhysicalPosition, ResUNet, bounded_pool_shape, physical_grid


class QueryBlock(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.cross_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.cross = nn.MultiheadAttention(channels, heads, dropout=0.0, batch_first=True)
        self.self_norm = nn.LayerNorm(channels)
        self.self_attention = nn.MultiheadAttention(channels, heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(),
                                 nn.Linear(4 * channels, channels))

    def forward(self, queries: Tensor, query_position: Tensor, memory: Tensor, position: Tensor) -> Tensor:
        query = self.cross_norm(queries)
        memory = self.memory_norm(memory)
        queries = queries + self.cross(query + query_position, memory + position, memory,
                                       need_weights=False)[0]
        value = self.self_norm(queries)
        queries = queries + self.self_attention(value + query_position, value + query_position,
                                                value, need_weights=False)[0]
        return queries + self.ffn(self.ffn_norm(queries))


def mask_centers_mm(mask_logits: Tensor, affine: Tensor) -> Tensor:
    """Differentiable soft centroids in physical coordinates, without a voxel grid allocation."""
    # Full CT masks can contain millions of voxels; FP16 accumulation overflows.
    weights = mask_logits.float().sigmoid()
    affine = affine.to(dtype=weights.dtype)
    mass = weights.sum(dim=(-3, -2, -1)).clamp_min(1e-6)
    dimensions = weights.shape[-3:]
    coordinates = []
    for axis, size in enumerate(dimensions):
        shape = [1] * 5
        shape[axis + 2] = size
        axis_grid = torch.arange(size, device=weights.device, dtype=weights.dtype).reshape(shape)
        coordinates.append((weights * axis_grid).sum(dim=(-3, -2, -1)) / mass)
    xyz = torch.stack(tuple(reversed(coordinates)), dim=-1)
    return xyz @ affine[:, :3, :3].transpose(-1, -2) + affine[:, None, :3, 3]


class CoarseDetector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = ResUNet(config, fine=False)
        self.query_content = nn.Embedding(config.num_queries, config.query_dim)
        self.query_position = nn.Embedding(config.num_queries, config.query_dim)
        self.projections = nn.ModuleList(nn.Conv3d(channels, config.query_dim, 1)
                                         for channels in config.coarse_channels)
        self.position = PhysicalPosition(config.query_dim, config.coordinate_scale_mm)
        self.blocks = nn.ModuleList(QueryBlock(config.query_dim, config.heads)
                                    for _ in range(config.decoder_layers))
        self.output_norm = nn.LayerNorm(config.query_dim)
        self.classifier = nn.Linear(config.query_dim, config.num_classes + 1)
        self.mask_features = nn.Conv3d(config.coarse_channels[0], config.mask_dim, 1)
        self.mask_embeddings = nn.Sequential(nn.Linear(config.query_dim, config.query_dim),
                                             nn.GELU(), nn.Linear(config.query_dim, config.mask_dim))
        self.center_delta = nn.Sequential(nn.Linear(config.query_dim, config.query_dim), nn.GELU(),
                                          nn.Linear(config.query_dim, 3))
        self.visibility = nn.Linear(config.query_dim, 2)
        # Translation begins at the predicted visible-mask centroid and remains unbounded.
        nn.init.zeros_(self.center_delta[-1].weight)
        nn.init.zeros_(self.center_delta[-1].bias)

    def forward(self, volume: Tensor, affine: Tensor) -> dict:
        context = self.encoder(volume, affine)
        batch = volume.shape[0]
        queries = self.query_content.weight.unsqueeze(0).expand(batch, -1, -1)
        query_position = self.query_position.weight.unsqueeze(0).expand(batch, -1, -1)
        # Four default blocks visit the promised E4 / E3 / E2 / E3 scales.
        schedule = (4, 3, 2, 3)
        memories: dict[int, tuple[Tensor, Tensor]] = {}
        for index, block in enumerate(self.blocks):
            level = schedule[index % len(schedule)]
            if level not in memories:
                feature = context["features"][level]
                source_shape = tuple(feature.shape[-3:])
                shape = bounded_pool_shape(source_shape, self.config.max_global_tokens)
                if shape != source_shape:
                    feature = F.adaptive_avg_pool3d(feature, shape)
                # Pointwise projection commutes with pooling; project only the
                # bounded context rather than allocating a full 256-channel map.
                feature = self.projections[level](feature)
                position = self.position(physical_grid(shape, context["affine"],
                                                       context["strides"][level], source_shape)
                                         - context["position_origin_mm"][:, None])
                memories[level] = (feature.flatten(2).transpose(1, 2), position)
            queries = block(queries, query_position, *memories[level])
        queries = self.output_norm(queries)
        mask_features = self.mask_features(context["decoded"])
        mask_logits = torch.einsum("bqc,bcdhw->bqdhw", self.mask_embeddings(queries), mask_features)
        centers = mask_centers_mm(mask_logits, context["affine"]) + self.center_delta(queries)
        return {"class_logits": self.classifier(queries), "mask_logits": mask_logits,
                "queries": queries, "centers_mm": centers,
                "visibility_logits": self.visibility(queries), "context": context}
