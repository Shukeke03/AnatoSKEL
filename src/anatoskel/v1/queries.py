"""Geometry-aware typed instance queries and fixed-index latent anchors.

Identity stays a soft output head, never an input to the shape prior. All
multiscale QD features refer to the same world point, including tile overlaps.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import V1Config
from .attention import (SwiGLU, PhysicalPosition, SelfBlock, CrossBlock,
                        physical_sample, feature_world_grid, safe_padding_mask)


def bounded_shape(shape: tuple[int, int, int], limit: int) -> tuple[int, int, int]:
    result = list(shape)
    while math.prod(result) > limit:
        axis = max(range(3), key=lambda i: result[i])
        result[axis] -= 1
    return tuple(result)


class GeometryBlock(nn.Module):
    def __init__(self, config: V1Config):
        super().__init__()
        width = config.query_dim
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, config.heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, config.heads, batch_first=True)
        self.point_norm = nn.LayerNorm(width)
        self.points_attention = nn.MultiheadAttention(width, config.heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = SwiGLU(width, config.ffn_hidden)
        self.level_weights = nn.Linear(width, 5)
        self.level_content_norm = nn.LayerNorm(width)
        self.level_content_score = nn.Linear(width, 1)
        self.level_position_score = nn.Linear(3, 5)
        self.coordinate_scale_mm = config.shape_scale_mm
        self.center_update = nn.Linear(width, 3)
        nn.init.zeros_(self.center_update.weight)
        nn.init.zeros_(self.center_update.bias)

    def forward(self, queries: Tensor, memory: Tensor, memory_valid: Tensor,
                points: Tensor, point_valid: Tensor, position: Tensor,
                max_step_mm: float, relative_points_mm: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch, instances, types, width = queries.shape
        value = self.self_norm(queries + position[:, :, None])
        flat = value.reshape(batch, instances * types, width)
        queries = queries + self.self_attention(flat, flat, flat, need_weights=False)[0].reshape_as(queries)
        memory_value = self.memory_norm(memory).masked_fill(~memory_valid[..., None], 0.)
        padding, empty = safe_padding_mask(memory_valid)
        cross = self.cross(self.cross_norm(queries).reshape(batch, instances * types, width),
                           memory_value, memory_value, key_padding_mask=padding,
                           need_weights=False)[0].reshape_as(queries)
        queries = queries + cross.masked_fill(empty[:, None, None, None], 0.)
        # Query-dependent aggregation over five levels at identical positions.
        query_score = self.level_weights(self.point_norm(queries))[:, :, :, None, :]
        content_score = self.level_content_score(self.level_content_norm(points)).squeeze(-1)[:, :, None]
        score = query_score + content_score
        if relative_points_mm is not None:
            spatial_score = self.level_position_score(relative_points_mm / self.coordinate_scale_mm)
            score = score + spatial_score[:, :, None]
        weights = score.softmax(-1)
        aggregate = torch.einsum("bnkps,bnpsc->bnkpc", weights, points)
        aggregate = aggregate.reshape(batch * instances * types, points.shape[2], width)
        valid = point_valid[:, :, None].expand(-1, -1, types, -1).reshape(batch * instances * types, -1)
        padding, empty = safe_padding_mask(valid)
        aggregate = aggregate.masked_fill(~valid[..., None], 0.)
        q = self.point_norm(queries).reshape(batch * instances * types, 1, width)
        delta = self.points_attention(q, aggregate, aggregate, key_padding_mask=padding,
                                      need_weights=False)[0].squeeze(1)
        queries = queries + delta.masked_fill(empty[:, None], 0.).reshape_as(queries)
        queries = queries + self.ffn(self.ffn_norm(queries))
        step = self.center_update(queries[:, :, 0]).float().tanh() * max_step_mm
        return queries, step


class GeometryQueries(nn.Module):
    def __init__(self, config: V1Config):
        super().__init__()
        self.config = config.validate()
        fs, width = config.feature_size, config.query_dim
        self.level_projections = nn.ModuleList(nn.Linear(fs * 2**i, width) for i in range(5))
        self.context_projection = nn.Linear(fs * 4, width)
        self.position = PhysicalPosition(width, config.shape_scale_mm)
        self.context_blocks = nn.ModuleList(SelfBlock(width, config.heads, config.ffn_hidden)
                                             for _ in range(config.context_layers))
        self.context_slots = nn.Parameter(torch.randn(config.context_tokens, width) * .02)
        self.context_level_embedding = nn.Parameter(torch.randn(3, width) * .02)
        self.context_aggregators = nn.ModuleList(CrossBlock(width, config.heads, config.ffn_hidden)
                                                 for _ in range(config.context_layers))
        self.slot_content = nn.Parameter(torch.randn(config.num_queries, 3, width) * .02)
        self.proposal_query = nn.Parameter(torch.randn(config.num_queries, width) * .02)
        self.proposal_score = nn.Linear(width, 1)
        self.proposal_projection = nn.Linear(config.detail_channels, width)
        self.heatmap = nn.Conv3d(config.detail_channels, 1, 1)
        self.global_blocks = nn.ModuleList(GeometryBlock(config) for _ in range(config.global_layers))
        self.local_blocks = nn.ModuleList(GeometryBlock(config) for _ in range(config.local_layers))
        self.point_offsets = nn.Linear(width, config.query_points * 3)
        nn.init.zeros_(self.point_offsets.weight)
        nn.init.zeros_(self.point_offsets.bias)
        # Fixed spatial sample offsets avoid collapsed equal samples at startup.
        index = torch.arange(config.query_points, dtype=torch.float32)
        phase = index * math.pi * (3 - math.sqrt(5))
        z = 1 - 2 * (index + .5) / config.query_points
        ring = torch.sqrt((1 - z.square()).clamp_min(0))
        offsets = torch.stack((ring * phase.cos(), ring * phase.sin(), z), -1) * 8.
        self.register_buffer("fixed_offsets_mm", offsets)
        self.output_norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, config.num_classes + 1)
        self.extent = nn.Linear(width, 6)
        self.extent_scale_mm = config.local_size * config.local_spacing_mm / 2
        radius = 7. if config.local_size <= 32 else 25.
        nn.init.zeros_(self.extent.weight)
        nn.init.constant_(self.extent.bias, math.log(math.expm1(radius / self.extent_scale_mm)))
        self.visibility = nn.Linear(width, 2)

    def _memory(self, context: dict) -> tuple[Tensor, Tensor, Tensor]:
        """Learned context slots cross-attend physical S8/S16/S32 memories.

        Returned positions are a diagnostic reference only. Learned aggregated
        slots are not spatial cells and cannot be masked as visible anatomy.
        """
        memories, validities = [], []
        for level in (2, 3, 4):
            feature = context["features"][level]
            # S8 is pooled before projection; coarser stages normally fit whole.
            shape = bounded_shape(tuple(feature.shape[-3:]), 512)
            stride = context["strides"][level]
            valid_level = F.avg_pool3d(context["valid_mask"].float(), stride, stride)
            if valid_level.shape[-3:] != feature.shape[-3:]:
                valid_level = F.interpolate(valid_level, size=feature.shape[-3:], mode="nearest")
            numerator = F.adaptive_avg_pool3d(feature * valid_level.to(feature.dtype), shape)
            denominator = F.adaptive_avg_pool3d(valid_level, shape)
            pooled = numerator / denominator.clamp_min(1e-6).to(numerator.dtype)
            valid = denominator.flatten(1) > 0
            tokens = self.level_projections[level](pooled.flatten(2).transpose(1, 2))
            world = feature_world_grid(shape, context["affine"], stride,
                                       context["offsets_vox"][level], tuple(feature.shape[-3:]))
            tokens = tokens + self.position(world - context["position_origin_mm"][:, None]).to(tokens.dtype)
            tokens = tokens + self.context_level_embedding[level - 2].to(tokens.dtype)
            memories.append(tokens.masked_fill(~valid[..., None], 0.))
            validities.append(valid)
        raw_memory = torch.cat(memories, 1)
        raw_valid = torch.cat(validities, 1)
        memory = self.context_slots[None].expand(raw_memory.shape[0], -1, -1)
        for block in self.context_aggregators:
            memory = block(memory, raw_memory, raw_valid)
        observed = raw_valid.any(-1)
        memory_valid = observed[:, None].expand(-1, self.config.context_tokens)
        memory = memory.masked_fill(~memory_valid[..., None], 0.)
        reference = context["position_origin_mm"][:, None].expand(-1, self.config.context_tokens, -1)
        return memory, memory_valid, reference

    def _multi_tile_memory(self, contexts: list[dict], anchor: Tensor) -> tuple[Tensor, Tensor]:
        """Sample one bounded physical memory across all tiles; overlap is fused."""
        corners = []
        for context in contexts:
            support = context["valid_mask"][0, 0]
            bounds = []
            for axis in (2, 1, 0):
                active = support.any(tuple(i for i in range(3) if i != axis)).nonzero().flatten()
                bounds.append(torch.stack((active[0], active[-1])).to(anchor))
            grid = torch.cartesian_prod(*bounds)
            with torch.autocast(device_type=anchor.device.type, enabled=False):
                world = grid.float() @ context["affine"][0, :3, :3].float().transpose(-1, -2) + context["affine"][0, :3, 3].float()
            corners.append(world)
        corners = torch.cat(corners)
        lower, upper = corners.amin(0), corners.amax(0)
        span = (upper - lower).clamp_min(1.)
        shape = bounded_shape(tuple(max(1, int(v / 8) + 1) for v in span.flip(0).tolist()), self.config.context_tokens)
        axes = [lower[2 - axis] + (torch.arange(count, device=anchor.device) + .5) / count * span[2 - axis]
                for axis, count in enumerate(shape)]
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        world = torch.stack((xx, yy, zz), -1).reshape(1, 1, -1, 3)
        # Guarantee one actual observed key even for a narrow oblique support.
        support = contexts[0]["valid_mask"][0, 0]
        middle = support.nonzero()[support.sum().item() // 2].flip(0).float()
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            observed = middle.float() @ contexts[0]["affine"][0, :3, :3].float().transpose(-1, -2) + contexts[0]["affine"][0, :3, 3].float()
        world[:, :, 0] = observed
        total, coverage = None, None
        for context in contexts:
            sampled = physical_sample(context["features"][2], world, context["affine"],
                                      context["strides"][2], context["offsets_vox"][2])
            valid = physical_sample(context["valid_mask"].float(), world, context["affine"]) > .999
            # Smooth centre weighting reduces edge-padding influence. Physical
            # overlap contributes once to memory, independent of tile count.
            with torch.autocast(device_type=anchor.device.type, enabled=False):
                inverse = torch.linalg.inv(context["affine"].float())[0]
                vox = torch.einsum("ij,bnpj->bnpi", inverse[:3, :3], world.float()) + inverse[:3, 3]
            shape_xyz = vox.new_tensor(tuple(reversed(context["volume"].shape[-3:])))
            relative = (vox + .5) / shape_xyz * 2 - 1
            weight = torch.exp(-2 * relative.square().sum(-1, keepdim=True)) * valid.to(sampled.dtype)
            total = sampled * weight if total is None else total + sampled * weight
            coverage = weight if coverage is None else coverage + weight
        memory_valid = coverage[:, 0, :, 0] > 0
        memory = self.context_projection((total / coverage.clamp_min(1e-8))[:, 0])
        memory = memory + self.position(world[:, 0] - anchor[:, None]).to(memory.dtype)
        memory = memory.masked_fill(~memory_valid[..., None], 0.)
        for block in self.context_blocks:
            memory = block(memory, memory_valid)
        return memory, memory_valid

    def _seed(self, context: dict, memory: Tensor, valid: Tensor, world: Tensor) -> Tensor:
        heatmap = self.heatmap(context["decoded"])
        foreground = context["mask_logits"].sigmoid()
        # Proposal positions use a finer lattice than the global context budget;
        # a 256-cell whole-scan memory cannot separate adjacent vertebrae.
        shape = bounded_shape(tuple(context["decoded"].shape[-3:]), min(4096, self.config.num_queries * 64))
        valid_weight = F.adaptive_avg_pool3d(context["valid_mask"].float(), shape)
        pooled = F.adaptive_avg_pool3d(context["decoded"] * context["valid_mask"].to(context["decoded"].dtype), shape)
        pooled = pooled / valid_weight.clamp_min(1e-6).to(pooled.dtype)
        proposal_memory = self.proposal_projection(pooled.flatten(2).transpose(1, 2))
        valid = valid_weight.flatten(1) > 0
        world = feature_world_grid(shape, context["affine"], 1., 0., tuple(context["decoded"].shape[-3:]))
        proposal_memory = proposal_memory + self.position(world - context["position_origin_mm"][:, None]).to(proposal_memory.dtype)
        evidence = F.adaptive_avg_pool3d(heatmap.sigmoid() * foreground * context["valid_mask"], shape)
        evidence = (evidence / valid_weight.clamp_min(1e-6)).flatten(1)
        score = self.proposal_score(proposal_memory).squeeze(-1) + evidence
        attention = torch.einsum("nq,bkq->bnk", self.proposal_query, proposal_memory) / math.sqrt(proposal_memory.shape[-1])
        # Physical nonmaximum suppression selects separated neighbourhoods;
        # each local soft argmax is trained by downstream geometric losses.
        used = torch.zeros_like(valid)
        previous = []
        chosen = []
        for slot in range(self.config.num_queries):
            candidate = score + attention[:, slot]
            if previous:
                distances = torch.cdist(world.float(), torch.stack(previous, 1).float()).amin(-1)
                candidate = candidate - torch.exp(-distances.square() / (2 * 16.**2)) * 4.
            candidates_valid = valid & ~used
            if (~candidates_valid.any(-1)).any():
                candidates_valid = torch.where(candidates_valid.any(-1)[:, None], candidates_valid, valid)
            index = candidate.masked_fill(~candidates_valid, -torch.inf).argmax(-1)
            selected = world[torch.arange(world.shape[0], device=world.device), index]
            # Local soft argmax retains gradients for the attention proposal and
            # heatmap heads while physical NMS keeps separate instances apart.
            distance = torch.linalg.vector_norm(world - selected[:, None], dim=-1)
            neighbourhood = (distance <= 24.) & valid
            local_score = candidate.masked_fill(~neighbourhood, -torch.inf)
            local_weights = local_score.softmax(-1)
            refined = torch.einsum("bk,bkc->bc", local_weights, world)
            chosen.append(refined)
            previous.append(selected)
            used.scatter_(1, index[:, None], True)
        return torch.stack(chosen, 1)

    def _points(self, context: dict, queries: Tensor, centers: Tensor,
                radius_mm: float) -> tuple[Tensor, Tensor, Tensor]:
        batch, count, _, width = queries.shape
        learned = self.point_offsets(queries.mean(2)).reshape(batch, count, self.config.query_points, 3).tanh() * radius_mm
        world = centers[:, :, None] + self.fixed_offsets_mm + learned
        sampled = []
        for i, (feature, stride, offset) in enumerate(zip(context["features"], context["strides"], context["offsets_vox"])):
            value = physical_sample(feature, world, context["affine"], stride, offset)
            sampled.append(self.level_projections[i](value.to(queries.dtype)))
        valid = physical_sample(context["valid_mask"].float(), world, context["affine"]).squeeze(-1) > .999
        return torch.stack(sampled, -2), valid, world - centers[:, :, None]

    def _heads(self, typed: Tensor, centers: Tensor, anchors: Tensor) -> dict:
        queries = self.output_norm(typed.mean(2))
        widths = F.softplus(self.extent(typed[:, :, 0]).float()) * self.extent_scale_mm
        lower, upper = widths.chunk(2, -1)
        return {"queries": queries, "typed_queries": typed,
                "centers_mm": centers, "anchors_mm": anchors,
                "class_logits": self.classifier(self.output_norm(typed[:, :, 1])),
                "visible_bounds_mm": torch.stack((centers - lower, centers + upper), -2),
                "visibility_logits": self.visibility(queries)}

    def forward_global(self, context: dict) -> dict:
        memory, memory_valid, world = self._memory(context)
        centers = self._seed(context, memory, memory_valid, world)
        anchors = centers.clone()
        typed = self.slot_content[None].expand(memory.shape[0], -1, -1, -1)
        typed = typed + self.position(centers - context["position_origin_mm"][:, None])[:, :, None].to(typed.dtype)
        for block in self.global_blocks:
            points, point_valid, relative = self._points(context, typed, centers, 12.)
            position = self.position(centers - context["position_origin_mm"][:, None]).to(typed.dtype)
            typed, step = block(typed, memory, memory_valid, points, point_valid, position, 16., relative)
            centers = centers + step
        output = self._heads(typed, centers, anchors)
        output.update(memory=memory, memory_valid=memory_valid, memory_world_mm=world,
                      memory_is_spatial=False, memory_reference_mm=context["position_origin_mm"],
                      context=context, heatmap_logits=self.heatmap(context["decoded"]),
                      mask_logits=context["mask_logits"])
        return output

    def refine_instances(self, global_output: dict, selected: Tensor,
                         tile_contexts: list[dict], tile_owners: Tensor) -> dict:
        """Refine M selected instances using J singleton-batch CT tile contexts.

        selected[M,2] identifies (scan batch, query index); tile_owners[J] maps
        each tile to an index in selected. One fixed anchor and one latent
        predictor remain shared across every tile belonging to that instance.
        """
        if selected.ndim != 2 or selected.shape[-1] != 2 or len(tile_contexts) != len(tile_owners):
            raise ValueError("Expected selected[M,2] and one owner per tile")
        if selected.shape[0] == 0:
            raise ValueError("At least one selected instance is required")
        if any(ctx["volume"].shape[0] != 1 for ctx in tile_contexts):
            raise ValueError("Each tile context must have batch size one")
        if (tile_owners < 0).any() or (tile_owners >= selected.shape[0]).any():
            raise ValueError("Tile owner index outside selected instances")
        device = global_output["queries"].device
        selected = selected.to(device=device, dtype=torch.long)
        tile_owners = tile_owners.to(device=device, dtype=torch.long)
        outputs = []
        memory_lists, valid_lists = [], []
        for owner, (scan, query) in enumerate(selected.tolist()):
            typed = global_output["typed_queries"][scan:scan + 1, query:query + 1]
            centers = global_output["centers_mm"][scan:scan + 1, query:query + 1]
            anchors = global_output["anchors_mm"][scan:scan + 1, query:query + 1]
            indices = (tile_owners == owner).nonzero().flatten().tolist()
            if not indices:
                raise ValueError("Each selected instance needs at least one tile")
            local_memory, local_valid = self._multi_tile_memory([tile_contexts[j] for j in indices], anchors[:, 0])
            # Keep direct whole-scan context for identity, neighbours, and the
            # final latent/diffusion conditioner, beyond its initial query trace.
            memory = torch.cat((global_output["memory"][scan:scan + 1], local_memory), 1)
            memory_valid = torch.cat((global_output["memory_valid"][scan:scan + 1], local_valid), 1)
            # The same world sample is read from every covering tile. Average
            # overlapping CT evidence, rather than repeating a geometry query.
            for block in self.local_blocks:
                point_sum = None
                counts = None
                for j in indices:
                    point, valid, relative = self._points(tile_contexts[j], typed, centers, 6.)
                    weight = valid[..., None, None].to(point.dtype)
                    point_sum = point * weight if point_sum is None else point_sum + point * weight
                    counts = weight if counts is None else counts + weight
                points = point_sum / counts.clamp_min(1)
                valid = counts[..., 0, 0] > 0
                position = self.position(centers - anchors[:, :, :]).to(typed.dtype)
                typed, step = block(typed, memory, memory_valid, points, valid, position, 4., relative)
                centers = centers + step
            output = self._heads(typed, centers, anchors)
            outputs.append({key: value.squeeze(1) for key, value in output.items()})
            memory_lists.append(memory.squeeze(0))
            valid_lists.append(memory_valid.squeeze(0))
        result = {key: torch.cat([output[key] for output in outputs], 0) for key in outputs[0]}
        length = max(value.shape[0] for value in memory_lists)
        result["memory"] = torch.stack([F.pad(value, (0, 0, 0, length - value.shape[0])) for value in memory_lists])
        result["memory_valid"] = torch.stack([F.pad(value, (0, length - value.shape[0]), value=False) for value in valid_lists])
        result["selected"] = selected
        return result


class LatentAnchor(nn.Module):
    """Fixed learned slots preserve latent-index correspondence across cases."""
    def __init__(self, config: V1Config):
        super().__init__()
        width = config.query_dim
        self.slots = nn.Parameter(torch.randn(config.latent_tokens, width) * .02)
        self.blocks = nn.ModuleList(CrossBlock(width, config.heads, config.ffn_hidden)
                                   for _ in range(config.latent_predictor_layers))
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, config.latent_dim)
        self.contrastive_head = nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                             nn.Linear(width, config.contrastive_dim))

    def forward(self, queries: Tensor, memory: Tensor,
                memory_valid: Tensor | None = None) -> Tensor:
        if queries.ndim != 2 or memory.ndim != 3 or queries.shape[0] != memory.shape[0]:
            raise ValueError("Expected instance queries[B,Q] and memory[B,K,Q]")
        latent = self.slots[None] + queries[:, None]
        for block in self.blocks:
            latent = block(latent, memory, memory_valid)
        return self.output(self.norm(latent))

    def contrastive(self, features: Tensor) -> Tensor:
        return F.normalize(self.contrastive_head(features), dim=-1)
