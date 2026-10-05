"""Rigid anatomical parameters and CT-conditioned implicit bone surfaces.

Coordinates are RAS millimetres. Tensor volumes are ``[B,C,D,H,W]``; their
affines map voxel ``(x=W, y=H, z=D)`` centres to world coordinates. Distance
fields use the configured distance scale, without a final sigmoid or tanh.
The prior is learned from this project's training data, not the SKEL assets.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def rotation_6d_to_matrix(value: Tensor) -> Tensor:
    """Convert two column vectors to SO(3), including degenerate inputs.

    A deterministic orthogonal fallback prevents zero or parallel predictions
    from creating non-finite matrices. The identity encoding is [1,0,0,0,1,0].
    """
    if value.shape[-1] != 6:
        raise ValueError("Rotation encoding must have six components")
    value = value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    first, second = value[..., :3], value[..., 3:]
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    fallback_first = torch.zeros_like(first)
    fallback_first[..., 0] = 1
    first = torch.where(first_norm > 1e-6, first / first_norm.clamp_min(1e-6), fallback_first)
    orthogonal = second - (first * second).sum(-1, keepdim=True) * first
    second_norm = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)
    basis = F.one_hot(first.abs().argmin(-1), num_classes=3).to(first.dtype)
    fallback_second = basis - (basis * first).sum(-1, keepdim=True) * first
    fallback_second = F.normalize(fallback_second, dim=-1, eps=1e-6)
    second = torch.where(second_norm > 1e-6,
                         orthogonal / second_norm.clamp_min(1e-6), fallback_second)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def axis_angle_to_matrix(omega: Tensor) -> Tensor:
    """Rodrigues rotation with a finite derivative at the zero update."""
    omega = omega.float() if omega.dtype in (torch.float16, torch.bfloat16) else omega
    x, y, z = omega.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    skew = skew.reshape(*omega.shape[:-1], 3, 3)
    angle = torch.linalg.vector_norm(omega, dim=-1)
    a = torch.sinc(angle / math.pi)[..., None, None]
    b = (0.5 * torch.sinc(angle / (2 * math.pi)).square())[..., None, None]
    identity = torch.eye(3, dtype=omega.dtype, device=omega.device)
    with torch.autocast(device_type=omega.device.type, enabled=False):
        return identity + a * skew + b * (skew @ skew)


def canonical_to_world(points: Tensor, rotation: Tensor, translation: Tensor) -> Tensor:
    dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=points.device.type, enabled=False):
        return (torch.einsum("bnij,bnpj->bnpi", rotation.to(dtype), points.to(dtype))
                + translation.to(dtype)[..., None, :])


def world_to_canonical(points: Tensor, rotation: Tensor, translation: Tensor) -> Tensor:
    dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=points.device.type, enabled=False):
        return torch.einsum("bnji,bnpj->bnpi", rotation.to(dtype),
                            points.to(dtype) - translation.to(dtype)[..., None, :])


def world_to_voxel(points: Tensor, affine: Tensor) -> Tensor:
    """Inverse full affine; supports oblique, reflected and anisotropic CTs."""
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("World points must be [B,N,P,3]")
    if affine.shape != (points.shape[0], 4, 4):
        raise ValueError("Affine must be [B,4,4]")
    dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=points.device.type, enabled=False):
        inverse = torch.linalg.inv(affine.to(dtype=dtype))
        return (torch.einsum("bij,bnpj->bnpi", inverse[:, :3, :3], points.to(dtype))
                + inverse[:, None, None, :3, 3])


def sample_volume(volume: Tensor, voxel_points: Tensor,
                  stride: float | tuple[float, float, float] = 1,
                  mode: str = "bilinear") -> Tensor:
    """Sample at input-voxel centres, with strided-convolution centre alignment.

    A 3-tuple stride is ordered (z,y,x), matching Conv3d. With odd kernels and
    symmetric padding, feature index j represents input voxel centre j*stride.
    The half-voxel term implements grid_sample's align_corners=False contract.
    """
    if volume.ndim != 5 or volume.shape[0] != voxel_points.shape[0]:
        raise ValueError("Volume must be [B,C,D,H,W] with matching batch")
    if isinstance(stride, (tuple, list)):
        stride_xyz = voxel_points.new_tensor(tuple(reversed(stride)))
    else:
        stride_xyz = voxel_points.new_full((3,), float(stride))
    shape_xyz = voxel_points.new_tensor(tuple(reversed(volume.shape[-3:])))
    grid = 2 * (voxel_points / stride_xyz + 0.5) / shape_xyz - 1
    # Native 3D grid_sample has incomplete low-precision kernel support (in
    # particular BF16). Sample in FP32 even if cached Conv3d features came from
    # AMP. Casts remain differentiable back to the original feature tensors.
    sampling_dtype = torch.float32 if volume.dtype in (torch.float16, torch.bfloat16) \
        else volume.dtype
    grid = grid.to(sampling_dtype).unsqueeze(1)  # [B,1,N,P,3]
    with torch.autocast(device_type=volume.device.type, enabled=False):
        sampled = F.grid_sample(volume.to(sampling_dtype), grid, mode=mode,
                                padding_mode="zeros", align_corners=False)
    return sampled[:, :, 0].permute(0, 2, 3, 1)  # [B,N,P,C]


class SkipFieldMLP(nn.Module):
    """Four Softplus(20) hidden layers with an input skip at hidden layer 3."""

    def __init__(self, input_dim: int, hidden: int, zero_output: bool = False):
        super().__init__()
        self.layers = nn.ModuleList((nn.Linear(input_dim, hidden), nn.Linear(hidden, hidden),
                                     nn.Linear(hidden + input_dim, hidden),
                                     nn.Linear(hidden, hidden)))
        self.activation = nn.Softplus(beta=20)
        self.output = nn.Linear(hidden, 1)
        if zero_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        value = inputs
        for index, layer in enumerate(self.layers):
            if index == 2:
                value = torch.cat((value, inputs), -1)
            value = self.activation(layer(value))
        return self.output(value).squeeze(-1)


class AnatomicalShapePrior(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.coordinate_scale_mm = config.coordinate_scale_mm
        self.num_classes = config.num_classes
        self.beta_dim = config.beta_dim
        self.class_embedding = nn.Embedding(config.num_classes, config.class_dim)
        self.decoder = SkipFieldMLP(3 + config.beta_dim + config.class_dim,
                                    config.field_hidden)

    def forward(self, canonical_points: Tensor, beta: Tensor, class_ids: Tensor) -> Tensor:
        if canonical_points.ndim != 4 or canonical_points.shape[-1] != 3:
            raise ValueError("Canonical points must be [B,N,P,3]")
        if beta.shape != (*canonical_points.shape[:2], self.beta_dim):
            raise ValueError("Shape codes must be [B,N,beta_dim]")
        if class_ids.shape != canonical_points.shape[:2]:
            raise ValueError("Class ids must be [B,N]")
        if torch.any((class_ids < 0) | (class_ids >= self.num_classes)):
            raise ValueError("Prior accepts vertebral classes only, excluding no-object")
        count = canonical_points.shape[-2]
        shape = beta.unsqueeze(-2).expand(-1, -1, count, -1)
        anatomical_class = self.class_embedding(class_ids).unsqueeze(-2)
        anatomical_class = anatomical_class.expand(-1, -1, count, -1)
        inputs = torch.cat((canonical_points / self.coordinate_scale_mm,
                            shape, anatomical_class), -1)
        return self.decoder(inputs)


class SharedRefinement(nn.Module):
    """The same point attention and relative-pose attention run each round."""

    def __init__(self, config: Any):
        super().__init__()
        width = config.query_dim
        point_dim = 4 * config.sample_channels + 3 + 1 + 1 + 1 + 1 + 2
        self.point_projection = nn.Sequential(nn.Linear(point_dim, width), nn.GELU())
        self.point_norm = nn.LayerNorm(width)
        self.cross_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, config.heads, batch_first=True,
                                                    dropout=0)
        self.relation_norm = nn.LayerNorm(width)
        # Relative rotation 9 + translation 3 + each bone's visibility logits 2.
        self.relation_bias = nn.Sequential(nn.Linear(16, width), nn.GELU(),
                                           nn.Linear(width, config.heads))
        self.relation_attention = nn.MultiheadAttention(width, config.heads,
                                                       batch_first=True, dropout=0)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))
        self.delta = nn.Sequential(nn.Linear(width, 128), nn.GELU(),
                                   nn.Linear(128, 6 + config.beta_dim))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.heads = config.heads
        self.coordinate_scale_mm = config.coordinate_scale_mm

    def forward(self, state: dict[str, Tensor], samples: dict[str, Tensor]) -> dict[str, Tensor]:
        query = state["queries"]
        batch, bones, width = query.shape
        point_inputs = torch.cat((samples["features"],
                                  samples["canonical_points"] / self.coordinate_scale_mm,
                                  samples["prior"].unsqueeze(-1),
                                  samples["field"].unsqueeze(-1),
                                  samples["mask_probability"].unsqueeze(-1),
                                  samples["valid"].unsqueeze(-1).to(query.dtype),
                                  samples["hu"]), -1)
        point_tokens = self.point_norm(self.point_projection(point_inputs))
        point_tokens = point_tokens.reshape(batch * bones, -1, width)
        point_valid = samples["valid"].reshape(batch * bones, -1)
        any_valid = point_valid.any(-1)
        # MHA cannot accept a row with every key masked. The synthetic unmasked
        # key produces no update because its resulting attention is zeroed.
        safe_valid = point_valid.clone()
        safe_valid[:, 0] |= ~any_valid
        attention, _ = self.cross_attention(self.cross_norm(query).reshape(batch * bones, 1, width),
                                            point_tokens, point_tokens,
                                            key_padding_mask=~safe_valid, need_weights=False)
        attention = attention * any_valid[:, None, None].to(attention.dtype)
        query = query + attention.reshape(batch, bones, width)

        rotation, translation = state["rotation"], state["translation"]
        with torch.autocast(device_type=query.device.type, enabled=False):
            relative_rotation = torch.einsum("bnki,bmkj->bnmij", rotation.float(), rotation.float())
            difference = translation.float()[:, None, :, :] - translation.float()[:, :, None, :]
            relative_translation = torch.einsum("bnji,bnmj->bnmi", rotation.float(), difference)
        visible = state["visibility_logits"].sigmoid()
        first_visible = visible[:, :, None, :].expand(-1, -1, bones, -1)
        second_visible = visible[:, None, :, :].expand(-1, bones, -1, -1)
        relations = torch.cat((relative_rotation.flatten(-2),
                               relative_translation / self.coordinate_scale_mm,
                               first_visible, second_visible), -1)
        bias = self.relation_bias(relations).permute(0, 3, 1, 2)
        normalized = self.relation_norm(query)
        relation_update, _ = self.relation_attention(normalized, normalized, normalized,
                                                     attn_mask=bias.reshape(batch * self.heads,
                                                                            bones, bones),
                                                     need_weights=False)
        query = query + relation_update
        query = query + self.ffn(self.ffn_norm(query))
        delta = self.delta(query)
        with torch.autocast(device_type=query.device.type, enabled=False):
            updated_rotation = rotation.float() @ axis_angle_to_matrix(delta[..., 3:6]).float()
        result = dict(state)
        result.update(queries=query, translation=translation + delta[..., :3],
                      rotation=updated_rotation,
                      beta=state["beta"] + delta[..., 6:])
        return result


class AnatomicalReconstructor(nn.Module):
    """Independent rigid bone placements plus trainable canonical shape fields.

    The prior is trainable initially. Call ``freeze_prior(pretrained=True)``
    only after loading a prior checkpoint fitted from the training split.
    Ordinary marching cubes is deliberately outside this training module.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.parameters_mlp = nn.Sequential(nn.Linear(config.query_dim, config.query_dim),
                                             nn.GELU(), nn.Linear(config.query_dim, 128),
                                             nn.GELU())
        self.rotation_head = nn.Linear(128, 6)
        self.translation_head = nn.Linear(128, 3)
        self.beta_head = nn.Linear(128, config.beta_dim)
        self.visibility_head = nn.Linear(128, 2)
        for head in (self.rotation_head, self.translation_head, self.beta_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.rotation_head.bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))
        self.prior = AnatomicalShapePrior(config)
        self.feature_projections = nn.ModuleList(
            nn.Conv3d(channels, config.sample_channels, 1) for channels in config.fine_channels[1:])
        residual_dim = (4 * config.sample_channels + 3 + config.beta_dim
                        + config.class_dim + 1 + 2 + 1)
        self.residual = SkipFieldMLP(residual_dim, config.field_hidden, zero_output=True)
        self.refinement = SharedRefinement(config)
        candidate_count = max(4 * config.refinement_points, 128)
        # Fixed candidates make repeated inference reproducible. Cube extent is
        # 65 mm at the default scale and is specific to the vertebra task.
        candidates = torch.quasirandom.SobolEngine(3, scramble=False).draw(candidate_count)
        candidates = (candidates * 2 - 1) * (0.65 * config.coordinate_scale_mm)
        self.register_buffer("canonical_candidates", candidates)
        self.prior_pretrained = False

    def freeze_prior(self, pretrained: bool = False) -> None:
        if not pretrained:
            raise ValueError("Cannot freeze an untrained shape prior; load a verified prior checkpoint")
        self.prior.requires_grad_(False)
        self.prior_pretrained = True

    def unfreeze_prior(self) -> None:
        self.prior.requires_grad_(True)

    def prepare_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Project CT features once for repeated field queries; does not mutate input."""
        if "anatomical_features" in context:
            return context
        features = context["features"]
        if len(features) != 5:
            raise ValueError("Anatomical reconstruction requires five fine encoder feature levels")
        prepared = dict(context)
        prepared["anatomical_features"] = [projection(feature)
                                           for projection, feature in
                                           zip(self.feature_projections, features[1:], strict=True)]
        return prepared

    def forward(self, context: dict[str, Any], queries: Tensor, class_ids: Tensor,
                anchors_mm: Tensor, refinements: bool = True) -> dict[str, Tensor]:
        if queries.ndim != 3 or queries.shape[-1] != self.config.query_dim:
            raise ValueError("Queries must be [B,N,query_dim]")
        if anchors_mm.shape != (*queries.shape[:2], 3) or class_ids.shape != queries.shape[:2]:
            raise ValueError("Anchor or class dimensions do not match queries")
        if queries.shape[1] < 1:
            raise ValueError("Filter empty detections before reconstructing")
        if torch.any((class_ids < 0) | (class_ids >= self.config.num_classes)):
            raise ValueError("Reconstruction excludes the no-object class")
        parameters = self.parameters_mlp(queries)
        state = {"rotation": rotation_6d_to_matrix(self.rotation_head(parameters)),
                 "translation": anchors_mm.float() + self.translation_head(parameters).float(),
                 "beta": self.beta_head(parameters), "queries": queries,
                 "class_ids": class_ids, "visibility_logits": self.visibility_head(parameters)}
        if refinements and self.config.refinement_rounds:
            prepared = self.prepare_context(context)
            for _ in range(self.config.refinement_rounds):
                canonical = self.surface_candidates(state)
                world = canonical_to_world(canonical, state["rotation"], state["translation"])
                samples = self.field(prepared, state, world)
                state = self.refinement(state, samples)
        return state

    @torch.no_grad()
    def surface_candidates(self, state: dict[str, Tensor]) -> Tensor:
        """Select low-|prior| canonical candidates without differentiating top-k."""
        batch, bones = state["class_ids"].shape
        candidates = self.canonical_candidates[None, None].expand(batch, bones, -1, -1)
        values = []
        # Chunking bounds temporary hidden activations for 32-bone scans.
        for chunk in candidates.split(1024, dim=2):
            values.append(self.prior(chunk, state["beta"], state["class_ids"]).abs())
        nearest = torch.cat(values, -1).topk(self.config.refinement_points, largest=False).indices
        return candidates.gather(2, nearest[..., None].expand(-1, -1, -1, 3))

    def field(self, context: dict[str, Any], state: dict[str, Tensor],
              points_world: Tensor) -> dict[str, Tensor]:
        prepared = self.prepare_context(context)
        canonical = world_to_canonical(points_world.float(), state["rotation"], state["translation"])
        prior = self.prior(canonical, state["beta"], state["class_ids"])
        batch, bones, count = points_world.shape[:3]
        packed_shape = context.get("bone_batch_shape")
        if packed_shape is not None:
            if tuple(packed_shape) != (batch, bones):
                raise ValueError("Packed ROI bone_batch_shape must match state [B,N]")
            if context["volume"].shape[0] != batch * bones:
                raise ValueError("Packed ROI context requires B*N volumes and affines")
            # Each bone has its own CT crop and affine while relation attention
            # keeps the original patient/bone dimensions [B,N].
            sampling_world = points_world.reshape(batch * bones, 1, count, 3)
            sampling_bones = 1
        else:
            sampling_world = points_world
            sampling_bones = bones
        voxel = world_to_voxel(sampling_world, context["affine"])
        volume = context["volume"]
        shape_xyz = voxel.new_tensor(tuple(reversed(volume.shape[-3:])))
        valid = ((voxel >= -0.5) & (voxel < shape_xyz - 0.5)).all(-1)
        if "valid_mask" in context:
            observed = sample_volume(context["valid_mask"].to(volume.dtype), voxel,
                                     mode="nearest").squeeze(-1) > 0.5
            valid = valid & observed
        strides = context.get("strides", (1, 2, 4, 8, 16))
        features = torch.cat([sample_volume(feature, voxel, stride)
                              for feature, stride in zip(prepared["anatomical_features"],
                                                         strides[1:], strict=True)], -1)
        intensity = sample_volume(volume, voxel)
        # Input CT intensity is clipped [-1000,2000] HU and normalized [-1,1].
        # The second channel is its square, an explicit nonlinear intensity cue.
        hu = torch.cat((intensity, intensity.square()), -1)
        mask_probability = self._mask_probability(context, voxel, sampling_bones)
        if packed_shape is not None:
            features = features.reshape(batch, bones, count, -1)
            hu = hu.reshape(batch, bones, count, -1)
            valid = valid.reshape(batch, bones, count)
            mask_probability = mask_probability.reshape(batch, bones, count)
        beta = state["beta"].unsqueeze(2).expand(-1, -1, count, -1)
        class_embedding = self.prior.class_embedding(state["class_ids"])
        class_embedding = class_embedding.unsqueeze(2).expand(-1, -1, count, -1)
        residual_inputs = torch.cat((features, canonical / self.config.coordinate_scale_mm,
                                     beta, class_embedding, prior.unsqueeze(-1), hu,
                                     valid.unsqueeze(-1).to(features.dtype)), -1)
        # Invalid observations cannot deform anatomy; callers must also ignore
        # invalid field targets and avoid exporting invented FOV boundary caps.
        residual = self.residual(residual_inputs) * valid.to(prior.dtype)
        field = prior + residual
        return {"field": field, "prior": prior, "residual": residual, "valid": valid,
                "features": features, "hu": hu, "canonical_points": canonical,
                "mask_probability": mask_probability}

    @staticmethod
    def _mask_probability(context: dict[str, Any], voxel: Tensor, bones: int) -> Tensor:
        logits = context.get("mask_logits")
        if logits is None:
            return voxel.new_zeros(voxel.shape[:-1])
        if logits.ndim != 5 or logits.shape[1] not in (1, bones):
            raise ValueError("mask_logits must be [B,N,D,H,W] or [B,1,D,H,W]")
        stride = context.get("mask_stride", 1)
        sampled = sample_volume(logits.sigmoid(), voxel, stride)
        if logits.shape[1] == 1:
            return sampled.squeeze(-1)
        return sampled.diagonal(dim1=1, dim2=3).permute(0, 2, 1)
