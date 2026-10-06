"""One CT encoder, one latent per instance, one fused patient-coordinate field."""
from __future__ import annotations

from dataclasses import replace
import torch
from torch import nn

from anatoskel.models.backend import sample_volume, world_to_voxel, SkipFieldMLP
from .config import V1Config
from .encoder import CTEncoder
from .queries import GeometryQueries, LatentAnchor
from .shape import ShapeAutoencoder
from .diffusion import ResidualDiffusion
from .tiles import blend_residuals, tile_weights, overlap_consistency


class AnatoSKELV1(nn.Module):
    def __init__(self, config=None, *, load_pretrained=True):
        super().__init__()
        self.config = (config or V1Config()).validate()
        encoder_config = self.config if load_pretrained else replace(self.config, pretrained_encoder=None)
        self.encoder = CTEncoder(encoder_config)
        self.geometry = GeometryQueries(self.config)
        self.anchor_predictor = LatentAnchor(self.config)
        self.shape = ShapeAutoencoder(self.config)
        self.diffusion = ResidualDiffusion(self.config)
        self.mask_embedding = nn.Linear(self.config.query_dim, self.config.detail_channels)
        self.distance_embedding = nn.Linear(self.config.query_dim, self.config.detail_channels)
        inputs = self.config.detail_channels+3+self.config.latent_dim+4
        self.detail_field = SkipFieldMLP(inputs, self.config.field_hidden, zero_output=True)
        self.local_contrast_projection = nn.Linear(self.config.detail_channels,
                                                   self.config.contrastive_dim)
        self.register_buffer("shape_trained", torch.tensor(False))
        self.register_buffer("anchor_trained", torch.tensor(False))
        self.register_buffer("ct_geometry_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("diffusion_trained", torch.tensor(False))
        self.register_buffer("adapters_trained", torch.tensor(False))

    def encode(self, volume, affine, valid):
        hu = (volume.float()+1)*1500-1000
        soft_window = ((hu+175)/425).clamp(0, 1)*valid
        return self.encoder(soft_window, affine, valid_mask=valid, bone_volume=volume)

    def detect(self, context):
        result = self.geometry.forward_global(context)
        result["mask_logits"] = torch.einsum(
            "bnc,bcdhw->bndhw", self.mask_embedding(result["queries"]), context["decoded"])
        return result

    def prepare_instances(self, detection, selected, tile_contexts, tile_owners):
        local = self.geometry.refine_instances(detection, selected, tile_contexts, tile_owners)
        local["z_anchor"] = self.anchor_predictor(local["queries"], local["memory"],
                                                 local["memory_valid"])
        local["z"] = local["z_anchor"]
        for context, owner in zip(tile_contexts, tile_owners):
            index = int(owner)
            query = local["queries"][index:index+1]
            context["mask_logits"] = torch.einsum(
                "bc,bcdhw->bdhw", self.mask_embedding(query), context["decoded"])[:, None]
            context["tsdf_mm"] = context["tsdf_mm"] + torch.einsum(
                "bc,bcdhw->bdhw", self.distance_embedding(query), context["decoded"])[:, None]
        return local

    def sample_instances(self, state, *, seed=42, differentiable_tail=False):
        result = dict(state)
        result["z"] = self.diffusion.sample(
            state["z_anchor"], state["memory"], state["memory_valid"], seed=seed,
            adapt=bool(self.adapters_trained) or differentiable_tail,
            differentiable_tail=differentiable_tail)
        return result

    def field(self, state, contexts, points_world, *, shape_memory=None):
        """A single instance [1,P,3]; all contexts share its fixed shape frame."""
        if points_world.ndim != 3 or points_world.shape[0] != 1:
            raise ValueError("Field expects one instance and [1,P,3] world points")
        u = (points_world.float()-state["anchors_mm"][:, None].float())/self.config.shape_scale_mm
        prior = self.shape.decode(state["z"], u, memory=shape_memory).float()
        residuals, weights = [], []
        for context in contexts:
            voxel = world_to_voxel(points_world[:, None], context["affine"])
            detail = sample_volume(context["decoded"], voxel).squeeze(1)
            hu = sample_volume(context["bone_volume"], voxel).squeeze(1)
            visible = sample_volume(context["valid_mask"].float(), voxel,
                                    mode="nearest").squeeze(1).squeeze(-1) > .5
            mask = sample_volume(context["mask_logits"].sigmoid(), voxel).squeeze(1)
            weight, _ = tile_weights(points_world[0], context["affine"][0],
                                      context["volume"].shape[-1])
            latent = state["z"].mean(1)[:, None].expand(-1, points_world.shape[1], -1)
            inputs = torch.cat((detail, u, latent, prior[..., None], hu,
                                visible[..., None].to(detail), mask), -1)
            residuals.append(self.detail_field(inputs)[0])
            weights.append(weight*visible[0])
        if not residuals:
            raise ValueError("At least one observed tile is required")
        residuals, weights = torch.stack(residuals), torch.stack(weights)
        residual, covered = blend_residuals(residuals, weights)
        return {"sdf_mm": prior+residual[None], "prior_mm": prior,
                "valid": covered[None], "tile_loss": overlap_consistency(residuals, weights),
                "residuals": residuals, "weights": weights}

    def configure_stage(self, stage):
        self.requires_grad_(False)
        self.eval()
        if stage == "shape":
            self.anchor_trained.fill_(False)
            self.ct_geometry_steps.zero_()
            self.diffusion_trained.fill_(False)
            self.adapters_trained.fill_(False)
            self.diffusion.reset_statistics()
            self.shape.requires_grad_(True).train()
        elif stage == "ct":
            if not bool(self.shape_trained):
                raise ValueError("Train/load complete shape AE before CT training")
            self.shape.freeze()
            # Updating the CT reference changes the residual distribution.
            self.diffusion_trained.fill_(False)
            self.adapters_trained.fill_(False)
            self.diffusion.reset_statistics()
            for module in (self.encoder, self.geometry, self.anchor_predictor,
                           self.mask_embedding, self.detail_field,
                           self.distance_embedding,
                           self.local_contrast_projection):
                module.requires_grad_(True).train()
        elif stage == "diffusion":
            if not bool(self.anchor_trained):
                raise ValueError("Train/load fixed CT anchor before diffusion")
            self.shape.freeze()
            self.adapters_trained.fill_(False)
            with torch.no_grad():
                for block in self.diffusion.blocks:
                    block.adapter.up.weight.zero_()
                    block.adapter.up.bias.zero_()
            self.diffusion.train_main().train()
        elif stage == "partial":
            if not bool(self.diffusion_trained):
                raise ValueError("Train/load base diffusion before partial adapters")
            self.shape.freeze()
            self.diffusion.train_adapters()
        else:
            raise ValueError("Unknown training stage")
        return [p for p in self.parameters() if p.requires_grad]


def instance_state(state, index):
    """Slice tensors whose first dimension is the instance dimension."""
    count = len(state["z_anchor"])
    return {key: value[index:index+1] if isinstance(value, torch.Tensor)
            and value.ndim and value.shape[0] == count else value
            for key, value in state.items()}
