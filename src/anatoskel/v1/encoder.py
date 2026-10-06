"""Shared VoCo-compatible Swin encoder and CT detail decoder.

The one-channel Transformer input is the pretrained window in [0,1]. Wide-HU
input is an explicit separate branch. Loading a partial encoder checkpoint
returns measured coverage, never silently labels a random decoder pretrained.
"""
from __future__ import annotations

import math
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from monai.networks.nets import SwinUNETR

from .config import V1Config


class SafeInstanceNorm3d(nn.InstanceNorm3d):
    """Preserve pretrained parameters and define a one-voxel affine limit."""
    def forward(self, x: Tensor) -> Tensor:
        if math.prod(x.shape[-3:]) == 1:
            if self.affine:
                return x * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]
            return x
        return super().forward(x)


def _safe_norms(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.InstanceNorm3d):
            norm = SafeInstanceNorm3d(child.num_features, eps=child.eps,
                                      momentum=child.momentum, affine=child.affine,
                                      track_running_stats=child.track_running_stats)
            norm.load_state_dict(child.state_dict())
            setattr(module, name, norm)
        else:
            _safe_norms(child)


def _groups(channels: int) -> int:
    # Keep at least two channels per group even at a one-voxel bottleneck.
    return next(g for g in (8, 4, 2, 1) if channels % g == 0 and channels // g >= 2)


class DetailResidual(nn.Module):
    def __init__(self, incoming: int, outgoing: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(incoming, outgoing, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(outgoing), outgoing)
        self.conv2 = nn.Conv3d(outgoing, outgoing, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(outgoing), outgoing)
        self.skip = nn.Identity() if incoming == outgoing and stride == 1 else nn.Conv3d(incoming, outgoing, 1, stride=stride, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        value = F.silu(self.norm1(self.conv1(x)))
        return F.silu(self.norm2(self.conv2(value)) + self.skip(x))


def valid_physical_center(valid: Tensor, affine: Tensor) -> Tensor:
    """Centre of valid support, independent of padding or image translation."""
    with torch.autocast(device_type=valid.device.type, enabled=False):
        weights = valid.float().squeeze(1)
        mass = weights.sum((-3, -2, -1)).clamp_min(1.)
        coordinates = []
        for axis, size in enumerate(weights.shape[-3:]):
            marginal = weights.sum(tuple(i + 1 for i in range(3) if i != axis))
            grid = torch.arange(size, device=weights.device, dtype=torch.float32)
            coordinates.append((marginal * grid).sum(-1) / mass)
        xyz = torch.stack(tuple(reversed(coordinates)), -1)
        return torch.einsum("bij,bj->bi", affine.float()[:, :3, :3], xyz) + affine.float()[:, :3, 3]


class CTEncoder(nn.Module):
    def __init__(self, config: V1Config):
        super().__init__()
        self.config = config.validate()
        backbone = SwinUNETR(in_channels=1, out_channels=1,
                             feature_size=config.feature_size,
                             depths=config.swin_depths, num_heads=config.swin_heads,
                             window_size=config.window_size, use_v2=True,
                             use_checkpoint=config.gradient_checkpointing,
                             downsample="merging")
        # These exact prefixes match the official VoCo encoder. Its discarded
        # decoder/head have no role in the custom high-resolution field branch.
        for name in ("swinViT", "encoder1", "encoder2", "encoder3", "encoder4", "encoder10"):
            setattr(self, name, getattr(backbone, name))
        _safe_norms(self)
        self.normalize = backbone.normalize
        fs = config.feature_size
        bone_full = max(8, fs // 2)
        self.bone_branch = nn.Sequential(DetailResidual(2, bone_full),
                                         DetailResidual(bone_full, bone_full))
        self.bone_down = DetailResidual(bone_full, fs)
        self.bone_down_refine = DetailResidual(fs, fs)
        # Upsample S32->S16->S8->S4->S2->S1. All decoder norms are GN/SiLU.
        decoder_widths = [max(config.detail_channels, round(v * fs / 48))
                          for v in (256, 128, 64, 48)] + [config.detail_channels]
        skip_channels = [fs * 8, fs * 4, fs * 2, fs * 2, fs + bone_full]
        previous = fs * 16
        self.decoder = nn.ModuleList()
        for channels, skip in zip(decoder_widths, skip_channels):
            self.decoder.append(DetailResidual(previous + skip, channels))
            previous = channels
        self.mask_head = nn.Conv3d(config.detail_channels, 1, 1)
        self.tsdf_head = nn.Conv3d(config.detail_channels, 1, 1)
        self.pretrained_report: dict | None = None
        if config.pretrained_encoder:
            self.load_voco_checkpoint(config.pretrained_encoder)

    def _block(self, module: nn.Module, *inputs: Tensor) -> Tensor:
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *inputs, use_reentrant=False, preserve_rng_state=False)
        return module(*inputs)

    def _decode(self, block: nn.Module, value: Tensor, skip: Tensor) -> Tensor:
        # Recompute interpolation and concatenation too: retaining their large
        # full-resolution temporary would defeat most decoder memory savings.
        def forward(previous, residual):
            up = F.interpolate(previous, size=residual.shape[-3:], mode="trilinear", align_corners=False)
            return block(torch.cat((up, residual), 1))
        return self._block(forward, value, skip)

    def forward(self, volume: Tensor, affine: Tensor, valid_mask: Tensor | None = None,
                bone_volume: Tensor | None = None) -> dict:
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError("CT must be [B,1,D,H,W]")
        if affine.shape != (volume.shape[0], 4, 4):
            raise ValueError("Affine must be [B,4,4], mapping XYZ voxel centres to RAS mm")
        if any(v % 32 for v in volume.shape[-3:]):
            raise ValueError("Swin volume spatial dimensions must be divisible by 32")
        valid_mask = torch.ones_like(volume, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
        bone_volume = volume if bone_volume is None else bone_volume
        if valid_mask.shape != volume.shape or bone_volume.shape != volume.shape:
            raise ValueError("Valid support and wide-HU input must match CT")
        if not valid_mask.flatten(1).any(-1).all():
            raise ValueError("Each input must contain at least one observed CT voxel")
        if not torch.isfinite(volume).all() or not torch.isfinite(bone_volume).all() or not torch.isfinite(affine).all():
            raise ValueError("CT and affine must be finite")
        affine = affine.to(device=volume.device, dtype=torch.float32)
        masked = volume * valid_mask.to(volume.dtype)
        hidden = self.swinViT(masked, self.normalize)
        full = self._block(self.encoder1, masked)
        features = [self._block(self.encoder2, hidden[0]), self._block(self.encoder3, hidden[1]),
                    self._block(self.encoder4, hidden[2]), hidden[3], self._block(self.encoder10, hidden[4])]
        bone = self._block(self.bone_branch, torch.cat((bone_volume * valid_mask.to(bone_volume.dtype),
                                                       valid_mask.to(bone_volume.dtype)), 1))
        bone_s2 = self._block(self.bone_down_refine, self._block(self.bone_down, F.avg_pool3d(bone, 2, 2)))
        skips = [features[3], features[2], features[1],
                 torch.cat((features[0], bone_s2), 1), torch.cat((full, bone), 1)]
        decoded = features[-1]
        for block, skip in zip(self.decoder, skips):
            decoded = self._decode(block, decoded, skip)
        decoded = decoded * valid_mask.to(decoded.dtype)
        return {"volume": volume, "bone_volume": bone_volume, "affine": affine,
                "valid_mask": valid_mask, "features": features,
                "strides": [2, 4, 8, 16, 32],
                "offsets_vox": [.5, 1.5, 3.5, 7.5, 15.5],
                "decoded": decoded, "mask_logits": self.mask_head(decoded),
                "tsdf_mm": self.tsdf_head(decoded),
                "position_origin_mm": valid_physical_center(valid_mask, affine)}

    def load_voco_checkpoint(self, path: str | Path, minimum_coverage: float = .95) -> dict:
        """Load only official encoder tensors, with shape and coverage checks.

        Accepted containers are state_dict/student/model/raw tensor dicts and
        prefixes module./backbone./student./encoder. Unknown namespaces are not
        guessed. ``minimum_coverage`` applies separately to Swin and CNN skips.
        """
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for container in ("state_dict", "student", "model"):
            if isinstance(payload, dict) and container in payload and isinstance(payload[container], dict):
                payload = payload[container]
                break
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint must contain a tensor state dict")
        expected = self.state_dict()
        allowed = ("swinViT.", "encoder1.", "encoder2.", "encoder3.", "encoder4.", "encoder10.")
        selected, mismatched = {}, []
        for key, value in payload.items():
            if not isinstance(value, Tensor):
                continue
            changed = True
            while changed:
                changed = False
                for prefix in ("module.", "backbone.", "student.", "encoder."):
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        changed = True
                        break
            if not key.startswith(allowed) or key not in expected:
                continue
            if value.shape != expected[key].shape:
                mismatched.append(key)
            else:
                selected[key] = value
        coverage = {}
        for group, prefixes in (("swin", ("swinViT.",)), ("residual_skips", allowed[1:])):
            keys = [key for key in expected if key.startswith(prefixes)]
            total = sum(expected[key].numel() for key in keys)
            coverage[group] = sum(expected[key].numel() for key in keys if key in selected) / max(total, 1)
        if mismatched or any(v < minimum_coverage for v in coverage.values()):
            raise ValueError(f"Incompatible/incomplete VoCo encoder: coverage={coverage}; shape mismatches={mismatched[:8]}")
        self.load_state_dict(selected, strict=False)
        self.pretrained_report = {"path": str(path), "coverage": coverage,
                                  "loaded_tensors": len(selected),
                                  "decoder_pretrained": False, "bone_branch_pretrained": False}
        return self.pretrained_report
